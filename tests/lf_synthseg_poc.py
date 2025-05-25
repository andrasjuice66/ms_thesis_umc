#!/usr/bin/env python3
"""
LF-SynthSeg  ·  bootstrap-intensity, multi-modal
Author : ChatGPT · 2025-05

Creates synthetic glioma-like lesions that have realistic,
*modality-specific* contrast: the intensities for each region are
bootstrapped from the target image itself.

Dependencies
------------
pip install nibabel numpy scipy scikit-image tqdm
"""

import os, random
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    affine_transform,
    gaussian_filter,
    distance_transform_edt,
)
from skimage.draw import ellipsoid


# ─────────────────────── user-tunable contrast profiles ──────────────────────
# For each MRI modality we define a gain (brightness factor) applied to the
# bootstrapped voxels of every tumour sub-region.
GAIN = {
    "T1":   {"ED": 0.9, "ET": 1.0, "NCR": 0.5},
    "T2":   {"ED": 1.2, "ET": 1.1, "NCR": 0.4},
    "FLAIR":{"ED": 1.3, "ET": 1.2, "NCR": 0.3},
}

# You can change those numbers to mimic a different look
# (e.g. set ET gain much higher on gad-enhanced T1).

# ────────────────────────────── I/O ───────────────────────────────────────────
def load_nii(path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine, img.header


def save_nii(data, affine, hdr, path):
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, hdr), str(path))


# ───────────────── geometry helpers ───────────────────────────────────────────
def mask_center(mask):
    pts = np.column_stack(np.where(mask > 0))
    return pts.mean(axis=0)


def random_ellipsoid_in(tpl_mask, min_r=4, sigma=1.0, max_tries=100):
    """Soft ellipsoid whose centre is chosen *first*, radii second."""
    dist   = distance_transform_edt(tpl_mask)
    inside = np.column_stack(np.where(tpl_mask))
    if inside.size == 0:
        raise RuntimeError("template mask is empty")

    for _ in range(max_tries):
        cz, cy, cx = inside[np.random.randint(len(inside))]
        room = dist[cz, cy, cx]
        if room < min_r + 1:
            continue

        rx = random.uniform(min_r, 0.9 * room)
        ry = random.uniform(min_r, 0.9 * room)
        rz = random.uniform(min_r, 0.9 * room)

        core = ellipsoid(rx, ry, rz).astype(np.uint8)
        sz   = np.array(core.shape)
        z0, y0, x0 = cz - sz[0]//2, cy - sz[1]//2, cx - sz[2]//2
        z1, y1, x1 = z0 + sz[0], y0 + sz[1], x0 + sz[2]

        if (z0 < 0 or y0 < 0 or x0 < 0 or
            z1 > tpl_mask.shape[0] or y1 > tpl_mask.shape[1] or x1 > tpl_mask.shape[2]):
            continue

        canvas = np.zeros_like(tpl_mask, np.uint8)
        canvas[z0:z1, y0:y1, x0:x1] = core
        return (gaussian_filter(canvas.astype(float), sigma) > 0.1).astype(np.uint8)

    raise RuntimeError("failed to fit ellipsoid in template")


def random_affine(scale=(0.8, 1.2), rot=15):
    theta, phi, gamma = np.deg2rad(np.random.uniform(-rot, rot, 3))
    s = random.uniform(*scale)
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[ np.cos(phi), 0, np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [-np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma),  np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    S  = np.diag([s, s, s, 1])
    return Rz @ Ry @ Rx @ S


def apply_affine(vol, M, order=0):
    c = mask_center(vol)
    T = np.eye(4);  T[:3, 3]  = -c
    Tinv = np.eye(4);  Tinv[:3, 3] = c
    A = Tinv @ M @ T
    return affine_transform(vol, A[:3, :3], offset=A[:3, 3], order=order)


# ─────────────── intensity bootstrap ─────────────────────────────────────────
def bootstrap_like(img, tissue_mask, tumour_mask,
                   jitter=0.05, local=True, margin=20):
    donor = tissue_mask.copy()
    if local:
        z, y, x = np.where(tumour_mask)
        z0, y0, x0 = max(z.min()-margin,0), max(y.min()-margin,0), max(x.min()-margin,0)
        z1, y1, x1 = min(z.max()+margin+1, img.shape[0]), \
                     min(y.max()+margin+1, img.shape[1]), \
                     min(x.max()+margin+1, img.shape[2])
        window = np.zeros_like(tissue_mask, bool)
        window[z0:z1, y0:y1, x0:x1] = True
        donor &= window

    vals = img[donor]
    if vals.size == 0:
        raise RuntimeError("empty donor pool")
    k = tumour_mask.sum()
    sampled = np.random.choice(vals, k, replace=True)
    sampled *= np.random.uniform(1-jitter, 1+jitter, k)
    out = np.zeros_like(img)
    out[tumour_mask>0] = sampled
    return out


# ─────────────────── synthesis for ONE subject ───────────────────────────────
def synthesize_one(vol_t2, tpl_masks, thr=0.7, max_tries=150):
    CSF_tpl, GM_tpl, WM_tpl = tpl_masks
    masks, done = [None]*3, [False]*3      # NCR, ET, ED

    for _ in range(max_tries):
        for i, tpl in enumerate([CSF_tpl, GM_tpl, WM_tpl]):     # NCR / ET / ED
            if done[i]: continue
            m = apply_affine(random_ellipsoid_in(tpl), random_affine())
            if (m & tpl).sum() / (m.sum()+1e-6) >= thr:
                masks[i], done[i] = m, True
        if all(done): break
    else:
        raise RuntimeError("placement failed")

    m_ncr, m_et, m_ed = masks
    I_ed  = bootstrap_like(vol_t2, WM_tpl,  m_ed )
    I_et  = bootstrap_like(vol_t2, GM_tpl,  m_et )
    I_ncr = bootstrap_like(vol_t2, CSF_tpl, m_ncr)
    tumour = 0.6*I_ed + 0.25*I_et + 0.15*I_ncr

    label = np.zeros(vol_t2.shape, np.uint8)
    label[m_ncr>0] = 1; label[m_et>0] = 2; label[m_ed>0] = 3
    return tumour, label


# ───────────────────────── modality-specific synthesis ───────────────────────
def make_tumour_for_modality(mod, subj_vol, tpl_masks, tumour_label):
    gm_mask, wm_mask, csf_mask = tpl_masks[1], tpl_masks[2], tpl_masks[0]

    ED  = bootstrap_like(subj_vol, wm_mask,  tumour_label==3)
    ET  = bootstrap_like(subj_vol, gm_mask,  tumour_label==2)
    NCR = bootstrap_like(subj_vol, csf_mask, tumour_label==1)

    g = GAIN[mod]
    tumour = g["ED"]*ED + g["ET"]*ET + g["NCR"]*NCR
    return tumour


# ─────────────────────────────── main ─────────────────────────────────────────
def main():
    # reproducibility (comment out for random every run)
    random.seed(42); np.random.seed(42)

    os.makedirs(OUT_DIR, exist_ok=True)

    # load subject volumes
    vol_t1, aff, hdr = load_nii(SUBJ_T1)
    vol_t2, _, _     = load_nii(SUBJ_T2)
    vol_flair, _, _  = load_nii(SUBJ_FLAIR)

    # load template tissues
    load_mask = lambda m,t: load_nii(Path(TPL_DIR)/f"{m}_{t}.nii.gz")[0].astype(bool)
    TPL = {m:[load_mask(m,t) for t in ("CSF","GM","WM")] for m in ("T1","T2","FLAIR")}

    # geometry driven by T2
    tumour_T2, tumour_lbl = synthesize_one(vol_t2, TPL["T2"])

    tumour_T1    = make_tumour_for_modality("T1",    vol_t1,    TPL["T1"],    tumour_lbl)
    tumour_FLAIR = make_tumour_for_modality("FLAIR", vol_flair, TPL["FLAIR"], tumour_lbl)

    # paste and save
    save_nii(np.where(tumour_lbl, tumour_T1,    vol_t1),    aff, hdr, Path(OUT_DIR)/"sub-T1_syn.nii.gz")
    save_nii(np.where(tumour_lbl, tumour_T2,    vol_t2),    aff, hdr, Path(OUT_DIR)/"sub-T2_syn.nii.gz")
    save_nii(np.where(tumour_lbl, tumour_FLAIR, vol_flair), aff, hdr, Path(OUT_DIR)/"sub-FLAIR_syn.nii.gz")
    save_nii(tumour_lbl, aff, hdr, Path(OUT_DIR)/"tumour_mask.nii.gz")
    print("Synthetic volumes written to", OUT_DIR)


# ───────────────────── hard-coded paths (edit) ────────────────────────────────
t1_image   = Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/OpenNeuro/BoldVariability/sub-100_T1w.nii.gz")
t2_image   = Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC221031_T2w.nii.gz")
flair_image= Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC210526_T2w.nii.gz")

SUBJ_T1   = t1_image
SUBJ_T2   = t2_image
SUBJ_FLAIR= flair_image
TPL_DIR   = "C:/Projects/thesis_project/brain_age_pred/data/templates"
OUT_DIR   = "C:/Projects/thesis_project/brain_age_pred/data/tumour_simulation_lf_synthseg"

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()