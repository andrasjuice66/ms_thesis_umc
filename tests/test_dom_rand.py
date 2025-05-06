# ──────────────────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────────────────
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImaged, EnsureChannelFirstd

#  << your DomainRandomizer implementation lives in domain_randomizer.py >>
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer

# ──────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────
IMG_PATH = (r"C:\Projects\thesis_project\Data\brain_age_preprocessed\NIMH_RV"
            r"\sub-ON02693_ses-01_acq-MPRAGE_T1w.nii.gz")
OUT_DIR  = "./brain_age_pred/tests/doom_rand_test_imgs"
os.makedirs(OUT_DIR, exist_ok=True)

#  A fixed random seed makes the run (reasonably) reproducible
torch.manual_seed(42)
np.random.seed(42)

#  All augmentation *switches* in DomainRandomizer and the flag that
#  controls the TorchIO path.
TRANSFORM_SWITCHES = [
    "flip",
    "affine",
    "elastic",
    "contrast",
    "gamma",
    "blur",
    "bias",
    "resolution",
    "spike",
    "ghosting",
    "crop",
]

#  Shape the random crop should take when we test it
CROP_SHAPE = (150, 190, 150)   # (z, y, x)
# ──────────────────────────────────────────────────────────────────────────


def load_volume(path: str) -> torch.Tensor:
    """
    Load the NIfTI volume and return a torch tensor with shape (1, 1, Z, Y, X)
    so that it matches MONAI/TorchIO conventions (B, C, Z, Y, X).
    """
    loader = LoadImaged(keys=["img"], image_only=False)
    channel_first = EnsureChannelFirstd(keys=["img"])

    data = loader({"img": path})
    data = channel_first(data)
    vol = data["img"]         # torch.Tensor, shape (1, Z, Y, X)
    vol = vol.unsqueeze(0)    # add batch dimension: (1, 1, Z, Y, X)
    return vol


def build_randomizer(active_flag: str) -> DomainRandomizer:
    """
    Build a DomainRandomizer instance with *only* `active_flag` turned on
    and probability 1.0.  Everything else is disabled.
    """
    # --- 1) start with all switches OFF ----------------------------------
    kwargs = dict(
        enable_flip       = False,
        enable_affine     = False,
        enable_elastic    = False,
        enable_contrast   = False,
        enable_gamma      = False,
        enable_blur       = False,
        enable_bias       = False,
        enable_resolution = False,
        enable_spike      = False,
        enable_ghosting   = False,
        output_shape      = None,     # crop disabled by default
        use_torchio       = True,     # needed for elastic/spike/ghosting
        use_gpu           = False,    # keep it simple; CPU works everywhere
    )

    # --- 2) enable the flag we want to visualise -------------------------
    if active_flag == "flip":        kwargs["enable_flip"]       = True
    elif active_flag == "affine":    kwargs["enable_affine"]     = True
    elif active_flag == "elastic":   kwargs["enable_elastic"]    = True
    elif active_flag == "contrast":  kwargs["enable_contrast"]   = True
    elif active_flag == "gamma":     kwargs["enable_gamma"]      = True
    elif active_flag == "blur":      kwargs["enable_blur"]       = True
    elif active_flag == "bias":      kwargs["enable_bias"]       = True
    elif active_flag == "resolution":kwargs["enable_resolution"] = True
    elif active_flag == "spike":     kwargs["enable_spike"]      = True
    elif active_flag == "ghosting":  kwargs["enable_ghosting"]   = True
    elif active_flag == "crop":
        kwargs["output_shape"] = CROP_SHAPE

    # --- 3) make sure the chosen transform is *certainly* applied --------
    prob_dict = {flag: (1.0 if flag == active_flag else 0.0)
                 for flag in TRANSFORM_SWITCHES}

    # --- 4) create the randomizer ---------------------------------------
    return DomainRandomizer(transform_probs=prob_dict, **kwargs)


def show_side_by_side(orig_vol: torch.Tensor,
                      aug_vol : torch.Tensor,
                      name     : str,
                      slice_id : int = None) -> None:
    """
    Take the centre axial slice (or the one provided) from both 3-D volumes
    and plot them.
    """
    # drop batch & channel -> (Z, Y, X)
    orig = orig_vol.squeeze().cpu().numpy()
    aug  = aug_vol.squeeze().cpu().numpy()

    if slice_id is None:
        slice_id = orig.shape[0] // 2          # axial centre

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
    for ax, img, title in zip(axes,
                              (orig[slice_id], aug[slice_id]),
                              ("Original", name.capitalize())):
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    plt.close()
    print(f"[✓] Saved {name}.png")


def main() -> None:
    print("Loading volume …")
    volume = load_volume(IMG_PATH)

    # loop through every augmentation
    for flag in TRANSFORM_SWITCHES:
        print(f"Applying {flag} …")
        dr = build_randomizer(flag)
        # DomainRandomizer expects a dict with key "image"
        out_dict = dr({"image": volume.squeeze(0)})   # remove batch for DR
        aug_vol  = out_dict["image"].unsqueeze(0)      # add batch back
        show_side_by_side(volume, aug_vol, flag)

    print(f"All figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()