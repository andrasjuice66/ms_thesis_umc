#!/usr/bin/env python3
"""
Preprocess pipeline in Python (concurrent CPU, serialized GPU):
 1. Orientation check & swap (FSL)
 2. robustfov           (FSL)
 3. DenoiseImage        (ANTsPy)
 4. N4BiasFieldCorrection (ANTsPy)
 5. SynthStrip          (Docker, bounded concurrency)
 6. SynthMorph affine   (Docker, bounded concurrency)

Final outputs keep the original filename; intermediates are deleted.
"""
import argparse
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
import torch    
import ants  # ANTsPy

CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA available:", CUDA_AVAILABLE)
NUM_GPUS = int(torch.cuda.device_count())
print("NUM_GPUS: ", NUM_GPUS)
_gpu_sem = threading.BoundedSemaphore(NUM_GPUS)

def run_cmd(cmd, **kwargs):
    logging.debug("CMD: " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)

def check_and_swap_orientation(logger, inp: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    orient = subprocess.check_output(
        ["fslorient", "-getorient", str(inp)]
    ).decode().strip()
    if orient == "NEUROLOGICAL":
        logger.info(f"Swapping orientation: {inp.name}")
        if inp != out:
            out.write_bytes(inp.read_bytes())
        run_cmd(["fslorient", "-swaporient", str(out)])
    else:
        if inp != out:
            logger.info(f"No swap needed; copying → {out.name}")
            out.write_bytes(inp.read_bytes())
        else:
            logger.info(f"No swap needed: {inp.name}")
    return out

def run_denoise_image(logger, inp: Path, out: Path) -> Path:
    """Run DenoiseImage using ANTsPy."""
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Denoising: {inp.name}")
    img = ants.image_read(str(inp))
    den = ants.denoise_image(img, noise_model="Rician")
    ants.image_write(den, str(out))
    logger.info(f"Denoised → {out.name}")
    return out

def run_n4_bias_field_corr(logger, inp: Path, out: Path) -> Path:
    """Run N4BiasFieldCorrection using ANTsPy."""
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"N4 bias‑field corr: {inp.name}")
    img = ants.image_read(str(inp))
    corr = ants.n4_bias_field_correction(img)
    ants.image_write(corr, str(out))
    logger.info(f"N4 → {out.name}")
    return out

def run_robust_fov(logger, inp: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cropping FOV: {inp.name}")
    run_cmd(["robustfov", "-i", str(inp), "-r", str(out)])
    logger.info(f"FOV → {out.name}")
    return out

def run_synthstrip(logger, inp: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    mount_dir = inp.parent
    logger.info(f"SynthStrip: {inp.name}")
    with _gpu_sem:
        run_cmd([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{mount_dir}:/data",
            "freesurfer/synthstrip",
            "-i", f"/data/{inp.name}",
            "-o", f"/data/{out.name}"
        ])
        
    logger.info(f"Skull‑strip → {out.name}")
    return out

def run_synthmorph_affine(logger, inp: Path, mni: Path, out: Path, xfm: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    xfm.parent.mkdir(parents=True, exist_ok=True)
    subj_dir = inp.parent
    mni_dir  = mni.parent
    logger.info(f"SynthMorph affine: {inp.name} → {mni.name}")
    with _gpu_sem:
        # Check GPU before        
        run_cmd([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{subj_dir}:/moving",
            "-v", f"{mni_dir}:/fixed",
            "freesurfer/synthmorph", "register",
            "-m", "affine",
            "-o", f"/moving/{out.name}",
            "-t", f"/moving/{xfm.name}",
            "/moving/" + inp.name,
            "/fixed/"  + mni.name
        ])
        
    
    logger.info(f"Affine → {out.name}, xfm → {xfm.name}")

def find_images(root: Path):
    for p in root.rglob("*"):
        if p.suffixes and p.suffixes[-1] in (".nii", ".gz") \
           and any(mod in p.name.lower() for mod in ("t1", "t2", "flair")):
            yield p

def strip_suffixes(name: str) -> str:
    for suf in (".nii.gz", ".nii", ".gz"):
        if name.endswith(suf):
            return name[:-len(suf)]
    return name

def process_image(img: Path, base_root: Path, out_root: Path, mni_img: Path):
    logger = logging.getLogger()
    original_name = img.name
    stem = strip_suffixes(img.name)

    # 1) Orientation
    oriented = out_root / f"{stem}_oriented.nii.gz"
    check_and_swap_orientation(logger, img, oriented)

    # 2) FOV (moved earlier in the pipeline)
    fov = out_root / f"{stem}_fov.nii.gz"
    run_robust_fov(logger, oriented, fov)
    
    # 3) Denoise (now after FOV)
    denoised = out_root / f"{stem}_den.nii.gz"
    run_denoise_image(logger, fov, denoised)

    # 4) N4 bias‑field correction (now after FOV)
    n4 = out_root / f"{stem}_n4.nii.gz"
    run_n4_bias_field_corr(logger, denoised, n4)

    # 5) Skull‑strip (now uses n4 as input)
    brain = out_root / f"{stem}_brain.nii.gz"
    run_synthstrip(logger, n4, brain)

    # 6) Affine → final
    final = out_root / original_name
    xfm   = out_root / f"{stem}.lta"
    run_synthmorph_affine(logger, brain, mni_img, final, xfm)

    # Cleanup
    for tmp in (oriented, fov, brain, xfm, denoised, n4):
        try:
            tmp.unlink()
            logger.debug(f"Removed intermediate: {tmp.name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not remove {tmp.name}: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_root", type=Path,
                   help="Base folder, e.g. /mnt/neuroRT")
    p.add_argument("dataset", help="Dataset name, e.g. IXI")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger()

    data_root = args.base_root / args.dataset
    out_root  = args.base_root / "brain_age_preprocessed" / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    mni_img = args.base_root / "Standard" / "MNI152_T1_1mm_Brain.nii.gz"
    if not mni_img.exists():
        logger.error(f"MNI template missing: {mni_img}")
        raise FileNotFoundError(mni_img)

    print("Searching for images in ", data_root)
    images = list(find_images(data_root))
    total  = len(images)
    if total == 0:
        logger.warning(f"No images in {data_root}")
        return

    logger.info(f"Found {total} images. CPU threads: {os.cpu_count()}, "
                f"GPU slots: {NUM_GPUS}")
    start = time.time()

    done = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 12) as exe:
        futures = {
            exe.submit(process_image, img, args.base_root, out_root, mni_img): img
            for img in images
        }
        for future in as_completed(futures):
            done += 1
            img = futures[future]
            try:
                future.result()
                logger.info(f"[{done}/{total}] ✔ {img.name}")
            except Exception as e:
                logger.error(f"[{done}/{total}] ✖ {img.name}: {e}")

            elapsed = time.time() - start
            avg     = elapsed / done
            eta     = (total - done) * avg
            logger.info(
                f"Elapsed {timedelta(seconds=int(elapsed))}, "
                f"ETA {timedelta(seconds=int(eta))}"
            )

    logger.info(f"✅ All done. Final files in {out_root}")

if __name__ == "__main__":
    main()