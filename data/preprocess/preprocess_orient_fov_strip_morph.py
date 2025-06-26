"""
Preprocess + W&B logging (project=thesis_preprocess), resume support,
and restart‑aware logic that only checks for the final image.
"""
import argparse
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import ants
import torch
import wandb

# → log into W&B up‑front
wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")

NUM_GPUS = torch.cuda.device_count()
_gpu_sem = threading.BoundedSemaphore(1)
CACHE_FILE_NAME = "image_paths.txt"


def run_cmd(cmd, **kw):
    logging.debug("CMD: " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def check_and_swap_orientation(log, inp, out):
    out.parent.mkdir(exist_ok=True, parents=True)
    orient = subprocess.check_output(
        ["fslorient", "-getorient", str(inp)]
    ).decode().strip()
    if orient == "NEUROLOGICAL":
        log.info(f"Swapping orient: {inp.name}")
        out.write_bytes(inp.read_bytes())
        run_cmd(["fslorient", "-swaporient", str(out)])
    else:
        if inp != out:
            log.info(f"Copy (no swap): {inp.name} → {out.name}")
            out.write_bytes(inp.read_bytes())
        else:
            log.info(f"No orient change: {inp.name}")
    return out


def run_robust_fov(log, inp, out):
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"robustfov: {inp.name}")
    run_cmd(["robustfov", "-i", str(inp), "-r", str(out)])
    return out


def run_denoise(log, inp, out):
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Denoise: {inp.name}")
    img = ants.image_read(str(inp))
    den = ants.denoise_image(img, noise_model="Rician")
    ants.image_write(den, str(out))
    return out


def run_n4(log, inp, out):
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"N4: {inp.name}")
    img = ants.image_read(str(inp))
    corr = ants.n4_bias_field_correction(img)
    ants.image_write(corr, str(out))
    return out


def run_synthstrip(log, inp, out):
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthStrip: {inp.name}")
    with _gpu_sem:
        run_cmd([
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{inp.parent}:/data",
            "freesurfer/synthstrip:1.7-gpu",
            "-i", f"/data/{inp.name}",
            "-o", f"/data/{out.name}",
            "-g"
        ])
    return out


def run_synthmorph_affine(log, inp, mni, out, xfm):
    out.parent.mkdir(exist_ok=True, parents=True)
    xfm.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthMorph affine: {inp.name}")
    with _gpu_sem:
        run_cmd([
            "docker", "run", "--rm", "--gpus", "all",
            "-e", "TF_CPP_MIN_LOG_LEVEL=2",
            "-v", f"{inp.parent}:/moving",
            "-v", f"{mni.parent}:/fixed",
            "freesurfer/synthmorph", "register",
            "-g",
            "-m", "affine",
            "-o", f"/moving/{out.name}",
            "-t", f"/moving/{xfm.name}",
            f"/moving/{inp.name}", f"/fixed/{mni.name}"
        ])
    return out


def find_images(root):
    """
    Walk `root` recursively, logging each candidate and yielding it.
    Only includes T1w .nii.gz files that do NOT contain 'ses-2'.
    """
    log = logging.getLogger()
    log.info(f"Scanning directory for images: {root}")
    for p in root.rglob("*"):
        if (p.name.endswith(".nii.gz") and 
            "T1w" in p.name and 
            "ses-2" not in p.name):
            log.info(f"Found image: {p}")
            yield p


def strip_suffixes(n):
    for s in (".nii.gz", ".nii", ".gz"):
        if n.endswith(s):
            return n[:-len(s)]
    return n


def process_image(img, base, out_root, mni):
    log = logging.getLogger()
    stem = strip_suffixes(img.name)
    
    # Final output file
    final = out_root / img.name
    
    # Skip if final output already exists
    if final.exists():
        log.info(f"Skipping (final exists): {final.name}")
        return
    
    # Copy input file to local destination directory for processing
    local_input = out_root / f"{stem}_input.nii.gz"
    log.info(f"Copying {img.name} to local directory for processing")
    shutil.copy2(img, local_input)
    
    try:
        # Process using local paths
        o = out_root / f"{stem}_oriented.nii.gz"
        check_and_swap_orientation(log, local_input, o)

        f = out_root / f"{stem}_fov.nii.gz"
        run_robust_fov(log, o, f)

        # DENOISING STEP - COMMENTED OUT
        # d = out_root / f"{stem}_den.nii.gz"
        # run_denoise(log, f, d)

        # N4 BIAS FIELD CORRECTION STEP - COMMENTED OUT
        # n4 = out_root / f"{stem}_n4.nii.gz"
        # run_n4(log, d, n4)

        # Using fov result directly for brain extraction since denoising and n4 are skipped
        b = out_root / f"{stem}_brain.nii.gz"
        run_synthstrip(log, f, b)

        # Final registration step
        xfm = out_root / f"{stem}.lta"
        run_synthmorph_affine(log, b, mni, final, xfm)
        
        log.info(f"Successfully processed: {img.name}")

    finally:
        # Clean up all intermediate files
        intermediate_files = [
            local_input,  # The copied input file
            out_root / f"{stem}_oriented.nii.gz",
            out_root / f"{stem}_fov.nii.gz", 
            out_root / f"{stem}_brain.nii.gz",
            out_root / f"{stem}.lta"
        ]
        
        for temp_file in intermediate_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    log.debug(f"Cleaned up: {temp_file.name}")
            except OSError as e:
                log.warning(f"Could not clean up {temp_file.name}: {e}")


def main(data_root, out_root):
    # Hardcoded paths
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger()
    
    out_root.mkdir(exist_ok=True, parents=True)

    # Cache file for image paths
    cache_file = out_root / CACHE_FILE_NAME

    # Load or build list of images
    if cache_file.exists():
        log.info(f"Loading cached image paths from {cache_file}")
        with cache_file.open("r") as f:
            all_imgs = [Path(line.strip()) for line in f if line.strip()]
        log.info(f"Loaded {len(all_imgs)} image paths from cache")
    else:
        all_imgs = list(find_images(data_root))
        log.info(f"Discovered {len(all_imgs)} images; saving to cache {cache_file}")
        with cache_file.open("w") as f:
            for img in all_imgs:
                f.write(str(img) + "\n")

    total = len(all_imgs)
    if total == 0:
        log.warning(f"No images in {data_root}")
        return

    # only check for final .nii.gz
    pending = [img for img in all_imgs
               if not (out_root / img.name).exists()]

    done0 = total - len(pending)
    log.info(f"{total} images ({done0} done); pending: {len(pending)}")
    log.info(f"CPUs: {os.cpu_count()}, GPUs: {NUM_GPUS}")

    # W&B run
    run = wandb.init(
        project="thesis_preprocess",
        name=f"{out_root.name}",
        id=wandb.util.generate_id(),
        resume="allow",
        config={
            "dataset": out_root.name,
            "total": total,
            "cpus": os.cpu_count(),
            "gpus": NUM_GPUS
        }
    )
    table = wandb.Table(columns=["#", "image", "status", "avg_sec"])
    if done0:
        wandb.log({"images_done": done0, "elapsed_sec": 0}, step=done0)

    start = time.time()
    done = done0
    mni = Path("/mnt/c/Projects/thesis_project/Data/MNI152_T1_1mm_Brain.nii.gz")

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
        futures = {
            exe.submit(process_image, img, data_root.parent, out_root, mni): img
            for img in pending
        }
        for fut in as_completed(futures):
            done += 1
            img = futures[fut]
            try:
                fut.result()
                st = "✔"
                log.info(f"[{done}/{total}] ✔ {img.name}")
            except Exception as e:
                st = "✖"
                log.error(f"[{done}/{total}] ✖ {img.name}: {e}")

            elapsed = time.time() - start
            avg = elapsed / done if done else 0
            eta = (total - done) * avg

            wandb.log({
                "images_done": done,
                "elapsed_sec": elapsed,
                "eta_sec": eta
            }, step=done)

            table.add_data(done, img.name, st, avg)
            if done % 10 == 0 or done == total:
                wandb.log({"table": table}, commit=False)

            log.info(f"Elapsed {timedelta(seconds=int(elapsed))}, "
                     f"ETA {timedelta(seconds=int(eta))}")

    run.finish()
    log.info("✅ All done.")


if __name__ == "__main__":
    data_root = Path("/mnt/c/Projects/thesis_project/Data/CoRR")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/CoRR_")
    main(data_root, out_root)

    data_root = Path("/mnt/c/Projects/thesis_project/Data/SALD")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/SALD_")
    main(data_root, out_root)

    data_root = Path("/mnt/c/Projects/thesis_project/Data/PanGen")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/PanGen_")
    main(data_root, out_root)

    data_root = Path("/mnt/c/Projects/thesis_project/Data/AOMIC_ID1000")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/AOMIC_ID1000_")
    main(data_root, out_root)


