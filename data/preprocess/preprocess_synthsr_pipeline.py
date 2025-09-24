"""
SynthSR + SynthStrip + SynthMorph Pipeline
Preprocess pipeline: SynthSR → SynthStrip → SynthMorph registration
Maintains directory structure and includes W&B logging.
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

import wandb

# → log into W&B up‑front
wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")

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


def run_synthsr(log, inp, out, threads=8, use_gpu=True):
    """
    Run SynthSR using Docker with mackenzieasnyder/synthsr:latest
    """
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthSR: {inp.name}")
    
    # Convert paths to Docker-compatible format
    in_dir = inp.parent
    out_dir = out.parent
    in_dir_abs = in_dir.absolute()
    out_dir_abs = out_dir.absolute()

    cmd = ["docker", "run", "--rm"]
    
    # Add GPU support if requested
    if use_gpu:
        cmd.extend(["--gpus", "all"])
    
    # Use the correct command structure for this Docker image
    cmd.extend([
        "-v", f"{in_dir_abs}:/input",
        "-v", f"{out_dir_abs}:/output",
        "mackenzieasnyder/synthsr:latest",
        "python", "./scripts/predict_command_line.py",
        f"/input/{inp.name}",
        f"/output/{out.name}",
        "--threads", str(threads)
    ])
    
    # Add --cpu flag if GPU is not requested
    if not use_gpu:
        cmd.append("--cpu")
    
    with _gpu_sem:
        run_cmd(cmd)
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
            "-g",
            "--no-csf"  # No CSF flag
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
    Only includes .nii.gz files that contain 'uni' or 't1' in their filename (case-insensitive),
    but excludes files that contain 'map' in their filename.
    """
    log = logging.getLogger()
    log.info(f"Scanning directory for images: {root}")
    for p in root.rglob("*"):
        if (p.name.lower().endswith(".nii.gz")):
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
    
    # Maintain directory structure from input
    rel_path = img.relative_to(base)
    output_dir = out_root / rel_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Final output file (maintains directory structure)
    final = output_dir / img.name
    
    # Skip if final output already exists
    if final.exists():
        log.info(f"Skipping (final exists): {final}")
        return
    
    # Copy input file to output directory for processing
    local_input = output_dir / f"{stem}_input.nii.gz"
    log.info(f"Copying {img} to {local_input} for processing")
    shutil.copy2(img, local_input)
    
    try:
        # Step 1: Orient
        o = output_dir / f"{stem}_oriented.nii.gz"
        check_and_swap_orientation(log, local_input, o)

        # Step 2: FOV
        f = output_dir / f"{stem}_fov.nii.gz"
        run_robust_fov(log, o, f)

        # Step 3: SynthSR (super-resolution)
        sr = output_dir / f"{stem}_synthsr.nii.gz"
        run_synthsr(log, f, sr)

        # Step 4: SynthStrip (skull stripping)
        b = output_dir / f"{stem}_brain.nii.gz"
        run_synthstrip(log, sr, b)

        # Step 5: SynthMorph (final registration)
        xfm = output_dir / f"{stem}.lta"
        run_synthmorph_affine(log, b, mni, final, xfm)
        
        log.info(f"Successfully processed: {img}")

    finally:
        # Clean up all intermediate files
        intermediate_files = [
            local_input,  # The copied input file
            output_dir / f"{stem}_oriented.nii.gz",
            output_dir / f"{stem}_fov.nii.gz",
            output_dir / f"{stem}_synthsr.nii.gz",
            output_dir / f"{stem}_brain.nii.gz",
            output_dir / f"{stem}.lta"
        ]
        
        cleanup_success = 0
        cleanup_failed = 0
        
        for temp_file in intermediate_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    log.info(f"Cleaned up: {temp_file}")
                    cleanup_success += 1
                else:
                    log.debug(f"File already removed: {temp_file}")
            except Exception as e:
                log.warning(f"Could not clean up {temp_file}: {e}")
                cleanup_failed += 1
        
        if cleanup_failed > 0:
            log.warning(f"Cleanup completed: {cleanup_success} successful, {cleanup_failed} failed")
        else:
            log.info(f"Cleanup completed: {cleanup_success} intermediate files removed")


def main(data_root, out_root):
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

    # Check for final outputs (maintaining directory structure)
    pending = []
    for img in all_imgs:
        rel_path = img.relative_to(data_root)
        final_output = out_root / rel_path
        if not final_output.exists():
            pending.append(img)

    done0 = total - len(pending)
    log.info(f"{total} images ({done0} done); pending: {len(pending)}")
    log.info(f"CPUs: {os.cpu_count()}")

    # W&B run - explicitly set to online mode only
    try:
        run = wandb.init(
            project="thesis_preprocess",
            name=f"{out_root.name}_synthsr_pipeline",
            id=wandb.util.generate_id(),
            resume="allow",
            mode="online",
            config={
                "dataset": out_root.name,
                "pipeline": "SynthSR + SynthStrip + SynthMorph",
                "total": total,
                "cpus": os.cpu_count()
            }
        )
        wandb_enabled = True
        log.info("W&B logging enabled (online mode)")
    except Exception as e:
        log.warning(f"Failed to initialize W&B in online mode: {e}")
        log.info("Continuing without W&B logging")
        run = None
        wandb_enabled = False

    if wandb_enabled:
        table = wandb.Table(columns=["#", "image", "status", "avg_sec"])
        if done0:
            wandb.log({"images_done": done0, "elapsed_sec": 0}, step=done0)

    start = time.time()
    done = done0
    mni = Path("/mnt/c/Projects/thesis_project/Data/MNI152_T1_1mm_Brain.nii.gz")

    with ThreadPoolExecutor(2) as exe:  # Reduced to 2 due to GPU memory constraints
        futures = {
            exe.submit(process_image, img, data_root, out_root, mni): img
            for img in pending
        }
        for fut in as_completed(futures):
            done += 1
            img = futures[fut]
            try:
                fut.result()
                st = "✔"
                log.info(f"[{done}/{total}] ✔ {img}")
            except Exception as e:
                st = "✖"
                log.error(f"[{done}/{total}] ✖ {img}: {e}")

            elapsed = time.time() - start
            avg = elapsed / done if done else 0
            eta = (total - done) * avg

            if wandb_enabled:
                wandb.log({
                    "images_done": done,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta
                }, step=done)

                table.add_data(done, str(img), st, avg)
                if done % 10 == 0 or done == total:
                    wandb.log({"table": table}, commit=False)

            log.info(f"Elapsed {timedelta(seconds=int(elapsed))}, "
                     f"ETA {timedelta(seconds=int(eta))}")

    if wandb_enabled:
        run.finish()
    log.info("✅ All done.")


if __name__ == "__main__":

    data_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed_no_csf/OpenNeuro/")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_mp_rage/OpenNeuro/")
    main(data_root, out_root)
    