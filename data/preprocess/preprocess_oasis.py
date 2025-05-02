"""
Preprocess + W&B logging (project=thesis_preprocess), resume support,
and restart‑aware logic that only checks for the final image.
Filters images to only include subjects from the provided CSV file.
"""
import argparse
import csv
import logging
import os
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


def load_subject_ids(csv_path):
    """
    Load subject IDs from the CSV file.
    Returns a set of subject IDs for efficient lookup.
    """
    subject_ids = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_ids.add(row['OASIS3_id'])
    return subject_ids


def find_images(root, allowed_subjects=None):
    """
    Walk `root` recursively and yield filtered images:
    - Only includes subjects in allowed_subjects if provided
    - Only includes the lowest session number (d####) for each subject
    - Excludes images with run-02, run-03, or acq-TSE in their names
    """
    log = logging.getLogger()
    log.info(f"Scanning directory for images: {root}")
    
    if allowed_subjects:
        log.info(f"Filtering for {len(allowed_subjects)} allowed subjects")
    
    # Dict to track lowest session per subject
    subject_sessions = {}
    all_image_paths = []
    
    # First pass: find all images and determine lowest session for each subject
    for p in root.rglob("*"):
        if not (p.suffixes and p.suffixes[-1] in (".nii", ".gz")):
            continue
            
        if not any(m in p.name.lower() for m in ("t1", "t2", "flair")):
            continue
            
        if any(exclude in p.name for exclude in ["SCIC_T2w", "SCIC_T1w", "T2star", 
                                                "run-02", "run-03", "acq-TSE"]):
            continue
        
        path_str = str(p)
        
        # Check if subject is allowed
        if allowed_subjects:
            subject_found = False
            for subject_id in allowed_subjects:
                if subject_id in path_str:
                    subject_found = True
                    break
            if not subject_found:
                continue
        
        # Extract subject ID and session from path
        # Path format: .../OASIS3/OAS30559/OAS30559_MR_d0431/...
        path_parts = path_str.split('/')
        subject_id = None
        session_id = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('OAS3') and i+1 < len(path_parts):
                # This is likely the subject ID folder
                subject_id = part
                # Next part should contain session info like "OAS30559_MR_d0431"
                if '_MR_d' in path_parts[i+1]:
                    session_part = path_parts[i+1]
                    # Extract session number
                    try:
                        # Find the part after "_MR_d" and convert to integer
                        session_number = int(session_part.split('_MR_d')[1])
                        session_id = session_number
                        break
                    except (IndexError, ValueError):
                        continue
        
        if subject_id and session_id:
            # Update lowest session
            if subject_id not in subject_sessions or session_id < subject_sessions[subject_id]:
                subject_sessions[subject_id] = session_id
                
            # Store all valid image paths for later filtering
            all_image_paths.append((p, subject_id, session_id))
    
    # Second pass: yield only images from the lowest session for each subject
    for p, subject_id, session_id in all_image_paths:
        if session_id == subject_sessions[subject_id]:
            log.info(f"Found image: {p} (subject: {subject_id}, session: {session_id})")
            yield p


def strip_suffixes(n):
    for s in (".nii.gz", ".nii", ".gz"):
        if n.endswith(s):
            return n[:-len(s)]
    return n


def process_image(img, base, out_root, mni):
    log = logging.getLogger()
    stem = strip_suffixes(img.name)

    o = out_root / f"{stem}_oriented.nii.gz"
    if not o.exists():
        check_and_swap_orientation(log, img, o)

    f = out_root / f"{stem}_fov.nii.gz"
    if not f.exists():
        run_robust_fov(log, o, f)

    # Using fov result directly for brain extraction since denoising and n4 are skipped
    b = out_root / f"{stem}_brain.nii.gz"
    if not b.exists():
        run_synthstrip(log, f, b)  # Changed from n4 to f (fov output)

    final = out_root / img.name
    xfm   = out_root / f"{stem}.lta"
    if not final.exists():
        run_synthmorph_affine(log, b, mni, final, xfm)
    else:
        log.info(f"Skipping affine (final exists): {final.name}")

    # remove intermediates + .lta
    for t in (o, f, b, xfm):  # Removed d and n4 from cleanup list
        try:
            t.unlink()
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_root", type=Path)
    p.add_argument("dataset", help="e.g. CamCAN")
    p.add_argument("--subjects_csv", type=Path, 
                  help="Path to CSV file with subject IDs to include")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger()

    data_root = args.base_root / args.dataset
    out_root = args.base_root / "brain_age_preprocessed" / args.dataset
    out_root.mkdir(exist_ok=True, parents=True)

    # Load subject IDs from CSV if specified
    allowed_subjects = None
    if args.subjects_csv and args.subjects_csv.exists():
        log.info(f"Loading subject IDs from {args.subjects_csv}")
        allowed_subjects = load_subject_ids(args.subjects_csv)
        log.info(f"Loaded {len(allowed_subjects)} subject IDs")

    # Cache file for image paths
    cache_file = out_root / CACHE_FILE_NAME

    # Load or build list of images
    if cache_file.exists() and not allowed_subjects:
        log.info(f"Loading cached image paths from {cache_file}")
        with cache_file.open("r") as f:
            all_imgs = [Path(line.strip()) for line in f if line.strip()]
        log.info(f"Loaded {len(all_imgs)} image paths from cache")
    else:
        # If we're using allowed_subjects, we'll regenerate the cache
        all_imgs = list(find_images(data_root, allowed_subjects))
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
        name=args.dataset,
        id=wandb.util.generate_id(),
        resume="allow",
        config={
            "dataset": args.dataset,
            "total": total,
            "cpus": os.cpu_count(),
            "gpus": NUM_GPUS,
            "filtered": bool(allowed_subjects)
        }
    )
    table = wandb.Table(columns=["#", "image", "status", "avg_sec"])
    if done0:
        wandb.log({"images_done": done0, "elapsed_sec": 0}, step=done0)

    start = time.time()
    done = done0
    mni = args.base_root / "Standard" / "MNI152_T1_1mm_Brain.nii.gz"

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
        futures = {
            exe.submit(process_image, img, args.base_root, out_root, mni): img
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
    main()