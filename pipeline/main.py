#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MRI pipeline (Preprocess + BrainAge) driven by YAML config.
- Preprocess (optional): orientation check -> robust FOV -> SynthSR -> SynthStrip -> SynthMorph (affine to MNI)
- Brain Age: MedNeXt encoder ensemble prediction with optional bias correction

Usage:
  python mri_pipeline.py --config ./config.yaml

Notes:
- Docker images required:
  * mackenzieasnyder/synthsr:latest
  * freesurfer/synthstrip:1.7 (or :1.7-gpu if docker_gpu=true)
  * freesurfer/synthmorph
- FSL: fslorient, robustfov must be on PATH
- Python: torch, monai, torchio, numpy, pyyaml, wandb

W&B:
- API key is hardcoded as requested. For security, consider using environment variables instead.
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from monai.transforms import Compose, LoadImaged, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
import torchio

import torch
import torch.nn as nn
from create_mednext_encoder_v1 import create_mednext_encoder_v1

import yaml

# ===== W&B setup (API key hardcoded as requested) =====
WANDB_API_KEY = "2abdb867a9244072f2237704a3cacc77fa548dd8"

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Ensure local modules (e.g., nnunet_mednext.py) can be imported when placed next to this script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# =========================
# Utilities and shell helpers
# =========================

_docker_sem = threading.BoundedSemaphore(1)

def run_cmd(cmd, **kw):
    logging.debug("CMD: " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)

def check_binary_exists(binary: str):
    try:
        subprocess.run([binary, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        logging.warning(f"Binary not found or not executable: {binary}")

def check_docker_available():
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("Docker is not available on PATH. Please install Docker and try again.")

def strip_suffixes(n: str) -> str:
    for s in (".nii.gz", ".nii", ".gz"):
        if n.endswith(s):
            return n[:-len(s)]
    return n

# =========================
# Preprocess steps (FSL + Docker)
# =========================

def check_and_swap_orientation(log, inp: Path, out: Path) -> Path:
    out.parent.mkdir(exist_ok=True, parents=True)
    orient = subprocess.check_output(["fslorient", "-getorient", str(inp)]).decode().strip()
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

def run_robust_fov(log, inp: Path, out: Path) -> Path:
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"robustfov: {inp.name}")
    run_cmd(["robustfov", "-i", str(inp), "-r", str(out)])
    return out

def run_synthsr(log, inp: Path, out: Path, threads: int = 8, use_gpu: bool = False) -> Path:
    """
    SynthSR via docker: mackenzieasnyder/synthsr:latest
    CPU by default.
    """
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthSR: {inp.name}")
    in_dir_abs = inp.parent.absolute()
    out_dir_abs = out.parent.absolute()

    cmd = ["docker", "run", "--rm"]
    if use_gpu:
        cmd.extend(["--gpus", "all"])
    cmd.extend([
        "-v", f"{in_dir_abs}:/input",
        "-v", f"{out_dir_abs}:/output",
        "mackenzieasnyder/synthsr:latest",
        "python", "./scripts/predict_command_line.py",
        f"/input/{inp.name}",
        f"/output/{out.name}",
        "--threads", str(threads)
    ])
    if not use_gpu:
        cmd.append("--cpu")

    with _docker_sem:
        run_cmd(cmd)
    return out

def run_synthstrip(log, inp: Path, out: Path, use_gpu: bool = False) -> Path:
    """
    SynthStrip via docker.
    - CPU:  freesurfer/synthstrip:1.7
    - GPU:  freesurfer/synthstrip:1.7-gpu (adds --gpus all and -g flag)
    """
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthStrip: {inp.name}")
    image = "freesurfer/synthstrip:1.7"
    cmd = ["docker", "run", "--rm"]
    if use_gpu:
        image = "freesurfer/synthstrip:1.7-gpu"
        cmd.extend(["--gpus", "all"])
    cmd.extend([
        "-v", f"{inp.parent}:/data",
        image,
        "-i", f"/data/{inp.name}",
        "-o", f"/data/{out.name}",
    ])
    if use_gpu:
        cmd.append("-g")
    # No CSF flag
    cmd.append("--no-csf")

    with _docker_sem:
        run_cmd(cmd)
    return out

def run_synthmorph_affine(log, moving: Path, fixed: Path, out: Path, xfm: Path, use_gpu: bool = False) -> Path:
    """
    SynthMorph affine registration via docker. Default CPU (no '-g').
    """
    out.parent.mkdir(exist_ok=True, parents=True)
    xfm.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"SynthMorph affine: moving={moving.name}, fixed={fixed.name}")

    cmd = ["docker", "run", "--rm"]
    if use_gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        "-e", "TF_CPP_MIN_LOG_LEVEL=2",
        "-v", f"{moving.parent}:/moving",
        "-v", f"{fixed.parent}:/fixed",
        "freesurfer/synthmorph",
        "register",
    ]
    if use_gpu:
        cmd += ["-g"]
    cmd += [
        "-m", "affine",
        "-o", f"/moving/{out.name}",
        "-t", f"/moving/{xfm.name}",
        f"/moving/{moving.name}", f"/fixed/{fixed.name}"
    ]

    with _docker_sem:
        run_cmd(cmd)
    return out

# =========================
# Brain Age (MedNeXt encoder)
# =========================

def create_mednext_encoder():

    class MedNeXtEncReg(nn.Module):
        def __init__(self, *args, **kwargs):
            super(MedNeXtEncReg, self).__init__()
            self.mednextv1 = create_mednext_encoder_v1(
                num_input_channels=1,
                num_classes=1,
                model_id='B',
                kernel_size=3,
                deep_supervision=True
            )
            self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.regression_fc = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            x = self.mednextv1(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, start_dim=1)
            age_estimate = self.regression_fc(x)
            return age_estimate.squeeze()
    return MedNeXtEncReg

def build_monai_torchio_transforms():

    def masking_method(x):
        return x > 0

    x, y, z = (160, 192, 160)
    p = 1.0
    monai_transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z))
    ]
    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=masking_method, keys=["image"], include=['image'])]
    )
    return Compose(monai_transforms + [val_torchio_transforms])

def brain_age_predict(
    images: List[Path],
    device_str: str,
    models_dir: Path,
    model_pattern: str = "BrainAge_{}.pth",
    folds: Tuple[int, ...] = (1,2,3,4,5),
) -> Dict[Path, float]:
    """
    Returns dict: image_path -> BA (float).
    No correction applied here; correction is handled by caller using CA values (if available).
    """
    import torch
    import numpy as np
    from monai.data import CacheDataset
    from torch.utils.data import DataLoader

    data_dicts = [{'image': str(p), 'label': 0.0} for p in images]  # label not used

    transforms = build_monai_torchio_transforms()
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.0, num_workers=0)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=(device_str == "cuda" and torch.cuda.is_available()))

    MedNeXtEncReg = create_mednext_encoder()
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    fold_preds = []
    for i in folds:
        model_path = models_dir / model_pattern.format(i)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        model = MedNeXtEncReg().to(device)
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch_data in dataloader:
                images_tensor = batch_data['image'].to(device)
                pred = model(images_tensor)
                preds.append(pred.detach().cpu().numpy())
        fold_preds.append(np.array(preds).reshape(-1))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import numpy as np
    avg_preds = np.median(np.stack(fold_preds, axis=0), axis=0)  # (N,)

    results = {}
    for idx, p in enumerate(images):
        results[p] = float(avg_preds[idx])
    return results

# =========================
# Orchestration per image
# =========================

def preprocess_image(
    img: Path,
    out_root: Path,
    mni: Path,
    threads: int,
    docker_gpu: bool
) -> Path:
    """
    Returns the final MNI-aligned brain-extracted NIfTI path.
    """
    log = logging.getLogger()
    stem = strip_suffixes(img.name)
    output_dir = out_root / img.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    final = output_dir / f"{stem}_preproc_mni_brain.nii.gz"
    if final.exists():
        log.info(f"Skipping preprocess (exists): {final}")
        return final

    local_input = output_dir / f"{stem}_input.nii.gz"
    log.info(f"Copying input for processing: {img} -> {local_input}")
    shutil.copy2(img, local_input)

    o = output_dir / f"{stem}_oriented.nii.gz"
    f = output_dir / f"{stem}_fov.nii.gz"
    sr = output_dir / f"{stem}_synthsr.nii.gz"
    b = output_dir / f"{stem}_brain.nii.gz"
    xfm = output_dir / f"{stem}.lta"

    try:
        check_and_swap_orientation(log, local_input, o)
        run_robust_fov(log, o, f)
        run_synthsr(log, f, sr, threads=threads, use_gpu=docker_gpu)
        run_synthstrip(log, sr, b, use_gpu=docker_gpu)
        run_synthmorph_affine(log, b, mni, final, xfm, use_gpu=docker_gpu)
        log.info(f"Preprocess OK: {final}")
        return final
    finally:
        for p in [local_input, o, f, sr, b, xfm]:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logging.warning(f"Could not remove {p}: {e}")

def process_one_image(
    img: Path,
    mni: Optional[Path],
    out_root: Path,
    do_preprocess: bool,
    docker_gpu: bool,
    threads: int
) -> Tuple[Path, Path]:
    """
    Returns: (original_input_path, image_used_for_brain_age)
    """
    log = logging.getLogger()
    img = img.resolve()
    out_root.mkdir(exist_ok=True, parents=True)

    if do_preprocess:
        if not (mni and mni.exists()):
            raise FileNotFoundError("MNI template is required for preprocessing but was not found.")
        base_for_next = preprocess_image(img, out_root, mni, threads=threads, docker_gpu=docker_gpu)
    else:
        base_for_next = img

    return (img, base_for_next)

# =========================
# Input discovery and CSV handling
# =========================

def list_nifti_images(root_or_file: Path) -> List[Path]:
    if root_or_file.is_file():
        return [root_or_file]
    imgs = [p for p in root_or_file.rglob("*") if p.name.lower().endswith(".nii.gz")]
    return imgs

def read_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "age"}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")
        for row in reader:
            rows.append(row)
    return rows

def resolve_image_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()

# Add this function around line 385, before the list_nifti_images function
def analyze_cohorts(results_rows, log=None):
    """
    Group results by modality and age ranges.
    Returns statistics for each cohort combination.
    """
    if not log:
        log = logging.getLogger()
    
    # Extract modality from filename or path (T1, T2, etc.)
    for row in results_rows:
        filepath = row.get("input_image", "")
        filename = os.path.basename(filepath).lower()
        
        # Try to detect modality from filename
        if "t1" in filename:
            row["modality"] = "T1"
        elif "t2" in filename:
            row["modality"] = "T2"
        elif "flair" in filename:
            row["modality"] = "FLAIR"
        else:
            row["modality"] = "Unknown"
    
    # Define age cohorts
    age_cohorts = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
    modalities = set(row["modality"] for row in results_rows)
    
    # Group by modality and age cohort
    cohort_stats = {}
    for modality in modalities:
        cohort_stats[modality] = {}
        for min_age, max_age in age_cohorts:
            cohort_name = f"{min_age}-{max_age}"
            cohort_stats[modality][cohort_name] = {
                "count": 0,
                "mae_raw": 0.0,
                "mae_corrected": 0.0,
                "sum_error_raw": 0.0,
                "sum_error_corrected": 0.0
            }
    
    # Calculate statistics for each cohort
    for row in results_rows:
        modality = row["modality"]
        
        # Skip rows without chronological age
        ca_str = row.get("ChronologicalAge", "")
        if not ca_str:
            continue
            
        try:
            ca = float(ca_str)
            ba_raw = float(row.get("BrainAge_raw", 0))
            ba_corr = float(row.get("BrainAge_corrected", 0))
            
            # Find the appropriate age cohort
            for min_age, max_age in age_cohorts:
                if min_age <= ca < max_age:
                    cohort_name = f"{min_age}-{max_age}"
                    stats = cohort_stats[modality][cohort_name]
                    stats["count"] += 1
                    stats["sum_error_raw"] += abs(ba_raw - ca)
                    stats["sum_error_corrected"] += abs(ba_corr - ca)
                    break
        except (ValueError, TypeError):
            continue
    
    # Calculate MAE for each cohort
    for modality in cohort_stats:
        for cohort_name, stats in cohort_stats[modality].items():
            if stats["count"] > 0:
                stats["mae_raw"] = stats["sum_error_raw"] / stats["count"]
                stats["mae_corrected"] = stats["sum_error_corrected"] / stats["count"]
    
    return cohort_stats

# =========================
# W&B helpers
# =========================

def wandb_start_run(enabled: bool, cfg: dict):
    if not enabled or not _WANDB_AVAILABLE:
        return None
    try:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.login(key=WANDB_API_KEY, relogin=True)
        run = wandb.init(
            project=cfg.get("wandb", {}).get("project", "mri-pipeline"),
            name=cfg.get("wandb", {}).get("run_name", None),
            tags=cfg.get("wandb", {}).get("tags", None),
            notes=cfg.get("wandb", {}).get("notes", None),
            config=cfg
        )
        return run
    except Exception as e:
        logging.warning(f"W&B init failed, continuing without logging: {e}")
        return None

def wandb_log(run, data: dict):
    if run is None:
        return
    try:
        wandb.log(data)
    except Exception as e:
        logging.debug(f"W&B log error: {e}")

def wandb_finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass

# =========================
# Config and main
# =========================

def validate_and_normalize_config(cfg: dict) -> dict:
    # Basic checks
    pipeline = cfg.get("pipeline", {})
    preprocess = cfg.get("preprocess", {})
    brain_age = cfg.get("brain_age", {})
    single_image = cfg.get("single_image", {})
    wandb_cfg = cfg.get("wandb", {})

    input_path = pipeline.get("input")
    csv_path = pipeline.get("csv")
    if bool(input_path) == bool(csv_path):
        raise ValueError("Exactly one of pipeline.input or pipeline.csv must be provided (not both or neither).")

    # Normalize paths
    def norm_path(p):
        return None if p in (None, "", "null") else str(Path(p).resolve())

    pipeline["output_dir"] = norm_path(pipeline.get("output_dir"))
    pipeline["results_csv"] = norm_path(pipeline.get("results_csv"))
    pipeline["csv_base_dir"] = norm_path(pipeline.get("csv_base_dir"))
    pipeline["input"] = norm_path(pipeline.get("input"))
    pipeline["csv"] = norm_path(pipeline.get("csv"))

    preprocess["mni"] = norm_path(preprocess.get("mni"))
    brain_age["models_dir"] = norm_path(brain_age.get("models_dir"))

    # Defaults
    pipeline.setdefault("max_workers", 1)
    preprocess.setdefault("enabled", True)
    preprocess.setdefault("docker_gpu", False)
    preprocess.setdefault("threads", 8)
    brain_age.setdefault("device", "cuda")
    brain_age.setdefault("folds", [1,2,3,4,5])
    single_image.setdefault("age", None)
    wandb_cfg.setdefault("enabled", True)

    # Put back
    cfg["pipeline"] = pipeline
    cfg["preprocess"] = preprocess
    cfg["brain_age"] = brain_age
    cfg["single_image"] = single_image
    cfg["wandb"] = wandb_cfg
    return cfg

def main():
    parser = argparse.ArgumentParser(description="MRI pipeline (YAML-config): [Preprocess] -> BrainAge")
    parser.add_argument("--config", type=Path, default=Path("./config.yaml"), help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = validate_and_normalize_config(cfg)

    # Pull sections
    pipeline = cfg["pipeline"]
    preprocess_cfg = cfg["preprocess"]
    brainage_cfg = cfg["brain_age"]
    single_cfg = cfg["single_image"]
    wandb_cfg = cfg["wandb"]

    output_dir = Path(pipeline["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Tools availability
    need_docker = preprocess_cfg["enabled"]
    need_fsl = preprocess_cfg["enabled"]

    if need_docker:
        check_docker_available()
    if need_fsl:
        for bin_ in ["fslorient", "robustfov"]:
            check_binary_exists(bin_)

    if preprocess_cfg["enabled"]:
        if not preprocess_cfg["mni"] or not Path(preprocess_cfg["mni"]).exists():
            raise FileNotFoundError(f"MNI template not found: {preprocess_cfg['mni']}")

    # BrainAge model checks
    models_dir = Path(brainage_cfg["models_dir"])
    for i in range(1, 6):
        mp = models_dir / f"BrainAge_{i}.pth"
        if not mp.exists():
            raise FileNotFoundError(f"Missing brain-age model file: {mp}")

    # Start W&B
    run = wandb_start_run(enabled=bool(wandb_cfg.get("enabled", True)), cfg=cfg)
    t0 = time.time()
    wandb_log(run, {"event": "run_start"})

    # Resolve inputs
    rows = []
    input_images: List[Path] = []
    per_row_meta = []

    if pipeline["csv"]:
        csv_path = Path(pipeline["csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        rows = read_csv_rows(csv_path)
        base_dir = Path(pipeline["csv_base_dir"]).resolve() if pipeline["csv_base_dir"] else csv_path.parent.resolve()

        for row in rows:
            img_p = resolve_image_path(row["image_path"], base_dir)
            if not img_p.exists():
                log.warning(f"CSV image path not found, skipping: {row['image_path']} (resolved: {img_p})")
                continue
            try:
                ca = float(row["age"])
            except Exception as e:
                log.warning(f"Invalid age in CSV for {img_p}: {row.get('age')} ({e}); skipping row.")
                continue
            input_images.append(img_p)
            per_row_meta.append({
                "subject_id": row.get("subject_id", ""),
                "image_path": img_p,
                "age": ca,
                "raw_row": row
            })
    else:
        inp = Path(pipeline["input"])
        if not inp.exists():
            raise FileNotFoundError(f"Input not found: {inp}")
        input_images = list_nifti_images(inp)
        if not input_images:
            log.error(f"No .nii.gz files found in: {inp}")
            sys.exit(1)

    # Process images
    start = time.time()
    input_to_ba_used: Dict[Path, Path] = {}
    status: Dict[Path, str] = {}
    total = len(input_images)

    with ThreadPoolExecutor(max_workers=max(1, int(pipeline["max_workers"]))) as exe:
        futures = {}
        for img in input_images:
            futures[exe.submit(
                process_one_image,
                img=img.resolve(),
                mni=Path(preprocess_cfg["mni"]).resolve() if preprocess_cfg["enabled"] else None,
                out_root=output_dir.resolve(),
                do_preprocess=bool(preprocess_cfg["enabled"]),
                docker_gpu=bool(preprocess_cfg["docker_gpu"]),
                threads=int(preprocess_cfg["threads"])
            )] = img

        done = 0
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                orig, ba_img = fut.result()
                input_to_ba_used[orig] = ba_img
                status[img] = "OK"
                done += 1
                log.info(f"[{done}/{total}] ✔ {img}")
            except Exception as e:
                status[img] = f"ERROR: {e}"
                done += 1
                log.error(f"[{done}/{total}] ✖ {img}: {e}")
            elapsed = time.time() - start
            avg = elapsed / max(done, 1)
            eta = (total - done) * avg
            log.info(f"Elapsed {timedelta(seconds=int(elapsed))}, ETA {timedelta(seconds=int(eta))}")

    # Brain Age predictions
    ba_inputs = [input_to_ba_used[p] for p in input_images if p in input_to_ba_used]
    seen = set()
    ba_inputs_unique: List[Path] = []
    for pth in ba_inputs:
        if pth not in seen:
            seen.add(pth)
            ba_inputs_unique.append(pth)

    if not ba_inputs_unique:
        logging.error("No images available for BrainAge prediction.")
        wandb_log(run, {"error": "no_images_for_brainage"})
        wandb_finish(run)
        sys.exit(1)

    ba_predictions: Dict[Path, float] = brain_age_predict(
        images=ba_inputs_unique,
        device_str=str(brainage_cfg["device"]).lower(),
        models_dir=models_dir.resolve(),
        folds=tuple(brainage_cfg.get("folds", [1,2,3,4,5])),
    )

    # Reporting
    results_rows = []
    if pipeline["csv"]:
        abs_errors_corr = []
        abs_errors_raw = []
        print("\nPredicted Brain Ages (CSV mode):")
        for meta in per_row_meta:
            orig = meta["image_path"].resolve()
            ca = float(meta["age"])
            subj = meta["subject_id"]

            if orig not in input_to_ba_used:
                print(f"{subj or ''}\t{orig} -> No result ({status.get(orig, 'UNKNOWN')})")
                continue

            used = input_to_ba_used[orig]
            if used not in ba_predictions:
                print(f"{subj or ''}\t{orig} -> Prediction missing for used path: {used}")
                continue

            ba = ba_predictions[used]
            ba_corr = ba + (ca * 0.062) - 2.96 if ca > 18 else ba
            bad_corr = ba_corr - ca
            print(f"{subj or ''}\t{orig} -> BrainAge_raw={ba:.2f} | BrainAge_corr={ba_corr:.2f} (CA={ca:.2f}, Δ={bad_corr:.2f}) [via {used.name}]")

            results_rows.append({
                "subject_id": subj,
                "input_image": str(orig),
                "used_for_prediction": str(used),
                "preprocess_enabled": str(bool(preprocess_cfg["enabled"])),
                "BrainAge_raw": f"{ba:.6f}",
                "ChronologicalAge": f"{ca:.6f}",
                "BrainAge_corrected": f"{ba_corr:.6f}",
                "BrainAgeDelta_corrected": f"{bad_corr:.6f}",
                "Status": status.get(orig, "")
            })
            abs_errors_corr.append(abs(bad_corr))
            abs_errors_raw.append(abs(ba - ca))

        if abs_errors_corr:
            mae_corr = sum(abs_errors_corr) / len(abs_errors_corr)
            mae_raw = sum(abs_errors_raw) / len(abs_errors_raw)
            print(f"\nCSV summary over {len(abs_errors_corr)} rows:")
            print(f"MAE (corrected) = {mae_corr:.4f} years")
            print(f"MAE (raw)       = {mae_raw:.4f} years")
            wandb_log(run, {"mae_corrected": mae_corr, "mae_raw": mae_raw, "rows": len(abs_errors_corr)})
        else:
            print("\nCSV summary: No valid rows to compute MAE.")
            wandb_log(run, {"mae_corrected": None, "mae_raw": None, "rows": 0})

    else:
        print("\nPredicted Brain Ages (single-image mode):")
        ca = single_cfg.get("age", None)
        if ca is not None:
            try:
                ca = float(ca)
            except Exception:
                ca = None

        for orig in input_images:
            orig = orig.resolve()
            if orig in input_to_ba_used:
                used = input_to_ba_used[orig]
                ba = ba_predictions[used]
                if ca is not None:
                    ba_corr = ba + (ca * 0.062) - 2.96 if ca > 18 else ba
                    bad_corr = ba_corr - ca
                    print(f"{orig} -> BrainAge={ba_corr:.2f} years (CA={ca:.2f}, Δ={bad_corr:.2f}) [via {used.name}]")
                    wandb_log(run, {"brain_age_corrected": ba_corr, "brain_age_raw": ba, "delta_corrected": bad_corr})
                    results_rows.append({
                        "subject_id": "",
                        "input_image": str(orig),
                        "used_for_prediction": str(used),
                        "preprocess_enabled": str(bool(preprocess_cfg["enabled"])),
                        "BrainAge_raw": f"{ba:.6f}",
                        "ChronologicalAge": f"{ca:.6f}",
                        "BrainAge_corrected": f"{ba_corr:.6f}",
                        "BrainAgeDelta_corrected": f"{bad_corr:.6f}",
                        "Status": status.get(orig, "")
                    })
                else:
                    print(f"{orig} -> BrainAge={ba:.2f} years [via {used.name}]")
                    wandb_log(run, {"brain_age_raw": ba})
                    results_rows.append({
                        "subject_id": "",
                        "input_image": str(orig),
                        "used_for_prediction": str(used),
                        "preprocess_enabled": str(bool(preprocess_cfg["enabled"])),
                        "BrainAge_raw": f"{ba:.6f}",
                        "ChronologicalAge": "",
                        "BrainAge_corrected": f"{ba:.6f}",
                        "BrainAgeDelta_corrected": "",
                        "Status": status.get(orig, "")
                    })
            else:
                print(f"{orig} -> No result ({status.get(orig, 'UNKNOWN')})")

    # Log a W&B table with results (if any)
    if results_rows and _WANDB_AVAILABLE and run is not None:
        columns = [
            "subject_id", "InputImage", "UsedForPrediction",
            "PreprocessEnabled", "Modality",
            "BrainAge_raw", "ChronologicalAge",
            "BrainAge_corrected", "BrainAgeDelta_corrected",
            "Status"
        ]
        table = wandb.Table(columns=columns)
        for r in results_rows:
            table.add_data(
                r.get("subject_id", ""),
                r.get("input_image", ""),
                r.get("used_for_prediction", ""),
                r.get("preprocess_enabled", ""),
                r.get("modality", "Unknown"),
                r.get("BrainAge_raw", ""),
                r.get("ChronologicalAge", ""),
                r.get("BrainAge_corrected", ""),
                r.get("BrainAgeDelta_corrected", ""),
                r.get("Status", "")
            )
        wandb_log(run, {"results": table})

    # Optional results CSV
    if pipeline["results_csv"]:
        out_csv = Path(pipeline["results_csv"])
        out_csv.parent.mkdir(exist_ok=True, parents=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject_id", "InputImage", "UsedForPrediction",
                "PreprocessEnabled", "Modality",
                "BrainAge_raw", "ChronologicalAge",
                "BrainAge_corrected", "BrainAgeDelta_corrected",
                "Status"
            ])
            for r in results_rows:
                writer.writerow([
                    r.get("subject_id", ""), r.get("input_image", ""), r.get("used_for_prediction", ""),
                    r.get("preprocess_enabled", ""), r.get("modality", "Unknown"),
                    r.get("BrainAge_raw", ""), r.get("ChronologicalAge", ""),
                    r.get("BrainAge_corrected", ""), r.get("BrainAgeDelta_corrected", ""),
                    r.get("Status", "")
                ])
        print(f"\nResults CSV written to: {out_csv}")
        wandb_log(run, {"results_csv": str(out_csv)})

    # Cohort analysis
    if results_rows:
        cohort_stats = analyze_cohorts(results_rows, log)
        print("\nCohort Analysis (by modality and age range):")
        for modality in cohort_stats:
            print(f"\n{modality}:")
            print("Age Range  | Count | MAE (Raw) | MAE (Corrected)")
            print("-" * 50)
            for cohort_name, stats in cohort_stats[modality].items():
                if stats["count"] > 0:
                    print(f"{cohort_name:10} | {stats['count']:5d} | {stats['mae_raw']:.4f} | {stats['mae_corrected']:.4f}")
        
        # Log cohort stats to W&B
        if run is not None:
            for modality in cohort_stats:
                for cohort_name, stats in cohort_stats[modality].items():
                    if stats["count"] > 0:
                        wandb_log(run, {
                            f"cohort/{modality}/{cohort_name}/count": stats["count"],
                            f"cohort/{modality}/{cohort_name}/mae_raw": stats["mae_raw"],
                            f"cohort/{modality}/{cohort_name}/mae_corrected": stats["mae_corrected"]
                        })

    # Finish W&B
    runtime = time.time() - t0
    wandb_log(run, {"event": "run_end", "runtime_sec": runtime})
    wandb_finish(run)

if __name__ == "__main__":
    main()