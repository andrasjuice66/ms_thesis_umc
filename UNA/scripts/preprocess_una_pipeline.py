"""
UNA (Unsupervised Anomaly) Preprocessing Pipeline
Applies UNA model to all images while maintaining directory structure.
Includes W&B logging and parallel processing.
"""
import logging
import os
import shutil
import sys
import threading
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import wandb

# Add UNA to path
current_dir = Path(__file__).parent.parent.parent
una_repo_path = current_dir / "UNA"
sys.path.append(str(una_repo_path))

# Import UNA utilities
try:
    import utils.test_utils as utils
    from utils.misc import viewVolume, make_dir
except ImportError as e:
    print(f"Error importing UNA utilities: {e}")
    print(f"Make sure UNA is properly installed at: {una_repo_path}")
    sys.exit(1)

# W&B login
wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")

_gpu_sem = threading.BoundedSemaphore(1)
CACHE_FILE_NAME = "image_paths.txt"


def find_images(root):
    """
    Walk `root` recursively, logging each candidate and yielding it.
    Only includes .nii.gz files.
    """
    log = logging.getLogger()
    log.info(f"Scanning directory for images: {root}")
    for p in root.rglob("*.nii.gz"):
        log.info(f"Found image: {p}")
        yield p


def strip_suffixes(n):
    for s in (".nii.gz", ".nii", ".gz"):
        if n.endswith(s):
            return n[:-len(s)]
    return n


def run_una_inference(log, inp, out, model_path, gen_cfg, model_cfg, win_size=[160, 160, 160]):
    """
    Run UNA inference on a single image and save only the T1 output
    
    Args:
        log: Logger instance
        inp (Path): Input image path
        out (Path): Output image path
        model_path (str): Path to UNA model weights
        gen_cfg (str): Generator config path
        model_cfg (str): Model config path
        win_size (list): Window size for processing
    """
    out.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"UNA inference: {inp.name}")
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create temporary output directory for intermediate files
        temp_dir = out.parent / f"temp_{strip_suffixes(inp.name)}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        with _gpu_sem:
            # Read the original image
            log.info(f"Preparing image: {inp.name}")
            _, img, _, aff = utils.prepare_image(str(inp), win_size=win_size, im_only=True, device=device)
            
            # Create flipped version (required by UNA's pipeline)
            img_flip = torch.flip(img, dims=[0])
            flip_path = temp_dir / f"{strip_suffixes(inp.name)}_flip_reg2orig.nii.gz"
            viewVolume(img_flip, aff, names=[f"{strip_suffixes(inp.name)}_flip_reg2orig"], save_dir=str(temp_dir))
            
            # Read the flipped image back (to match UNA's pipeline)
            _, img_flip_reg2orig, _, _ = utils.prepare_image(str(flip_path), win_size=win_size, spacing=None, im_only=True, device=device)
            
            # Run inference
            log.info(f"Running UNA inference on: {inp.name}")
            outs = utils.evaluate_image(img, img_flip_reg2orig, ckp_path=model_path, device=device, gen_cfg=gen_cfg, model_cfg=model_cfg)
            
            # Save ONLY the T1 output as the final result
            if 'T1' in outs and isinstance(outs['T1'], torch.Tensor):
                viewVolume(outs['T1'], aff, names=[strip_suffixes(out.name)], save_dir=str(out.parent))
                log.info(f"Successfully processed: {inp.name} -> {out.name}")
            else:
                log.error(f"No T1 output found for {inp.name}")
                raise ValueError("No T1 output from UNA inference")
        
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            log.info(f"Cleaned up temporary directory: {temp_dir}")
            
    except Exception as e:
        log.error(f"UNA inference failed for {inp.name}: {e}")
        # Clean up temporary directory on error
        temp_dir = out.parent / f"temp_{strip_suffixes(inp.name)}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    
    return out


def process_image(img, base, out_root, model_path, gen_cfg, model_cfg, win_size):
    """
    Process a single image through UNA pipeline while maintaining directory structure
    """
    log = logging.getLogger()
    
    # Maintain directory structure from input
    rel_path = img.relative_to(base)
    output_dir = out_root / rel_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Final output file (maintains directory structure and original filename)
    final = output_dir / img.name
    
    # Skip if final output already exists
    if final.exists():
        log.info(f"Skipping (final exists): {final}")
        return
    
    try:
        # Run UNA inference directly on the input image
        run_una_inference(log, img, final, model_path, gen_cfg, model_cfg, win_size)
        log.info(f"Successfully processed: {img}")
        
    except Exception as e:
        log.error(f"Failed to process {img}: {e}")
        # Remove partial output file if it exists
        if final.exists():
            final.unlink()
        raise


def main(data_root, out_root, model_path, win_size):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger()
    
    out_root.mkdir(exist_ok=True, parents=True)
    
    # UNA config files
    gen_cfg = current_dir / "UNA" / "cfgs" / "generator" / "test" / "test.yaml"
    model_cfg = current_dir / "UNA" / "cfgs" / "trainer" / "test" / "test.yaml"
    
    # Verify required files exist
    for required_file in [model_path, gen_cfg, model_cfg]:
        if not Path(required_file).exists():
            log.error(f"Required file not found: {required_file}")
            return
    
    log.info(f"Using UNA model: {model_path}")
    log.info(f"Window size: {win_size}")
    
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
    log.info(f"GPU available: {torch.cuda.is_available()}")
    
    # W&B run - explicitly set to online mode only
    try:
        run = wandb.init(
            project="thesis_preprocess",
            name=f"{out_root.name}_una_pipeline",
            id=wandb.util.generate_id(),
            resume="allow",
            mode="online",
            config={
                "dataset": out_root.name,
                "pipeline": "UNA",
                "total": total,
                "cpus": os.cpu_count(),
                "model_path": str(model_path),
                "win_size": win_size
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
    
    # Use only 1 worker due to GPU memory constraints and UNA's requirements
    with ThreadPoolExecutor(1) as exe:
        futures = {
            exe.submit(process_image, img, data_root, out_root, str(model_path), str(gen_cfg), str(model_cfg), win_size): img
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
            avg = elapsed / (done - done0) if (done - done0) > 0 else 0
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
    # Hardcoded configuration
    data_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed_no_csf/IXI")
    out_root = Path("/mnt/c/Projects/thesis_project/Data/brain_age_una/IXI")
    model_path = Path("/mnt/c/Projects/thesis_project/brain_age_pred/UNA/assets/una-001.pth")
    win_size = [182, 218, 182]
    
    main(data_root, out_root, model_path, win_size)