#!/usr/bin/env python3
"""
preprocess_mri.py

Flattened preprocessing pipeline for T1/T2/FLAIR .nii.gz files:
  1) SynthStrip (Docker)
  2) N4BiasFieldCorrection (antsx/ants Docker)
  3) SynthMorph (Docker, affine only)

Collects all final outputs into a single flat directory,
calculates and prints a running ETA during processing.
"""

import os
import sys
import time
import argparse
import subprocess
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import docker

def format_secs(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h: parts.append(f"{int(h)}h")
    if m: parts.append(f"{int(m)}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)

def format_docker_path(path: Path) -> str:
    """
    Convert Windows paths (including extended-length UNC) to
    Docker-compatible mount points under WSL2/desktop.
    """
    p = str(path)

    # 1) Strip extended-length UNC prefix '\\?\UNC\server\share\…'
    UNC_EXT = '\\\\?\\UNC\\'
    if p.startswith(UNC_EXT):
        # drop '\\?\UNC\' → 'server\share\…', then re‑prefix '\\server\share…'
        p = '\\\\' + p[len(UNC_EXT):]

    # 2) Strip extended-length local prefix '\\?\C:\…'
    LOCAL_EXT = '\\\\?\\'
    if p.startswith(LOCAL_EXT):
        p = p[len(LOCAL_EXT):]

    # 3) Handle UNC shares '\\server\share\…'
    if p.startswith('\\\\'):
        parts = p.lstrip('\\').split('\\')
        server, share, *rest = parts
        docker_path = f"/run/desktop/mnt/host/{server}/{share}"
        if rest:
            docker_path += "/" + "/".join(rest)
        return docker_path

    # 4) Handle local drive-letter paths 'C:\…'
    p_unix = p.replace('\\', '/')
    if len(p_unix) > 1 and p_unix[1] == ':':
        drive = p_unix[0].lower()
        rest = p_unix[2:]  # includes leading '/'
        return f"/mnt/{drive}{rest}"

    # 5) Fallback: return unchanged
    return p

def process_file(args):
    (infile, outroot,
     synthstrip_img, ants_img, synthmorph_img,
     template, cleanup) = args

    t0 = time.perf_counter()
    infile = Path(infile)
    base = infile.stem.replace(".nii", "")
    outroot = Path(outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    stripped  = outroot / f"{base}_brain.nii.gz"
    n4corr    = outroot / f"{base}_brain_N4.nii.gz"
    final_aff = outroot / f"{base}_brain_N4_affine.nii.gz"

    is_windows = platform.system() == "Windows"

    try:
        docker_infile    = format_docker_path(infile)
        docker_outroot   = format_docker_path(outroot)
        docker_template  = format_docker_path(Path(template))
        docker_stripped  = format_docker_path(stripped)
        docker_n4corr    = format_docker_path(n4corr)
        docker_final_aff = format_docker_path(final_aff)

        # 1) SynthStrip
        cmd = ["docker", "run", "--rm"]
        if not is_windows:
            cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
        cmd += [
            "-v", f"{docker_infile}:{docker_infile}",
            "-v", f"{docker_outroot}:{docker_outroot}",
            synthstrip_img,
            "synthstrip", docker_infile, docker_stripped
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 2) N4 bias correction via antsx/ants Docker
        cmd = ["docker", "run", "--rm"]
        if not is_windows:
            cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
        cmd += [
            "-v", f"{docker_outroot}:{docker_outroot}",
            ants_img,
            "N4BiasFieldCorrection",
            "-d", "3",
            "-i", docker_stripped,
            "-o", docker_n4corr
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3) SynthMorph (affine only)
        cmd = ["docker", "run", "--rm"]
        if not is_windows:
            cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
        cmd += [
            "-v", f"{docker_n4corr}:{docker_n4corr}",
            "-v", f"{docker_template}:{docker_template}",
            "-v", f"{docker_outroot}:{docker_outroot}",
            synthmorph_img,
            "synthmorph",
            "--fixed", docker_template,
            "--moving", docker_n4corr,
            "--output", docker_final_aff,
            "--only-affine"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # cleanup intermediates
        if cleanup:
            stripped.unlink(missing_ok=True)
            n4corr.unlink(missing_ok=True)

        return (infile, time.perf_counter() - t0, None)

    except Exception as e:
        return (infile, None, e)

def main():
    parser = argparse.ArgumentParser(
        description="Flattened MRI preprocessing: SynthStrip → N4 → SynthMorph"
    )
    parser.add_argument("--input_dir",   required=True,
                        help="Root to traverse for .nii.gz (T1/T2/FLAIR).")
    parser.add_argument("--output_dir",  required=True,
                        help="Flat directory for all final preprocessed images.")
    parser.add_argument("--template",    required=True,
                        help="Fixed-image template (e.g. MNI152).")
    parser.add_argument("--synthstrip_image", default="synthstrip:latest",
                        help="SynthStrip Docker image.")
    parser.add_argument("--ants_image", default="antsx/ants:latest",
                        help="ANTS Docker image for N4BiasFieldCorrection.")
    parser.add_argument("--synthmorph_image", default="synthmorph:latest",
                        help="SynthMorph Docker image.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers.")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Keep intermediate files.")
    args = parser.parse_args()

    inroot  = Path(args.input_dir).expanduser().resolve()
    outroot = Path(args.output_dir).expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    mods = ("t1", "t2", "t1w", "t2w", "flair")
    files = []
    for fn in inroot.rglob("*.nii.gz"):
        if any(m in fn.name.lower() for m in mods):
            files.append(str(fn))

    if not files:
        print("No matching .nii.gz files found. Exiting.")
        sys.exit(0)

    total_files = len(files)
    print(f"Found {total_files} files. Starting preprocessing with {args.workers} workers...\n")

    times = []
    done = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = [
            exe.submit(
                process_file,
                (f, str(outroot),
                 args.synthstrip_image,
                 args.ants_image,
                 args.synthmorph_image,
                 args.template,
                 not args.no_cleanup)
            )
            for f in files
        ]
        for fut in as_completed(futures):
            infile, dt, err = fut.result()
            if err:
                errors.append((infile, err))
                print(f"[ERROR] {infile}: {err}")
            else:
                done += 1
                times.append(dt)
                avg = sum(times) / len(times)
                remaining = total_files - done
                eta = avg * remaining
                print(
                    f"[OK {done}/{total_files}] "
                    f"{Path(infile).name} → {format_secs(dt)}  "
                    f"ETA: {format_secs(eta)}"
                )

    print(f"\nFinished {done}/{total_files} files.")
    if times:
        total_time = sum(times)
        print(
            f" Total elapsed (sum of individual): {format_secs(total_time)}  "
            f"Avg/file: {format_secs(total_time/len(times))}"
        )
    if errors:
        print(f"\n{len(errors)} failures:")
        for f, e in errors:
            print(f"  {f}: {e}")

if __name__ == "__main__":
    main()
    #python strip_morph_n4_preprocess.py --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\data\OpenNeuro\BoldVariability" --output_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\brainage_preprocessed_data\OpenNeuro\BoldVariability" --template "C:\Projects\thesis_project\ms_thesis_umc\data\preprocess\MNI152lin_T1_1mm.nii.gz" --synthstrip_image freesurfer/synthstrip:latest --ants_image antsx/ants:latest --synthmorph_image freesurfer/synthmorph:latest --workers 8
    #python strip_morph_n4_preprocess.py --input_dir "C:\Projects\thesis_project\Data\IXI" --output_dir "C:\Projects\thesis_project\Data\IXI_processed" --template "C:\Projects\thesis_project\ms_thesis_umc\data\preprocess\MNI152lin_T1_1mm.nii.gz" --synthstrip_image freesurfer/synthstrip:latest --ants_image antsx/ants:latest --synthmorph_image freesurfer/synthmorph:latest --workers 8