import subprocess
import re

def run_wsl_command(logger, cmd, use_fsl=True):
    """Run a command in WSL and return the output"""
    if use_fsl==True:
        wsl_cmd = f'wsl -d Ubuntu-22.04 --user selen bash -ls -c {cmd}'
    else:
        wsl_cmd = f'wsl -d Ubuntu-22.04 {cmd}'
    logger.info(f"Running WSL command: {wsl_cmd}")
    #login = subprocess.run("wsl -d Ubuntu-22.04 --user selen", shell=True, text=True)
    result = subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"WSL command failed: {result.stderr}")
        raise Exception(f"WSL command failed: {result.stderr}")
    return result.stdout.strip()

def windows_to_wsl_path(win_path):
    """Convert Windows path to WSL path"""
    # Replace backslashes with forward slashes
    path = str(win_path).replace('\\\\', '/').replace('\\', '/')
    
    # Convert drive letter (e.g., C:) to WSL path (/mnt/c)
    if re.match(r'^[A-Za-z]:', path):
        drive_letter = path[0].lower()
        path = f"/mnt/{drive_letter}{path[2:]}"
    
    return path