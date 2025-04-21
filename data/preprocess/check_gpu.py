import torch
import subprocess
import os

def get_gpu_info():
    print("=== CUDA Available ===")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("\n=== CUDA Device Info ===")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        # Get device properties
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**2:.2f} MB")
            print(f"  GPU Memory Usage:")
            print(f"    Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"    Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            
        # Try to get nvidia-smi output
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
            print("\n=== nvidia-smi Output ===")
            print(nvidia_smi)
        except:
            print("\nCouldn't get nvidia-smi output")

get_gpu_info()