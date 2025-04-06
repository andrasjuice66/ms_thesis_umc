import torch
import time

def benchmark_matrix_mult(size=5000, iterations=10, use_cuda=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # Create large matrices
    matrix_a = torch.randn(size, size, device=device)
    matrix_b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(2):
        _ = torch.matmul(matrix_a, matrix_b)
    
    # Synchronize if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        result = torch.matmul(matrix_a, matrix_b)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Wait for GPU operation to complete
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    return avg_time, device.type

def main():
    print("Running benchmark...")
    
    # Test whether CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Running benchmarks on both CPU and GPU.")
        
        # For smaller matrices to get meaningful CPU comparison
        cpu_time, _ = benchmark_matrix_mult(size=2000, iterations=3, use_cuda=False)
        gpu_time, _ = benchmark_matrix_mult(size=2000, iterations=3, use_cuda=True)
        
        print(f"CPU time: {cpu_time:.10f} seconds per iteration")
        print(f"GPU time: {gpu_time:.20f} seconds per iteration")
        #print(f"GPU is {cpu_time/gpu_time:.20f}x faster than CPU")
        
        # For larger matrices to show GPU power
        print("\nRunning with larger matrices:")
        cpu_time, _ = benchmark_matrix_mult(size=50000, iterations=1, use_cuda=False)
        gpu_time, _ = benchmark_matrix_mult(size=50000, iterations=3, use_cuda=True)
        
        print(f"CPU time: {cpu_time:.10f} seconds per iteration")
        print(f"GPU time: {gpu_time:.10f} seconds per iteration")
        print(f"GPU is {cpu_time/gpu_time:.10f}x faster than CPU")
    else:
        print("CUDA is not available. Running benchmark on CPU only.")
        cpu_time, _ = benchmark_matrix_mult(size=2000, iterations=3, use_cuda=False)
        print(f"CPU time: {cpu_time:.10f} seconds per iteration")

if __name__ == "__main__":
    main()
