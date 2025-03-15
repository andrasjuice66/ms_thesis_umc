import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial

# Create a sample 3D brain volume
def create_sample_brain_volume(size=128):
    volume = np.zeros((size, size, size))
    x, y, z = np.indices((size, size, size))
    center = (size//2, size//2, size//2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    volume[r < size//3] = 0.8
    volume[r < size//4] = 0.6
    volume[r < size//6] = 0.4
    volume += np.random.normal(0, 0.05, volume.shape)
    return volume

# Create a dataset of brain volumes
def create_dataset(num_samples=10, size=128):
    return [create_sample_brain_volume(size) for _ in range(num_samples)]

# Augmentation functions
def random_rotation(volume):
    """Apply random rotation to the volume"""
    axes = [(0, 1), (0, 2), (1, 2)]
    axis = axes[np.random.choice(len(axes))]
    angle = np.random.uniform(-15, 15)
    k = int(angle / 90)
    return np.rot90(volume, k, axis)

def random_flip(volume):
    """Apply random flip to the volume"""
    axis = np.random.randint(0, 3)
    return np.flip(volume, axis)

def random_brightness(volume):
    """Apply random brightness adjustment"""
    factor = np.random.uniform(0.8, 1.2)
    return volume * factor

def random_contrast(volume):
    """Apply random contrast adjustment"""
    factor = np.random.uniform(0.8, 1.2)
    mean = np.mean(volume)
    return (volume - mean) * factor + mean

def random_noise(volume):
    """Add random noise to the volume"""
    noise_level = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, noise_level, volume.shape)
    return volume + noise

def random_gamma(volume):
    """Apply random gamma correction"""
    gamma = np.random.uniform(0.8, 1.2)
    # Ensure volume is non-negative before applying power
    volume_positive = np.maximum(volume, 0)
    return np.power(volume_positive, gamma)

def apply_augmentations(volume):
    """Apply a series of random augmentations to a volume"""
    # List of augmentation functions
    augmentations = [
        random_rotation,
        random_flip,
        random_brightness,
        random_contrast,
        random_noise,
        random_gamma
    ]
    
    # Randomly select which augmentations to apply
    num_augs = np.random.randint(1, len(augmentations) + 1)
    selected_augs = np.random.choice(augmentations, num_augs, replace=False)
    
    # Apply selected augmentations
    augmented = volume.copy()
    for aug_func in selected_augs:
        augmented = aug_func(augmented)
    
    return augmented

# Simulate a training batch with augmentation
def get_batch(dataset, batch_size=4):
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    batch = [dataset[i] for i in indices]
    return batch

# Sequential augmentation (inefficient)
def sequential_augmentation(batch):
    return [apply_augmentations(volume) for volume in batch]

# Parallel augmentation with threading
def threaded_augmentation(batch):
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        return list(executor.map(apply_augmentations, batch))

# Parallel augmentation with multiprocessing
def multiprocess_augmentation(batch):
    with ProcessPoolExecutor(max_workers=min(len(batch), multiprocessing.cpu_count())) as executor:
        return list(executor.map(apply_augmentations, batch))

# Vectorized augmentation (batch-wise operations)
def vectorized_augmentation(batch):
    # Convert batch to numpy array
    batch_array = np.array(batch)
    
    # Apply batch-wise operations
    # Random brightness (same factor for all volumes in batch)
    factor = np.random.uniform(0.8, 1.2)
    batch_array = batch_array * factor
    
    # Random contrast (same factor for all volumes in batch)
    factor = np.random.uniform(0.8, 1.2)
    mean = np.mean(batch_array, axis=(1, 2, 3), keepdims=True)
    batch_array = (batch_array - mean) * factor + mean
    
    # Random noise (different noise for each volume)
    noise_level = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, noise_level, batch_array.shape)
    batch_array = batch_array + noise
    
    return list(batch_array)

# Benchmark different augmentation strategies
def benchmark_augmentation_strategies():
    print("Creating dataset...")
    dataset = create_dataset(num_samples=20, size=64)
    batch_size = 8
    batch = get_batch(dataset, batch_size)
    
    print(f"Benchmarking augmentation strategies with batch size {batch_size}...")
    
    # Sequential
    start_time = time.time()
    sequential_result = sequential_augmentation(batch)
    sequential_time = time.time() - start_time
    print(f"Sequential augmentation: {sequential_time:.4f} seconds")
    
    # Threaded
    start_time = time.time()
    threaded_result = threaded_augmentation(batch)
    threaded_time = time.time() - start_time
    print(f"Threaded augmentation: {threaded_time:.4f} seconds (Speedup: {sequential_time/threaded_time:.2f}x)")
    
    # Multiprocessing
    start_time = time.time()
    multiprocess_result = multiprocess_augmentation(batch)
    multiprocess_time = time.time() - start_time
    print(f"Multiprocessing augmentation: {multiprocess_time:.4f} seconds (Speedup: {sequential_time/multiprocess_time:.2f}x)")
    
    # Vectorized
    start_time = time.time()
    vectorized_result = vectorized_augmentation(batch)
    vectorized_time = time.time() - start_time
    print(f"Vectorized augmentation: {vectorized_time:.4f} seconds (Speedup: {sequential_time/vectorized_time:.2f}x)")
    
    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(batch[0][:, :, batch[0].shape[2]//2], cmap='gray')
    axes[0].set_title('Original')
    
    # Sequential
    axes[1].imshow(sequential_result[0][:, :, sequential_result[0].shape[2]//2], cmap='gray')
    axes[1].set_title('Sequential')
    
    # Threaded
    axes[2].imshow(threaded_result[0][:, :, threaded_result[0].shape[2]//2], cmap='gray')
    axes[2].set_title('Threaded')
    
    # Multiprocessing
    axes[3].imshow(multiprocess_result[0][:, :, multiprocess_result[0].shape[2]//2], cmap='gray')
    axes[3].set_title('Multiprocessing')
    
    # Vectorized
    axes[4].imshow(vectorized_result[0][:, :, vectorized_result[0].shape[2]//2], cmap='gray')
    axes[4].set_title('Vectorized')
    
    # Performance comparison
    times = [sequential_time, threaded_time, multiprocess_time, vectorized_time]
    labels = ['Sequential', 'Threaded', 'Multiprocessing', 'Vectorized']
    
    axes[5].bar(labels, times)
    axes[5].set_title('Execution Time (seconds)')
    axes[5].set_ylabel('Time (s)')
    
    # Speedup comparison
    speedups = [1, sequential_time/threaded_time, sequential_time/multiprocess_time, sequential_time/vectorized_time]
    
    axes[6].bar(labels, speedups)
    axes[6].set_title('Speedup (relative to Sequential)')
    axes[6].set_ylabel('Speedup factor')
    
    # Hide the last subplot
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'sequential': sequential_time,
        'threaded': threaded_time,
        'multiprocessing': multiprocess_time,
        'vectorized': vectorized_time
    }

# Demonstrate a PyTorch-like data loader with prefetching
def simulate_training_with_prefetching():
    dataset = create_dataset(num_samples=20, size=64)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, prefetch_factor=2)
    
    # Simulate 5 training iterations
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        # Simulate model forward pass and backward pass
        time.sleep(0.1)  # Simulate computation time
        print(f"Iteration {i+1}: Processed batch of {len(batch)} volumes")
    
    total_time = time.time() - start_time
    print(f"Total training time with prefetching: {total_time:.4f} seconds")

# Define DataLoader class
class DataLoader:
    def __init__(self, dataset, batch_size=4, num_workers=2, prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = []
        
    def __iter__(self):
        self.prefetch_queue = []
        # Prefetch initial batches
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for _ in range(self.prefetch_factor):
                batch = get_batch(self.dataset, self.batch_size)
                future = executor.submit(threaded_augmentation, batch)
                self.prefetch_queue.append(future)
        return self
    
    def __next__(self):
        if not self.prefetch_queue:
            raise StopIteration
        
        # Get the next batch from the queue
        next_batch = self.prefetch_queue.pop(0).result()
        
        # Prefetch another batch
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            batch = get_batch(self.dataset, self.batch_size)
            future = executor.submit(threaded_augmentation, batch)
            self.prefetch_queue.append(future)
        
        return next_batch

# Demonstrate progressive augmentation
def visualize_progressive_augmentation():
    volume = create_sample_brain_volume(size=64)
    epochs = [1, 3, 5, 7, 10]
    
    fig, axes = plt.subplots(1, len(epochs) + 1, figsize=(15, 3))
    
    # Original
    axes[0].imshow(volume[:, :, volume.shape[2]//2], cmap='gray')
    axes[0].set_title('Original')
    
    # Progressive augmentations
    for i, epoch in enumerate(epochs):
        augmented = progressive_augmentation(volume, epoch, max_epochs=10)
        axes[i+1].imshow(augmented[:, :, augmented.shape[2]//2], cmap='gray')
        axes[i+1].set_title(f'Epoch {epoch}')
    
    plt.tight_layout()
    plt.show()

def progressive_augmentation(volume, epoch, max_epochs=10):
    """Apply progressively more complex augmentations as training progresses"""
    # Calculate the intensity of augmentations based on current epoch
    intensity = min(1.0, epoch / (max_epochs * 0.7))  # Reach max intensity at 70% of training
    
    # Apply basic augmentations always
    result = random_flip(volume)
    
    # Apply more complex augmentations with increasing probability
    if np.random.random() < intensity:
        result = random_rotation(result)
    
    if np.random.random() < intensity * 0.8:
        result = random_brightness(result)
    
    if np.random.random() < intensity * 0.6:
        result = random_contrast(result)
    
    if np.random.random() < intensity * 0.4:
        result = random_gamma(result)
    
    # Apply noise with decreasing intensity as training progresses
    noise_level = 0.05 * (1 - intensity * 0.5)  # Reduce noise over time
    noise = np.random.normal(0, noise_level, result.shape)
    result = result + noise
    
    return result

# Move all execution code inside if __name__ == '__main__'
if __name__ == '__main__':
    # Run the benchmark
    benchmark_results = benchmark_augmentation_strategies()
    
    # Demonstrate a PyTorch-like data loader with prefetching
    print("\nSimulating a PyTorch-like DataLoader with prefetching...")
    simulate_training_with_prefetching()
    
    # Demonstrate progressive augmentation
    print("\nDemonstrating progressive augmentation...")
    visualize_progressive_augmentation()
    
    print("\nOptimization strategies for online data augmentation in medical imaging:")
    print("1. Parallel processing with threading and multiprocessing")
    print("2. Vectorized batch operations")
    print("3. Prefetching with data loaders")
    print("4. Progressive augmentation strategies")
    print("5. Selective application of computationally expensive augmentations")