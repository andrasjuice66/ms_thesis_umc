import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define a simple dataset simulating a few MRI images.
class MRIDataset(Dataset):
    def __init__(self):
        # Create 10 dummy MRI images (1 channel, 256x256) and random labels (0 or 1)
        self.data = [torch.randn(1, 256, 256) for _ in range(10)]
        self.labels = [torch.randint(0, 2, (1,)).item() for _ in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data remains on the CPU until moved within the prefetcher.
        return self.data[idx], self.labels[idx]

# Define a GPU-based transformation function.
def gpu_transform(image):
    # For demonstration, perform a 90-degree rotation on the GPU.
    # Replace or extend this with your actual augmentation.
    return torch.rot90(image, k=1, dims=[1, 2])

# Define a minimal CNN for a classification task.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)  # Two output classes

    def forward(self, x):
        # Ensure proper shape before passing through the model
        if len(x.shape) == 4 and x.shape[1] == 1:
            pass  # Shape is correct: [batch_size, channels, height, width]
        elif len(x.shape) == 4 and x.shape[2] == 1:
            # Shape is [batch_size, height, channels, width], needs to be rearranged
            x = x.permute(0, 2, 1, 3)
        
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create a prefetcher class that asynchronously loads and augments data.
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        # Create a separate stream for async GPU transfers, if using CUDA.
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        # If a CUDA stream is available, perform the copy and transformation asynchronously.
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(self.device, non_blocking=True)
                self.next_target = torch.tensor(self.next_target, device=self.device, non_blocking=True)
                # Apply the GPU transformation.
                self.next_input = gpu_transform(self.next_input)
        else:
            # If no GPU device is available, perform synchronous copy.
            self.next_input = self.next_input.to(self.device)
            self.next_target = torch.tensor(self.next_target, device=self.device)
            self.next_input = gpu_transform(self.next_input)

    def next(self):
        if self.stream is not None:
            # Ensure the data transfer is complete.
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            # Associate the tensor with the current stream.
            input.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create the dataset on CPU and set up the DataLoader with pinned memory.
    dataset = MRIDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=0)

    # Instantiate our model and move it to GPU.
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize the prefetcher to load batches asynchronously.
    prefetcher = DataPrefetcher(loader, device)
    input_batch, target_batch = prefetcher.next()

    iteration = 0
    # Training loop over the data.
    while input_batch is not None:
        iteration += 1
        # Print shape for debugging
        print(f"Input batch shape: {input_batch.shape}")
        
        optimizer.zero_grad()
        output = model(input_batch)
        loss = loss_fn(output, target_batch)
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration}: Loss = {loss.item():.4f}")

        # Fetch the next batch asynchronously while the training step is finishing.
        input_batch, target_batch = prefetcher.next()

if __name__ == "__main__":
    main()