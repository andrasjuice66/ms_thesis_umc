import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import sys
from scipy.stats import norm

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.utils.utils import read_csv
from brain_age_pred.training.metrics import calculate_metrics

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        
        
def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop



def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    return loss

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out

# Function to create a test dataloader
def create_test_dataloader(csv_path, data_dir, batch_size=8, num_workers=4):
    """Create a DataLoader for test data with targets"""
    # Read CSV file to get file paths and age labels
    file_paths, ages, sample_weights, sexes, modalities = read_csv(csv_path, data_dir)
    
    print(f"Loaded {len(file_paths)} samples from {csv_path}")
    print(f"Age range: {min(ages):.2f} - {max(ages):.2f}, mean: {np.mean(ages):.2f}")
    
    # Create dataset
    test_dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=None,
        mode="test",
        cache_size=0,
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader, file_paths, ages, sexes, modalities

def inference_with_dataloader():
    # Configure paths and parameters
    model_path = '/home/ajoos/model_files/sfcn_original_ckp.p'
    test_csv_path = '/home/ajoos/brain_age_pred/data/labels/test.csv'
    data_dir = '/scratch-shared/ajoos/'
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running on device: {device}")
    
    # Create test dataloader
    test_loader, file_paths, true_ages, sexes, modalities = create_test_dataloader(
        test_csv_path, data_dir, batch_size=batch_size
    )
    
    print(f"Loaded test dataloader with {len(test_loader)} batches ({len(file_paths)} samples)")
    
    # Load model
    model = SFCN()
    model = torch.nn.DataParallel(model)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model.to(device)
    model.eval()
    
    # Parameters for age prediction with soft labels
    bin_range = [42, 82]
    bin_step = 1
    sigma = 1
    bin_centers = bin_range[0] + float(bin_step) / 2 + bin_step * np.arange(int((bin_range[1] - bin_range[0]) / bin_step))
    
    # Evaluation loop
    all_predictions = []
    all_targets = []
    all_losses = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get inputs and targets
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)
            
            # Generate soft labels for the targets
            soft_targets = []
            for age in targets.cpu().numpy():
                soft_target, _ = num2vect(age, bin_range, bin_step, sigma)
                soft_targets.append(soft_target)
            
            soft_targets = torch.tensor(np.array(soft_targets), dtype=torch.float32).to(device)
            
            # Forward pass
            outputs = model(inputs)
            log_probs = outputs[0]
            
            # Calculate loss
            loss = my_KLDivLoss(log_probs, soft_targets).item()
            
            # Convert log probabilities to expected age
            probs = torch.exp(log_probs)
            predictions = torch.sum(probs * torch.tensor(bin_centers, device=device), dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_losses.append(loss)
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)}, Loss: {loss:.4f}")
    
    # Convert to numpy arrays for analysis
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate comprehensive metrics using the project's metrics function
    metrics = calculate_metrics(predictions, targets, modalities, sexes)
    
    # Print detailed metrics
    print("\nEvaluation Results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Create and save visualizations
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Brain Age Prediction Results (MAE={metrics["mae"]:.2f}, R²={metrics["r2"]:.2f})')
    plt.grid(True)
    plt.savefig('brain_age_prediction_results.png')
    
    # Create brain age delta plot
    plt.figure(figsize=(10, 6))
    brain_age_delta = predictions - targets
    plt.hist(brain_age_delta, bins=30)
    plt.xlabel('Brain Age Delta (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title(f'Brain Age Delta Distribution (Mean={metrics["brain_age_delta"]:.2f})')
    plt.grid(True)
    plt.savefig('brain_age_delta_distribution.png')
    
    # If there are multiple modalities, create modality-specific plots
    if modalities is not None and len(np.unique(modalities)) > 1:
        plt.figure(figsize=(12, 8))
        unique_modalities = np.unique(modalities)
        for i, modality in enumerate(unique_modalities):
            mask = np.array(modalities) == modality
            plt.scatter(targets[mask], predictions[mask], alpha=0.5, label=f'{modality} (MAE={metrics.get(f"{modality}_mae", 0):.2f})')
        
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Brain Age Prediction by Modality')
        plt.legend()
        plt.grid(True)
        plt.savefig('brain_age_prediction_by_modality.png')
    
    # If there are sex values, create sex-specific plots
    if sexes is not None and len(np.unique(sexes)) > 1:
        plt.figure(figsize=(12, 8))
        unique_sexes = np.unique(sexes)
        for i, sex in enumerate(unique_sexes):
            mask = np.array(sexes) == sex
            plt.scatter(targets[mask], predictions[mask], alpha=0.5, label=f'{sex} (MAE={metrics.get(f"{sex}_mae", 0):.2f})')
        
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Brain Age Prediction by Sex')
        plt.legend()
        plt.grid(True)
        plt.savefig('brain_age_prediction_by_sex.png')
    
    # Save age-specific MAE bar chart
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_specific_mae = {}
    for i in range(len(age_bins) - 1):
        bin_start = age_bins[i]
        bin_end = age_bins[i+1]
        metric_key = f"mae_{bin_start}_{bin_end}"
        if metric_key in metrics:
            age_specific_mae[f"{bin_start}-{bin_end}"] = metrics[metric_key]
    
    if age_specific_mae:
        plt.figure(figsize=(10, 6))
        plt.bar(age_specific_mae.keys(), age_specific_mae.values())
        plt.xlabel('Age Range')
        plt.ylabel('MAE')
        plt.title('Age-Specific Mean Absolute Error')
        plt.grid(axis='y')
        plt.savefig('age_specific_mae.png')
    
    # Save all plots
    plt.close('all')
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_age': targets,
        'predicted_age': predictions,
        'brain_age_delta': predictions - targets,
    })
    if sexes is not None:
        results_df['sex'] = sexes
    if modalities is not None:
        results_df['modality'] = modalities
    
    results_df.to_csv('brain_age_prediction_results.csv', index=False)
    print("Results saved to brain_age_prediction_results.csv")
    
    # Return metrics for further analysis if needed
    return metrics, results_df

if __name__ == "__main__":
    inference_with_dataloader()