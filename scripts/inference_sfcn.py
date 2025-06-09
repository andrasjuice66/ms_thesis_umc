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
import json
import wandb
from datetime import datetime
from scipy.stats import norm

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
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

def create_test_dataloader(csv_path, data_dir, transform=None, batch_size=8, num_workers=4):
    """Create a DataLoader for test data with targets"""
    # Read CSV file to get file paths and age labels
    file_paths, ages, sample_weights, sexes, modalities = read_csv(csv_path, data_dir)
    
    # Create dataset
    test_dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=transform,
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

def create_eval_transforms(device, use_domain_rand=False, use_tumor=False):
    """Create evaluation-specific transforms"""
    if not use_domain_rand:
        return None
    
    # Default domain randomization config for evaluation
    eval_rand_cfg = {
        "use_domain_randomization": True,
        "transform_probs": {
            "flip": 0.5,
            "affine": 0.8,
            "contrast": 0.6,
            "gamma": 0.5,
            "blur": 0.4,
            "bias": 0.5,
            "scale_int": 0.4,
            "shift_int": 0.4,
            "hist_shift": 0.3,
            "noise": 0.4,
            "rician": 0.3,
            "gibbs": 0.3,
            "resolution": 0.5,
            "coarse_do": 0.3,
            "crop": 1.0,
            "tumor": 0.3 if use_tumor else 0.0,
        },
        "output_shape": (160, 192, 160),
    }
    
    eval_tumor_cfg = {
        "use_tumor_simulation": use_tumor,
        "prob": 0.3,
        "use_age_based_segmentation": False,  # Simplified for inference
        "perlin_res": [2, 2, 2],
        "tumor_size_factor_range": [0.5, 2.0],
        "use_fluid_dynamics": True,
    } if use_tumor else {}
    
    eval_transform = DomainRandomizer(
        device=device,
        use_tumor_simulation=use_tumor,
        tumor_config=eval_tumor_cfg,
        **eval_rand_cfg,
    )
    
    return eval_transform

def run_single_evaluation(model, test_loader, device, bin_range, bin_step, sigma, bin_centers):
    """Run a single evaluation on the test loader"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
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
    
    return np.array(all_predictions), np.array(all_targets), np.mean(all_losses)

def run_multi_fold_evaluation(model, csv_path, data_dir, device, transform, n_folds, eval_name, 
                              bin_range, bin_step, sigma, bin_centers, batch_size=8):
    """Run evaluation multiple times with different augmentations and average results"""
    print(f"Running {n_folds}-fold {eval_name} evaluation...")
    
    all_metrics = []
    
    for fold in range(n_folds):
        print(f"{eval_name} evaluation fold {fold+1}/{n_folds}")
        
        # Create test dataloader with transform
        test_loader, file_paths, ages, sexes, modalities = create_test_dataloader(
            csv_path, data_dir, transform=transform, batch_size=batch_size
        )
        
        # Run evaluation
        predictions, targets, avg_loss = run_single_evaluation(
            model, test_loader, device, bin_range, bin_step, sigma, bin_centers
        )
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets, modalities, sexes)
        metrics["loss"] = avg_loss
        all_metrics.append(metrics)
    
    # Average metrics across folds
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
    
    print(f"{eval_name} evaluation results (averaged over {n_folds} folds):")
    print(f"MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}")
    print(f"MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
    print(f"R²: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    
    return avg_metrics

def inference_with_3fold_evaluation():
    # Configure paths and parameters
    model_path = '/home/ajoos/model_files/sfcn_original_ckp.p'
    test_csv_path = '/home/ajoos/brain_age_pred/data/labels/test.csv'
    data_dir = '/scratch-shared/ajoos/'
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"sfcn_original_{timestamp}"
    
    WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'
    wandb.login(key=WANDB_API)
    wandb.init(
        project="brainage-inference",
        name=experiment_name,
        config={
            "model": "SFCN",
            "model_path": model_path,
            "test_csv": test_csv_path,
            "batch_size": batch_size,
            "device": str(device),
            "evaluation_type": "3fold_inference",
            "bin_range": [42, 82],
            "bin_step": 1,
            "sigma": 1
        },
        reinit=True,
    )
    
    print(f"Running on device: {device}")
    print(f"W&B experiment: {experiment_name}")
    
    try:
        # Load model
        model = SFCN()
        model = torch.nn.DataParallel(model)
        print(f"Loading model from {model_path}")
        
        # Try the original simple approach first
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            # If there's a device mismatch, try loading to CPU first
            if "device" in str(e).lower():
                print("Device mismatch detected, loading to CPU first...")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                raise
        
        model.to(device)
        model.eval()
        
        # Parameters for age prediction with soft labels
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        bin_centers = bin_range[0] + float(bin_step) / 2 + bin_step * np.arange(int((bin_range[1] - bin_range[0]) / bin_step))
        
        # Read file info for consistent dataset creation
        file_paths, ages, sample_weights, sexes, modalities = read_csv(test_csv_path, data_dir)
        print(f"Loaded {len(file_paths)} samples from {test_csv_path}")
        print(f"Age range: {min(ages):.2f} - {max(ages):.2f}, mean: {np.mean(ages):.2f}")
        
        # Log dataset info to W&B
        wandb.log({
            "dataset/num_samples": len(file_paths),
            "dataset/age_min": min(ages),
            "dataset/age_max": max(ages),
            "dataset/age_mean": np.mean(ages),
            "dataset/age_std": np.std(ages),
        })
        
        print("Starting 3-fold evaluation...")
        
        # 1. Normal test evaluation
        print("=== 1/3: Normal test evaluation ===")
        normal_test_loader, _, _, _, _ = create_test_dataloader(
            test_csv_path, data_dir, transform=None, batch_size=batch_size
        )
        normal_predictions, normal_targets, normal_loss = run_single_evaluation(
            model, normal_test_loader, device, bin_range, bin_step, sigma, bin_centers
        )
        normal_metrics = calculate_metrics(normal_predictions, normal_targets, modalities, sexes)
        normal_metrics["loss"] = normal_loss
        print(f"Normal test results: MAE={normal_metrics['mae']:.4f}, R²={normal_metrics['r2']:.4f}")
        
        # Log normal test results to W&B
        wandb.log({f"test/{k}": v for k, v in normal_metrics.items()})
        
        # 2. Domain randomized test evaluation (10 folds)
        print("=== 2/3: Domain randomized test evaluation ===")
        dom_rand_transform = create_eval_transforms(device, use_domain_rand=True, use_tumor=False)
        dom_rand_metrics = run_multi_fold_evaluation(
            model, test_csv_path, data_dir, device, dom_rand_transform, 
            n_folds=10, eval_name="domain_randomized", 
            bin_range=bin_range, bin_step=bin_step, sigma=sigma, bin_centers=bin_centers,
            batch_size=batch_size
        )
        
        # Log domain randomized results to W&B
        wandb.log({f"test_dom_rand/{k}": v for k, v in dom_rand_metrics.items()})
        
        # 3. Domain randomized + tumor simulation test evaluation (10 folds)
        print("=== 3/3: Domain randomized + tumor simulation test evaluation ===")
        dom_rand_tumor_transform = create_eval_transforms(device, use_domain_rand=True, use_tumor=True)
        dom_rand_tumor_metrics = run_multi_fold_evaluation(
            model, test_csv_path, data_dir, device, dom_rand_tumor_transform, 
            n_folds=10, eval_name="domain_rand_tumor", 
            bin_range=bin_range, bin_step=bin_step, sigma=sigma, bin_centers=bin_centers,
            batch_size=batch_size
        )
        
        # Log domain randomized + tumor results to W&B
        wandb.log({f"test_dom_rand_tumor/{k}": v for k, v in dom_rand_tumor_metrics.items()})
        
        # Log summary comparison to W&B
        wandb.log({
            "evaluation_summary/normal_mae": normal_metrics["mae"],
            "evaluation_summary/dom_rand_mae": dom_rand_metrics["mae"],
            "evaluation_summary/dom_rand_tumor_mae": dom_rand_tumor_metrics["mae"],
            "evaluation_summary/dom_rand_mae_std": dom_rand_metrics["mae_std"],
            "evaluation_summary/dom_rand_tumor_mae_std": dom_rand_tumor_metrics["mae_std"],
            "evaluation_summary/normal_r2": normal_metrics["r2"],
            "evaluation_summary/dom_rand_r2": dom_rand_metrics["r2"],
            "evaluation_summary/dom_rand_tumor_r2": dom_rand_tumor_metrics["r2"],
            "evaluation_summary/normal_loss": normal_metrics["loss"],
            "evaluation_summary/dom_rand_loss": dom_rand_metrics["loss"],
            "evaluation_summary/dom_rand_tumor_loss": dom_rand_tumor_metrics["loss"],
        })
        
        # Save evaluation results
        eval_results = {
            "normal": normal_metrics,
            "domain_randomized": dom_rand_metrics,
            "domain_rand_tumor": dom_rand_tumor_metrics,
        }
        
        with open('sfcn_3fold_evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print("=== Evaluation Summary ===")
        print(f"Normal test MAE: {normal_metrics['mae']:.4f}")
        print(f"Domain rand test MAE: {dom_rand_metrics['mae']:.4f} ± {dom_rand_metrics['mae_std']:.4f}")
        print(f"Domain rand + tumor test MAE: {dom_rand_tumor_metrics['mae']:.4f} ± {dom_rand_tumor_metrics['mae_std']:.4f}")
        
        # Create visualization comparing all three evaluations
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Normal evaluation
        plt.subplot(1, 3, 1)
        plt.scatter(normal_targets, normal_predictions, alpha=0.5)
        plt.plot([min(normal_targets), max(normal_targets)], [min(normal_targets), max(normal_targets)], 'r--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title(f'Normal Test\n(MAE={normal_metrics["mae"]:.2f}, R²={normal_metrics["r2"]:.2f})')
        plt.grid(True)
        
        # Plot 2: MAE comparison
        plt.subplot(1, 3, 2)
        mae_values = [normal_metrics['mae'], dom_rand_metrics['mae'], dom_rand_tumor_metrics['mae']]
        mae_stds = [0, dom_rand_metrics['mae_std'], dom_rand_tumor_metrics['mae_std']]
        labels = ['Normal', 'Domain Rand', 'Dom Rand + Tumor']
        plt.bar(labels, mae_values, yerr=mae_stds, capsize=5)
        plt.ylabel('MAE')
        plt.title('MAE Comparison')
        plt.grid(axis='y')
        
        # Plot 3: R² comparison
        plt.subplot(1, 3, 3)
        r2_values = [normal_metrics['r2'], dom_rand_metrics['r2'], dom_rand_tumor_metrics['r2']]
        r2_stds = [0, dom_rand_metrics['r2_std'], dom_rand_tumor_metrics['r2_std']]
        plt.bar(labels, r2_values, yerr=r2_stds, capsize=5)
        plt.ylabel('R²')
        plt.title('R² Comparison')
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('sfcn_3fold_evaluation_comparison.png', dpi=300, bbox_inches='tight')
        
        # Log the plot to W&B
        wandb.log({"evaluation_plots": wandb.Image('sfcn_3fold_evaluation_comparison.png')})
        plt.close()
        
        # Save detailed normal test results
        results_df = pd.DataFrame({
            'file_path': file_paths,
            'true_age': normal_targets,
            'predicted_age': normal_predictions,
            'brain_age_delta': normal_predictions - normal_targets,
        })
        if sexes is not None:
            results_df['sex'] = sexes
        if modalities is not None:
            results_df['modality'] = modalities
        
        results_df.to_csv('sfcn_normal_test_results.csv', index=False)
        print("Results saved to sfcn_normal_test_results.csv and sfcn_3fold_evaluation_results.json")
        
        return eval_results
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        wandb.finish()

# Keep the original function for backward compatibility
def inference_with_dataloader():
    return inference_with_3fold_evaluation()

if __name__ == "__main__":
    inference_with_3fold_evaluation()