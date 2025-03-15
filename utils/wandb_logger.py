"""
Weights & Biases logger for brain age prediction.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
import os


class WandbLogger:
    """
    Logger for Weights & Biases.
    """
    
    def __init__(
        self,
        project: str = "brain-age-prediction",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        dir: Optional[str] = None,
        mode: str = "online"
    ):
        """
        Initialize the logger.
        
        Args:
            project: Project name
            entity: Entity name
            name: Run name
            config: Run configuration
            tags: Run tags
            notes: Run notes
            group: Run group
            job_type: Run job type
            dir: Directory to store W&B files
            mode: W&B mode ('online', 'offline', 'disabled')
        """
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            job_type=job_type,
            dir=dir,
            mode=mode
        )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Metrics to log
            step: Step number
        """
        self.run.log(metrics, step=step)
    
    def log_predictions(self, predictions: np.ndarray, targets: np.ndarray, epoch: int):
        """
        Log predictions vs. targets plot to W&B.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            epoch: Current epoch
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot identity line
        min_val = min(np.min(predictions), np.min(targets))
        max_val = max(np.max(predictions), np.max(targets))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Plot predictions vs. targets
        ax.scatter(targets, predictions, alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel('True Age')
        ax.set_ylabel('Predicted Age')
        ax.set_title(f'Predictions vs. Targets (Epoch {epoch+1})')
        
        # Add correlation coefficient
        correlation = np.corrcoef(predictions, targets)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes)
        
        # Log to W&B
        self.run.log({f"predictions_plot_epoch_{epoch+1}": wandb.Image(fig)})
        
        # Close figure to free memory
        plt.close(fig)
    
    def log_model(self, model_path: str, name: Optional[str] = None):
        """
        Log model to W&B.
        
        Args:
            model_path: Path to model file
            name: Model name
        """
        artifact = wandb.Artifact(
            name=name or os.path.basename(model_path),
            type="model"
        )
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish the W&B run."""
        self.run.finish() 