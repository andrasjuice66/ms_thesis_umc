"""
Optimizers and schedulers for brain age prediction models.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    StepLR
)
from typing import Dict, List, Optional, Union, Iterable


def get_optimizer(
    params: Iterable[torch.nn.Parameter],
    optimizer_type: str = "adam",
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get the specified optimizer.
    
    Args:
        params: Model parameters
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Optimizer
    """
    optimizers = {
        "adam": optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999))
        ),
        "adamw": optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999))
        ),
        "sgd": optim.SGD(
            params,
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get("nesterov", True)
        ),
        "rmsprop": optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.0),
            alpha=kwargs.get("alpha", 0.99)
        )
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizers[optimizer_type]


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get the specified learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Scheduler or None if scheduler_type is "none"
    """
    if scheduler_type == "none":
        return None
    
    schedulers = {
        "cosine": CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 10),
            eta_min=kwargs.get("eta_min", 1e-6)
        ),
        "plateau": ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 5),
            min_lr=kwargs.get("min_lr", 1e-6)
        ),
        "onecycle": OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            total_steps=kwargs.get("total_steps", None),
            epochs=kwargs.get("epochs", None),
            steps_per_epoch=kwargs.get("steps_per_epoch", None),
            pct_start=kwargs.get("pct_start", 0.3),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4)
        ),
        "step": StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1)
        )
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return schedulers[scheduler_type]
