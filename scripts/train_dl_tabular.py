#!/usr/bin/env python
"""
Deep Learning training pipeline for brain age prediction using tabular segmentation features.
Uses a ResNet-style MLP with skip connections, batch normalization, and dropout.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from time import perf_counter
from contextlib import contextmanager
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed

warnings.filterwarnings('ignore')


class BrainAgeDataset(Dataset):
    """PyTorch Dataset for tabular brain age data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ResidualBlock(nn.Module):
    """Residual block for tabular data with skip connection."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for feature importance."""
    
    def __init__(self, features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(features, features // 4),
            nn.ReLU(),
            nn.Linear(features // 4, features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att_weights = self.attention(x)
        return x * att_weights


class TabularResNet(nn.Module):
    """
    ResNet-style architecture for tabular data with:
    - Multiple residual blocks
    - Batch normalization
    - Dropout regularization
    - Optional attention mechanism
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128, 64],
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Input projection
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)
            )
        
        # Optional attention
        if use_attention:
            self.attention = AttentionBlock(hidden_dims[-1])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
    
    def forward(self, x):
        # Input processing
        x = self.input_bn(x)
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Output
        x = self.output_head(x)
        return x.squeeze(-1)


class TabularMLP(nn.Module):
    """
    Simple but powerful MLP for tabular data with:
    - Multiple hidden layers
    - Batch normalization
    - Dropout
    - LeakyReLU activation
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [1024, 512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Input batch norm
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class DeepTabularBrainAgePredictor:
    """
    Deep Learning pipeline for brain age prediction using segmentation features.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.icv_resid_params = {}
        self.eps = 1e-8
        
        # Initialize logger
        self.logger = setup_logger(
            "dl-tabular-brain-age",
            log_file=Path(config['output']['log_dir']) / "train.log"
        )
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize W&B if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'brain-age-dl-tabular'),
                name=config.get('experiment_name'),
                config=config
            )
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Experiment: {config.get('experiment_name')}")
    
    @contextmanager
    def _log_section(self, title: str):
        start = perf_counter()
        self.logger.info(f"[start] {title}")
        try:
            yield
        finally:
            self.logger.info(f"[done]  {title} in {perf_counter() - start:.2f}s")
    
    def _log_dataframe_overview(self, df: pd.DataFrame, name: str):
        self.logger.info(f"{name}: shape={df.shape}, columns={len(df.columns)}")
    
    def _csv_line_from_index(self, idx: Union[int, np.integer]) -> int:
        try:
            return int(idx) + 2
        except Exception:
            return -1
    
    def _validate_labels(self, labels_df: pd.DataFrame, split_name: str, source_csv: str) -> pd.DataFrame:
        """Validate labels - same as train_ml.py"""
        required_cols = ['subject_id', 'age', 'sex', 'modality', 'dataset', 'image_path']
        missing_cols = [c for c in required_cols if c not in labels_df.columns]
        if missing_cols:
            self.logger.error(f"[{split_name}] Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        labels_df = labels_df.copy()
        age_num = pd.to_numeric(labels_df['age'], errors='coerce')
        labels_df['age'] = age_num
        labels_df = labels_df[labels_df['age'].notna()]
        
        missing_path = labels_df[labels_df['image_path'].isna() | 
                                  (labels_df['image_path'].astype(str).str.strip() == '')]
        if len(missing_path) > 0:
            labels_df = labels_df[~labels_df.index.isin(missing_path.index)]
        
        return labels_df
    
    def load_segmentation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load segmentation data - same as train_ml.py"""
        self.logger.info("Loading segmentation data...")
        
        with self._log_section("Read label CSVs"):
            train_csv = self.config['data']['train_csv']
            val_csv = self.config['data']['val_csv']
            test_csv = self.config['data']['test_csv']
            labels_train = pd.read_csv(train_csv)
            labels_val = pd.read_csv(val_csv)
            labels_test = pd.read_csv(test_csv)
        
        labels_train = self._validate_labels(labels_train, 'train', train_csv)
        labels_val = self._validate_labels(labels_val, 'val', val_csv)
        labels_test = self._validate_labels(labels_test, 'test', test_csv)
        
        seg_data_dir = Path(self.config['data']['segmented_data_dir'])
        
        def load_split(labels_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            rows = []
            missing = 0
            with self._log_section(f"Load segmentation CSVs [{split_name}]"):
                for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"seg {split_name}"):
                    seg_csv_path = seg_data_dir / (row['image_path'].replace('.nii.gz', '.csv'))
                    if seg_csv_path.exists():
                        try:
                            seg_df = pd.read_csv(seg_csv_path)
                            if len(seg_df) > 0:
                                seg_row = seg_df.iloc[0].to_dict()
                                seg_row.update({
                                    'subject_id': row['subject_id'],
                                    'age': row['age'],
                                    'sex': row['sex'],
                                    'modality': row['modality'],
                                    'dataset': row['dataset'],
                                    'image_path': row['image_path']
                                })
                                rows.append(seg_row)
                        except Exception as e:
                            pass
                    else:
                        missing += 1
            
            df = pd.DataFrame(rows)
            self.logger.info(f"[{split_name}] loaded {len(df)} samples, {missing} missing")
            return df
        
        train_df = load_split(labels_train, 'train')
        val_df = load_split(labels_val, 'val')
        test_df = load_split(labels_test, 'test')
        
        return train_df, val_df, test_df
    
    def _engineer_features(self, X: pd.DataFrame, volumetric_features: List[str], 
                          split_name: str = 'train') -> pd.DataFrame:
        """Feature engineering - same as train_ml.py"""
        df = X.copy()
        cols_lower = {c.lower(): c for c in df.columns}
        
        # Find ICV column
        icv_aliases = {
            'total intracranial', 'intracranial volume',
            'estimated total intracranial volume', 'total_intracranial', 'etiv'
        }
        icv_col = None
        for lc, orig in cols_lower.items():
            if lc in icv_aliases:
                icv_col = orig
                break
        
        def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a / (b.replace(0, np.nan) + self.eps)
        
        def regex_strip_side(name: str) -> str:
            return re.sub(r'^(left|right)\s+', '', name, flags=re.IGNORECASE)
        
        def pick(name_variants: List[str]) -> Optional[str]:
            for v in name_variants:
                if v in df.columns:
                    return v
            return None
        
        # ICV normalization
        if icv_col is not None:
            base_feats = [c for c in volumetric_features if c != icv_col and c in df.columns]
            for col in base_feats:
                df[f'{col}_nicv'] = safe_div(df[col], df[icv_col])
        
        # ICV residualization
        if icv_col is not None:
            base_feats = [c for c in volumetric_features if c != icv_col and c in df.columns]
            icv = df[icv_col]
            
            if split_name == 'train':
                self.icv_resid_params = {}
                icv_mean = icv.mean()
                icv_var = icv.var(ddof=0) + self.eps
                
                for col in base_feats:
                    y = df[col]
                    cov = ((y - y.mean()) * (icv - icv_mean)).mean()
                    b = cov / icv_var
                    a = y.mean() - b * icv_mean
                    self.icv_resid_params[col] = (float(a), float(b))
                    df[f'{col}_icv_resid'] = y - (a + b * icv)
            else:
                for col in base_feats:
                    if col in self.icv_resid_params:
                        a, b = self.icv_resid_params[col]
                        df[f'{col}_icv_resid'] = df[col] - (a + b * df[icv_col])
        
        # Left-right bilateral features
        all_cols = df.columns.tolist()
        left_cols = [c for c in all_cols if re.match(r'^(?i)left\s+', c)]
        base_to_pair = {}
        for lc in left_cols:
            base = regex_strip_side(lc).lower()
            candidates = [c for c in all_cols if re.match(r'^(?i)right\s+', c) 
                         and regex_strip_side(c).lower() == base]
            if candidates:
                base_to_pair[lc] = candidates[0]
        
        for lcol, rcol in base_to_pair.items():
            df[f'{lcol}_bilateral'] = df[lcol] + df[rcol]
            df[f'{lcol}_diff'] = df[lcol] - df[rcol]
            df[f'{lcol}_asymmetry'] = safe_div(df[lcol] - df[rcol], df[lcol] + df[rcol])
            
            if icv_col is not None:
                ln, rn = f'{lcol}_nicv', f'{rcol}_nicv'
                if ln in df.columns and rn in df.columns:
                    df[f'{lcol}_bilateral_nicv'] = df[ln] + df[rn]
                    df[f'{lcol}_asymmetry_nicv'] = safe_div(df[ln] - df[rn], df[ln] + df[rn])
            
            lr, rr = f'{lcol}_icv_resid', f'{rcol}_icv_resid'
            if lr in df.columns and rr in df.columns:
                df[f'{lcol}_bilateral_resid'] = df[lr] + df[rr]
                df[f'{lcol}_asymmetry_resid'] = safe_div(df[lr] - df[rr], df[lr] + df[rr])
        
        # Tissue/system composites
        cortex_cerebral = [c for c in all_cols if 'cerebral cortex' in c.lower()]
        cortex_cerebellum = [c for c in all_cols if 'cerebellum cortex' in c.lower()]
        wm_cerebral = [c for c in all_cols if 'cerebral white matter' in c.lower()]
        wm_cerebellum = [c for c in all_cols if 'cerebellum white matter' in c.lower()]
        
        sub_tokens = ['thalamus', 'caudate', 'putamen', 'pallidum', 
                     'hippocampus', 'amygdala', 'accumbens', 'ventral dc']
        subcort_cols = [c for c in all_cols if any(t in c.lower() for t in sub_tokens)]
        
        if cortex_cerebral:
            df['cortex_total'] = df[cortex_cerebral].sum(axis=1)
        if cortex_cerebellum:
            df['cerebellum_cortex_total'] = df[cortex_cerebellum].sum(axis=1)
        if wm_cerebral:
            df['cerebral_wm_total'] = df[wm_cerebral].sum(axis=1)
        if wm_cerebellum:
            df['cerebellum_wm_total'] = df[wm_cerebellum].sum(axis=1)
        if subcort_cols:
            df['gm_subcort_total'] = df[subcort_cols].sum(axis=1)
        
        if all(c in df.columns for c in ['cortex_total', 'gm_subcort_total', 'cerebellum_cortex_total']):
            df['gm_total'] = df['cortex_total'] + df['gm_subcort_total'] + df['cerebellum_cortex_total']
        
        if all(c in df.columns for c in ['cerebral_wm_total', 'cerebellum_wm_total']):
            df['wm_total'] = df['cerebral_wm_total'] + df['cerebellum_wm_total']
        
        # Ventricular features
        lat_L = pick([c for c in all_cols if c.lower() == 'left lateral ventricle'])
        lat_R = pick([c for c in all_cols if c.lower() == 'right lateral ventricle'])
        if lat_L and lat_R:
            df['lat_ventricles_total'] = df[lat_L] + df[lat_R]
        
        # More composites
        bg_cols = [c for c in all_cols if any(t in c.lower() 
                  for t in ['caudate', 'putamen', 'pallidom', 'accumbens'])]
        if bg_cols:
            df['basal_ganglia_total'] = df[bg_cols].sum(axis=1)
        
        limbic_cols = [c for c in all_cols if any(t in c.lower() 
                      for t in ['hippocampus', 'amygdala'])]
        if limbic_cols:
            df['limbic_total'] = df[limbic_cols].sum(axis=1)
        
        # Ratios
        if icv_col is not None:
            for name in ['cortex_total', 'wm_total', 'limbic_total']:
                if name in df.columns:
                    df[f'{name}_nicv'] = safe_div(df[name], df[icv_col])
            
            if all(c in df.columns for c in ['cortex_total', 'wm_total']):
                df['cortex_to_wm'] = safe_div(df['cortex_total'], df['wm_total'])
        
        # Nonlinear features
        for base in ['cortex_total_nicv', 'wm_total_nicv', 'limbic_total_nicv']:
            if base in df.columns:
                df[f'{base}__sq'] = df[base] ** 2
        
        # Interactions
        inter_pairs = [
            ('cortex_total_nicv', 'wm_total_nicv'),
            ('limbic_total_nicv', 'cortex_total_nicv'),
        ]
        for a, b in inter_pairs:
            if a in df.columns and b in df.columns:
                df[f'{a}__x__{b}'] = df[a] * df[b]
        
        return df
    
    def preprocess_features(self, df: pd.DataFrame, split_name: str = 'train') -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess features - same as train_ml.py"""
        target_col = 'age'
        exclude_cols = ['subject_id', 'image_path', 'age', 'subject', 'sex', 'modality', 'dataset']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        volumetric_features = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[volumetric_features].copy()
        y = df[target_col].copy()
        
        X[volumetric_features] = X[volumetric_features].fillna(X[volumetric_features].median())
        
        with self._log_section("Feature engineering"):
            X = self._engineer_features(X, volumetric_features, split_name=split_name)
        
        if not self.feature_names:
            self.feature_names = X.columns.tolist()
            self.logger.info(f"Feature count: {len(self.feature_names)}")
        
        return X, y
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X
        return X.reindex(columns=self.feature_names, fill_value=0)
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create the neural network model."""
        arch_type = self.config['model'].get('architecture', 'resnet')
        
        if arch_type == 'resnet':
            model = TabularResNet(
                input_dim=input_dim,
                hidden_dims=self.config['model'].get('hidden_dims', [512, 256, 128, 64]),
                dropout=self.config['model'].get('dropout', 0.3),
                use_attention=self.config['model'].get('use_attention', True)
            )
        elif arch_type == 'mlp':
            model = TabularMLP(
                input_dim=input_dim,
                hidden_dims=self.config['model'].get('hidden_dims', [1024, 512, 256, 128]),
                dropout=self.config['model'].get('dropout', 0.3)
            )
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
        
        return model.to(self.device)
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer, criterion, scheduler=None) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          self.config['training'].get('grad_clip', 1.0))
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluate model."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                outputs = model(features)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return mae, mse, r2, all_preds, all_targets
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> nn.Module:
        """Train the deep learning model."""
        self.logger.info("Training deep learning model...")
        
        # Scale features
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = MinMaxScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = BrainAgeDataset(X_train_scaled, y_train.values)
        val_dataset = BrainAgeDataset(X_val_scaled, y_val.values)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create model
        model = self.create_model(X_train.shape[1])
        self.logger.info(f"Model architecture: {self.config['model'].get('architecture')}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer
        opt_type = self.config['training'].get('optimizer', 'adamw')
        if opt_type == 'adamw':
            optimizer = AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 1e-5)
            )
        else:
            optimizer = Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )
        
        # Loss function
        criterion = nn.L1Loss()  # MAE loss
        
        # Scheduler
        scheduler_type = self.config['training'].get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.config['training']['epochs']
            )
        else:
            scheduler = None
        
        # Training loop
        best_val_mae = float('inf')
        patience_counter = 0
        patience = self.config['training'].get('early_stopping_patience', 20)
        
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_mae, val_mse, val_r2, _, _ = self.evaluate(model, val_loader)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_mae)
                else:
                    scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, "
                f"Val R²: {val_r2:.4f}"
            )
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_mae': val_mae,
                    'val_mse': val_mse,
                    'val_r2': val_r2,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, Path(self.config['output']['output_dir']) / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        checkpoint = torch.load(Path(self.config['output']['output_dir']) / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Best validation MAE: {checkpoint['val_mae']:.4f}")
        
        return model
    
    def generate_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray, 
                               results: Dict):
        """Generate visualizations."""
        output_dir = Path(self.config['output']['output_dir'])
        
        # Prediction scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Age', fontsize=14)
        plt.ylabel('Predicted Age', fontsize=14)
        plt.title(f"Deep Learning Brain Age Prediction\nMAE: {results['test_mae']:.2f}, R²: {results['test_r2']:.3f}", 
                 fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residuals plot
        residuals = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('True Age', fontsize=14)
        plt.ylabel('Residuals (Predicted - True)', fontsize=14)
        plt.title('Residuals Plot', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution of errors
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals (years)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Prediction Errors', fontsize=16)
        plt.axvline(x=0, color='r', linestyle='--', lw=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config.get('use_wandb', False):
            wandb.log({
                "prediction_scatter": wandb.Image(str(output_dir / 'prediction_scatter.png')),
                "residuals_plot": wandb.Image(str(output_dir / 'residuals_plot.png')),
                "error_distribution": wandb.Image(str(output_dir / 'error_distribution.png'))
            })
    
    def run_pipeline(self):
        """Run the complete training pipeline."""
        self.logger.info("Starting deep learning tabular pipeline...")
        
        # Load data
        train_df, val_df, test_df = self.load_segmentation_data()
        
        # Preprocess
        X_train, y_train = self.preprocess_features(train_df, split_name='train')
        X_val, y_val = self.preprocess_features(val_df, split_name='val')
        X_test, y_test = self.preprocess_features(test_df, split_name='test')
        
        X_val = self._align_features(X_val)
        X_test = self._align_features(X_test)
        
        self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Features: {X_train.shape[1]}")
        
        # Train model
        model = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        test_dataset = BrainAgeDataset(X_test_scaled, y_test.values)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        test_mae, test_mse, test_r2, y_pred, y_true = self.evaluate(model, test_loader)
        
        results = {
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'config': self.config
        }
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"TRAINING COMPLETE")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Test MAE: {test_mae:.3f}")
        self.logger.info(f"Test RMSE: {np.sqrt(test_mse):.3f}")
        self.logger.info(f"Test R²: {test_r2:.3f}")
        
        # Save results
        output_dir = Path(self.config['output']['output_dir'])
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Visualizations
        self.generate_visualizations(y_true, y_pred, results)
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'final_test_mae': test_mae,
                'final_test_rmse': np.sqrt(test_mse),
                'final_test_r2': test_r2
            })
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train DL model on tabular features')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['experiment_name'] = f"dl_tabular_{timestamp}"
    
    # Create output directory
    output_dir = Path(config['output']['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output']['output_dir'] = str(output_dir)
    config['output']['log_dir'] = str(output_dir)
    
    # Run pipeline
    predictor = DeepTabularBrainAgePredictor(config)
    predictor.run_pipeline()


if __name__ == "__main__":
    main()

