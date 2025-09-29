#!/usr/bin/env python
"""
Traditional ML training pipeline for brain age prediction using tabular segmentation features.
Supports XGBoost, LightGBM, Random Forest, and ensemble methods.
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
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, KFold
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import optuna
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

class TabularBrainAgePredictor:
    """
    Comprehensive traditional ML pipeline for brain age prediction using segmentation features.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.results = {}
        self.icv_resid_params = {}   # per-feature {col: (intercept, slope)} fit on train
        self.eps = 1e-8              # small epsilon for divisions
        
        # Initialize logger
        self.logger = setup_logger(
            "tabular-brain-age", 
            log_file=Path(config['output']['log_dir']) / "train.log"
        )
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize W&B if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'brain-age-tabular'),
                name=config.get('experiment_name'),
                config=config
            )
        # Run summary
        tr_cfg = self.config.get('training', {})
        enabled_models = [m for m, cfg in self.config.get('models', {}).items() if cfg.get('enabled', True)]
        self.logger.info(
            f"Experiment '{self.config.get('experiment_name')}' seed={self.config.get('seed', 42)} "
            f"cv_folds={tr_cfg.get('cv_folds', 5)} n_trials={tr_cfg.get('n_trials', 100)} "
            f"optimize={tr_cfg.get('optimize_hyperparams', False)} models={enabled_models}"
        )

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
        if df.isnull().any().any():
            na = df.isnull().sum()
            na = na[na > 0].sort_values(ascending=False).head(10)
            if not na.empty:
                self.logger.info(f"{name}: top missing columns: {na.to_dict()}")
    
    def _csv_line_from_index(self, idx: Union[int, np.integer]) -> int:
        """Return 1-based CSV line number (including header) for a 0-based DataFrame index."""
        try:
            return int(idx) + 2
        except Exception:
            return -1

    def _validate_labels(self, labels_df: pd.DataFrame, split_name: str, source_csv: str) -> pd.DataFrame:
        """Validate labels and log precise issues with CSV line numbers and file path."""
        required_cols = ['subject_id', 'age', 'sex', 'modality', 'dataset', 'image_path']
        missing_cols = [c for c in required_cols if c not in labels_df.columns]
        if missing_cols:
            self.logger.error(f"[{split_name}] Labels file '{source_csv}' missing required columns: {missing_cols}")
            raise ValueError(f"Missing columns in {source_csv}: {missing_cols}")

        # Age numeric validation
        age_num = pd.to_numeric(labels_df['age'], errors='coerce')
        invalid_age = labels_df[age_num.isna()]
        for idx, row in invalid_age.iterrows():
            line_no = self._csv_line_from_index(idx)
            self.logger.error(
                f"[{split_name}] Labels '{source_csv}' line {line_no}: invalid age='{row.get('age')}' "
                f"subject_id={row.get('subject_id')} image_path={row.get('image_path')}"
            )
        # Replace age with numeric and drop invalids
        labels_df = labels_df.copy()
        labels_df['age'] = age_num
        if len(invalid_age) > 0:
            self.logger.warning(f"[{split_name}] Dropping {len(invalid_age)} rows with invalid age from '{source_csv}'")
            labels_df = labels_df[labels_df['age'].notna()]

        # Age plausible range (optional but helpful)
        out_of_range = labels_df[(labels_df['age'] < 0) | (labels_df['age'] > 120)]
        for idx, row in out_of_range.iterrows():
            line_no = self._csv_line_from_index(idx)
            self.logger.warning(
                f"[{split_name}] Labels '{source_csv}' line {line_no}: implausible age={row.get('age')} "
                f"subject_id={row.get('subject_id')} image_path={row.get('image_path')}"
            )

        # image_path presence
        missing_path = labels_df[labels_df['image_path'].isna() | (labels_df['image_path'].astype(str).str.strip() == '')]
        for idx, row in missing_path.iterrows():
            line_no = self._csv_line_from_index(idx)
            self.logger.error(
                f"[{split_name}] Labels '{source_csv}' line {line_no}: missing image_path "
                f"subject_id={row.get('subject_id')}"
            )
        if len(missing_path) > 0:
            self.logger.warning(f"[{split_name}] Dropping {len(missing_path)} rows with missing image_path from '{source_csv}'")
            labels_df = labels_df[~labels_df.index.isin(missing_path.index)]

        # Duplicates
        dup_mask = labels_df.duplicated(subset=['image_path'], keep=False)
        dups = labels_df[dup_mask]
        if len(dups) > 0:
            sample = min(10, len(dups))
            self.logger.warning(f"[{split_name}] Found {len(dups)} duplicate image_path rows in '{source_csv}'. Showing first {sample}:")
            for idx, row in dups.head(sample).iterrows():
                line_no = self._csv_line_from_index(idx)
                self.logger.warning(
                    f"  line {line_no}: subject_id={row.get('subject_id')} image_path={row.get('image_path')}"
                )

        # Missing sex (will be imputed later but log explicitly)
        missing_sex = labels_df[labels_df['sex'].isna() | (labels_df['sex'].astype(str).str.strip() == '')]
        for idx, row in missing_sex.iterrows():
            line_no = self._csv_line_from_index(idx)
            self.logger.warning(
                f"[{split_name}] Labels '{source_csv}' line {line_no}: missing sex subject_id={row.get('subject_id')}"
            )

        return labels_df
    
    def load_segmentation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load CSV labels for train/val/test and join with per-image segmentation features.
        Returns three dataframes: train_df, val_df, test_df.
        """
        self.logger.info("Loading segmentation data...")

        # Load label CSVs
        with self._log_section("Read label CSVs"):
            train_csv = self.config['data']['train_csv']
            val_csv = self.config['data']['val_csv']
            test_csv = self.config['data']['test_csv']
            labels_train = pd.read_csv(train_csv)
            labels_val = pd.read_csv(val_csv)
            labels_test = pd.read_csv(test_csv)

        # Validate labels with detailed logs
        labels_train = self._validate_labels(labels_train, 'train', train_csv)
        labels_val = self._validate_labels(labels_val, 'val', val_csv)
        labels_test = self._validate_labels(labels_test, 'test', test_csv)

        self._log_dataframe_overview(labels_train, "labels_train")
        self._log_dataframe_overview(labels_val, "labels_val")
        self._log_dataframe_overview(labels_test, "labels_test")

        seg_data_dir = Path(self.config['data']['segmented_data_dir'])

        def load_split(labels_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            rows: List[dict] = []
            missing = 0
            failed = 0
            with self._log_section(f"Load segmentation CSVs [{split_name}]"):
                for i, (_, row) in enumerate(tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"seg {split_name}", leave=False)):
                    image_path = row['image_path']
                    seg_csv_path = seg_data_dir / (image_path.replace('.nii.gz', '.csv'))
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
                            failed += 1
                            csv_line = self._csv_line_from_index(i)
                            self.logger.warning(f"[{split_name}] Failed to load {seg_csv_path} (from labels line {csv_line}): {e}")
                    else:
                        missing += 1
                        if missing <= 10:
                            csv_line = self._csv_line_from_index(i)
                            self.logger.warning(f"[{split_name}] Segmentation file not found: {seg_csv_path} (labels line {csv_line})")
            if missing > 10:
                self.logger.warning(f"[{split_name}] {missing} segmentation CSVs missing (first 10 shown above)")
            if failed > 0:
                self.logger.warning(f"[{split_name}] {failed} segmentation CSVs failed to read")
            df = pd.DataFrame(rows)
            self._log_dataframe_overview(df, f"features_{split_name}")
            self.logger.info(f"[{split_name}] loaded {len(df)} samples with segmentation features")
            return df

        train_df = load_split(labels_train, 'train')
        val_df = load_split(labels_val, 'val')
        test_df = load_split(labels_test, 'test')

        self.logger.info(
            f"Loaded splits -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}"
        )
        return train_df, val_df, test_df
    
    def preprocess_features(self, df: pd.DataFrame, split_name: str = 'train') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features including scaling and encoding.
        """
        self.logger.info("Preprocessing features...")
        
        # Separate features and target
        target_col = 'age'
        exclude_cols = ['subject_id', 'image_path', 'age', 'subject', 'sex', 'modality', 'dataset']
        
        # Get volumetric features (all numeric columns except metadata)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        volumetric_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove categorical features - only use segmentation features
        categorical_features = []  # Empty list - no categorical features from labels
        
        self.logger.info(f"Found {len(volumetric_features)} volumetric features")
        self.logger.info(f"Using only segmentation features, excluding label metadata")
        
        # Create feature matrix - only volumetric features
        X = df[volumetric_features].copy()
        y = df[target_col].copy()
        
        # Handle missing values - only numeric imputation needed
        self.logger.info("Imputing missing: numeric=median")
        X[volumetric_features] = X[volumetric_features].fillna(X[volumetric_features].median())
        
        # No categorical encoding needed since we're not using categorical features
        
        # Feature engineering
        with self._log_section("Feature engineering"):
            X = self._engineer_features(X, volumetric_features, split_name=split_name)  
                  
        # Store feature names if not already set (train first)
        if not self.feature_names:
            self.feature_names = X.columns.tolist()
            self.logger.info(f"Feature count after preprocessing (train): {len(self.feature_names)}")
        
        return X, y

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X
        return X.reindex(columns=self.feature_names, fill_value=0)
    
    def _engineer_features(self, X: pd.DataFrame, volumetric_features: List[str], split_name: str = 'train') -> pd.DataFrame:
        """
        Create engineered features:
        - ICV normalization and residualization (fit on train, apply to val/test)
        - Left-right bilateral sums, differences, asymmetries
        - Tissue/system composites (cortex, WM, subcortex, ventricles, CSF, basal ganglia, limbic)
        - Ratios capturing atrophy patterns
        - Limited nonlinearities (log1p for skewed ratios; a few squared terms and interactions)
        """
        self.logger.info("Engineering additional features...")

        df = X.copy()
        cols_lower = {c.lower(): c for c in df.columns}

        # Identify ICV column (robust to naming)
        icv_aliases = {
            'total intracranial',
            'intracranial volume',
            'estimated total intracranial volume',
            'total_intracranial',
            'etiv'
        }
        icv_col = None
        for lc, orig in cols_lower.items():
            if lc in icv_aliases:
                icv_col = orig
                break

        # Helper functions
        def has_col(name: str) -> bool:
            return name in df.columns

        def pick(name_variants: List[str]) -> Optional[str]:
            """Pick the first existing column from a list of variants."""
            for v in name_variants:
                if v in df.columns:
                    return v
            return None

        def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a / (b.replace(0, np.nan) + self.eps)

        def regex_strip_side(name: str) -> str:
            """Remove leading 'left ' or 'right ' (case-insensitive) as whole word."""
            return re.sub(r'^(left|right)\s+', '', name, flags=re.IGNORECASE)

        # 1) ICV normalization
        if icv_col is not None:
            base_feats = [c for c in volumetric_features if c != icv_col and c in df.columns]
            for col in base_feats:
                df[f'{col}_nicv'] = safe_div(df[col], df[icv_col])

        # 2) ICV residualization (fit on train only)
        #    For each raw region f: fit f ~ a + b*ICV on train; store (a,b) and create f_icv_resid = f - (a + b*ICV)
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
                # val/test: apply stored params if available
                for col in base_feats:
                    if col in self.icv_resid_params:
                        a, b = self.icv_resid_params[col]
                        df[f'{col}_icv_resid'] = df[col] - (a + b * df[icv_col])

        # 3) Left-right bilateral/Asymmetry features
        #    Match left/right as a leading token only, to avoid 'lateral' false matches.
        all_cols = df.columns.tolist()
        left_cols = [c for c in all_cols if re.match(r'^(?i)left\s+', c)]
        # map by base name
        base_to_pair = {}
        for lc in left_cols:
            base = regex_strip_side(lc).lower()
            # find a right column with same base
            candidates = [c for c in all_cols if re.match(r'^(?i)right\s+', c) and regex_strip_side(c).lower() == base]
            if candidates:
                rc = candidates[0]
                base_to_pair[lc] = rc

        for lcol, rcol in base_to_pair.items():
            # Sums/Diffs/Asymmetry on raw
            df[f'{lcol}_bilateral'] = df[lcol] + df[rcol]
            df[f'{lcol}_diff'] = df[lcol] - df[rcol]
            df[f'{lcol}_asymmetry'] = safe_div(df[lcol] - df[rcol], df[lcol] + df[rcol])

            # Also on nicv if present
            if icv_col is not None:
                ln = f'{lcol}_nicv'
                rn = f'{rcol}_nicv'
                if ln in df.columns and rn in df.columns:
                    df[f'{lcol}_bilateral_nicv'] = df[ln] + df[rn]
                    df[f'{lcol}_asymmetry_nicv'] = safe_div(df[ln] - df[rn], df[ln] + df[rn])

            # Also on residuals if present
            lr = f'{lcol}_icv_resid'
            rr = f'{rcol}_icv_resid'
            if lr in df.columns and rr in df.columns:
                df[f'{lcol}_bilateral_resid'] = df[lr] + df[rr]
                df[f'{lcol}_asymmetry_resid'] = safe_div(df[lr] - df[rr], df[lr] + df[rr])

        # 4) Tissue/system composites
        # Cerebral cortex vs Cerebellum cortex
        cortex_cerebral = [c for c in all_cols if 'cerebral cortex' in c.lower()]
        cortex_cerebellum = [c for c in all_cols if 'cerebellum cortex' in c.lower()]
        wm_cerebral = [c for c in all_cols if 'cerebral white matter' in c.lower()]
        wm_cerebellum = [c for c in all_cols if 'cerebellum white matter' in c.lower()]

        # Subcortical nuclei
        sub_tokens = ['thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens', 'ventral dc']
        subcort_cols = [c for c in all_cols if any(t in c.lower() for t in sub_tokens)]

        # Ventricles and CSF
        lat_L = pick([c for c in all_cols if c.lower() == 'left lateral ventricle'])
        lat_R = pick([c for c in all_cols if c.lower() == 'right lateral ventricle'])
        inf_lat_L = pick([c for c in all_cols if c.lower() == 'left inferior lateral ventricle'])
        inf_lat_R = pick([c for c in all_cols if c.lower() == 'right inferior lateral ventricle'])
        third = pick(['3rd ventricle', 'third ventricle'])
        fourth = pick(['4th ventricle', 'fourth ventricle'])
        csf_col = pick(['csf'])

        # Totals
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

        if 'cortex_total' in df.columns and 'gm_subcort_total' in df.columns and 'cerebellum_cortex_total' in df.columns:
            df['gm_total'] = df['cortex_total'] + df['gm_subcort_total'] + df['cerebellum_cortex_total']

        if 'cerebral_wm_total' in df.columns and 'cerebellum_wm_total' in df.columns:
            df['wm_total'] = df['cerebral_wm_total'] + df['cerebellum_wm_total']

        # Ventricular totals
        lat_total = None
        if lat_L and lat_R:
            df['lat_ventricles_total'] = df[lat_L] + df[lat_R]
            lat_total = 'lat_ventricles_total'
        inf_lat_total = None
        if inf_lat_L and inf_lat_R:
            df['inf_lat_ventricles_total'] = df[inf_lat_L] + df[inf_lat_R]
            inf_lat_total = 'inf_lat_ventricles_total'

        v_parts = []
        for v in [lat_total, inf_lat_total, third, fourth]:
            if v and has_col(v):
                v_parts.append(v)
            elif isinstance(v, str) and v in df.columns:
                v_parts.append(v)
        if v_parts:
            df['ventricles_total'] = df[v_parts].sum(axis=1)

        if csf_col and 'ventricles_total' in df.columns:
            df['csf_total'] = df[csf_col] + df['ventricles_total']

        # A few system composites
        # Basal ganglia: caudate + putamen + pallidum + accumbens (bilateral, so summing all columns that match tokens)
        bg_cols = [c for c in all_cols if any(t in c.lower() for t in ['caudate', 'putamen', 'pallidum', 'accumbens'])]
        if bg_cols:
            df['basal_ganglia_total'] = df[bg_cols].sum(axis=1)
        limbic_cols = [c for c in all_cols if any(t in c.lower() for t in ['hippocampus', 'amygdala'])]
        if limbic_cols:
            df['limbic_total'] = df[limbic_cols].sum(axis=1)
        thal_cols = [c for c in all_cols if 'thalamus' in c.lower()]
        if thal_cols:
            df['thalamus_total'] = df[thal_cols].sum(axis=1)

        # 5) Ratios and biologically motivated markers
        if icv_col is not None:
            for name in ['cortex_total', 'wm_total', 'ventricles_total', 'csf_total', 'limbic_total', 'thalamus_total']:
                if name in df.columns:
                    df[f'{name}_nicv'] = safe_div(df[name], df[icv_col])

            # Additional ratios
            if 'hippocampus' in ' '.join(all_cols).lower():
                hip_cols = [c for c in all_cols if 'hippocampus' in c.lower()]
                df['hippocampus_total'] = df[hip_cols].sum(axis=1)
                df['hippocampus_total_nicv'] = safe_div(df['hippocampus_total'], df[icv_col])

            if 'cortex_total' in df.columns and 'wm_total' in df.columns:
                df['cortex_to_wm'] = safe_div(df['cortex_total'], df['wm_total'])

            if 'gm_subcort_total' in df.columns and 'cortex_total' in df.columns:
                df['subcortex_to_cortex'] = safe_div(df['gm_subcort_total'], df['cortex_total'])

            if 'thalamus_total' in df.columns and 'cortex_total' in df.columns:
                df['thalamus_to_cortex'] = safe_div(df['thalamus_total'], df['cortex_total'])

            if 'basal_ganglia_total' in df.columns and 'cortex_total' in df.columns:
                df['basal_ganglia_to_cortex'] = safe_div(df['basal_ganglia_total'], df['cortex_total'])

            # Ventricular expansion markers
            brain_parenchyma = None
            if 'cortex_total' in df.columns and 'wm_total' in df.columns:
                df['brain_parenchyma_basic'] = df['cortex_total'] + df['wm_total']
                brain_parenchyma = 'brain_parenchyma_basic'
            if 'ventricles_total' in df.columns and brain_parenchyma:
                df['ventricles_to_parenchyma'] = safe_div(df['ventricles_total'], df[brain_parenchyma])

            # hippocampus/cortex
            if 'hippocampus_total' in df.columns and 'cortex_total' in df.columns:
                df['hippocampus_to_cortex'] = safe_div(df['hippocampus_total'], df['cortex_total'])

        # 6) Limited nonlinearities (keep small to avoid overfitting)
        # log1p on skewed ratios
        for base in ['ventricles_total_nicv', 'csf_total_nicv', 'ventricles_to_parenchyma']:
            if base in df.columns:
                df[f'log1p_{base}'] = np.log1p(df[base].clip(lower=0))

        # Quadratic terms
        for base in ['cortex_total_nicv', 'wm_total_nicv', 'hippocampus_total_nicv', 'ventricles_total_nicv']:
            if base in df.columns:
                df[f'{base}__sq'] = df[base] ** 2

        # A few interactions
        inter_pairs = [
            ('ventricles_total_nicv', 'cortex_total_nicv'),
            ('hippocampus_total_nicv', 'cortex_total_nicv'),
            ('cortex_total_nicv', 'wm_total_nicv'),
        ]
        for a, b in inter_pairs:
            if a in df.columns and b in df.columns:
                df[f'{a}__x__{b}'] = df[a] * df[b]

        # 7) Keep existing simpler features (if not already created)
        # Cortical/WM/Subcortical totals (broad)
        cortical_regions = [col for col in volumetric_features if 'cortex' in col.lower() and col in df.columns]
        white_matter_regions = [col for col in volumetric_features if 'white matter' in col.lower() and col in df.columns]
        subcortical_regions = [col for col in volumetric_features if any(region in col.lower() for region in ['thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala']) and col in df.columns]

        if cortical_regions and 'total_cortical' not in df.columns:
            df['total_cortical'] = df[cortical_regions].sum(axis=1)
        if white_matter_regions and 'total_white_matter' not in df.columns:
            df['total_white_matter'] = df[white_matter_regions].sum(axis=1)
        if subcortical_regions and 'total_subcortical' not in df.columns:
            df['total_subcortical'] = df[subcortical_regions].sum(axis=1)

        # csf_ratio (legacy)
        if icv_col is not None and csf_col:
            df['csf_ratio'] = safe_div(df[csf_col], df[icv_col])

        self.logger.info(f"Engineered features -> shape now: {df.shape[1]} columns")
        return df
        
    def setup_models(self) -> Dict:
        """
        Initialize all models to be tested.
        """
        self.logger.info("Setting up models...")
        
        mcfg = self.config.get('models', {})

        models: Dict[str, object] = {}

        # XGBoost
        xgb_cfg = mcfg.get('xgboost', {})
        if xgb_cfg.get('enabled', True):
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=xgb_cfg.get('n_estimators', 1000),
                max_depth=xgb_cfg.get('max_depth', 6),
                learning_rate=xgb_cfg.get('learning_rate', 0.1),
                subsample=xgb_cfg.get('subsample', 0.8),
                colsample_bytree=xgb_cfg.get('colsample_bytree', 0.8),
                random_state=self.config.get('seed', 42),
                n_jobs=-1,
                tree_method=xgb_cfg.get('tree_method', 'hist')
            )

        # LightGBM
        lgb_cfg = mcfg.get('lightgbm', {})
        if lgb_cfg.get('enabled', True):
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=lgb_cfg.get('n_estimators', 1000),
                max_depth=lgb_cfg.get('max_depth', 6),
                learning_rate=lgb_cfg.get('learning_rate', 0.1),
                subsample=lgb_cfg.get('subsample', 0.8),
                colsample_bytree=lgb_cfg.get('colsample_bytree', 0.8),
                random_state=self.config.get('seed', 42),
                n_jobs=-1,
                verbose=-1
            )

        # Random Forest
        rf_cfg = mcfg.get('random_forest', {})
        if rf_cfg.get('enabled', True):
            models['random_forest'] = RandomForestRegressor(
                n_estimators=rf_cfg.get('n_estimators', 500),
                max_depth=rf_cfg.get('max_depth', 10),
                random_state=self.config.get('seed', 42),
                n_jobs=-1
            )

        # Extra Trees (always available; no YAML toggle provided in current config)
        models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=self.config.get('seed', 42),
            n_jobs=-1
        )

        # Ridge
        ridge_cfg = mcfg.get('ridge', {})
        if ridge_cfg.get('enabled', True):
            models['ridge'] = Ridge(alpha=ridge_cfg.get('alpha', 1.0), random_state=self.config.get('seed', 42))

        # Lasso
        lasso_cfg = mcfg.get('lasso', {})
        if lasso_cfg.get('enabled', True):
            models['lasso'] = Lasso(alpha=lasso_cfg.get('alpha', 1.0), random_state=self.config.get('seed', 42))

        # ElasticNet and SVR are included by default
        models['elastic_net'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config.get('seed', 42))
        models['svr'] = SVR(kernel='rbf', C=1.0, epsilon=0.1)

        return models
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}...")

        cv_folds = self.config.get('training', {}).get('cv_folds', 5)
        n_trials = self.config.get('training', {}).get('n_trials', 100)
        seed = self.config.get('seed', 42)

        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),  # Reduced max
                    'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced max depth
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),  # Reduced max
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # Reduced max
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),  # Reduced max
                    'tree_method': 'hist',
                    'random_state': seed,
                    'n_jobs': 1,  # Single thread per model to avoid memory explosion
                }
                model = xgb.XGBRegressor(**params)

            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),  # Reduced max
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 256, log=True),  # Reduced max
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # Reduced max
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),  # Reduced max
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                    'random_state': seed,
                    'n_jobs': 1,  # Single thread per model
                }
                model = lgb.LGBMRegressor(**params, verbose=-1)

            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),  # Reduced max
                    'max_depth': trial.suggest_int('max_depth', 5, 20),  # Reduced max depth
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),  # Removed 'auto'
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': seed,
                    'n_jobs': 1,  # Single thread per model
                }
                model = RandomForestRegressor(**params)

            elif model_name == 'extra_trees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),  # Reduced max
                    'max_depth': trial.suggest_int('max_depth', 5, 20),  # Reduced max depth
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),  # Removed 'auto'
                    'random_state': seed,
                    'n_jobs': 1,  # Single thread per model
                }
                model = ExtraTreesRegressor(**params)

            elif model_name == 'svr':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                    'kernel': 'rbf'
                }
                model = SVR(**params)

            elif model_name == 'ridge':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 100.0, log=True),
                    'random_state': seed
                }
                model = Ridge(**params)

            elif model_name == 'lasso':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
                    'random_state': seed
                }
                model = Lasso(**params)

            elif model_name == 'elastic_net':
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                    'random_state': seed
                }
                model = ElasticNet(**params)

            else:
                raise ValueError(f"Unknown model for optimization: {model_name}")

            # Use fewer CV folds and sequential processing to reduce memory usage
            cv_scores = cross_val_score(
                model, X, y,
                cv=KFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=seed),  # Cap at 5 folds
                scoring='neg_mean_absolute_error',
                n_jobs=1  # Sequential processing to avoid memory explosion
            )
            return -cv_scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        
        def _trial_callback(study_obj, trial_obj):
            try:
                self.logger.info(
                    f"[{model_name}] trial {trial_obj.number} value={trial_obj.value:.4f} "
                    f"best={study_obj.best_value:.4f} params={trial_obj.params}"
                )
            except Exception:
                pass

        self.logger.info(f"{model_name}: tuning with {n_trials} trials, {min(cv_folds, 5)}-fold CV")
        study.optimize(objective, n_trials=n_trials, callbacks=[_trial_callback])

        self.logger.info(f"Best parameters for {model_name}: {study.best_params}")
        return study.best_params
    
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train on train; evaluate on val and test provided by YAML splits.
        """
        self.logger.info("Training and evaluating models on provided splits...")

        tr_cfg = self.config.get('training', {})
        seed = self.config.get('seed', 42)
        do_opt = tr_cfg.get('optimize_hyperparams', False)

        self.logger.info(
            f"Split sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)} features={X_train.shape[1]}"
        )
        
        # Scale features
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
        self.logger.info(f"Scaler: {scaler.__class__.__name__}")
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), 
            columns=X_val.columns, 
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        self.scalers['feature_scaler'] = scaler
        
        # Setup models
        models = self.setup_models()
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Hyperparameter optimization for selected models
            if do_opt and model_name in ['xgboost', 'lightgbm', 'random_forest', 'extra_trees', 'svr', 'ridge', 'lasso', 'elastic_net']:
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train, model_name)
                model.set_params(**best_params)
            
            # Train model with proper API for each model type
            try:
                if model_name == 'xgboost':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
                        verbose=False
                    )
                elif model_name == 'lightgbm':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                        self.logger.info(f"lightgbm best_iteration={model.best_iteration_}")
                else:
                    model.fit(X_train_scaled, y_train)
                    
            except Exception as e:
                self.logger.warning(f"Error with early stopping for {model_name}, using standard fit: {e}")
                model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[model_name] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'predictions': {
                    'y_test': y_test.values,
                    'y_test_pred': y_test_pred
                }
            }
            
            # Store model
            self.models[model_name] = model
            
            self.logger.info(f"{model_name} - Test MAE: {test_mae:.3f}, Test R²: {test_r2:.3f}")
            
            # Log to W&B
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'{model_name}/train_mae': train_mae,
                    f'{model_name}/val_mae': val_mae,
                    f'{model_name}/test_mae': test_mae,
                    f'{model_name}/train_r2': train_r2,
                    f'{model_name}/val_r2': val_r2,
                    f'{model_name}/test_r2': test_r2,
                })
        
        return results
    
    def analyze_feature_importance(self) -> Dict:
        """
        Analyze feature importance across models.
        """
        self.logger.info("Analyzing feature importance...")
        
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
            else:
                continue
            
            # Create importance dataframe
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_data[model_name] = feature_importance
            
            # Log top features
            self.logger.info(f"Top 10 features for {model_name}:")
            for _, row in feature_importance.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_data
    
    def create_ensemble(self, results: Dict) -> Dict:
        """
        Create ensemble predictions from best models.
        """
        self.logger.info("Creating ensemble model...")
        
        # Select best models based on validation MAE
        model_scores = {name: res['val_mae'] for name, res in results.items()}
        best_models = sorted(model_scores.items(), key=lambda x: x[1])[:3]  # Top 3 models
        
        self.logger.info(f"Ensemble models: {[name for name, _ in best_models]}")
        
        # Simple average ensemble
        ensemble_preds = np.zeros_like(results[best_models[0][0]]['predictions']['y_test'])
        
        for model_name, _ in best_models:
            ensemble_preds += results[model_name]['predictions']['y_test_pred']
        
        ensemble_preds /= len(best_models)
        
        # Evaluate ensemble
        y_test = results[best_models[0][0]]['predictions']['y_test']
        ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
        ensemble_r2 = r2_score(y_test, ensemble_preds)
        
        self.logger.info(f"Ensemble - Test MAE: {ensemble_mae:.3f}, Test R²: {ensemble_r2:.3f}")
        
        return {
            'test_mae': ensemble_mae,
            'test_r2': ensemble_r2,
            'predictions': ensemble_preds,
            'models_used': [name for name, _ in best_models]
        }
    
    def generate_visualizations(self, results: Dict, ensemble_results: Dict, importance_data: Dict):
        """
        Generate comprehensive visualizations.
        """
        self.logger.info("Generating visualizations...")
        
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model comparison
        plt.figure(figsize=(12, 8))
        model_names = list(results.keys())
        test_maes = [results[name]['test_mae'] for name in model_names]
        test_r2s = [results[name]['test_r2'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(model_names, test_maes)
        ax1.set_title('Test MAE by Model')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(model_names, test_r2s)
        ax2.set_title('Test R² by Model')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance for best model
        if importance_data:
            best_model = min(results.keys(), key=lambda x: results[x]['test_mae'])
            if best_model in importance_data:
                plt.figure(figsize=(12, 10))
                top_features = importance_data[best_model].head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Features - {best_model}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(output_dir / f'feature_importance_{best_model}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Prediction scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= 6:  # Only plot first 6 models
                break
            
            y_test = result['predictions']['y_test']
            y_pred = result['predictions']['y_test_pred']
            
            axes[i].scatter(y_test, y_pred, alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[i].set_xlabel('True Age')
            axes[i].set_ylabel('Predicted Age')
            axes[i].set_title(f'{model_name} (MAE: {result["test_mae"]:.2f})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to W&B
        if self.config.get('use_wandb', False):
            wandb.log({
                "model_comparison": wandb.Image(str(output_dir / 'model_comparison.png')),
                "prediction_scatter": wandb.Image(str(output_dir / 'prediction_scatter.png'))
            })
    
    def save_results(self, results: Dict, ensemble_results: Dict, importance_data: Dict):
        """
        Save all results to files.
        """
        output_dir = Path(self.config['output']['output_dir'])
        
        # Save results summary
        summary = {
            'model_results': {
                name: {k: v for k, v in res.items() if k != 'predictions'}
                for name, res in results.items()
            },
            'ensemble_results': {k: v for k, v in ensemble_results.items() if k != 'predictions'},
            'best_model': min(results.keys(), key=lambda x: results[x]['test_mae']),
            'config': self.config
        }
        
        with open(output_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save feature importance
        if importance_data:
            for model_name, importance_df in importance_data.items():
                importance_df.to_csv(output_dir / f'feature_importance_{model_name}.csv', index=False)
        
        # Save models (pickle)
        import pickle
        with open(output_dir / 'trained_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_names': self.feature_names
            }, f)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def run_full_pipeline(self):
        """
        Run the complete training pipeline.
        """
        self.logger.info("Starting tabular ML pipeline...")
        
        # Load data using YAML splits
        train_df, val_df, test_df = self.load_segmentation_data()

        # Preprocess: fit on train, transform val/test, align to train features
        X_train, y_train = self.preprocess_features(train_df, split_name='train')
        X_val, y_val = self.preprocess_features(val_df, split_name='val')
        X_test, y_test = self.preprocess_features(test_df, split_name='test')

        X_val = self._align_features(X_val)
        X_test = self._align_features(X_test)

        # Train and evaluate
        results = self.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Feature importance
        importance_data = self.analyze_feature_importance()
        
        # Ensemble
        ensemble_results = self.create_ensemble(results)
        
        # Visualizations
        self.generate_visualizations(results, ensemble_results, importance_data)
        
        # Save results
        self.save_results(results, ensemble_results, importance_data)
        
        # Print summary
        best_model = min(results.keys(), key=lambda x: results[x]['test_mae'])
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"TRAINING COMPLETE")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Best model: {best_model}")
        self.logger.info(f"Best test MAE: {results[best_model]['test_mae']:.3f}")
        self.logger.info(f"Best test R²: {results[best_model]['test_r2']:.3f}")
        self.logger.info(f"Ensemble MAE: {ensemble_results['test_mae']:.3f}")
        self.logger.info(f"Ensemble R²: {ensemble_results['test_r2']:.3f}")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'best_test_mae': results[best_model]['test_mae'],
                'best_test_r2': results[best_model]['test_r2'],
                'ensemble_test_mae': ensemble_results['test_mae'],
                'ensemble_test_r2': ensemble_results['test_r2']
            })
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train tabular ML models for brain age prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add timestamp to experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['experiment_name'] = f"tabular_ml_{timestamp}"
    
    # Create output directory
    output_dir = Path(config['output']['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output']['output_dir'] = str(output_dir)
    config['output']['log_dir'] = str(output_dir)
    
    # Run pipeline
    predictor = TabularBrainAgePredictor(config)
    predictor.run_full_pipeline()


if __name__ == "__main__":
    main()
