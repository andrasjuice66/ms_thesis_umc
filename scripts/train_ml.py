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
    
    def load_segmentation_data(self) -> pd.DataFrame:
        """
        Load and combine all CSV segmentation files with demographics.
        """
        self.logger.info("Loading segmentation data...")
        
        # Load demographic data
        train_df = pd.read_csv(self.config['data']['train_csv'])
        val_df = pd.read_csv(self.config['data']['val_csv'])
        test_df = pd.read_csv(self.config['data']['test_csv'])
        
        # Combine all splits for now - we'll re-split later
        demo_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        self.logger.info(f"Loaded {len(demo_df)} demographic records")
        
        # Load segmentation features
        seg_data_dir = Path(self.config['data']['segmented_data_dir'])
        all_seg_data = []
        
        for _, row in demo_df.iterrows():
            # Construct path to segmentation CSV
            image_path = row['image_path']
            # Convert .nii.gz to .csv
            seg_csv_path = seg_data_dir / (image_path.replace('.nii.gz', '.csv'))
            
            if seg_csv_path.exists():
                try:
                    seg_df = pd.read_csv(seg_csv_path)
                    if len(seg_df) > 0:
                        # Add demographics to segmentation data
                        seg_row = seg_df.iloc[0].to_dict()  # First row contains the data
                        seg_row.update({
                            'subject_id': row['subject_id'],
                            'age': row['age'],
                            'sex': row['sex'],
                            'modality': row['modality'],
                            'dataset': row['dataset'],
                            'image_path': row['image_path']
                        })
                        all_seg_data.append(seg_row)
                except Exception as e:
                    self.logger.warning(f"Failed to load {seg_csv_path}: {e}")
            else:
                self.logger.warning(f"Segmentation file not found: {seg_csv_path}")
        
        combined_df = pd.DataFrame(all_seg_data)
        self.logger.info(f"Successfully loaded {len(combined_df)} samples with segmentation features")
        
        return combined_df
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features including scaling and encoding.
        """
        self.logger.info("Preprocessing features...")
        
        # Separate features and target
        target_col = 'age'
        exclude_cols = ['subject_id', 'image_path', 'age', 'subject']
        
        # Get volumetric features (all numeric columns except metadata)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        volumetric_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Get categorical features
        categorical_features = ['sex', 'modality', 'dataset']
        
        self.logger.info(f"Found {len(volumetric_features)} volumetric features")
        self.logger.info(f"Found {len(categorical_features)} categorical features")
        
        # Create feature matrix
        X = df[volumetric_features + categorical_features].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X[volumetric_features] = X[volumetric_features].fillna(X[volumetric_features].median())
        X[categorical_features] = X[categorical_features].fillna('unknown')
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col])
            else:
                X[col] = self.encoders[col].transform(X[col])
        
        # Feature engineering
        X = self._engineer_features(X, volumetric_features)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def _engineer_features(self, X: pd.DataFrame, volumetric_features: List[str]) -> pd.DataFrame:
        """
        Create additional engineered features.
        """
        self.logger.info("Engineering additional features...")
        
        # Calculate ratios and derived features
        if 'total intracranial' in volumetric_features:
            # Normalize all volumes by total intracranial volume
            for col in volumetric_features:
                if col != 'total intracranial' and 'total intracranial' in X.columns:
                    X[f'{col}_normalized'] = X[col] / (X['total intracranial'] + 1e-8)
        
        # Left-right asymmetry features
        left_features = [col for col in volumetric_features if 'left' in col.lower()]
        for left_col in left_features:
            right_col = left_col.replace('left', 'right')
            if right_col in volumetric_features:
                # Asymmetry index: (L - R) / (L + R)
                X[f'{left_col}_asymmetry'] = (X[left_col] - X[right_col]) / (X[left_col] + X[right_col] + 1e-8)
                # Total bilateral volume
                X[f'{left_col}_bilateral'] = X[left_col] + X[right_col]
        
        # Regional groupings
        cortical_regions = [col for col in volumetric_features if 'cortex' in col.lower()]
        white_matter_regions = [col for col in volumetric_features if 'white matter' in col.lower()]
        subcortical_regions = [col for col in volumetric_features if any(region in col.lower() for region in ['thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala'])]
        
        if cortical_regions:
            X['total_cortical'] = X[cortical_regions].sum(axis=1)
        if white_matter_regions:
            X['total_white_matter'] = X[white_matter_regions].sum(axis=1)
        if subcortical_regions:
            X['total_subcortical'] = X[subcortical_regions].sum(axis=1)
        
        # Age-related ratios
        if 'csf' in volumetric_features and 'total intracranial' in volumetric_features:
            X['csf_ratio'] = X['csf'] / (X['total intracranial'] + 1e-8)
        
        return X
    
    def setup_models(self) -> Dict:
        """
        Initialize all models to be tested.
        """
        self.logger.info("Setting up models...")
        
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        return models
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X, y, 
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            return -cv_scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        self.logger.info(f"Best parameters for {model_name}: {study.best_params}")
        return study.best_params
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train and evaluate all models.
        """
        self.logger.info("Training and evaluating models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=10, labels=False)
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, 
            stratify=pd.cut(y_train, bins=10, labels=False)
        )
        
        # Scale features
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
        
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
            
            # Hyperparameter optimization for key models
            if model_name in ['xgboost', 'lightgbm', 'random_forest'] and self.config.get('optimize_hyperparams', False):
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train, model_name)
                model.set_params(**best_params)
            
            # Train model with proper API for each model type
            try:
                if model_name == 'xgboost':
                    # Use callbacks for early stopping in newer XGBoost versions
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
                        verbose=False
                    )
                elif model_name == 'lightgbm':
                    # LightGBM still supports early_stopping_rounds
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                else:
                    # Standard fit for other models
                    model.fit(X_train_scaled, y_train)
                    
            except Exception as e:
                self.logger.warning(f"Error with early stopping for {model_name}, using standard fit: {e}")
                # Fallback to standard fit
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
        
        # Load data
        df = self.load_segmentation_data()
        
        # Preprocess
        X, y = self.preprocess_features(df)
        
        # Train and evaluate
        results = self.train_and_evaluate(X, y)
        
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
