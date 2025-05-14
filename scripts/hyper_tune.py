#!/usr/bin/env python
"""
Optuna HPO launcher for the brain-age project.

Search space:
  • lr            : 1e-5 – 3e-3  (log-uniform)
  • dropout       : 0.0 – 0.6    (uniform)
  • optimizer     : adam | adamw | sgd | rmsprop | radam | novograd

It re-uses the main training script (brain_age_train.py) without
touching any of its internals – it only hands over a temporary YAML.
"""

import os, sys, copy, yaml, optuna, torch
from pathlib import Path

# ─── paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))               # for `import brain_age_train`
BASE_CFG   = PROJECT_ROOT / "configs/tuning.yaml"  # original YAML
TUNE_DIR   = PROJECT_ROOT / "output" / "hpo_runs"

# import AFTER sys.path tweak
from train import main as train_entry     # <- rename your script

# ─── Optuna objective ─────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """Single HPO trial – must return validation loss/MAE."""
    # 1) sample hyper-parameters
    lr       = trial.suggest_loguniform("lr", 1e-5, 3e-3)
    dropout  = trial.suggest_float      ("dropout", 0.0, 0.7)
    optim    = trial.suggest_categorical("optimizer",
                                         ["adam", "adamw",
                                          "sgd", "rmsprop",
                                          "radam", "novograd"])

    # 2) load + patch base YAML in memory
    with BASE_CFG.open() as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["learning_rate"] = lr
    cfg["training"]["optimizer"]     = optim
    cfg["model"]["dropout_rate"]     = dropout

    # shrink epoch budget for HPO
    cfg["training"]["epochs"]                   = 15
    cfg["training"]["early_stopping_patience"]  = 5
    cfg["output"]["experiment_name"]            = f"trial_{trial.number:04d}"

    # lightweight W&B tracking (nice dashboards)
    cfg["wandb"]["use_wandb"] = True
    cfg["wandb"]["project"]   = "brainage-hpo"

    # 3) dump patched YAML to a temp file
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = TUNE_DIR / f"trial_{trial.number:04d}.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)

    # 4) launch *exactly* the same CLI entry point
    try:
        sys.argv = ["brain_age_train.py", str(cfg_path)]  # fake CLI call
        best_val = train_entry()                          # returns float
    except optuna.TrialPruned:
        raise                                             # propagate
    except Exception as e:
        # any crash ⇒ prune the trial so the optimiser can move on
        raise optuna.TrialPruned() from e
    finally:
        torch.cuda.empty_cache()

    return best_val   # minimised by Optuna


# ─── main launcher ────────────────────────────────────────────────────
def main():
    storage = f"sqlite:///{TUNE_DIR/'study.db'}"   # enables multi-process tuning
    pruner  = optuna.pruners.SuccessiveHalvingPruner(min_resource=3,
                                                     reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=42)  # Bayesian

    study = optuna.create_study(
        study_name  = "brainage-hpo",
        direction   = "minimize",
        storage     = storage,
        load_if_exists = True,
        sampler     = sampler,
        pruner      = pruner,
    )

    study.optimize(objective, n_trials=40, timeout=3*60*60)  # 3 h wall clock

    print("\n────────  Best trial  ────────")
    print("Value :", study.best_value)
    print("Params:", study.best_trial.params)


if __name__ == "__main__":
    main()