"""
Gradient-weighted Class Activation Mapping (GradCAM) for 3-D brain-age models.

Reference
---------
Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV.

Supported model types
---------------------
sfcn           – SFCN regression (feature_extractor[-2])
sfcn_class     – SFCN soft-classification (feature_extractor[-2])
brainagenext   – MedNeXt-based regression (mednextv1.bottleneck)
multitask      – Multi-task seg+age model (encoder.downs[-1])
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom as ndimage_zoom


logger = logging.getLogger(__name__)


def _relu_upsample_normalise(raw: np.ndarray, target_shape: tuple) -> np.ndarray:
    """ReLU → upsample to target_shape → min-max normalise to [0, 1]."""
    cam = np.maximum(raw, 0)
    if cam.shape != target_shape:
        factors = tuple(target_shape[i] / max(cam.shape[i], 1) for i in range(3))
        cam = ndimage_zoom(cam, factors, order=1)
    lo, hi = cam.min(), cam.max()
    if hi > lo:
        cam = (cam - lo) / (hi - lo)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────────── #
#  Core GradCAM engine                                                 #
# ──────────────────────────────────────────────────────────────────── #

class GradCAM3D:
    """
    GradCAM for 3-D volumetric models.

    Example::

        gcam = GradCAM3D(model, target_layer)
        cam  = gcam.generate(input_tensor, score_fn)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._hooks: list = []

    # ---------------------------------------------------------------- #
    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            self._activations = out.detach().clone()

        def bwd_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self._gradients = grad_out[0].detach().clone()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---------------------------------------------------------------- #
    def _backward(self, input_tensor: torch.Tensor, score_fn: Callable) -> bool:
        """Runs one forward+backward pass; returns True if hooks fired."""
        self._activations = None
        self._gradients   = None
        self._register_hooks()
        self.model.zero_grad()
        with torch.enable_grad():
            output = self.model(input_tensor)
            score_fn(output).backward()
        self._remove_hooks()
        return self._activations is not None and self._gradients is not None

    def generate_raw(
        self,
        input_tensor: torch.Tensor,
        score_fn: Callable,
    ) -> np.ndarray | None:
        """
        Returns the raw weighted-sum map at *feature-map* resolution
        (before ReLU, upsampling, or normalisation).

        This is the quantity that should be averaged across subjects before
        applying the non-linear ReLU step.  Returns None on failure.
        """
        if not self._backward(input_tensor, score_fn):
            return None
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1,C,1,1,1)
        raw = (weights * self._activations).sum(dim=1).squeeze()      # (d, h, w)
        return raw.cpu().numpy()

    def generate(
        self,
        input_tensor: torch.Tensor,
        score_fn: Callable,
    ) -> np.ndarray:
        """
        Full GradCAM pipeline for a single image:
        raw weighted sum → ReLU → upsample → normalise → [0,1].

        Parameters
        ----------
        input_tensor : (1, C, D, H, W) tensor on the model's device.
        score_fn     : callable(model_output) → scalar tensor for backprop.

        Returns
        -------
        cam : (D, H, W) float32 ndarray in [0, 1].
        """
        target_shape = tuple(input_tensor.shape[2:])
        raw = self.generate_raw(input_tensor, score_fn)
        if raw is None:
            return np.zeros(target_shape, dtype=np.float32)
        return _relu_upsample_normalise(raw, target_shape)


# ──────────────────────────────────────────────────────────────────── #
#  Model-type → (target_layer, score_fn)                               #
# ──────────────────────────────────────────────────────────────────── #

def get_gradcam_target(model: nn.Module, model_type: str):
    """
    Returns (target_layer, score_fn) for the given model type.

    target_layer : nn.Module whose output activations and gradients are captured.
    score_fn     : callable(model_output) → scalar tensor for backprop.

    Returns (None, None) when the model type is unsupported.
    """
    mtype = model_type.lower()

    if mtype == 'sfcn':
        # Second-to-last ConvBlock: last layer with spatial downsampling (256 ch)
        layer    = model.feature_extractor[-2]
        score_fn = lambda out: out.sum()

    elif mtype == 'sfcn_class':
        layer    = model.feature_extractor[-2]
        score_fn = lambda out: model.expected_age(out).sum()

    elif mtype == 'brainagenext':
        # MedNeXt bottleneck: deepest semantic representation (512 ch for model-B)
        layer    = model.mednextv1.bottleneck
        score_fn = lambda out: out.sum()

    elif mtype == 'multitask':
        # Deepest encoder block before any decoder upsampling
        layer    = model.encoder.downs[-1]
        score_fn = lambda out: out[1].sum()   # out = (seg_logits, age_pred)

    else:
        return None, None

    return layer, score_fn


# ──────────────────────────────────────────────────────────────────── #
#  GradCAM sample generation                                           #
# ──────────────────────────────────────────────────────────────────── #

def generate_gradcam_samples(
    model: nn.Module,
    model_type: str,
    test_ds,
    modalities_list: list,
    n_per_modality: int,
    device: torch.device,
    log: logging.Logger | None = None,
) -> list[dict]:
    """
    Runs GradCAM on the first ``n_per_modality`` samples for every unique modality.

    Parameters
    ----------
    model           : trained model in eval mode.
    model_type      : one of 'sfcn', 'sfcn_class', 'brainagenext', 'multitask'.
    test_ds         : PyTorch Dataset; items must have keys 'image', 'age'.
    modalities_list : modality label for each dataset index (same ordering).
    n_per_modality  : maximum number of GradCAM samples per modality.
    device          : torch device.
    log             : optional logger.

    Returns
    -------
    List of dicts with keys:
        'image'     – (D, H, W) ndarray, first MRI channel
        'cam'       – (D, H, W) ndarray, GradCAM heatmap in [0, 1]
        'pred_age'  – float
        'true_age'  – float
        'modality'  – str
    """
    layer, score_fn = get_gradcam_target(model, model_type)
    if layer is None:
        if log:
            log.warning(f"GradCAM not supported for model type '{model_type}'. Skipping.")
        return []

    # Select the first n_per_modality indices for each modality
    mod_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, mod in enumerate(modalities_list):
        mod_to_indices[str(mod)].append(idx)

    selected: list[tuple[int, str]] = []
    for mod in sorted(mod_to_indices):
        selected.extend((idx, mod) for idx in mod_to_indices[mod][:n_per_modality])

    gradcam = GradCAM3D(model, layer)
    model.eval()
    results: list[dict] = []

    for idx, mod in selected:
        try:
            sample     = test_ds[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device)   # (1, C, D, H, W)
            true_age   = float(sample['age'])

            cam = gradcam.generate(img_tensor, score_fn)

            with torch.no_grad():
                output = model(img_tensor)
                if model_type == 'multitask':
                    pred_age = float(output[1].cpu().flatten()[0].item())
                elif model_type == 'sfcn_class':
                    pred_age = float(model.expected_age(output).cpu().flatten()[0].item())
                else:
                    pred_age = float(output.cpu().flatten()[0].item())

            results.append({
                'image'    : img_tensor[0, 0].cpu().numpy(),  # (D, H, W) first channel
                'cam'      : cam,
                'pred_age' : pred_age,
                'true_age' : true_age,
                'modality' : mod,
            })

        except Exception as exc:
            if log:
                log.warning(f"GradCAM failed for sample {idx} (modality={mod}): {exc}")

    return results


# ──────────────────────────────────────────────────────────────────── #
#  Average GradCAM                                                      #
# ──────────────────────────────────────────────────────────────────── #

def generate_average_gradcam(
    model: nn.Module,
    model_type: str,
    test_ds,
    modalities_list: list,
    n_max_per_modality: int | None,
    device: torch.device,
    log: logging.Logger | None = None,
) -> dict:
    """
    Computes an average GradCAM heatmap per modality (and one for "All")
    by accumulating raw (pre-ReLU) feature-map responses across subjects,
    averaging, and then applying ReLU + upsample + normalise once.

    Averaging before ReLU preserves signed contributions: positive and
    negative activations can cancel where the model is inconsistent, so
    only the reliably-important regions remain bright in the average map.

    Parameters
    ----------
    n_max_per_modality : cap on the number of subjects used per modality.
                         Pass ``None`` to use every subject in the dataset.

    Returns
    -------
    dict keyed by modality string (plus "All") → {
        'cam'   : (D, H, W) float32 ndarray in [0, 1],
        'image' : (D, H, W) float32 ndarray, mean MRI for reference,
        'n'     : int, number of subjects averaged,
    }
    """
    layer, score_fn = get_gradcam_target(model, model_type)
    if layer is None:
        if log:
            log.warning(f"Average GradCAM: unsupported model type '{model_type}'.")
        return {}

    mod_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, mod in enumerate(modalities_list):
        mod_to_indices[str(mod)].append(idx)

    selected: list[tuple[int, str]] = []
    for mod in sorted(mod_to_indices):
        indices = (
            mod_to_indices[mod]
            if n_max_per_modality is None
            else mod_to_indices[mod][:n_max_per_modality]
        )
        selected.extend((idx, mod) for idx in indices)

    gradcam   = GradCAM3D(model, layer)
    model.eval()

    raw_per_mod: dict[str, list[np.ndarray]] = defaultdict(list)
    img_per_mod: dict[str, list[np.ndarray]] = defaultdict(list)
    input_shape: tuple | None = None

    for idx, mod in selected:
        try:
            sample     = test_ds[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device)
            if input_shape is None:
                input_shape = tuple(img_tensor.shape[2:])

            raw = gradcam.generate_raw(img_tensor, score_fn)
            if raw is None:
                continue

            raw_per_mod[mod].append(raw)
            img_per_mod[mod].append(img_tensor[0, 0].cpu().numpy())

        except Exception as exc:
            if log:
                log.warning(f"Average GradCAM failed for sample {idx} ({mod}): {exc}")

    if not raw_per_mod or input_shape is None:
        return {}

    results: dict = {}
    all_raws: list[np.ndarray] = []
    all_imgs:  list[np.ndarray] = []

    for mod, raws in sorted(raw_per_mod.items()):
        avg_raw = np.mean(raws, axis=0)
        results[mod] = {
            'cam'   : _relu_upsample_normalise(avg_raw, input_shape),
            'image' : np.mean(img_per_mod[mod], axis=0).astype(np.float32),
            'n'     : len(raws),
        }
        all_raws.extend(raws)
        all_imgs.extend(img_per_mod[mod])

    # "All" — average across every subject regardless of modality
    avg_raw_all = np.mean(all_raws, axis=0)
    results['All'] = {
        'cam'   : _relu_upsample_normalise(avg_raw_all, input_shape),
        'image' : np.mean(all_imgs, axis=0).astype(np.float32),
        'n'     : len(all_raws),
    }

    return results


# ──────────────────────────────────────────────────────────────────── #
#  Visualisation                                                        #
# ──────────────────────────────────────────────────────────────────── #

def _safe_fn(s: str) -> str:
    for ch in (' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(ch, '_')
    return s


def plot_gradcam_samples(
    model_name: str,
    test_set_name: str,
    samples: list[dict],
    output_dir: Path,
    use_wandb: bool = False,
) -> list[Path]:
    """
    Creates one figure per GradCAM sample showing three orthogonal centre-slices
    (Axial / Coronal / Sagittal) each with three columns:
    Original MRI | GradCAM heatmap | Overlay.

    Saves PNG files under ``<output_dir>/plots/gradcam/`` and optionally
    logs them to Weights & Biases.

    Returns list of saved figure paths.
    """
    plot_dir = Path(output_dir) / "gradcam"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    for i, s in enumerate(samples):
        img      = s['image']       # (D, H, W)
        heatmap  = s['cam']         # (D, H, W) in [0, 1]
        true_age = s['true_age']
        pred_age = s['pred_age']
        modality = s['modality']
        delta    = pred_age - true_age

        D, H, W = img.shape

        planes = {
            'Axial'    : (img[D // 2, :, :],  heatmap[D // 2, :, :]),
            'Coronal'  : (img[:, H // 2, :],  heatmap[:, H // 2, :]),
            'Sagittal' : (img[:, :, W // 2],  heatmap[:, :, W // 2]),
        }

        fig, axes = plt.subplots(3, 3, figsize=(11, 9))
        fig.suptitle(
            f"GradCAM  ·  {model_name}  ·  {test_set_name}\n"
            f"Modality: {modality}    True age: {true_age:.1f} yr    "
            f"Predicted: {pred_age:.1f} yr    \u0394 = {delta:+.1f} yr",
            fontsize=11, fontweight='bold',
        )

        col_titles = ["Original MRI", "GradCAM Heatmap", "Overlay"]
        for row_idx, (plane, (orig_sl, cam_sl)) in enumerate(planes.items()):
            orig_norm = orig_sl - orig_sl.min()
            if orig_norm.max() > 0:
                orig_norm = orig_norm / orig_norm.max()

            axes[row_idx][0].imshow(orig_norm.T,  cmap='gray', origin='lower', aspect='auto')
            axes[row_idx][1].imshow(cam_sl.T,      cmap='jet',  origin='lower', aspect='auto',
                                    vmin=0, vmax=1)
            axes[row_idx][2].imshow(orig_norm.T,  cmap='gray', origin='lower', aspect='auto')
            axes[row_idx][2].imshow(cam_sl.T,      cmap='jet',  origin='lower', aspect='auto',
                                    alpha=0.45, vmin=0, vmax=1)

            for col_idx in range(3):
                axes[row_idx][col_idx].axis('off')
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_titles[col_idx], fontsize=9)

            # Row label on the left-most column
            axes[row_idx][0].set_title(
                f"{plane}\n" + (col_titles[0] if row_idx == 0 else ""),
                fontsize=9,
            )

        # Shared colorbar for the heatmap column
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:, 1], shrink=0.6, pad=0.03)
        cbar.set_label('GradCAM activation', fontsize=8)

        plt.tight_layout()

        fname = (
            plot_dir
            / f"gradcam_{_safe_fn(model_name)}_{_safe_fn(test_set_name)}"
              f"_{_safe_fn(modality)}_{i:02d}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(fname)

        if use_wandb:
            import wandb
            wandb.log({
                f"plots/gradcam/{model_name}/{test_set_name}/{_safe_fn(modality)}_{i:02d}":
                    wandb.Image(str(fname))
            })

    return saved


def _resize_cam_to(cam: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Zoom ``cam`` to ``target_shape`` and re-normalise to [0, 1]."""
    if cam.shape == target_shape:
        return cam
    factors = tuple(target_shape[i] / max(cam.shape[i], 1) for i in range(3))
    out = ndimage_zoom(cam, factors, order=1)
    lo, hi = out.min(), out.max()
    if hi > lo:
        out = (out - lo) / (hi - lo)
    else:
        out = np.zeros_like(out)
    return out.astype(np.float32)


def plot_average_gradcam(
    model_name: str,
    test_set_name: str,
    avg_results: dict,
    output_dir: Path,
    atlas_path: str | Path | None = None,
    use_wandb: bool = False,
) -> Path | None:
    """
    Creates one summary figure showing the average GradCAM heatmap for each
    modality (plus "All") in a compact grid:

        Rows    : modalities in alphabetical order, then "All"
        Columns : Axial overlay | Coronal overlay | Sagittal overlay

    The background for each cell is the MNI152 atlas (if ``atlas_path`` is
    provided) so anatomical landmarks are visible; the average GradCAM
    activation is overlaid in jet colouring.  No colourbar is included.
    The figure is saved at 300 DPI for maximum sharpness.

    Returns the saved figure path, or None if avg_results is empty.
    """
    if not avg_results:
        return None

    plot_dir = Path(output_dir) / "gradcam"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MNI atlas ───────────────────────────────────────────────────────
    atlas_vol: np.ndarray | None = None
    if atlas_path is not None:
        try:
            import nibabel as nib
            atlas_nib = nib.load(str(atlas_path))
            atlas_vol = atlas_nib.get_fdata(dtype=np.float32)
            alo, ahi = atlas_vol.min(), atlas_vol.max()
            if ahi > alo:
                atlas_vol = (atlas_vol - alo) / (ahi - alo)
        except Exception as exc:
            logger.warning(f"Could not load atlas '{atlas_path}': {exc}. "
                           "Falling back to averaged MRI background.")
            atlas_vol = None

    # ── Row ordering: alphabetical modalities, then "All" ───────────────────
    mod_keys = sorted(k for k in avg_results if k != 'All')
    if 'All' in avg_results:
        mod_keys.append('All')

    n_rows = len(mod_keys)
    n_cols = 3
    plane_labels = ['Axial', 'Coronal', 'Sagittal']

    # Each subplot cell ~5 × 4 inches; 300 DPI gives crisp output
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        dpi=300,
    )
    fig.suptitle(
        f"Average GradCAM  ·  {model_name}  ·  {test_set_name}",
        fontsize=13, fontweight='bold',
    )

    for row_idx, mod in enumerate(mod_keys):
        entry  = avg_results[mod]
        cam    = entry['cam']    # (D, H, W) at model-input resolution, [0, 1]
        n_subj = entry['n']

        # Choose background and resize CAM to match
        if atlas_vol is not None:
            bg          = atlas_vol
            cam_display = _resize_cam_to(cam, bg.shape)
        else:
            bg          = entry['image']
            cam_display = cam

        D, H, W = bg.shape
        slices = [
            (bg[D // 2, :, :], cam_display[D // 2, :, :]),
            (bg[:, H // 2, :], cam_display[:, H // 2, :]),
            (bg[:, :, W // 2], cam_display[:, :, W // 2]),
        ]

        for col_idx, (bg_sl, cam_sl) in enumerate(slices):
            ax = axes[row_idx][col_idx]

            bg_norm = bg_sl - bg_sl.min()
            if bg_norm.max() > 0:
                bg_norm /= bg_norm.max()

            ax.imshow(bg_norm.T,  cmap='gray', origin='lower', aspect='auto',
                      interpolation='bilinear')
            ax.imshow(cam_sl.T,   cmap='jet',  origin='lower', aspect='auto',
                      alpha=0.45, vmin=0, vmax=1, interpolation='bilinear')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(plane_labels[col_idx], fontsize=10, fontweight='bold')

        axes[row_idx][0].set_ylabel(f"{mod}\n(n={n_subj})", fontsize=9)
        axes[row_idx][0].yaxis.set_visible(True)
        axes[row_idx][0].tick_params(left=False, labelleft=False)

    plt.tight_layout()

    fname = (
        plot_dir
        / f"gradcam_avg_{_safe_fn(model_name)}_{_safe_fn(test_set_name)}.png"
    )
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if use_wandb:
        import wandb
        wandb.log({
            f"plots/gradcam_avg/{model_name}/{test_set_name}": wandb.Image(str(fname))
        })

    return fname
