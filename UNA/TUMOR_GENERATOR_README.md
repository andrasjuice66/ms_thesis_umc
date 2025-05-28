# Simple Tumor Generator for UNA Framework

This document explains how to use the simplified tumor generator that creates synthetic tumors on brain images using Perlin noise, without requiring the complex dataset setup of the original UNA framework.

## Overview

The Simple Tumor Generator is a standalone tool that:
- Generates realistic tumor shapes using 3D Perlin noise
- Applies fluid dynamics to create more natural tumor boundaries
- Supports multiple MRI modalities (T1, T2, FLAIR)
- Works with both real brain images and synthetic data
- Requires minimal input: brain image + segmentation

## Requirements

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Input Data Requirements

1. **Brain Image**: NIfTI format (.nii.gz)
   - Any MRI modality (T1, T2, FLAIR)
   - Preprocessed and skull-stripped recommended

2. **Brain Segmentation**: NIfTI format (.nii.gz)
   - Labels: 0=background, 1=gray matter, 2=white matter, 3=CSF
   - Same dimensions as the brain image
   - Can be generated using tools like FreeSurfer, FSL, or SynthSeg

## Quick Start

### Method 1: Command Line Interface

```bash
# Basic usage
python scripts/simple_tumor_generator.py --input C:/Projects/thesis_project/Data/brain_age_preprocessed/OpenNeuro/BoldVariability/sub-007_T1w.nii.gz --seg C:/Projects/thesis_project/brain_age_pred/data/templates/seg_T1.nii.gz --modality T1 --output ./output

# Generate multiple tumors
python scripts/simple_tumor_generator.py \
    --input brain.nii.gz \
    --seg segmentation.nii.gz \
    --modality T1 \
    --output ./output \
    --num-tumors 3 \
    --tumor-size 1.5

# Use synthetic brain image
python scripts/simple_tumor_generator.py \
    --input dummy.nii.gz \
    --seg segmentation.nii.gz \
    --modality FLAIR \
    --output ./output \
    --use-synthetic
```

### Method 2: Python API

```python
from scripts.simple_tumor_generator import SimpleTumorGenerator
import nibabel as nib

# Load your data
input_img = nib.load('brain.nii.gz')
input_data = input_img.get_fdata()
seg_data = nib.load('segmentation.nii.gz').get_fdata()

# Initialize generator
generator = SimpleTumorGenerator(device='cuda')  # or 'cpu'

# Generate tumor
result = generator.generate_tumor_on_image(
    input_data, 
    seg_data, 
    modality='T1',
    tumor_size_factor=1.0
)

# Access results
diseased_image = result['diseased_image']
tumor_mask = result['tumor_mask']
tumor_probability = result['tumor_prob']
```

### Method 3: Run Examples

```bash
# Run all examples (synthetic data, multiple modalities)
python scripts/example_usage.py
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input brain image (NIfTI) | Required |
| `--seg` | Brain segmentation (NIfTI) | Required |
| `--modality` | Image modality (T1, T2, FLAIR) | Required |
| `--output` | Output directory | Required |
| `--tumor-size` | Tumor size factor (0.5-2.0) | 1.0 |
| `--num-tumors` | Number of tumors to generate | 1 |
| `--use-synthetic` | Generate synthetic brain from segmentation | False |

## Output Files

The generator creates the following files:

1. **`{modality}_with_tumor.nii.gz`**: Brain image with synthetic tumor
2. **`tumor_mask.nii.gz`**: Binary tumor mask
3. **`tumor_probability.nii.gz`**: Tumor probability map (0-1)
4. **`{modality}_original.nii.gz`**: Original input image (for comparison)

## Tumor Characteristics by Modality

### T1-weighted
- **Appearance**: Hypointense (darker than surrounding tissue)
- **Typical pathology**: Gliomas, metastases
- **Intensity**: 30-70% of white matter intensity

### T2-weighted
- **Appearance**: Hyperintense (brighter than surrounding tissue)
- **Typical pathology**: Edema, gliomas
- **Intensity**: 150-200% of white matter intensity

### FLAIR
- **Appearance**: Hyperintense (bright lesions, suppressed CSF)
- **Typical pathology**: White matter lesions, gliomas
- **Intensity**: 140-180% of white matter intensity

## Customization

### Tumor Parameters

You can modify tumor generation parameters by editing the `shape_gen_args` in `SimpleTumorGenerator`:

```python
generator = SimpleTumorGenerator(device='cpu')
generator.shape_gen_args.update({
    'perlin_res': [3, 3, 3],        # Higher resolution = more detailed shapes
    'mask_percentile_min': 85,       # Lower = larger tumors
    'mask_percentile_max': 99.5,     # Higher = smaller tumors
    'V_multiplier': 800,             # Higher = more fluid deformation
    'min_nt': 15,                    # More time steps = more deformation
    'max_nt': 25,
    'pathol_thres': 0.3              # Higher = smaller final mask
})
```

### Contrast Parameters

Modify tissue contrast values in the `get_contrast_values` method:

```python
# Example: Make tumors more prominent in T1
def get_contrast_values(self, modality):
    if modality.upper() == 'T1':
        mus = torch.tensor([0, 100, 150, 50], device=self.device)
        sigmas = torch.tensor([0, 10, 15, 8], device=self.device)  # Less noise
    # ... rest of the method
```

## Creating Brain Segmentations

If you don't have brain segmentations, you can create them using:

### Option 1: SynthSeg (Recommended)
```bash
# Install SynthSeg
pip install SynthSeg

# Generate segmentation
mri_synthseg --i brain.nii.gz --o segmentation.nii.gz --robust
```

### Option 2: FreeSurfer
```bash
# Run FreeSurfer recon-all
recon-all -i brain.nii.gz -s subject_id -all

# Convert to required format
mri_convert $SUBJECTS_DIR/subject_id/mri/aseg.mgz segmentation.nii.gz
```

### Option 3: FSL FAST
```bash
# Brain extraction first
bet brain.nii.gz brain_bet.nii.gz

# Tissue segmentation
fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o seg brain_bet.nii.gz
```

## Troubleshooting

### Common Issues

1. **"Generated tumor too small"**
   - Increase `--tumor-size` parameter
   - Check that segmentation has sufficient GM/WM regions
   - Reduce `pathol_thres` in parameters

2. **CUDA out of memory**
   - Use `device='cpu'` instead of `device='cuda'`
   - Reduce image size or use smaller `perlin_res`

3. **Poor tumor realism**
   - Increase `V_multiplier` for more fluid deformation
   - Adjust `min_nt`/`max_nt` for more evolution steps
   - Fine-tune contrast parameters

4. **Segmentation format issues**
   - Ensure segmentation has labels 0, 1, 2, 3
   - Check that segmentation and image have same dimensions
   - Verify both files are in NIfTI format

### Performance Tips

1. **Use GPU**: Set `device='cuda'` for faster generation
2. **Smaller images**: Downsample large images for faster processing
3. **Batch processing**: Generate multiple tumors in one script run
4. **Precompute**: Save segmentations to avoid recomputing

## Examples

### Generate T1 tumor on real data
```bash
python scripts/simple_tumor_generator.py \
    --input /path/to/T1.nii.gz \
    --seg /path/to/seg.nii.gz \
    --modality T1 \
    --output ./results \
    --tumor-size 1.2
```

### Generate multiple FLAIR lesions
```bash
python scripts/simple_tumor_generator.py \
    --input /path/to/FLAIR.nii.gz \
    --seg /path/to/seg.nii.gz \
    --modality FLAIR \
    --output ./results \
    --num-tumors 5 \
    --tumor-size 0.8
```

### Use synthetic brain
```bash
python scripts/simple_tumor_generator.py \
    --input dummy.nii.gz \
    --seg /path/to/seg.nii.gz \
    --modality T2 \
    --output ./results \
    --use-synthetic \
    --tumor-size 1.5
```

## Visualization

View results using medical image viewers:

```bash
# FSLeyes
fsleyes results/T1_with_tumor.nii.gz results/tumor_mask.nii.gz

# ITK-SNAP
itksnap results/T1_with_tumor.nii.gz -s results/tumor_mask.nii.gz

# 3D Slicer
# Open 3D Slicer and load the files through the GUI
```

## Citation

If you use this tumor generator in your research, please cite the original UNA paper:

```bibtex
@InProceedings{Liu_2025_UNA,
    author    = {Liu, Peirong and Aguila, Ana L. and Iglesias, Juan E.},
    title     = {Unraveling Normal Anatomy via Fluid-Driven Anomaly Randomization},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
}
```

## Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review the example scripts
3. Check the original UNA repository for more details
4. Open an issue with detailed error messages and data specifications 