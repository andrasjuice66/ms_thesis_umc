#!/usr/bin/env python3
"""
Segmentation Converter
======================

This script converts SynthSeg/FreeSurfer-style segmentations to the simple 4-label format 
expected by the tumor generator:
- 0: Background
- 1: Gray Matter (GM)
- 2: White Matter (WM)
- 3: Cerebrospinal Fluid (CSF)

Usage:
    python convert_segmentation.py input_seg.nii.gz output_seg.nii.gz
"""

import sys
import argparse
import numpy as np
import nibabel as nib


def get_synthseg_label_mapping():
    """
    Get mapping from SynthSeg labels to tissue types
    Based on the actual SynthSeg label table
    """
    # Gray Matter labels (cortical and subcortical)
    gm_labels = [
        3,   # left cerebral cortex
        42,  # right cerebral cortex
        8,   # left cerebellum cortex
        47,  # right cerebellum cortex
        10,  # left thalamus
        49,  # right thalamus
        11,  # left caudate
        50,  # right caudate
        12,  # left putamen
        51,  # right putamen
        13,  # left pallidum
        52,  # right pallidum
        17,  # left hippocampus
        53,  # right hippocampus
        18,  # left amygdala
        54,  # right amygdala
        26,  # left accumbens area
        58,  # right accumbens area
    ]
    
    # White Matter labels
    wm_labels = [
        2,   # left cerebral white matter
        41,  # right cerebral white matter
        7,   # left cerebellum white matter
        46,  # right cerebellum white matter
        16,  # brain-stem
        28,  # left ventral DC
        60,  # right ventral DC
    ]
    
    # CSF labels
    csf_labels = [
        4,   # left lateral ventricle
        43,  # right lateral ventricle
        5,   # left inferior lateral ventricle
        44,  # right inferior lateral ventricle
        14,  # 3rd ventricle
        15,  # 4th ventricle
        24,  # CSF
    ]
    
    return gm_labels, wm_labels, csf_labels


def get_freesurfer_label_mapping():
    """
    Get mapping from FreeSurfer labels to tissue types
    Based on FreeSurfer's LUT (Look-Up Table) - keeping for backward compatibility
    """
    # Gray Matter labels (cortical and subcortical)
    gm_labels = [
        # Cortical GM
        3, 42,  # Left/Right Cerebral Cortex
        17, 53,  # Left/Right Hippocampus
        18, 54,  # Left/Right Amygdala
        11, 50,  # Left/Right Caudate
        12, 51,  # Left/Right Putamen
        13, 52,  # Left/Right Pallidum
        26, 58,  # Left/Right Accumbens
        # Additional cortical regions (if present)
        1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
        2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
    ]
    
    # White Matter labels
    wm_labels = [
        2, 41,   # Left/Right Cerebral White Matter
        7, 46,   # Left/Right Cerebellum White Matter
        16,      # Brain Stem
        28,      # Left/Right Ventral DC
        60,      # Right Ventral DC
        # Additional WM regions
        251, 252, 253, 254, 255  # CC (Corpus Callosum) subdivisions
    ]
    
    # CSF labels
    csf_labels = [
        4, 43,   # Left/Right Lateral Ventricle
        5, 44,   # Left/Right Inf Lat Vent
        14, 15,  # 3rd/4th Ventricle
        24,      # CSF
        72,      # 5th Ventricle
    ]
    
    return gm_labels, wm_labels, csf_labels


def convert_segmentation(input_seg, method='synthseg'):
    """
    Convert segmentation to 4-label format
    
    Args:
        input_seg: numpy array of input segmentation
        method: conversion method ('synthseg', 'freesurfer', 'intensity', 'custom')
    
    Returns:
        numpy array with labels 0,1,2,3
    """
    output_seg = np.zeros_like(input_seg, dtype=np.uint8)
    
    if method == 'synthseg':
        gm_labels, wm_labels, csf_labels = get_synthseg_label_mapping()
        
        # Convert to tissue types
        for label in gm_labels:
            output_seg[input_seg == label] = 1  # GM
        
        for label in wm_labels:
            output_seg[input_seg == label] = 2  # WM
            
        for label in csf_labels:
            output_seg[input_seg == label] = 3  # CSF
            
        # Background remains 0
        
    elif method == 'freesurfer':
        gm_labels, wm_labels, csf_labels = get_freesurfer_label_mapping()
        
        # Convert to tissue types
        for label in gm_labels:
            output_seg[input_seg == label] = 1  # GM
        
        for label in wm_labels:
            output_seg[input_seg == label] = 2  # WM
            
        for label in csf_labels:
            output_seg[input_seg == label] = 3  # CSF
            
        # Background remains 0
        
    elif method == 'intensity':
        # Simple intensity-based conversion (less reliable)
        # This is a fallback method
        unique_labels = np.unique(input_seg)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        if len(unique_labels) >= 3:
            # Assume lowest non-zero is CSF, middle is GM, highest is WM
            sorted_labels = np.sort(unique_labels)
            csf_label = sorted_labels[0]
            gm_label = sorted_labels[len(sorted_labels)//3]
            wm_label = sorted_labels[2*len(sorted_labels)//3]
            
            output_seg[input_seg == csf_label] = 3  # CSF
            output_seg[input_seg == gm_label] = 1   # GM
            output_seg[input_seg == wm_label] = 2   # WM
        
    elif method == 'custom':
        # Custom mapping - you can modify this based on your specific labels
        label_mapping = {
            # Based on your specific labels - you may need to adjust these
            # Common mappings for your segmentation:
            2: 2,   # White matter (if label 2 is WM)
            3: 1,   # Gray matter (if label 3 is GM)
            4: 3,   # CSF (if label 4 is CSF)
            5: 3,   # Additional CSF
            # You can add more mappings based on your segmentation
            # Example for your labels:
            # 41: 2, 42: 1, 43: 3, 44: 3,  # Right hemisphere equivalents
            # Add more as needed...
        }
        
        # If you want to map multiple labels to the same tissue type:
        # GM labels (adjust based on your segmentation)
        gm_custom = [3, 42, 11, 12, 13, 17, 18, 26, 50, 51, 52, 53, 54, 58]
        # WM labels  
        wm_custom = [2, 41, 7, 46, 16, 28, 60]
        # CSF labels
        csf_custom = [4, 5, 14, 15, 24, 43, 44]
        
        for label in gm_custom:
            if label in np.unique(input_seg):
                output_seg[input_seg == label] = 1  # GM
        
        for label in wm_custom:
            if label in np.unique(input_seg):
                output_seg[input_seg == label] = 2  # WM
                
        for label in csf_custom:
            if label in np.unique(input_seg):
                output_seg[input_seg == label] = 3  # CSF
    
    return output_seg


def analyze_segmentation(seg_data):
    """Analyze segmentation to help determine the best conversion method"""
    unique_labels = np.unique(seg_data)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    print(f"Found {len(unique_labels)} unique non-zero labels")
    print(f"Label range: {unique_labels.min():.0f} - {unique_labels.max():.0f}")
    print(f"Unique labels: {sorted(unique_labels.astype(int))}")
    
    # Check if it looks like SynthSeg
    synthseg_indicators = [2, 3, 41, 42, 24]  # Common SynthSeg labels
    has_synthseg = sum(label in unique_labels for label in synthseg_indicators)
    
    # Check if it looks like FreeSurfer
    freesurfer_indicators = [1001, 1002, 2001, 2002]  # FreeSurfer cortical parcellation
    has_freesurfer = any(label in unique_labels for label in freesurfer_indicators)
    
    if has_synthseg >= 3:
        print("✅ Detected SynthSeg-style labeling")
        return 'synthseg'
    elif has_freesurfer:
        print("✅ Detected FreeSurfer-style labeling")
        return 'freesurfer'
    elif unique_labels.max() < 10:
        print("⚠️  Simple labeling detected - may need custom mapping")
        return 'custom'
    else:
        print("⚠️  Unknown labeling scheme - will try intensity-based conversion")
        return 'intensity'


def main():
    parser = argparse.ArgumentParser(description='Convert segmentation to 4-label format')
    parser.add_argument('input', help='Input segmentation file (.nii.gz)')
    parser.add_argument('output', help='Output segmentation file (.nii.gz)')
    parser.add_argument('--method', choices=['synthseg', 'freesurfer', 'intensity', 'custom'], 
                       help='Conversion method (auto-detected if not specified)')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Only analyze the segmentation without converting')
    
    args = parser.parse_args()
    
    # Load input segmentation
    print(f"Loading segmentation: {args.input}")
    try:
        seg_img = nib.load(args.input)
        seg_data = seg_img.get_fdata()
    except Exception as e:
        print(f"❌ Error loading segmentation: {e}")
        sys.exit(1)
    
    print(f"Shape: {seg_data.shape}")
    print(f"Data type: {seg_data.dtype}")
    
    # Analyze segmentation
    if args.method:
        method = args.method
        print(f"Using specified method: {method}")
    else:
        method = analyze_segmentation(seg_data)
        print(f"Auto-detected method: {method}")
    
    if args.analyze_only:
        print("\nAnalysis complete. Use --method to specify conversion method.")
        return
    
    # Convert segmentation
    print(f"\nConverting using method: {method}")
    converted_seg = convert_segmentation(seg_data, method)
    
    # Check results
    unique_output = np.unique(converted_seg)
    print(f"Output labels: {unique_output}")
    
    # Calculate statistics
    total_voxels = converted_seg.size
    for label in [0, 1, 2, 3]:
        count = np.sum(converted_seg == label)
        percentage = (count / total_voxels) * 100
        label_name = {0: 'Background', 1: 'Gray Matter', 2: 'White Matter', 3: 'CSF'}[label]
        print(f"  Label {label} ({label_name}): {count:,} voxels ({percentage:.1f}%)")
    
    # Check if conversion was successful
    has_gm = 1 in unique_output
    has_wm = 2 in unique_output
    
    if not (has_gm and has_wm):
        print("⚠️  WARNING: Conversion may not be successful - missing GM or WM labels")
        print("   You may need to use a different method or create custom mappings")
    else:
        print("✅ Conversion successful!")
    
    # Save converted segmentation
    print(f"\nSaving converted segmentation: {args.output}")
    converted_img = nib.Nifti1Image(converted_seg.astype(np.uint8), seg_img.affine, seg_img.header)
    nib.save(converted_img, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main() 