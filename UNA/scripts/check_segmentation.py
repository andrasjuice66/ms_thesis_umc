#!/usr/bin/env python3
"""
Segmentation Label Checker
==========================

This script checks if a segmentation file has the correct label values
for use with the tumor generator, and can also analyze SynthSeg/FreeSurfer segmentations.

Expected labels for tumor generation:
- 0: Background
- 1: Gray Matter (GM)
- 2: White Matter (WM)  
- 3: Cerebrospinal Fluid (CSF)

Usage:
    python check_segmentation.py segmentation.nii.gz
    python check_segmentation.py segmentation.nii.gz --check-synthseg
"""

import sys
import argparse
import numpy as np
import nibabel as nib


def get_synthseg_labels():
    """Get the expected SynthSeg labels and their tissue types"""
    synthseg_labels = {
        0: 'background',
        2: 'left cerebral white matter',
        3: 'left cerebral cortex', 
        4: 'left lateral ventricle',
        5: 'left inferior lateral ventricle',
        7: 'left cerebellum white matter',
        8: 'left cerebellum cortex',
        10: 'left thalamus',
        11: 'left caudate',
        12: 'left putamen',
        13: 'left pallidum',
        14: '3rd ventricle',
        15: '4th ventricle',
        16: 'brain-stem',
        17: 'left hippocampus',
        18: 'left amygdala',
        24: 'CSF',
        26: 'left accumbens area',
        28: 'left ventral DC',
        41: 'right cerebral white matter',
        42: 'right cerebral cortex',
        43: 'right lateral ventricle',
        44: 'right inferior lateral ventricle',
        46: 'right cerebellum white matter',
        47: 'right cerebellum cortex',
        49: 'right thalamus',
        50: 'right caudate',
        51: 'right putamen',
        52: 'right pallidum',
        53: 'right hippocampus',
        54: 'right amygdala',
        58: 'right accumbens area',
        60: 'right ventral DC'
    }
    
    # Tissue type mappings
    gm_labels = [3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]
    wm_labels = [2, 41, 7, 46, 16, 28, 60]
    csf_labels = [4, 43, 5, 44, 14, 15, 24]
    
    return synthseg_labels, gm_labels, wm_labels, csf_labels


def check_segmentation_labels(seg_path, verbose=True):
    """
    Check segmentation labels and return validation results
    
    Args:
        seg_path: Path to segmentation file
        verbose: Print detailed information
        
    Returns:
        dict with validation results
    """
    try:
        # Load segmentation
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        
        # Get unique labels
        unique_labels = np.unique(seg_data)
        expected_labels = np.array([0, 1, 2, 3])
        
        # Check labels
        missing_labels = set(expected_labels) - set(unique_labels)
        extra_labels = set(unique_labels) - set(expected_labels)
        
        # Calculate statistics
        label_stats = {}
        for label in unique_labels:
            count = np.sum(seg_data == label)
            percentage = (count / seg_data.size) * 100
            label_stats[int(label)] = {
                'count': count,
                'percentage': percentage
            }
        
        # Determine if valid
        has_required = 1 in unique_labels and 2 in unique_labels
        has_all_expected = len(missing_labels) == 0
        has_only_expected = len(extra_labels) == 0
        
        results = {
            'valid': has_required,
            'complete': has_all_expected,
            'clean': has_only_expected,
            'unique_labels': unique_labels,
            'missing_labels': list(missing_labels),
            'extra_labels': list(extra_labels),
            'label_stats': label_stats,
            'shape': seg_data.shape
        }
        
        if verbose:
            print(f"Segmentation file: {seg_path}")
            print(f"Shape: {seg_data.shape}")
            print(f"Data type: {seg_data.dtype}")
            print(f"Labels found: {sorted(unique_labels.astype(int))}")
            print(f"Expected labels: {expected_labels}")
            
            if missing_labels:
                print(f"❌ Missing labels: {sorted(missing_labels)}")
            else:
                print("✅ All expected labels present")
                
            if extra_labels:
                print(f"⚠️  Extra labels found: {sorted(extra_labels)}")
            else:
                print("✅ No unexpected labels")
            
            print("\nLabel Statistics:")
            label_names = {0: 'Background', 1: 'Gray Matter', 2: 'White Matter', 3: 'CSF'}
            for label in sorted(unique_labels):
                stats = label_stats[int(label)]
                name = label_names.get(int(label), f'Unknown({int(label)})')
                print(f"  Label {int(label)} ({name}): {stats['count']:,} voxels ({stats['percentage']:.1f}%)")
            
            print(f"\nValidation Results:")
            print(f"  ✅ Has required labels (GM=1, WM=2): {has_required}")
            print(f"  {'✅' if has_all_expected else '❌'} Has all expected labels: {has_all_expected}")
            print(f"  {'✅' if has_only_expected else '⚠️ '} Has only expected labels: {has_only_expected}")
            
            if has_required:
                print("\n🎉 Segmentation is VALID for tumor generation!")
            else:
                print("\n❌ Segmentation is NOT VALID - missing required GM/WM labels")
                print("   Consider using convert_segmentation.py to convert from SynthSeg/FreeSurfer format")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error loading segmentation: {e}")
        return {'valid': False, 'error': str(e)}


def check_synthseg_labels(seg_path, verbose=True):
    """
    Check if segmentation has SynthSeg labels and analyze tissue distribution
    
    Args:
        seg_path: Path to segmentation file
        verbose: Print detailed information
        
    Returns:
        dict with SynthSeg validation results
    """
    try:
        # Load segmentation
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        
        # Get SynthSeg label information
        synthseg_labels, gm_labels, wm_labels, csf_labels = get_synthseg_labels()
        
        # Get unique labels in the segmentation
        unique_labels = np.unique(seg_data).astype(int)
        
        # Check which SynthSeg labels are present
        present_synthseg = [label for label in unique_labels if label in synthseg_labels]
        missing_synthseg = [label for label in synthseg_labels.keys() if label not in unique_labels]
        unknown_labels = [label for label in unique_labels if label not in synthseg_labels]
        
        # Calculate tissue type statistics
        gm_voxels = sum(np.sum(seg_data == label) for label in gm_labels if label in unique_labels)
        wm_voxels = sum(np.sum(seg_data == label) for label in wm_labels if label in unique_labels)
        csf_voxels = sum(np.sum(seg_data == label) for label in csf_labels if label in unique_labels)
        bg_voxels = np.sum(seg_data == 0)
        total_voxels = seg_data.size
        
        # Check if it looks like SynthSeg
        is_synthseg = len(present_synthseg) >= 5  # At least 5 SynthSeg labels present
        
        results = {
            'is_synthseg': is_synthseg,
            'present_labels': present_synthseg,
            'missing_labels': missing_synthseg,
            'unknown_labels': unknown_labels,
            'tissue_stats': {
                'gm_voxels': gm_voxels,
                'wm_voxels': wm_voxels,
                'csf_voxels': csf_voxels,
                'bg_voxels': bg_voxels,
                'gm_percentage': (gm_voxels / total_voxels) * 100,
                'wm_percentage': (wm_voxels / total_voxels) * 100,
                'csf_percentage': (csf_voxels / total_voxels) * 100,
                'bg_percentage': (bg_voxels / total_voxels) * 100
            },
            'shape': seg_data.shape
        }
        
        if verbose:
            print(f"SynthSeg Analysis for: {seg_path}")
            print(f"Shape: {seg_data.shape}")
            print(f"Total unique labels: {len(unique_labels)}")
            print(f"Labels found: {sorted(unique_labels)}")
            
            if is_synthseg:
                print("✅ Detected SynthSeg-style segmentation")
            else:
                print("❌ Does not appear to be SynthSeg format")
            
            print(f"\nSynthSeg Labels Present ({len(present_synthseg)}):")
            for label in sorted(present_synthseg):
                count = np.sum(seg_data == label)
                percentage = (count / total_voxels) * 100
                structure = synthseg_labels[label]
                print(f"  {label:2d}: {structure:<30} ({count:,} voxels, {percentage:.1f}%)")
            
            if unknown_labels:
                print(f"\nUnknown Labels ({len(unknown_labels)}):")
                for label in sorted(unknown_labels):
                    count = np.sum(seg_data == label)
                    percentage = (count / total_voxels) * 100
                    print(f"  {label:2d}: Unknown structure        ({count:,} voxels, {percentage:.1f}%)")
            
            print(f"\nTissue Type Summary:")
            stats = results['tissue_stats']
            print(f"  Background: {stats['bg_voxels']:,} voxels ({stats['bg_percentage']:.1f}%)")
            print(f"  Gray Matter: {stats['gm_voxels']:,} voxels ({stats['gm_percentage']:.1f}%)")
            print(f"  White Matter: {stats['wm_voxels']:,} voxels ({stats['wm_percentage']:.1f}%)")
            print(f"  CSF: {stats['csf_voxels']:,} voxels ({stats['csf_percentage']:.1f}%)")
            
            if is_synthseg:
                print(f"\n💡 To convert to 4-label format, use:")
                print(f"   python convert_segmentation.py {seg_path} output.nii.gz --method synthseg")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error analyzing SynthSeg segmentation: {e}")
        return {'is_synthseg': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Check segmentation label values')
    parser.add_argument('segmentation', help='Path to segmentation file (.nii.gz)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode - minimal output')
    parser.add_argument('--check-synthseg', action='store_true', 
                       help='Check if segmentation is in SynthSeg format')
    
    args = parser.parse_args()
    
    if args.check_synthseg:
        # Check SynthSeg format
        results = check_synthseg_labels(args.segmentation, verbose=not args.quiet)
        success = results.get('is_synthseg', False)
    else:
        # Check 4-label format
        results = check_segmentation_labels(args.segmentation, verbose=not args.quiet)
        success = results.get('valid', False)
    
    # Exit with appropriate code
    if success:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Error


if __name__ == '__main__':
    main() 