import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import datetime

import numpy as np
import torch
import nibabel as nib

import utils.misc as utils
from Generator.datasets import BaseGen
from FluidAnomaly.DiffEqs.pde import AdvDiffPDE

# Default config files
default_gen_cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cfgs/generator/default.yaml')
demo_gen_cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cfgs/generator/test/demo_generator.yaml')

class SimpleGenerator(BaseGen):
    def __init__(self, gen_args, device='cpu'):
        self.gen_args = gen_args
        self.split = gen_args.split
        self.synth_args = self.gen_args.generator
        self.shape_gen_args = gen_args.pathology_shape_generator
        self.real_image_args = gen_args.real_image_generator
        self.synth_image_args = gen_args.synth_image_generator
        self.augmentation_steps = vars(gen_args.augmentation_steps)
        self.device = device
        
        # Initialize paths and names (needed by parent class)
        self.paths = {}
        self.names = []
        
        # Initialize tasks
        self.tasks = [key for (key, value) in vars(self.gen_args.task).items() if value]
        
        # Initialize grid without requiring paths
        self.size = self.synth_args.size
        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float, device=self.device)
        self.yy = torch.tensor(yy, dtype=torch.float, device=self.device)
        self.zz = torch.tensor(zz, dtype=torch.float, device=self.device)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float, device=self.device)
        self.xc = self.xx - self.c[0]
        self.yc = self.yy - self.c[1]
        self.zc = self.zz - self.c[2]
        
        # Initialize one-hot encoding
        self.prepare_one_hot()
        
        # Initialize PDE if needed
        if self.synth_args.augment_pathology:
            self.t = torch.from_numpy(np.arange(self.shape_gen_args.max_nt) * self.shape_gen_args.dt).to(self.device)
            with torch.no_grad():
                self.adv_pde = AdvDiffPDE(data_spacing=[1., 1., 1.], 
                                    perf_pattern='adv', 
                                    V_type='vector_div_free', 
                                    V_dict={},
                                    BC=self.shape_gen_args.bc, 
                                    dt=self.shape_gen_args.dt, 
                                    device=self.device
                                    )
        else:
            self.t, self.adv_pde = None, None
            
    def generate_deformation_dict(self, input_shape, input_affine):
        """Generate a deformation dictionary for the input image"""
        # Create identity transform
        A = torch.eye(3, dtype=torch.float, device=self.device)
        scaling_factor_distances = 1.0
        c2 = torch.tensor((np.array(input_shape) - 1) / 2, dtype=torch.float, device=self.device)
        
        # No nonlinear deformation
        F = None
        Fneg = None
        
        # Calculate grid coordinates
        xx2 = self.xx
        yy2 = self.yy
        zz2 = self.zz
        
        # Get margins
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = input_shape[0]
        y2 = input_shape[1]
        z2 = input_shape[2]
        
        # Convert affine to tensor if it's not already
        if isinstance(input_affine, np.ndarray):
            input_affine = torch.from_numpy(input_affine).float().to(self.device)
        
        return {
            'orig_shp': input_shape,
            'scaling_factor_distances': scaling_factor_distances,
            'A': A,
            'c2': c2,
            'F': F,
            'Fneg': Fneg,
            'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2],
            'aff_orig': input_affine,
            'flip2orig': torch.eye(4, dtype=torch.float, device=self.device)
        }
        
    def generate_sample(self, name, G, setups, deform_dict, res, target):
        """Override generate_sample to handle our simplified case"""
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']
        
        # Use the provided segmentation
        G = target['segmentation']
        Gr = torch.round(G).long()
        
        # Generate contrasts
        mus, sigmas = self.get_contrast(setups['photo_mode'])
        
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape, dtype=torch.float, device=self.device)
        
        if self.synth_args.pv:
            mask = (G!=Gr)
            SYN[mask] = 0
            Gv = G[mask]
            isv = torch.zeros(Gv.shape, dtype=torch.float, device=self.device)
            pw = (Gv<=3) * (3-Gv)
            isv += pw * mus[2] + pw * sigmas[2] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pg = (Gv<=3) * (Gv-2) + (Gv>3) * (4-Gv)
            isv += pg * mus[3] + pg * sigmas[3] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            pcsf = (Gv>=3) * (Gv-3)
            isv += pcsf * mus[4] + pcsf * sigmas[4] * torch.randn(Gv.shape, dtype=torch.float, device=self.device)
            SYN[mask] = isv
        
        SYN[SYN < 0] = 0
        
        # Generate pathology
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor) and target['pathology'].sum() > 0:
            wm_mask = (Gr==2)
            wm_mean = (SYN * wm_mask).sum() / wm_mask.sum()
            gm_mask = (Gr==1)
            gm_mean = (SYN * gm_mask).sum() / gm_mask.sum()
            
            target['pathology'][G < 1] = 0
            target['pathology_prob'][G < 1] = 0
            
            # Determine pathology direction based on modality
            pathol_direction = self.get_pathology_direction(setups.get('modality', 'T1'))
            
            # Encode pathology
            SYN = self.encode_pathology(SYN, target['pathology'], target['pathology_prob'], pathol_direction)
            SYN[SYN < 0.] = 0.
        else:
            pathol_direction = None
            target['pathology'] = torch.zeros_like(SYN)
            target['pathology_prob'] = torch.zeros_like(SYN)
        
        # Prepare flipped conditional input
        SYN_flip = torch.flip(SYN, [0])
        # Deform flip to non-flip space
        SYN_flip = self.deform_flip2orig(SYN_flip, deform_dict)
        
        return target['pathology'], target['pathology_prob'], self.augment_sample(
            name, SYN, SYN_flip, setups, deform_dict, res, target, 
            pathol_direction=pathol_direction, input_mode=setups.get('modality', 'T1')
        )

def generate_tumor(input_path, seg_path, modality, output_dir):
    """
    Generate a synthetic tumor using Perlin noise
    Args:
        input_path: Path to input nii.gz file
        seg_path: Path to segmentation nii.gz file (1=GM, 2=WM, 3=CSF)
        modality: One of ['T1', 'T2', 'FLAIR']
        output_dir: Directory to save the output
    """
    # Load configuration
    gen_args = utils.preprocess_cfg([default_gen_cfg_file, demo_gen_cfg_file])
    
    # Override some config settings for our use case
    gen_args.out_dir = output_dir
    gen_args.test_itr_limit = 1  # We only want to generate one sample
    gen_args.mild_samples = False  # We want clear tumors
    gen_args.all_samples = True  # Use all available samples
    
    # Set up device
    if gen_args.device_generator:
        device = gen_args.device_generator
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    print('Using device:', device)
    
    # Create output directory
    utils.make_dir(output_dir, reset=True)
    start_time = time.time()
    
    # Create our simple generator
    generator = SimpleGenerator(gen_args, device=device)
    
    # Load input image and segmentation
    input_img = nib.load(input_path)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    
    seg_data = nib.load(seg_path).get_fdata()
    
    # Create a sample
    subjects = {
        modality: torch.from_numpy(input_data).float().to(device),
        'aff': torch.from_numpy(affine).float().to(device),
        'shp': input_data.shape,
        'loc_idx': None,
        'segmentation': torch.from_numpy(seg_data).long().to(device),
        'pathology': torch.zeros_like(torch.from_numpy(input_data)).float().to(device),
        'pathology_prob': torch.zeros_like(torch.from_numpy(input_data)).float().to(device)
    }
    
    # Generate deformation dictionary
    deform_dict = generator.generate_deformation_dict(input_data.shape, affine)
    
    # Generate tumor
    sample = generator.generate_sample(
        name=os.path.basename(input_path),
        G=None,  # We don't use a generator
        setups={'photo_mode': False, 'spac': 1.0, 'modality': modality, 'thickness': 1.0},
        deform_dict=deform_dict,
        res=[1.0, 1.0, 1.0],
        target=subjects
    )
    
    # Save the results
    if sample is not None:
        save_dir = os.path.join(output_dir, f'{modality}_with_tumor')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the diseased image
        utils.viewVolume(
            sample['input'],
            affine,
            names=[f'{modality}_diseased'],
            save_dir=save_dir
        )
        
        # Save the tumor mask
        if 'pathology' in sample:
            utils.viewVolume(
                sample['pathology'],
                affine,
                names=['tumor_mask'],
                save_dir=save_dir
            )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Generation completed in {total_time_str}')

if __name__ == '__main__':
    # Hardcoded parameters - MODIFY THESE VALUES
    INPUT_PATH = "C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC110033_T1w.nii.gz"
    SEG_PATH = "C:/Projects/thesis_project/brain_age_pred/data/templates/seg_T1.nii.gz"  # Should have values: 1=GM, 2=WM, 3=CSF
    MODALITY = "T1"  # or "T2" or "FLAIR"
    OUTPUT_DIR = "C:/Projects/thesis_project/OtherRepos/UNA/output"
    
    # Run the generation
    generate_tumor(INPUT_PATH, SEG_PATH, MODALITY, OUTPUT_DIR)