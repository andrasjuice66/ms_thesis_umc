import numpy as np

GENERATION_LABELS = np.array([
    # Neutral (non-sided) labels --------------- ↓  n_neutral_labels = 5
      0,        # background – voxels outside the skull-stripped brain volume
     14,        # 3rd ventricle
     15,        # 4th ventricle
     16,        # brain-stem
     24,        # CSF / sub-arachnoid space

    # Left-hemisphere labels -------------------
      2,   # cerebral white matter - L
      3,   # cerebral cortex       - L
      4,   # lateral ventricle     - L
      5,   # inferior lat. vent.   - L
      7,   # cerebellar WM         - L
      8,   # cerebellar cortex     - L
     10,   # thalamus              - L
     11,   # caudate               - L
     12,   # putamen               - L
     13,   # pallidum              - L
     17,   # hippocampus           - L
     18,   # amygdala              - L
     26,   # accumbens             - L
     28,   # ventral DC            - L

    # Right-hemisphere labels ------------------
     41,   # cerebral white matter - R
     42,   # cerebral cortex       - R
     43,   # lateral ventricle     - R
     44,   # inferior lat. vent.   - R
     46,   # cerebellar WM         - R
     47,   # cerebellar cortex     - R
     49,   # thalamus              - R
     50,   # caudate               - R
     51,   # putamen               - R
     52,   # pallidum              - R
     53,   # hippocampus           - R
     54,   # amygdala              - R
     58,   # accumbens             - R
     60,   # ventral DC            - R
], dtype=np.int16)

N_NEUTRAL_LABELS = 5        # background + 3rdV + 4thV + brain-stem + CSF

GENERATION_CLASSES = np.array([
     # neutral ----------------------------------------------------------
      0,     # background
      1, 1, 2, 1,            # 3rd V, 4th V, brain-stem, CSF  (CSF-like tissue)

     # left-hemisphere -----------------------------------------------
      3,                   # cerebral WM
      4,                   # cortex
      1, 1,                # lat / inf-lat ventricles  (share CSF class)
      5, 6,                # cerebellar WM / cortex
      7, 8, 9, 10,         # thalamus / caudate / putamen / pallidum
     11, 12,               # hippocampus / amygdala
     13, 14,               # accumbens / ventral DC

     # right-hemisphere (mirror the left-side classes) ---------------
      3, 4,                # WM / cortex
      1, 1,                # ventricles (CSF)
      5, 6,                # cerebellar WM / cortex
      7, 8, 9, 10,         # thalamus / caudate / putamen / pallidum
     11, 12,               # hippocampus / amygdala
     13, 14                # accumbens / ventral DC
], dtype=np.int16)


print("Size: ", len(GENERATION_CLASSES), len(GENERATION_LABELS))