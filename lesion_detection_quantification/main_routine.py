import os
from glob import glob
import shutil

import numpy as np
import pandas as pd

import mask_quantification as mq
import mask_generation as mg
import helpers as hp

np.random.seed(0)

"""
In order for this script to run correctly, you must:
    1) download FSL and make sure its executables are added to your PATH
    2) download SPM12 and the LST add-on and make sure its executables are added to your path (and thus, you must have MATLAB installed as well)
    3) download kNN-TTP and make sure its executables are added to your path


"""


data_folder = '/Users/skyjones/Documents/lesion_detection_quantification_data/'
mni_reference_scan = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

generate_masks = True
evaluate_masks = True

n_cohorts = 3 # from the data, n unique training sets will be created and evaluated independently


##############


training_master = os.path.join(data_folder, 'lesion_training_data')
training_subs = glob(os.path.join(training_master, '*/'))
pt_names = [os.path.basename(os.path.normpath(i)) for i in training_subs]
pt_names.remove('bin')
pt_names = np.array(pt_names)

pt_goldens = np.array([os.path.join(training_master, i, 'axFLAIR_mask.nii.gz') for i in pt_names])
pt_t1s = np.array([os.path.join(training_master, i, 'axT1.nii.gz') for i in pt_names])
pt_flairs = np.array([os.path.join(training_master, i, 'axFLAIR.nii.gz') for i in pt_names])


if generate_masks:
    
    indices = np.arange(0,len(pt_names))
    training_indices = [i for i in hp.chunk(indices, n_cohorts)] # these are the training indices for a given cohort (roughly equally sized sublists composed of indices, with no repetitions)
    eval_indices = [[i for i in indices if i not in l] for l in training_indices] # these are the evaluation indices for a given cohort (each sublist is all the indices that are NOT in the corresponding trainign sublist)
    
    mask_master = os.path.join(data_folder, 'generated_masks')
    
    for i, (trainers, evaluators) in enumerate(zip(training_indices, eval_indices)):
        
        print(f'Generating masks for cohort {i} ({i+1} of {n_cohorts})')
        
        training_goldens = pt_goldens[[trainers]]
        training_t1s = pt_t1s[[trainers]]
        training_flairs = pt_flairs[[trainers]]
        training_names = pt_names[[trainers]]
        
        eval_goldens = pt_goldens[[evaluators]]
        eval_t1s = pt_t1s[[evaluators]]
        eval_flairs = pt_flairs[[evaluators]]
        eval_names = pt_names[[evaluators]]
        
        training_data = [training_names, training_flairs, training_t1s, training_goldens]
        evaluation_data = [eval_names, eval_flairs, eval_t1s]
        
        
        cohort_folder = os.path.join(mask_master, f'cohort_{i}')
        if os.path.exists(cohort_folder):
            shutil.rmtree(cohort_folder)
        os.mkdir(cohort_folder)
        
        cohort_bin_folder = os.path.join(data_folder, 'bin', f'cohort_{i}')
        if os.path.exists(cohort_bin_folder):
            shutil.rmtree(cohort_bin_folder)
        os.mkdir(cohort_bin_folder)
        
        
        bianca_write_folder = os.path.join(cohort_folder, 'bianca')
        bianca_bin_folder = os.path.join(cohort_bin_folder, 'bianca')
        print('Generating BIANCA masks')
        mg.generate_bianca_masks(training_data, evaluation_data, bianca_write_folder, bianca_bin_folder, mni_reference_scan)
        
        # not implemented
        mg.generate_knntpp_masks()
        mg.generate_default_lga_masks()
        mg.generate_default_lpa_masks()
        mg.generate_custom_lga_masks()
        mg.generate_custom_lpa_masks()
    
    
if evaluate_masks:
    pass