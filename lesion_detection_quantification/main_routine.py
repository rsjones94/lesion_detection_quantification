import os
from glob import glob
import shutil

import numpy as np
import pandas as pd
import nibabel as nib

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

generate_masks = False
evaluate_masks = True

n_cohorts = 4 # from the data, n unique training sets will be created and evaluated independently


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
    eval_indices = [i for i in hp.chunk(indices, n_cohorts)] # these are the evaluation indices for a given cohort (roughly equally sized sublists composed of indices, with no repetitions)
    training_indices = [[i for i in indices if i not in l] for l in eval_indices] # these are the training indices for a given cohort
    # (each sublist is all the indices that are NOT in the corresponding eval sublist)
    
    mask_master = os.path.join(data_folder, 'generated_masks')
    
    for i, (trainers, evaluators) in enumerate(zip(training_indices, eval_indices)):
        
        print(f'\n--------------------------------\nGenerating masks for cohort {i} ({i+1} of {n_cohorts})\n')
        
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
        
        
        """     
        bianca_write_folder = os.path.join(cohort_folder, 'bianca')
        bianca_bin_folder = os.path.join(cohort_bin_folder, 'bianca')
        print('Generating BIANCA masks')
        mg.generate_bianca_masks(training_data, evaluation_data, bianca_write_folder, bianca_bin_folder, mni_reference_scan)
        """
        
        # mg.generate_knntpp_masks() # not implemented
        # mg.generate_default_lga_masks() # not implemented
        
        standard_lpa_write_folder = os.path.join(cohort_folder, 'lpa_standard')
        standard_lpa_bin_folder = os.path.join(cohort_bin_folder, 'lpa_standard')
        print('Generating standard LPA masks')
        mg.generate_lpa_masks(training_data, evaluation_data, standard_lpa_write_folder, standard_lpa_bin_folder, model_type='default')
        
        
        # mg.generate_custom_lga_masks() # not implemented
        # mg.generate_custom_lpa_masks() # not implemented
    
    
if evaluate_masks:
    
    evaluation_folder = os.path.join(data_folder, 'quality_quantification')
    
    cohort_folders = glob(os.path.join(data_folder, 'generated_masks', '*/'))
    cohorts = [os.path.basename((os.path.normpath(i))) for i in cohort_folders]
    
    metrics_reference = {}
    
    for cohort_name, cohort_folder in zip(cohorts, cohort_folders):
        
        protocol_folders = glob(os.path.join(cohort_folder, '*/'))
        protocol_names = [os.path.basename((os.path.normpath(i))) for i in protocol_folders]
        
        cohort_eval_folder = os.path.join(evaluation_folder, cohort_name)
        if os.path.exists(cohort_eval_folder):
            shutil.rmtree(cohort_eval_folder)
        os.mkdir(cohort_eval_folder)
        
        for protocol_name, protocol_folder in zip(protocol_names, protocol_folders):
            print(f'\nEvaluating {cohort_name} ({protocol_name})')
            masks = glob(os.path.join(protocol_folder, '*.nii.gz'))
            
            out_csv_name = os.path.join(cohort_eval_folder, f'{protocol_name}.csv')
            stats_df = pd.DataFrame()
            
            
            
            for mask in masks:
                pt_name = os.path.basename(os.path.normpath(mask)).split('.')[0]
                golden_standard = os.path.join(data_folder, 'lesion_training_data', pt_name, 'axFLAIR_mask.nii.gz')
                
                # print(f'Generating metrics for {pt_name}:\n{mask} vs. {golden_standard}')
                print(f'Generating metrics for {pt_name}')
                
                mask_data = nib.load(mask).get_fdata()
                golden_data = nib.load(golden_standard).get_fdata()
                
                stats_dict = mq.quantify_mask_quality(golden_data, mask_data)
                the_order = ['pt']
                the_order.extend(stats_dict.keys())
                stats_dict['pt'] = pt_name
                
                stats_ser = pd.Series(stats_dict)
                stats_df = stats_df.append(stats_ser, ignore_index=True)
                stats_df = stats_df[the_order]
            
                stats_df.to_csv(out_csv_name, index=False)
                
            try:
                metrics_reference[protocol_name].append(stats_df)
            except KeyError:
                metrics_reference[protocol_name] = []
    
    average_folder = os.path.join(evaluation_folder, 'average')
    if os.path.exists(average_folder):
        shutil.rmtree(average_folder)
    os.mkdir(average_folder)
    for key, li in metrics_reference.items():
        protocol_df = pd.concat(li)
        mean_df = protocol_df.groupby('pt', as_index=False).mean()
        
        csv_name = os.path.join(average_folder, f'{key}.csv')
        mean_df.to_csv(csv_name)
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    