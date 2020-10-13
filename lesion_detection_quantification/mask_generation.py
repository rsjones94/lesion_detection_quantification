import os
import shutil
from glob import glob
import gzip
import subprocess

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn import metrics

import helpers as hp


PROTOTYPING = False

def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


def generate_bianca_masks(training_data, evaluation_data, write_folder, bin_folder, mni_ref):
    """
    
    Parameters
    ----------
    training_data : list of lists of filepaths
        the data used to generate the training model. [names, flairs, t1s, masks]
    evaluation_data : list of lists of filepaths. [names, flairs, t1s]
        the data for which masks will be generated.
    write_folder : filepath
        folder that evaluation masks will be written to
    bin_folder : filepath
        folder that helper files will be written to.
    mni_ref : filepath
        path to an MNI space reference brain (to generate omat for spatial probabilities).

    Returns
    -------
    None.

    """
    if PROTOTYPING:
        print(f'\n~~~~~~~~~~~~\nPrototyping is on. n={PROTOTYPING}\n~~~~~~~~~~~~\n')
    
    train_names, train_flairs, train_t1s, train_masks = training_data
    eval_names, eval_flairs, eval_t1s = evaluation_data
    
    
    if os.path.exists(bin_folder):
        shutil.rmtree(bin_folder)
    os.mkdir(bin_folder)
    
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)
    os.mkdir(write_folder)
    
    # step 1: generate model based on training set
    stripped_flairs = []
    stripped_and_registered_t1s = []
    mni_omats = []
    masks = []
    used_names = []
    training_folder = os.path.join(bin_folder, 'training')
    os.mkdir(training_folder)
    print('Preprocessing training data for BIANCA (skullstripping and registration)')
    for i, (name, flair, t1, mask) in enumerate(zip(train_names, train_flairs, train_t1s, train_masks)):
        print(f'{name} ({i+1} of {len(train_names)})')
        pt_folder = os.path.join(training_folder, name)
        os.mkdir(pt_folder)
        # first step is to skullstrip the scans and register the t1s to flair space
        stripped_flair = os.path.join(pt_folder, 'stripped_flair.nii.gz')
        hp.skullstrip(flair, stripped_flair)
        
        stripped_t1 = os.path.join(pt_folder, 'stripped_t1.nii.gz')
        hp.skullstrip(t1, stripped_t1)
        
        stripped_and_registered_t1 = os.path.join(pt_folder, 'stripped_and_registered_t1.nii.gz')
        hp.register_scan(stripped_flair, stripped_t1, stripped_and_registered_t1)
        
        omat = os.path.join(pt_folder, 'to_mni_omat.mat')
        stripped_flair_mni = os.path.join(pt_folder, 'stripped_flair_mni.nii.gz')
        hp.generate_omat(mni_ref, stripped_flair, stripped_flair_mni, omat)
        
        stripped_flairs.append(stripped_flair)
        stripped_and_registered_t1s.append(stripped_and_registered_t1)
        mni_omats.append(omat)
        masks.append(mask)
        
        used_names.append(name)
           
        if PROTOTYPING:
            if i >= PROTOTYPING: # for speeding up prototyping
                break
        
    bianca_master_name = os.path.join(bin_folder, 'bianca_master.txt')
    hp.generate_bianca_master(bianca_master_name, stripped_flairs, stripped_and_registered_t1s, masks, mni_omats)
    model_name = os.path.join(bin_folder, 'bianca_model')
    print('Generating BIANCA model')
    hp.construct_bianca_cmd(bianca_master_name, 1, 1, 3, 4, model_name)
    
    # step 2: using the model from step 1 and the training set, find the masking threshold that maximizes the average dice coefficient
    # maybe you could get results with a heuristic search but we'll just do it exhaustively (with a predetermined list of possible values)

    print('Generating training masks')
    training_mask_folder = os.path.join(bin_folder, 'training_masks')
    os.mkdir(training_mask_folder)
    classifier_name = model_name + '_classifier'
    prob_mask_names = []
    for i, (name, flair, t1, omat) in enumerate(zip(used_names, stripped_flairs, stripped_and_registered_t1s, mni_omats)):
        print(f'{name} ({i+1} of {len(used_names)})')
        prob_mask_name = os.path.join(training_mask_folder, f'{name}_probmask.nii.gz')
        mini_master = hp.generate_mini_master(training_mask_folder, flair, t1, omat)
        hp.execute_bianca(mini_master, classifier_name, 1, 1, 3, prob_mask_name)
        prob_mask_names.append(prob_mask_name)
    
    the_by = 0.05
    thresholds = np.arange(the_by, 1+the_by, the_by)
    dices = []
    
    print('Determining optimal threshold')
    for i, (p,t) in enumerate(zip(prob_mask_names, masks)):
        print(f'\n{i+1} of {len(prob_mask_names)}')
        print(f'Comparison:\n{p}\nvs.\n{t}')
        prob_mask = nib.load(p).get_fdata()
        true_mask = nib.load(t).get_fdata()
        
        y_true_f = true_mask.flatten()
        
        dices_for_this_pt = []
        for thresh in thresholds:
            print(round(thresh,2))
            bin_mask = prob_mask >= thresh
            y_pred_f = bin_mask.flatten()
            dice = metrics.f1_score(y_true_f, y_pred_f)
            dices_for_this_pt.append(dice)
        dices.append(np.array(dices_for_this_pt))
    dices = np.array(dices)
    means = dices.mean(0)
    amax = np.argmax(means)
    winning_thresh = thresholds[amax]
    print(f'\nOptimal threshold is {round(winning_thresh,2)}\n')
    
    plt.figure()
    for row in dices:
        plt.plot(thresholds, row, color='black', alpha=0.2)
    plt.plot(thresholds, means, color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Dice coefficient')
    plt.xlim(0,1)
    plt.title(f'Effect of threshold on cohort lesion mask quality\nOptimal threshold: {round(winning_thresh,2)}')
    
    figname = os.path.join(bin_folder, 'thresh_plot.png')
    plt.savefig(figname)    
        
    # step 3: preprocess evaluation data
    stripped_flairs_eval = []
    stripped_and_registered_t1s_eval = []
    mni_omats_eval = []
    used_names_eval = []
    eval_folder = os.path.join(bin_folder, 'eval')
    os.mkdir(eval_folder)
    print('Preprocessing evaluation scans')
    for i, (name, flair, t1) in enumerate(zip(eval_names, eval_flairs, eval_t1s)):
        print(f'{name} ({i+1} of {len(eval_names)})')
        pt_folder_eval = os.path.join(eval_folder, name)
        os.mkdir(pt_folder_eval)
        # first step is to skullstrip the scans and register the t1s to flair space
        stripped_flair_eval = os.path.join(pt_folder_eval, 'stripped_flair.nii.gz')
        hp.skullstrip(flair, stripped_flair_eval)
        
        stripped_t1_eval = os.path.join(pt_folder_eval, 'stripped_t1.nii.gz')
        hp.skullstrip(t1, stripped_t1_eval)
        
        stripped_and_registered_t1_eval = os.path.join(pt_folder_eval, 'stripped_and_registered_t1.nii.gz')
        hp.register_scan(stripped_flair_eval, stripped_t1_eval, stripped_and_registered_t1_eval)
        
        omat_eval = os.path.join(pt_folder_eval, 'to_mni_omat.mat')
        stripped_flair_mni_eval = os.path.join(pt_folder_eval, 'stripped_flair_mni.nii.gz')
        hp.generate_omat(mni_ref, stripped_flair_eval, stripped_flair_mni_eval, omat_eval)
        
        stripped_flairs_eval.append(stripped_flair_eval)
        stripped_and_registered_t1s_eval.append(stripped_and_registered_t1_eval)
        mni_omats_eval.append(omat_eval)
        
        used_names_eval.append(name)
        
        if PROTOTYPING:
            if i >= PROTOTYPING: # for speeding up prototyping
                break
            
    # step 4: using the model and the masking threshold from step 2, generate masks for the evaluation set
    print('Generating and thresholding evaluation masks')
    eval_mask_folder = os.path.join(bin_folder, 'eval_masks')
    os.mkdir(eval_mask_folder)
    for i, (name, flair, t1) in enumerate(zip(used_names_eval, stripped_flairs_eval, stripped_and_registered_t1s_eval)):
        print(f'\n{name} ({i+1} of {len(used_names_eval)})')
        eval_mask_name = os.path.join(eval_mask_folder, f'{name}_probmask.nii.gz')
        mini_master = hp.generate_mini_master(eval_mask_folder, flair, t1, omat)
        print('Generating probability mask')
        hp.execute_bianca(mini_master, classifier_name, 1, 1, 3, eval_mask_name)
        print('Thresholding mask')
        pmask = nib.load(eval_mask_name)
        fdata = pmask.get_fdata()
        
        bin_mask = (fdata >= winning_thresh).astype(int)
        bin_mask_name = os.path.join(write_folder, f'{name}.nii.gz')
        
        out = nib.Nifti1Image(bin_mask, pmask.affine, pmask.header)
        nib.save(out, bin_mask_name)
        

def generate_knntpp_masks():
    pass


def generate_lga_masks():
    pass


def generate_lpa_masks(training_data, evaluation_data, write_folder, bin_folder, model_type):
    """
    

    Parameters
    ----------
    training_data : list of lists of filepaths
        the data used to generate the training thresholds. [names, flairs, t1s, masks]
        note that t1s are not actually needed
        and can be a dummy variable
    evaluation_data : list of lists of filepaths. [names, flairs, t1s]
        the data for which masks will be generated. note that t1s are not actually needed
        and can be a dummy variable
    write_folder : filepath
        folder that evaluation masks will be written to
    bin_folder : filepath
        folder that helper files will be written to.
    model_type : str
        'default' or 'custom'

    Returns
    -------
    None.

    """
    
    if PROTOTYPING:
        print(f'\n~~~~~~~~~~~~\nPrototyping is on. n={PROTOTYPING}\n~~~~~~~~~~~~\n')

    
    if model_type != 'default':
        raise NotImplementedError
    
    train_names, train_flairs, train_t1s, train_masks = training_data
    eval_names, eval_flairs, eval_t1s = evaluation_data
    
    
    matlab_alias = "/Applications/MATLAB_R2020b.app/bin/matlab -nodesktop -nosplash -nojvm"
    
    
    if os.path.exists(bin_folder):
        shutil.rmtree(bin_folder)
    os.mkdir(bin_folder)
    
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)
    os.mkdir(write_folder)
    
    # step 1: generate probability masks for the training set
    
    training_folder = os.path.join(bin_folder, 'training')
    os.mkdir(training_folder)
    print(f'Generating training masks for LPA ({model_type})')
    unpacked_flairs = []
    probability_masks = []
    used_train_masks = []
    for i, (name, flair, t1, mask) in enumerate(zip(train_names, train_flairs, train_t1s, train_masks)):
        print(f'{name} ({i+1} of {len(train_names)})')
        pt_folder = os.path.join(training_folder, name)
        os.mkdir(pt_folder)
        
        # first step is unzip the FLAIR, move it to the pt_folder and then run LPA
        """
        f = gzip.open(flair, 'rb')
        file_content = f.read()
        f.close()
        """
        
        unzipped_flair = flair[:-3]
        unzipped_flair_base = os.path.basename(os.path.normpath(unzipped_flair))
        unzipped_flair_target = os.path.join(pt_folder, unzipped_flair_base)
        
        gunzip_shutil(flair, unzipped_flair)
        shutil.move(unzipped_flair, unzipped_flair_target)
        
        matlab_call = f"{matlab_alias} -nodisplay -r 'sky_ps_LST_lpa {unzipped_flair_target}'"
        os.system(matlab_call) # need to use interactive version of zsh, not the older default sh
        
        unpacked_flairs.append(unzipped_flair_target)
        
        prob_mask = os.path.join(pt_folder, f'ples_lpa_m{unzipped_flair_base}')
        probability_masks.append(prob_mask)
        
        used_train_masks.append(mask)
        
        if PROTOTYPING:
            if i >= PROTOTYPING: # for speeding up prototyping
                break
        
    

    # step 2: using the model from step 1 and the training set, find the masking threshold that maximizes the average dice coefficient
    # maybe you could get results with a heuristic search but we'll just do it exhaustively (with a predetermined list of possible values)
    
    the_by = 0.05
    thresholds = np.arange(the_by, 1+the_by, the_by)
    dices = []
    
    print('Determining optimal threshold')
    for i, (p,t) in enumerate(zip(probability_masks, used_train_masks)):
        print(f'\n{i+1} of {len(probability_masks)}')
        print(f'Comparison:\n{p}\nvs.\n{t}')
        prob_mask = nib.load(p).get_fdata()
        true_mask = nib.load(t).get_fdata()
        
        y_true_f = true_mask.flatten()
        
        dices_for_this_pt = []
        for thresh in thresholds:
            print(round(thresh,2))
            bin_mask = prob_mask >= thresh
            y_pred_f = bin_mask.flatten()
            dice = metrics.f1_score(y_true_f, y_pred_f)
            dices_for_this_pt.append(dice)
        dices.append(np.array(dices_for_this_pt))
    dices = np.array(dices)
    means = dices.mean(0)
    amax = np.argmax(means)
    winning_thresh = thresholds[amax]
    print(f'\nOptimal threshold is {round(winning_thresh,2)}\n')
    
    plt.figure()
    for row in dices:
        plt.plot(thresholds, row, color='black', alpha=0.2)
    plt.plot(thresholds, means, color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Dice coefficient')
    plt.xlim(0,1)
    plt.title(f'Effect of threshold on cohort lesion mask quality\nOptimal threshold: {round(winning_thresh,2)}')
    
    figname = os.path.join(bin_folder, 'thresh_plot.png')
    plt.savefig(figname)    
    
    
    # step 3: using the model and the masking threshold from step 2, generate masks for the evaluation set

    print('Generating and thresholding evaluation masks')
    eval_folder = os.path.join(bin_folder, 'eval')
    os.mkdir(eval_folder)
    for i, (name, flair) in enumerate(zip(eval_names, eval_flairs)):
        print(f'\n{name} ({i+1} of {len(eval_names)})')

        pt_folder_eval = os.path.join(eval_folder, name)
        os.mkdir(pt_folder_eval)
        
        unzipped_flair = flair[:-3]
        unzipped_flair_base = os.path.basename(os.path.normpath(unzipped_flair))
        unzipped_flair_target = os.path.join(pt_folder_eval, unzipped_flair_base)
        
        gunzip_shutil(flair, unzipped_flair)
        shutil.move(unzipped_flair, unzipped_flair_target)
        
        matlab_call = f"{matlab_alias} -nodisplay -r 'sky_ps_LST_lpa {unzipped_flair_target}'"
        os.system(matlab_call) # need to use interactive version of zsh, not the older default sh
        
        prob_mask = os.path.join(pt_folder_eval, f'ples_lpa_m{unzipped_flair_base}')
        
        print('Thresholding mask')
        pmask = nib.load(prob_mask)
        fdata = pmask.get_fdata()
        
        bin_mask = (fdata >= winning_thresh).astype(int)
        bin_mask_name = os.path.join(write_folder, f'{name}.nii.gz')
        
        out = nib.Nifti1Image(bin_mask, pmask.affine, pmask.header)
        nib.save(out, bin_mask_name)

        if PROTOTYPING:
            if i >= PROTOTYPING: # for speeding up prototyping
                break






