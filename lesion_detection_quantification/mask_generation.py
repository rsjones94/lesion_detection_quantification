import os
import shutil
from glob import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import helpers as hp

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
    print('Preprocessing for BIANCA (skullstripping and registration)')
    for i, (name, flair, t1, mask) in enumerate(zip(train_names, train_flairs, train_t1s, train_masks)):
        print(f'{name} ({i+1} of {len(train_names)})')
        pt_folder = os.path.join(bin_folder, name)
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
        
        #if i >= 2: # for speeding up prototyping
            #break
        
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
    
    thresholds = np.arange(0.02, 1.02, 0.02)
    dices = []
    
    print('Determining optimal threshold')
    for i, (p,t) in enumerate(zip(prob_mask_names, masks)):
        print(f'\n{i+1} of {len(prob_mask_names)}')
        print(f'Comparison:\n{p}\nvs.\n{t}')
        prob_mask = nib.load(p).get_fdata()
        true_mask = nib.load(t).get_fdata()
        dices_for_this_pt = []
        for thresh in thresholds:
            print(thresh)
            bin_mask = prob_mask >= thresh
            dice = hp.dice_coef(bin_mask, true_mask)
            dices_for_this_pt.append(dice)
        dices.append(np.array(dices_for_this_pt))
    dices = np.array(dices)
    means = dices.mean(0)
    amax = np.argmax(means)
    winning_thresh = thresholds[amax]
    print(f'Optimal threshold is {winning_thresh}')
    
    plt.figure()
    for row in dices:
        plt.plot(thresholds, row, color='black', alpha=0.2)
    plt.plot(thresholds, means, color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Dice coefficient')
    plt.xlim(0,1)
    plt.title(f'Effect of threshold on cohort lesion mask quality\nOptimal threshold: {winning_thresh}')
    
    figname = os.path.join(bin_folder, 'thresh_plot.png')
    plt.savefig(figname)    
        
    # step 3: using the model and the masking threshold from step 2, generate masks for the evaluation set


def generate_knntpp_masks():
    pass


def generate_default_lga_masks():
    pass


def generate_default_lpa_masks():
    pass
    
    
def generate_custom_lga_masks():
    pass


def generate_custom_lpa_masks():
    pass
    