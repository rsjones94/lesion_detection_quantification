import os
from glob import glob
import shutil

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from sklearn import neighbors
import matplotlib.cm as cm
from scipy import ndimage

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
evaluate_masks = False
generate_figures = True

n_repeats = 3 # not implemented. number of times to randomly reassort the samples
n_cohorts = 4 # from the data, n unique evaluation cohorts per repetition will be created


##############


training_master = os.path.join(data_folder, 'lesion_training_data')
training_subs = glob(os.path.join(training_master, '*/'))
pt_names = [os.path.basename(os.path.normpath(i)) for i in training_subs]
pt_names.remove('bin')
pt_names = np.array(pt_names)

pt_goldens = np.array([os.path.join(training_master, i, 'axFLAIR_mask.nii.gz') for i in pt_names])
pt_t1s = np.array([os.path.join(training_master, i, 'axT1.nii.gz') for i in pt_names])
pt_flairs = np.array([os.path.join(training_master, i, 'axFLAIR.nii.gz') for i in pt_names])

evaluation_folder = os.path.join(data_folder, 'quality_quantification')

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
        
        lga_write_folder = os.path.join(cohort_folder, 'lga')
        lga_bin_folder = os.path.join(cohort_bin_folder, 'lga')
        print('\n\n\nGenerating standard LPA masks')
        mg.generate_lga_masks(training_data, evaluation_data, lga_write_folder, lga_bin_folder) # not implemented
        
        standard_lpa_write_folder = os.path.join(cohort_folder, 'lpa_standard')
        standard_lpa_bin_folder = os.path.join(cohort_bin_folder, 'lpa_standard')
        print('\n\n\nGenerating standard LPA masks')
        mg.generate_lpa_masks(training_data, evaluation_data, standard_lpa_write_folder, standard_lpa_bin_folder, model_type='default')
          
        bianca_write_folder = os.path.join(cohort_folder, 'bianca')
        bianca_bin_folder = os.path.join(cohort_bin_folder, 'bianca')
        print('\n\n\nGenerating BIANCA masks')
        mg.generate_bianca_masks(training_data, evaluation_data, bianca_write_folder, bianca_bin_folder, mni_reference_scan)
        
        # mg.generate_knntpp_masks() # not implemented

        
    
    
if evaluate_masks:
    

    
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
                
                golden_nii = nib.load(golden_standard)
                golden_data = golden_nii.get_fdata()
                
                
                
                # get lesion volume data
                voxel_dims = golden_nii.header['pixdim'][1:4]
                voxel_vol = np.product(voxel_dims)
                labeled_lesions = measure.label(golden_data)
                try:
                    vols = measure.regionprops_table(labeled_lesions, properties=['area'])['area'] * voxel_vol
                    # yes the property is area, but it's 3d so it's actually volume. units are mm3
                    mean_vol = vols.mean()
                    median_vol = np.median(vols)
                    total_vol = vols.sum()
                except IndexError:
                    mean_vol = median_vol = total_vol = 0
                    
                
                stats_dict = mq.quantify_mask_quality(golden_data, mask_data)
                stats_dict['mean_vol'] = mean_vol
                stats_dict['median_vol'] = median_vol
                stats_dict['total_vol'] = total_vol
                

                
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
                metrics_reference[protocol_name] = [stats_df]
                
    average_folder = os.path.join(evaluation_folder, 'average')
    if os.path.exists(average_folder):
        shutil.rmtree(average_folder)
    os.mkdir(average_folder)
    for key, li in metrics_reference.items():
        protocol_df = pd.concat(li)
        mean_df = protocol_df.groupby('pt', as_index=False).mean()
        
        csv_name = os.path.join(average_folder, f'{key}.csv')
        mean_df.to_csv(csv_name)
        
if generate_figures:
    average_folder = os.path.join(evaluation_folder, 'average')
    figure_folder = os.path.join(data_folder, 'figures')
    result_csvs = os.listdir(average_folder)
    
    meanified_csv_name = os.path.join(figure_folder, 'master_comp.csv')
    meanified_df = pd.DataFrame()
    
    master_df = pd.DataFrame()
    
    for csv in result_csvs:
        the_csv = os.path.join(average_folder, csv)
        the_name = csv[:-4]
        result_df = pd.read_csv(the_csv, index_col='pt')
        result_df = result_df.drop(result_df.columns[0], axis=1)
        result_df['method'] = the_name
        
        master_df = master_df.append(result_df)
        
        meanified = result_df.mean()
        meanified['method'] = the_name
        meanified_df = meanified_df.append(meanified, ignore_index=True)
        
    meanified_df = meanified_df.set_index('method')
    meanified_df.to_csv(meanified_csv_name)
    
    
    methods = list(master_df['method'].unique())

    
    the_cols = list(master_df.columns.drop(['method', 'mean_vol', 'median_vol', 'total_vol']))
    
    n_subs  = len(the_cols)
    edges = [0,0]
    
    flipper = 1
    while edges[0] * edges[1] < n_subs:
        edges[flipper] += 1
        flipper ^= 1
    edge_x, edge_y = edges
    fig, axs = plt.subplots(nrows=edge_x, ncols=edge_y, figsize=(14,20))
    
    for i, ax in enumerate(fig.axes):
        try:
            c = the_cols[i]
        except IndexError:
            print('out of index!!')
            ax.set_visible(False)
            continue
        print(f'Boxplot for {c}')
        #master_df.boxplot(c, 'method')
        method_data = [master_df[master_df['method']==met][c].dropna() for met in methods]
        
        ax.boxplot(method_data)
        ax.set_xticklabels(methods)
        ax.set_title(f'{c}')
        ax.set_ylim(-.1,1.1)
    plt.tight_layout()
    
    master_fig_name = os.path.join(figure_folder, 'boxes.png')
    plt.savefig(master_fig_name)
    #plt.close('all')
    
    # plotting against volume
    for vol_type in ['median_vol', 'mean_vol', 'total_vol']:
        plt.figure()
        #plt.xscale('log')
        plot_name = os.path.join(figure_folder, f'methods_against_{vol_type}.png')
        for m in methods:
            
            sub_df = master_df[master_df['method'] == m]
            exes = sub_df[vol_type]
            whys = sub_df['f1_score']
            
            plt.scatter(exes, whys, label=m)
            
        plt.title(f'Segmentation quality as a function of {vol_type}')
        plt.xlabel('Volume (cubic mm)')
        plt.ylabel('F1-score')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_name)
        
    
    # now generate our comparison maps
    plt.style.use('dark_background')
    
    cohort_folders = glob(os.path.join(data_folder, 'generated_masks', '*/'))
    cohorts = [os.path.basename((os.path.normpath(i))) for i in cohort_folders]
    
    for cohort_name, cohort_folder in zip(cohorts, cohort_folders):
        figure_cohort_folder = os.path.join(figure_folder, cohort_name)
        if os.path.exists(figure_cohort_folder):
            shutil.rmtree(figure_cohort_folder)
        os.mkdir(figure_cohort_folder)

        protocol_folders = glob(os.path.join(cohort_folder, '*/'))
        protocol_names = [os.path.basename((os.path.normpath(i))) for i in protocol_folders]
        
        masks = glob(os.path.join(protocol_folders[0], '*.nii.gz'))
        masks = [os.path.basename(os.path.normpath(m)) for m in masks]
        
        
        n_slices_to_show = 9
        middy = (n_slices_to_show-1)/2
        for mask in masks:
            pt_name = os.path.basename(os.path.normpath(mask)).split('.')[0]
            print(f'Generating comparison map for {pt_name}')
            
            flair_name = golden_standard = os.path.join(data_folder, 'lesion_training_data', pt_name, 'axFLAIR.nii.gz')
            
            golden_standard = os.path.join(data_folder, 'lesion_training_data', pt_name, 'axFLAIR_mask.nii.gz')
            method_segmentations = [os.path.join(p, f'{pt_name}.nii.gz') for p in protocol_folders]
            
            maps = [None, golden_standard]
            map_names = ['FLAIR', 'golden_standard']
            
            maps.extend(method_segmentations)
            map_names.extend(protocol_names)
            
            comp_map_name = os.path.join(figure_cohort_folder, f'{pt_name}_comp_map.png')
            n_methods = len(protocol_names)
            
            flair_data = nib.load(flair_name).get_fdata()
            filt_flair, keepers = hp.filter_zeroed_axial_slices(flair_data)
            
            n_slices_to_remove = 8 # n slices to remove from top and bottom of viewbox (to reduce number of cerebellum and super high slices)
            idx = np.round(np.linspace(0 + n_slices_to_remove, filt_flair.shape[2] - 1 - n_slices_to_remove, n_slices_to_show)).astype(int)
            
            fig, axs = plt.subplots(nrows=n_methods+2, ncols=n_slices_to_show, figsize=(n_slices_to_show*2,n_slices_to_show*10))
            ind = 0
            for i, (mapped, map_name, axrow) in enumerate(zip(maps, map_names, axs)):
                print(f'({map_name})')
                if mapped is not None:
                    mapped_data = nib.load(mapped).get_fdata()
                    filt_mapped = mapped_data[:,:,keepers]
                for j, (ind, ax) in enumerate(zip(idx, axrow)):
                    my_cmap = cm.bwr
                    my_cmap.set_under('k', alpha=0)
                    
                    flair_slice = np.fliplr(ndimage.rotate(filt_flair[:,:,ind].T, 180))
                    
                    ax.imshow(flair_slice, cmap='gray')
                    
                    if mapped is not None:
                        map_slice = np.fliplr(ndimage.rotate(filt_mapped[:,:,ind].T, 180)) > 0.1
                        ax.imshow(map_slice, cmap=my_cmap, clim=[0.01, 1], interpolation='none')
                    ax.axis('off')
                    if j == middy:
                        ax.set_title(map_name)
                    
                    #i += 1
            
            plt.tight_layout()
            plt.savefig(comp_map_name)
            
                
        
    
    