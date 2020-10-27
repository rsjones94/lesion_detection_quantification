import os

import numpy as np


def chunk(xs, n):
    ys = list(xs)
    np.random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra
        
        
def skullstrip(inscan, outscan, fval=0.15):
    """
    

    Parameters
    ----------
    inscan : TYPE
        DESCRIPTION.
    outscan : TYPE
        DESCRIPTION.
    fval : TYPE, optional
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    None.

    """
    stripping_command = f"bet {inscan} {outscan} -f {fval}"
    os.system(stripping_command)
    

def generate_omat(ref, inscan, outscan, omat_name):
    """
    

    Parameters
    ----------
    ref : TYPE
        DESCRIPTION.
    inscan : TYPE
        DESCRIPTION.
    outscan : TYPE
        DESCRIPTION.
    omat_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    omat_cmd = f'flirt -in {inscan} -ref {ref} -out {outscan} -omat {omat_name}'
    os.system(omat_cmd)
    

def register_scan(ref, inscan, outscan):
    """
    

    Parameters
    ----------
    ref : TYPE
        DESCRIPTION.
    inscan : TYPE
        DESCRIPTION.
    outscan : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    register_command = f"flirt -in {inscan} -ref {ref} -out {outscan}"
    os.system(register_command)
    
    
def generate_bianca_master(master_name, flairs, t1s, masks, omats):
    """
    Generates a BIANCA master text file. Rows are:
        
        flair, t1, mask, omat
    

    Parameters
    ----------
    master_name : TYPE
        DESCRIPTION.
    flairs : TYPE
        DESCRIPTION.
    t1s : TYPE
        DESCRIPTION.
    masks : TYPE
        DESCRIPTION.
    omats : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    
    message_file = open(master_name, 'w')
    
    for flair, t1, mask, omat in zip(flairs, t1s, masks, omats):
        row = f'{flair} {t1} {mask} {omat}\n'
        message_file.write(row)
        
    message_file.close()
    

def construct_bianca_cmd(master_name, subject_index, skullstrip_col, mask_col, transformation_col, out_name, run_cmd=True):
    """
    Constructs a string that can be passed to the OS to execute BIANCA
    

    Parameters
    ----------
    master_name : str
        path to the master txt file.
    subject_index : int
        1-indexed index of the row of the subject to segment.
    skullstrip_col : int
        1-indexed index of the column that contains a skullstripping mask (usually use the FLAIR).
    mask_col : int
        1-indexed index of the column that contains the brain lesion mask.
    transformation_col : int
        1-indexed index of the column that contains the matrix that transforms your data to MNI space.
    out_name : str
        name of the trained BIANCA model to write.

    Returns
    -------
    executable string.

    """
    
    bianca = f'bianca --singlefile={master_name} --matfeaturenum={transformation_col} --spatialweight=1 --querysubjectnum={subject_index} --trainingnums=all --brainmaskfeaturenum={skullstrip_col} --selectpts=surround --trainingpts=equalpoints --labelfeaturenum={mask_col} --saveclassifierdata={out_name}_classifier -o {out_name}'
    if run_cmd:
        print(f'Executing BIANCA:\n{bianca}\n')
        os.system(bianca)
    return bianca


def generate_mini_master(parent_folder, flair, t1, trans):
    # returns the name of the output master file
    # flair, t1, transformation matrix
    
    master_name = os.path.join(parent_folder, 'bianca_master.txt')
    message_file = open(master_name, 'w')
    message_file.write(f'{flair} {t1} {trans}')
    message_file.close()
    return master_name


def execute_bianca(master, model, subject_index, skullstrip_col, transformation_col, outname):
    """
    Generates a BIANCA probability map given a master file and a pretrained BIANCA model.
    

    Parameters
    ----------
    master : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    subject_index : TYPE
        DESCRIPTION.
    skullstrip_col : TYPE
        DESCRIPTION.
    transformation_col : TYPE
        DESCRIPTION.
    outname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    cmd = f'bianca --singlefile={master} --querysubjectnum={subject_index} --brainmaskfeaturenum={skullstrip_col} --matfeaturenum={transformation_col} --spatialweight=1 --loadclassifierdata={model} -o {outname}'
    print(f'BIANCA execution: {cmd}')
    os.system(cmd)
    
    
def filter_zeroed_axial_slices(nii_data, thresh=0.99):
    # removes slices if the number of pixels that are lesser than or equal to 0 exceeds a % threshold, and replaces NaN with -1
    the_data = nii_data.copy()
    wherenan = np.isnan(the_data)
    the_data[wherenan] = -1
    
    if thresh:
        keep = []
        for i in range(the_data.shape[2]):
            d = the_data[:,:,i]
            
            near_zero = np.isclose(d,0)
            less_zero = (d <= 0)
            
            bad_pixels = np.logical_or(near_zero, less_zero)
            
            perc_bad = bad_pixels.sum() / d.size
            
            if not perc_bad >= thresh:
                keep.append(True)
            else:
                keep.append(False)
        
        new = the_data[:,:,keep]
        return new, keep
    else:
        return the_data, list(np.arange(0, nii_data.shape[2]))









