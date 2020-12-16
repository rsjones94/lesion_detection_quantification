#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:39:33 2020

@author: skyjones
"""

import os
import sys
import getopt
import glob
import shutil
from datetime import datetime
from contextlib import contextmanager
from time import time

import pandas as pd

infolder = '/Users/skyjones/Documents/lesion_training_data_export/'


##########

globber = os.path.join(infolder, '*/') # all subfolders
folders = glob.glob(globber)
folders = [f for f in folders if 'bin' not in f]

for i,f in enumerate(folders):
    
    print(f'\nOn {f} ({i+1} of {len(folders)})')
    
    flair = os.path.join(f, 'axFLAIR.nii.gz')
    t1 = os.path.join(f, 'axT1.nii.gz')
    
    reg_t1 = os.path.join(f, 'axT1_flairspace.nii.gz')
    
    register_command = f"flirt -in {t1} -ref {flair} -out {reg_t1}"
    os.system(register_command)