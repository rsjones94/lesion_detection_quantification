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
folers = [f for f in folders if f != 'bin']

for f in folders:
    flair = os.path.join(f, '')