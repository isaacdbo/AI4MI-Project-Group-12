#!/usr/bin/env python3.13
# Why are we using python3.7 and python3.10 in other scripts?
# For it to be compatible to Snellius? 

import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from multiprocessing import Pool
from shutil import copy

import numpy as np
import scipy as sp
import nibabel as nib
from numpy import pi as π

from utils import tqdm_

from sabotage import transform
from slice_segthor import sanity_ct, sanity_gt, slice_patient

# Select nifty images from the traning set

# Perform affine transformation on said CT images and Golden truth masks.
# Rotations
# Inversions
# Something else?
# TODO: what channels (maybe just the poor ones)

# Save edited images in seperate training folder

# Import images and create 2D slices

# Save 2D slices in seperate training folder

# Include images into the training

def main():
        pass

def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='''
                Sabotage params. Will copy all nifti from source to dest dir
                (including the scans),
                and modify on the fly the identified ground truth files.''')
        parser.add_argument('--source_dir', type=Path, required=True)
        parser.add_argument('--dest_dir', type=Path, required=True)

if __name__ == "__main__":
        main(get_args())