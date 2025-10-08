#!/usr/bin/env python3.13
# Why are we using python3.7 and python3.10 in other scripts?
# For it to be compatible to Snellius? 

import os
import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from multiprocessing import Pool
from shutil import copy
import warnings

import numpy as np
import scipy as sp
import nibabel as nib
import torchio as tio
from numpy import pi as π
from skimage.io import imsave
from skimage.transform import resize

from utils import tqdm_, map_

from sabotage import transform
from slice_segthor import sanity_ct, sanity_gt, slice_patient
from viewer import *

# Select nifty images from the training set
validation_set = ["01", "02", "13", "16", "21", "22", "28", "30", "35", "39"]

def ct_gt_augmentor():
        # Perform affine transformation on said CT images and Golden truth masks.
        augmented_dir = "data/segthor_train/train"

        # Rotations
        inversed_dir: Path = augmented_dir + "/Patient_27"
        channels = 5
        for K in range(channels):

                # Inverse Image
                transform(
                        "Patient_27.nii.gz", 
                        inversed_dir, 
                        inversed_dir + "Patient_27_inversed.nii.gz",
                        K,
                        affine="inv"
                )
                
                # Inverse the GT mask (label)
                transform(
                        "GT_fixed.nii.gz",
                        inversed_dir,
                        inversed_dir + "GT_inversed.nii.gz",
                        K,
                        affine="inv"
                )
        pass

        # Something else?

def augmented_slicer(id_):
        # Import images and create 2D slices
        augmented_ct = nib.load(augmented_dir + "Inverted_Patient_*.nii.gz")
        augmented_gt = nib.load(augmented_dir + "GT_inverted.nii.gz")

        assert sanity_ct(augmented_ct)
        assert sanity_gt(augmented_gt)

        augmented_dest = "data/SEGTHOR"
        slice_patient(id_, 
                augmented_dest, 
                augmented_dir,
                [256, 256]
                )
        pass

def axial_slicer():
        pass

def saggital_slicer():
        pass

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)

def slice_special_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / id_

    ct_path: Path = (id_path / f"{id_}_augmented.nii.gz") if not test_mode else (source_path / "test" / f"{id_}_augmented.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT_augmented.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct: np.ndarray = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}_augmented.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)
                print(f"Saved at {str(save_path / filename)}")

def main(args: argparse.Namespace):
        source_dir: str = args.source_dir
        train_dir: str = args.train_dir
        dest_dir: str = args.dest_dir

        # Get id list of patient in train dataset
        # ids: list[str] = sorted(map_(lambda p: p.name, (train_dir / 'train').glob('*')))
        validation_set = ["01", "02", "13", "16", "21", "22", "28", "30", "35", "39"]
        train_set = [
                f"Patient_{i:02d}"
                for i in range(1, 41)
                if f"{i:02d}" not in validation_set
        ] # I've never felt so lazy in my life

        # # Per patient, add 
        # for id in train_set:
        #         print(f"Processing data from {id}")
        #         subject = tio.Subject(
        #         image=tio.ScalarImage(f"{source_dir}/{id}/{id}.nii.gz"),
        #         label=tio.LabelMap(f"{source_dir}/{id}/GT_fixed.nii.gz")
        #         )

        #         transform = tio.RandomAffine(
        #         scales=(0.9, 1.1),
        #         degrees=10,
        #         translation=5,
        #         image_interpolation='linear'
        #         )

        #         # TODO: there are certainly better ways to integrate it into the 
        #         # system. Because the other scripts are really dependent on the file
        #         # names of these patients. We'd want to avoid overwriting files when
        #         # giving these to the training data...
        #         augmented = transform(subject)
        #         augmented.image.save(f"{dest_dir}/{id}/{id}_augmented.nii.gz")
        #         augmented.label.save(f"{dest_dir}/{id}/GT_augmented.nii.gz")
        #         print(f"Saved CT and GT images of {id} to {dest_dir}/{id}")

        for id in train_set:
              print(id)
              slice_special_patient(id, dest_dir, source_dir, [256,256])
              print(f"Processed {id}")
                      

def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='''
                Sabotage params. Will copy all nifti from source to dest dir
                (including the scans),
                and modify on the fly the identified ground truth files.''')
        parser.add_argument('--source_dir', type=Path, required=False, default="/home/scur0607/AI4MI-Project-Group-12/data/segthor_train/train")
        parser.add_argument('--train_dir', type=Path, required=False, help="Location of the train set of SEGTHOR. Used to only create augmented samples of the training data",default="/home/scur0607/AI4MI-Project-Group-12/data/SEGTHOR_CLEAN/train/img")
        parser.add_argument('--dest_dir', type=Path, required=False, default="/home/scur0607/AI4MI-Project-Group-12/data/segthor_train/train")
        args = parser.parse_args()
        
        return args

if __name__ == "__main__":
        main(get_args())