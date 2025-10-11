import os
import nibabel as nib
import numpy as np

# Directories
bctv_raw_directory = r"data/RawData/Training"
bctv_filtered_directory = r"pretraining_folder/filtered_data"

# ---------- SANITY CHECKS ---------- #

def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], f"Invalid dtype: {ct.dtype}"
    assert -1000 <= ct.min(), f"CT min out of range: {ct.min()}"
    assert ct.max() <= 31743, f"CT max out of range: {ct.max()}"

    assert 0.896 <= dx <= 1.37, f"dx out of range: {dx}"
    assert dx == dy, f"dx != dy ({dx}, {dy})"
    assert 2 <= dz <= 3.7, f"dz out of range: {dz}"

    assert (x, y) == (512, 512), f"Shape mismatch: {(x, y)}"
    assert x == y
    assert 135 <= z <= 284, f"Z dimension out of range: {z}"

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape, f"GT shape {gt.shape} != CT shape {ct.shape}"
    assert gt.dtype == np.uint8, f"GT dtype should be uint8, got {gt.dtype}"

    unique_labels = set(np.unique(gt))
    assert unique_labels.issubset({0, 5, 8}), f"Unexpected labels: {unique_labels}"

    return True


# ---------- FILTER FUNCTION ---------- #

def filter_labels(label_path, save_path, keep_labels=[0, 5, 8]):
    """
    Filter a NIfTI label file to keep only certain label values.
    """
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata().astype(np.uint8)

    # Apply filtering
    filtered = np.zeros_like(label_data)
    for lbl in keep_labels:
        filtered[label_data == lbl] = lbl

    # Run sanity check for label maps only
    try:
        sanity_gt(filtered, label_data)
    except AssertionError as e:
        print(f"Sanity check failed for GT in {label_path}: {e}")

    # Save filtered result
    nib.save(nib.Nifti1Image(filtered, label_nii.affine, label_nii.header), save_path)
    print(f"Saved filtered file to: {save_path}")


# ---------- MAIN LOOP ---------- #

for path, folders, files in os.walk(bctv_raw_directory):
    for filename in files:
        if not filename.endswith(('.nii', '.nii.gz')):
            continue  # Skip non-NIfTI files

        file_path = os.path.join(path, filename)
        print(f"Reading: {file_path}")

        # Create mirrored directory structure
        relative_path = os.path.relpath(path, bctv_raw_directory)
        save_dir = os.path.join(bctv_filtered_directory, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)

        # Filter and sanity-check
        filter_labels(file_path, save_path)

