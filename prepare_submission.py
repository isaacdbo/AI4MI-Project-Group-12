import shutil
from pathlib import Path
import nibabel as nib

# Paths
pred_dir = Path("stitched_data")        # where preds are generated
pred_dest_dir = Path("val/pred")        # where preds should be copied
gt_src_root = Path("data/segthor_fixed/train")
gt_dest_dir = Path("val/gt")

# Make sure destinations exist
pred_dest_dir.mkdir(parents=True, exist_ok=True)
gt_dest_dir.mkdir(parents=True, exist_ok=True)

# Loop through prediction files
for pred_file in pred_dir.glob("Patient_*.nii.gz"):
    # Extract patient ID (XX)
    patient_id = pred_file.stem.split("_")[1]  # e.g. "XX"
    patient_id = patient_id.split(".")[0]

    # --- Copy prediction file ---
    dest_pred = pred_dest_dir / pred_file.name
    shutil.copy(pred_file, dest_pred)
    print(f"Copied pred {pred_file} -> {dest_pred}")

    # --- Copy GT file ---
    src_gt = gt_src_root / f"Patient_{patient_id}" / "GT.nii.gz"
    dest_gt = gt_dest_dir / f"Patient_{patient_id}.nii.gz"

    if src_gt.exists():
        shutil.copy(src_gt, dest_gt)
        print(f"Copied GT   {src_gt} -> {dest_gt}")

        # --- Shape check ---
        gt_img = nib.load(dest_gt)
        pred_img = nib.load(dest_pred)

        gt_shape = gt_img.shape
        pred_shape = pred_img.shape

        if gt_shape != pred_shape:
            print(f" Shape mismatch for Patient_{patient_id}: GT {gt_shape} vs Pred {pred_shape}")
        else:
            print(f"Shape match for Patient_{patient_id}: {gt_shape}")
    else:
        print(f" GT not found for {pred_file.name} ({src_gt})")
