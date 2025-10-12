import os
import shutil

base_dir = r"pretraining_folder/filtered_data"

img_dir = os.path.join(base_dir, "img")
label_dir = os.path.join(base_dir, "label")

# Ensure both folders exist
assert os.path.exists(img_dir), f"Missing: {img_dir}"
assert os.path.exists(label_dir), f"Missing: {label_dir}"

# Iterate through all label files (assuming one label per patient)
for label_file in os.listdir(label_dir):
    if not label_file.endswith((".nii", ".nii.gz")):
        continue

    # Extract patient number
    patient_no = label_file.replace("label", "").replace(".nii.gz", "").replace(".nii", "")

    # Find matching image file
    img_file = f"img{patient_no}.nii.gz"
    if not os.path.exists(os.path.join(img_dir, img_file)):
        img_file = f"img{patient_no}.nii"  # fallback if not gzipped

    # Create patient folder
    patient_dir = os.path.join(base_dir, patient_no)
    os.makedirs(patient_dir, exist_ok=True)

    # Move or copy both files into the patient folder
    src_label = os.path.join(label_dir, label_file)
    dst_label = os.path.join(patient_dir, label_file)
    shutil.move(src_label, dst_label)

    src_img = os.path.join(img_dir, img_file)
    dst_img = os.path.join(patient_dir, img_file)
    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
    else:
        print(f"Missing image for patient {patient_no}")

    print(f"Moved patient {patient_no} files into {patient_dir}")
