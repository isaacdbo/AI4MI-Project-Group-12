from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects, ball
from skimage.morphology import disk
from skimage.measure import label
import numpy as np

# this one only post processes the heart and aorta class
def postprocess_2d_slice_without_class_1_or_class_3(pred_slice):
    pred_clean = pred_slice.copy()
    
    for class_id in [2, 4]:
        mask = (pred_slice == class_id)
        
        if mask.sum() == 0:
            continue
        
        labeled = label(mask)
        if labeled.max() > 0:
            component_sizes = np.bincount(labeled.flat)[1:]
            largest = np.argmax(component_sizes) + 1
            
            pred_clean[pred_clean == class_id] = 0
            pred_clean[labeled == largest] = class_id
    
    return pred_clean

def postprocess_per_class(pred_slice):
    out = np.zeros_like(pred_slice, dtype=pred_slice.dtype)

    def smooth(mask, closing_radius=1, fill=True):
        if fill:
            mask = binary_fill_holes(mask)
        if closing_radius > 0:
            mask = binary_closing(mask, structure=disk(closing_radius))
        return mask

    m1 = (pred_slice == 1)
    out[m1] = 1

    m3 = (pred_slice == 3)
    out[m3] = 3

    m2 = (pred_slice == 2)
    if m2.any():
        m2 = smooth(m2, closing_radius=1, fill=True)
    out[m2] = 2

    m4 = (pred_slice == 4)
    if m4.any():
        m4 = smooth(m4, closing_radius=1, fill=True)
    out[m4] = 4

    return out