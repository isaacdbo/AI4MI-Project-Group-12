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