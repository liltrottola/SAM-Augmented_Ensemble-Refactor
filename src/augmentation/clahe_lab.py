import skimage.color
import numpy as np
import cv2

def clahe_lab(image , mask, clip_limit=2.0):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel of the LAB color space.

    Parameters:
    - image: Input RGB image
    - mask: Not used in this function, but included for compatibility with the augmentation framework.
    - clip_limit: CLAHE clipLimit (default 2.0, the standard value used so far).

    Returns:
    - Augmented image after applying CLAHE to the L channel. 
    """

    # Convert the input image from RGB to LAB color space
    # numpy array is expected to be in the range [0, 1] for skimage.color.rgb2lab
    # 
    src_lab = skimage.color.rgb2lab(image / 255.0)  # Convert RGB to LAB color space

    src_l = src_lab[:, :, 0]  # Extract the L channel

    # l is in range [0, 100], we need to convert it to [0, 255] for CLAHE
    l_uint = ((src_l / 100) * 255).astype(np.uint8)  # Convert L channel to uint8 for CLAHE

    # Create a CLAHE object (tileGridSize=(8, 8), clipLimit from the argument, default 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_uint)

    # Convert the CLAHE result back to the LAB color space
    src_lab[:, :, 0] = (l_clahe/ 255.0) * 100  # Replace the L channel with the CLAHE result (range [0, 100])

    # Convert the augmented image back to RGB color space
    aug_rgb = skimage.color.lab2rgb(src_lab)
    aug = (np.clip(aug_rgb, 0, 1) * 255).astype(np.uint8)  # Convert back to uint8

    return aug , mask