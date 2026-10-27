import numpy as np
import cv2

from src.augmentation.depth_prior_methods import depth_p1p99
from src.augmentation.normal_prior_methods import _prepare
from src.augmentation.clahe_lab import clahe_lab

'''
    Light DA: photometric variants of a training image, the mask is not changed.
    Pure functions (no I/O): tI is the RGB image (H, W, 3) uint8, the output is (H, W, 3) uint8.
    No min-max normalization of the output: only the final clip to [0, 255].
'''


def resize_normals(normals, h, w):
    '''
        DSINE normals (H', W', 3) float32 in [-1, 1] -> (h, w, 3).
        Bilinear resize of each component, then the vectors are re-normalized to unit length
        (the interpolation shortens them). If the size is already (h, w) the normals are returned unchanged.
    '''
    if normals.shape[:2] == (h, w):
        return normals
    normals = cv2.resize(normals.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return _prepare(normals)


def relight(tI, normals, light):
    '''
        Lambertian relighting with the DSINE normals :
            l = light / |light|
            shading = 0.4 + 0.8 * clip(n . l, 0, 1)
            out = clip(tI / 255 * shading, 0, 1) -> uint8
        Parameters:
            - tI: RGB image (H, W, 3) uint8.
            - normals: DSINE normals (H', W', 3) float32, resized to (H, W) if needed.
            - light: light direction (lx, ly, lz), normalized here.
    '''
    h, w = tI.shape[:2]
    normals = resize_normals(normals, h, w)

    # Normalize light direction
    lx, ly, lz = light
    l_len = np.sqrt(lx**2 + ly**2 + lz**2)
    lx_n, ly_n, lz_n = lx / l_len, ly / l_len, lz / l_len

    # Lambertian shading
    shading = (normals[:, :, 0] * lx_n) + (normals[:, :, 1] * ly_n) + (normals[:, :, 2] * lz_n)
    shading = np.clip(shading, 0.0, 1.0)
    shading = 0.4 + (shading * 0.8)

    # Apply the same shading to the 3 channels
    img = tI.astype(np.float32) / 255.0
    aug_img = img * shading[:, :, None]
    return np.clip(aug_img * 255.0, 0, 255).astype(np.uint8)


def attenuate(tI, depth, floor):
    '''
        Depth-guided attenuation: the far regions get darker, the near ones are unchanged.
            DP = depth_p1p99(depth)            computed at the DEPTH resolution (near = 1, far = 0)
            DP -> bilinear resize to (H, W)
            A = floor + (1 - floor) * DP       A in [floor, 1]
            out = clip(tI * A, 0, 255) -> uint8
        Parameters:
            - tI: RGB image (H, W, 3) uint8.
            - depth: raw DA3 depth (H', W') float.
            - floor: minimum brightness factor kept in the farthest points (0 = RGB * DP, 1 = unchanged).
    '''
    h, w = tI.shape[:2]
    DP = depth_p1p99(depth).astype(np.float32)
    DP = cv2.resize(DP, (w, h), interpolation=cv2.INTER_LINEAR)

    A = floor + (1 - floor) * DP
    aug_img = tI.astype(np.float32) * A[:, :, None]
    return np.clip(aug_img, 0, 255).astype(np.uint8)


def clahe(tI, clip_limit):
    '''
        CLAHE on the L channel of LAB (clahe_lab.py), tile 8x8, with the given clip_limit.
    '''
    aug_img, _ = clahe_lab(tI, None, clip_limit)
    return aug_img
