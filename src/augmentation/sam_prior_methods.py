import numpy as np
import skimage
from sklearn.decomposition import PCA

# Same augmentation methods as sam_methods.py (same names = same output folders and sweep methods),
# built from the SAM priors saved by run_sam_generation.py instead of running SAM inside every method.
# Signature f(tI, priors) -> final uint8 image:
# tI = RGB uint8 at the original size; priors = {prior name: map} with the maps of run_sam_generation.py
# (keys = folder names: "segPrior", "boundaryPrior", "areaPrior", "logits"), at the image size.


def save_like_pipeline(aug_img):
    """global min-max over the 3 channels, then uint8."""
    aug_img_scaled = skimage.exposure.rescale_intensity(aug_img, in_range='image', out_range=(0, 1))
    return skimage.img_as_ubyte(aug_img_scaled)


def SAMAug(tI, priors):
    """G += segPrior, B += boundaryPrior."""
    tI = skimage.img_as_float(tI).copy()
    tI[:, :, 1] = tI[:, :, 1] + priors["segPrior"]
    tI[:, :, 2] = tI[:, :, 2] + priors["boundaryPrior"]
    return save_like_pipeline(tI)


def ourSAMAug(tI, priors):
    """As SAMAug, plus R += areaPrior (relative area of the masks)."""
    tI = skimage.img_as_float(tI).copy()
    tI[:, :, 0] = tI[:, :, 0] + priors["areaPrior"]
    tI[:, :, 1] = tI[:, :, 1] + priors["segPrior"]
    tI[:, :, 2] = tI[:, :, 2] + priors["boundaryPrior"]
    return save_like_pipeline(tI)


def RG_segPrior(tI, priors):
    """Original R, G, B = segPrior."""
    tI = skimage.img_as_float(tI)
    return save_like_pipeline(np.dstack((tI[:, :, 0], tI[:, :, 1], priors["segPrior"])))


def SV_segPrior(tI, priors):
    """Same output as SV_segPrior of sam_methods.py: (G, B, segPrior) of the RGB image.
    NOTE: the original computes the HSV image but then takes the channels from the RGB one,
    so despite the name S and V are NOT used; kept as it is to match the trained models."""
    tI = skimage.img_as_float(tI)
    return save_like_pipeline(np.dstack((tI[:, :, 1], tI[:, :, 2], priors["segPrior"])))


def RG_logits(tI, priors):
    """Original R, G (0-255), B = min-max of the SAM logits * 255."""
    logits = priors["logits"]
    b = (logits - logits.min()) * 255 / (logits.max() - logits.min())
    return save_like_pipeline(np.dstack((tI[:, :, 0], tI[:, :, 1], b)))


def PCA_segPrior(tI, priors):
    """PCA 3->2 of the RGB (fitted on each image), B = segPrior."""
    tI = skimage.img_as_float(tI)
    h, w, c = tI.shape
    flat_image = tI.reshape(-1, c)
    pca = PCA(n_components=2)
    flat_image_pca = pca.fit_transform(flat_image)
    image_pca = flat_image_pca.reshape(h, w, 2)
    image_pca_scaled = (image_pca - np.min(image_pca)) / (np.max(image_pca) - np.min(image_pca)) * 255
    image_pca_scaled = image_pca_scaled.astype(np.uint8)
    seg = priors["segPrior"]
    r = image_pca_scaled[:, :, 0]
    g = image_pca_scaled[:, :, 1]
    b = (255 * (seg - seg.min()) / (seg.max() - seg.min())).astype(np.uint8)
    return save_like_pipeline(np.dstack((r, g, b)))
