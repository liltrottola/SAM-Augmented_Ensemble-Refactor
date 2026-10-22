import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def raw_normal_generation(model, image_path, device, fov=60.0):
    '''
        Generates a raw surface normal map from an input image using a pre-trained DSINE model.
        Each pixel holds a unit vector (x, y, z) in camera coordinates.
        Parameters:
            - model: the DSINE model (from load_dsine_model).
            - image_path: path of the input image.
            - device: torch device.
            - fov: field of view in degrees, used to build the camera intrinsics
                (principal point at the image center). DSINE default: 60.
        Returns:
            - normals: float32 numpy array (H, W, 3) in [-1, 1], at the ORIGINAL image size
                (padding is removed), so it is aligned with the image and the mask.
    '''
    # DSINE modules are importable only after load_dsine_model has set sys.path
    import utils.utils as utils
    from utils.projection import intrins_from_fov

    img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # pad the input as DSINE expects
    _, _, H, W = img.shape
    lrtb = utils.get_padding(H, W)
    img = _normalize(F.pad(img, lrtb, mode="constant", value=0.0))

    intrins = intrins_from_fov(new_fov=fov, H=H, W=W, device=device).unsqueeze(0)
    intrins[:, 0, 2] += lrtb[0]
    intrins[:, 1, 2] += lrtb[2]

    pred = model(img, intrins=intrins)[-1]
    pred = pred[:, :, lrtb[2]:lrtb[2] + H, lrtb[0]:lrtb[0] + W]
    normals = pred.detach().cpu().permute(0, 2, 3, 1).numpy()[0].astype(np.float32)

    if not np.isfinite(normals).all():
        raise ValueError("Normal map contains non-finite values (NaN or Inf). Please check the input image and model.")

    return normals
