import numpy as np

from src.augmentation.depth_prior_methods import RG_B, save_like_pipeline

'''
    Normal priors: 1-channel maps computed from the raw DSINE normals (H, W, 3), float32 in [-1, 1].
    Signature f(normals) -> (H, W) float32 in [0, 1]; the function name is also the output folder name.
    Aux methods (end of the file): f(tI, priors) -> uint8 image, used by run_normal_augmentation.py.
'''

def nrm01(a):
    """ 0-1 normalization : p1 p99 percentile normalization """
    low, high = np.percentile(a, (1, 99))
    return np.clip((a - low) / (high - low + 1e-12), 0, 1)


def _prepare(normals):
    """float32 and unit norm; null or non-finite vectors become zero vectors."""
    n = normals.astype(np.float32, copy=False)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    valid = np.isfinite(norm) & (norm > 1e-8)
    return np.divide(n, norm, out=np.zeros_like(n, dtype=np.float32), where=valid)


def angular_difference_map(normals, dy, dx):
    """Angle (rad) between N(p) and N(p + (dy, dx)); NaN where the neighbour is outside the image."""
    h, w, _ = normals.shape
    result = np.full((h, w), np.nan, dtype=np.float32)
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y0 >= y1 or x0 >= x1:
        return result
    center = normals[y0:y1, x0:x1]
    neighbor = normals[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    dot = np.clip(np.sum(center * neighbor, axis=2), -1.0, 1.0)  # clip: arccos domain
    result[y0:y1, x0:x1] = np.arccos(dot).astype(np.float32)
    return result


def get_neighbor_angles(normals, window_size):
    """Stack (H, W, N_neighbours) of the angular differences with all the neighbours in the window."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    radius = window_size // 2
    angles = [angular_difference_map(normals, dy, dx)
              for dy in range(-radius, radius + 1)
              for dx in range(-radius, radius + 1)
              if not (dy == 0 and dx == 0)]
    return np.stack(angles, axis=2)


def normal_variation_mean(normals, window_size=7):
    """Mean angular variation in the window (ignores the neighbours outside the image)."""
    return np.nanmean(get_neighbor_angles(normals, window_size), axis=2).astype(np.float32)


def normal_variation_trimmed(normals, window_size=7, lower_percentile=1, upper_percentile=95):
    """Mean angular variation without the extreme angles (below p1 and above p95 of each window)."""
    angles = get_neighbor_angles(normals, window_size)
    p_low = np.nanpercentile(angles, lower_percentile, axis=2)
    p_high = np.nanpercentile(angles, upper_percentile, axis=2)
    valid = (angles >= p_low[:, :, None]) & (angles <= p_high[:, :, None]) & np.isfinite(angles)
    return np.nanmean(np.where(valid, angles, np.nan), axis=2).astype(np.float32)


# ---------------------------------------------------------------- priors (called by name from the yaml)

def normal_nz(normals):
    """Nz component: (Nz + 1) / 2. Alignment of the surface with the optical axis (no per-image normalization)."""
    nz = _prepare(normals)[:, :, 2]
    return np.clip((nz + 1.0) / 2.0, 0, 1).astype(np.float32)


def normal_mean7(normals):
    """Boundary prior: mean angular variation 7x7, p1-p99 per image."""
    return nrm01(normal_variation_mean(_prepare(normals), 7)).astype(np.float32)


def normal_trimmed7(normals):
    """Boundary prior: trimmed mean angular variation 7x7, p1-p99 per image."""
    return nrm01(normal_variation_trimmed(_prepare(normals), 7, 1, 95)).astype(np.float32)


# ---------------------------------------------------------------- aux methods (called by name from the yaml)

def RG_normal_nz(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["normal_nz"]))


def RG_normal_mean7(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["normal_mean7"]))
