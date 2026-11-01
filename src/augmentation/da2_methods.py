import numpy as np
import skimage
import scipy.ndimage
import cv2
import skimage.exposure
import skimage.color
import torchstain

def width_shift(image, mask, siz=352):
    '''
    Random horizontal translation.
    '''
    # --- Random sampling (matches MATLAB rand) ---
    # a*interval may be non-integer (e.g. 35.2 with siz=352) → MATLAB uses
    # bilinear interpolation for sub-pixel shifts. We match with order=1.
    interval = siz * 0.1
    a = np.random.randint(1, 10)               # {1,...,9} like MATLAB 1+floor(rand*9)
    direction = 1 if np.random.rand() < 0.5 else -1
    dx = direction * interval * a              # float, may be non-integer

    # --- Shift on X axis ---
    # For RGB (H, W, 3): shift = [dy, dx, dchannel] = [0, dx, 0]
    # order=1 (bilinear) on the image → matches MATLAB imtranslate for float shifts
    # mode='constant', cval=0 → empty borders = black (MATLAB FillValues default)
    image_shifted = scipy.ndimage.shift(
        image, [0, dx, 0],
        order=1, mode='constant', cval=0,
    )

    # For the mask: order=0 (nearest neighbor) to preserve binary values.
    # With order=1 bilinear, intermediate values would appear along mask edges,
    # which are neither foreground nor background → would corrupt the ground truth.
    mask_shifted = scipy.ndimage.shift(
        mask, [0, dx],
        order=0, mode='constant', cval=0,
    )

    return image_shifted, mask_shifted

def height_shift(image, mask, siz=352):
    '''
    Random vertical translation.
    '''
    # --- Random sampling (matches MATLAB rand) ---
    # Same pattern as width_shift: a*interval may be non-integer → order=1 on image.
    interval = siz * 0.1
    a = np.random.randint(1, 10)               # {1,...,9} like MATLAB 1+floor(rand*9)
    direction = 1 if np.random.rand() < 0.5 else -1
    dy = direction * interval * a              # float, may be non-integer

    # --- Shift on Y axis ---
    # For RGB (H, W, 3): shift = [dy, dx, dchannel] = [dy, 0, 0]
    # order=1 on image, mode/cval like MATLAB FillValues default
    image_shifted = scipy.ndimage.shift(
        image, [dy, 0, 0],
        order=1, mode='constant', cval=0,
    )

    # For the mask: order=0 (nearest) to preserve binary values
    mask_shifted = scipy.ndimage.shift(
        mask, [dy, 0],
        order=0, mode='constant', cval=0,
    )

    return image_shifted, mask_shifted

def rotation(image, mask):
    '''
    Random rotation of {45°, 90°, 135°, 180°}, random direction.
    '''
    # --- Random sampling (matches MATLAB) ---
    # MATLAB: c=1+floor(rand*4) → c in {1,2,3,4}; ang*c → {45,90,135,180}
    ang = 45
    c = np.random.randint(1, 5)                # {1,...,4}
    direction = 1 if np.random.rand() < 0.5 else -1
    angle = direction * ang * c         # ±45°, ±90°, ±135°, ±180°

    # --- Rotation ---
    # skimage.transform.rotate: order=1 (bilinear) = MATLAB imwarp default.
    # resize=False replicates MATLAB affineOutputView(size(im),tform):
    #   output has the SAME size as input (cropped at borders).
    # preserve_range=True keeps dtype uint8 [0,255] (else skimage normalises to float [0,1]).
    image_rotated = skimage.transform.rotate(
        image, angle,
        order=1, mode='constant', cval=0,
        resize=False, preserve_range=True,
    ).astype(image.dtype)

    # For the mask: order=0 (nearest) to preserve binary values
    mask_rotated = skimage.transform.rotate(
        mask, angle,
        order=0, mode='constant', cval=0,
        resize=False, preserve_range=True,
    ).astype(mask.dtype)

    return image_rotated, mask_rotated

def shear(image, mask):
    '''
    Shear random: XShear o YShear, ±45° (4 opzioni equiprobabili).
    '''
    # --- Random sampling ---
    # MATLAB: c=1+floor(rand)=1 always (likely bug), ang=45 → fixed ±45° shear
    # 4 equiprobable options (d ∈ [0,1) split in 4 quartiles)
    ang = 45
    d = np.random.rand()
    if d < 0.25:
        axis, sign = 'X', +1
    elif d < 0.5:
        axis, sign = 'X', -1
    elif d < 0.75:
        axis, sign = 'Y', +1
    else:
        axis, sign = 'Y', -1

    shear_rad = np.deg2rad(ang * sign)
    tan_s = np.tan(shear_rad)

    # --- Matrix construction (matches MATLAB randomAffine2d) ---
    # skimage AffineTransform(shear=θ) uses opposite sign convention vs MATLAB
    # (matrix[0,1] = -tan(θ) instead of +tan(θ)).
    # Explicit matrix to match MATLAB exactly.
    if axis == 'X':
        # MATLAB XShear: xval' = xval + tan(θ)·y, y' = y
        matrix = np.array([
            [1, tan_s, 0],
            [0, 1,     0],
            [0, 0,     1],
        ])
    else:  # 'Y'
        # MATLAB YShear: xval' = xval, y' = tan(θ)·x + y
        matrix = np.array([
            [1,     0, 0],
            [tan_s, 1, 0],
            [0,     0, 1],
        ])
    tform = skimage.transform.AffineTransform(matrix=matrix)

    # --- Apply warp ---
    # skimage.warp applies the INVERSE transform.
    # We pass tform.inverse to match MATLAB imwarp.
    image_sheared = skimage.transform.warp(
        image, tform.inverse,
        order=1, mode='constant', cval=0,
        preserve_range=True,
    ).astype(image.dtype)

    mask_sheared = skimage.transform.warp(
        mask, tform.inverse,
        order=0, mode='constant', cval=0,
        preserve_range=True,
    ).astype(mask.dtype)

    return image_sheared, mask_sheared

def random_flip(image, mask):
    '''
        Random flip (50% vertical, 50% horizontal).
    '''
    # MATLAB uses flip() = vertical (axis 0) and fliplr() = horizontal (axis 1).
    if np.random.rand() <0.5:
        return np.flipud(image) , np.flipud(mask) #vertical flip
    else:
        return np.fliplr(image) , np.fliplr(mask) #horizontal flip

def brightness_uniform(image, mask):
    '''
    Uniform brightness: ±[25,50) added to all RGB channels.
    '''

    # MATLAB: l = floor((1+rand)*25) → {25,...,49}; random sign
    l = int(np.floor((1 + np.random.rand()) * 25))   # {25,...,49}
    sign = 1 if np.random.rand() < 0.5 else -1

    delta = sign*l

    # uint8 + delta in NumPy wraps around (250+25=19!), MATLAB saturates (250+25=255).
    # Cast to int16, sum, clip to [0,255], back to uint8

    image_bright = np.clip(image.astype(np.int16) + delta , 0, 255).astype(np.uint8)

    return image_bright, mask

def brightness_per_channel(image, mask):
    u = int(np.floor((1 + np.random.rand()) * 25))
    v = int(np.floor((1 + np.random.rand()) * 25))
    z = int(np.floor((1 + np.random.rand()) * 25))

    sign = 1 if np.random.rand() < 0.5 else -1

    result = image.astype(np.int16)          # room for overflow
    result[:, :, 0] = result[:, :, 0] + sign * u   # R channel
    result[:, :, 1] = result[:, :, 1] + sign * v   # G channel
    result[:, :, 2] = result[:, :, 2] + sign * z   # B channel
    image_bright = np.clip(result, 0, 255).astype(np.uint8)

    return image_bright, mask

def speckle_noise(image, mask):
    '''
        Speckle noise (multiplicative), variance 0.05 (MATLAB imnoise default).
    '''
    # MATLAB imnoise(im,'speckle'): J = I + n*I, n ~ Gaussian mean 0, var 0.05.
    # skimage random_noise does the same but defaults to var=0.01.
    # random_noise returns float in [0,1] → we convert back to uint8.
    noisy = skimage.util.random_noise(image, mode='speckle', var=0.05)
    image_noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)

    # Mask unchanged
    return image_noisy, mask

def _motion_kernel(length: float, angle: float) -> np.ndarray:
    """Motion-blur PSF of a camera translating uniformly along a straight segment.

    Physical model: during the exposure the camera moves at constant speed along a
    segment of ``length`` pixels, centred on the origin, at ``angle`` degrees
    (counter-clockwise from the horizontal). The kernel value of each pixel is the
    fraction of the trajectory that falls on it. The trajectory is sampled with
    20000 points (midpoints of equal sub-segments) and each point is splatted
    bilinearly on its 4 neighbouring pixels; the kernel is then normalised to sum 1.

    Parameters
    ----------
    length : float
        Motion length in pixels (must be > 0).
    angle : float
        Motion angle in degrees, counter-clockwise from the horizontal.

    Returns
    -------
    np.ndarray
        2-D float64 kernel summing to 1, odd-sized, with the motion segment centred on the central pixel.
    """
    # Direction of motion; rounded so that cos(90 deg) is exactly 0 (not 6e-17),
    # otherwise a vertical motion would leave a near-zero extra column
    cosphi = np.round(np.cos(np.deg2rad(angle)), 12)
    sinphi = np.round(np.sin(np.deg2rad(angle)), 12)

    # Number of samples along the trajectory. The sampled kernel differs from the
    # exact one (continuous trajectory)
    n_points = 20000

    # Half trajectory: midpoints of n_points/2 sub-segments of [0, length/2];
    # the other half is its mirror, so the kernel is exactly point-symmetric
    n_half = n_points // 2
    t = (np.arange(n_half) + 0.5) / n_half * (length / 2.0)
    t = np.concatenate([t, -t])

    # Point coordinates: x to the right, y downwards (image rows) -> minus sign on
    # sin to have the angle counter-clockwise on screen
    x = t * cosphi
    y = -t * sinphi

    # Kernel grid (2r+1) x (2r+1) centred on pixel (r, r), large enough for the segment
    r = int(np.ceil(length / 2.0)) + 1
    h = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float64)

    # Bilinear splat: each point gives weight to its 4 neighbouring pixels,
    # proportional to the overlap (1 - distance) along x and along y
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    col = x0.astype(int) + r
    row = y0.astype(int) + r
    np.add.at(h, (row,     col),     (1 - fx) * (1 - fy))
    np.add.at(h, (row,     col + 1), fx * (1 - fy))
    np.add.at(h, (row + 1, col),     (1 - fx) * fy)
    np.add.at(h, (row + 1, col + 1), fx * fy)

    # Crop the all-zero rows/columns, symmetrically around the centre so the kernel
    # stays odd-sized and centred (filter2D anchors it on the central pixel):
    # a horizontal/vertical motion collapses to a 1-D vector
    rows = np.nonzero(h.sum(axis=1))[0]
    cols = np.nonzero(h.sum(axis=0))[0]
    ry = max(r - rows.min(), rows.max() - r)
    rx = max(r - cols.min(), cols.max() - r)
    h = h[r - ry:r + ry + 1, r - rx:r + rx + 1]

    # Normalise to sum 1
    h = h / h.sum()

    return h

def contrast_blur(image, mask, func=1, direction=1, harsh=1):
    xval = np.linspace(0, 1, num=256)
    if func == 1:
        # Rational function
        if direction == 1:
            if harsh == 1:
                a, b = -5, -3
            else:
                a, b = -2, -1
        else:
            if harsh == 1:
                a, b = 2.8, 3.8
            else:
                a, b = 1.5, 2.5
        
        #k controls the slope of the curve, when k>0 the contrast decreases, when k<0 the contrast increases
        k= a + (b-a)*np.random.rand()
        yval = ((xval-1/2)* np.sqrt(1-k/4)) / np.sqrt(1- ((xval-1/2) ** 2) * k) + 0.5
    
    elif func == 2:
        if direction == 1:
            if harsh == 1:
                a = 1.8
                b = 2.3
            else:
                a = 1.2
                b = 1.7
            
        else:
            if harsh == 1:
                a = 0.25
                b = 0.5
            else:
                a = 0.6
                b = 0.9
    
        #q controls the slope of the curve, when q<1 the contrast decreases, when q>1 the contrast increases
        q = a + (b - a)* np.random.rand()

        yval = np.where(
            xval < 0.5,
            0.5 * (xval / 0.5)**q,
            1 - 0.5 * ((1 - xval) / 0.5)**q
        )
    else:
        raise ValueError(f"Invalid func value: {func}. Expected 1 or 2.")
    
    yval = yval*255

    # applying contrast
    out_ch1 = yval[image[:, :, 0]].astype(np.uint8)
    out_ch2 = yval[image[:, :, 1]].astype(np.uint8)
    out_ch3 = yval[image[:, :, 2]].astype(np.uint8)

    mask_after_lut = yval[mask.astype(np.uint8)]  # shape (H, W), dtype float

    # motion blur
    length = 3 + np.random.rand() * 4
    angle  = np.random.rand() * 360
    h = _motion_kernel(length, angle)

    out_ch1 = cv2.filter2D(out_ch1, -1, h, borderType=cv2.BORDER_REPLICATE)
    out_ch2 = cv2.filter2D(out_ch2, -1, h, borderType=cv2.BORDER_REPLICATE)
    out_ch3 = cv2.filter2D(out_ch3, -1, h, borderType=cv2.BORDER_REPLICATE)
    
    # Mask: MATLAB modifies it — on hold pending prof feedback. Kept unchanged for now.
    mask_blurred = cv2.filter2D(mask_after_lut, -1, h, borderType=cv2.BORDER_REPLICATE)
    mask_out = np.clip(np.round(mask_blurred), 0, 255).astype(np.uint8)

    # append modified image
    image_out = np.stack([out_ch1, out_ch2, out_ch3], axis=-1)  # shape (H, W, 3) uint8

    # For now: contrast/blur applied ONLY to image, mask unchanged.
    return image_out, mask

def shadows(image, mask):
    '''
    Gradual shadow on one side (left or right): each column × its corresponding
    yval ∈ [0.2, 1].
    '''

    H, W = image.shape[:2]                     # rows and columns
    xval = np.linspace(0, 1, W)               # MATLAB linspace(0,1,siz)
    direction = np.random.randint(0, 2)        # MATLAB randi(2)-1 → {0,1}
    if direction:
        yval = 0.2 + np.sqrt(xval / 0.5) * 0.8         # shadow on the left
    else:
        yval = 0.2 + np.sqrt((-xval + 1) / 0.5) * 0.8  # shadow on the right
    yval[yval > 1] = 1.0                        # cap at 1 (MATLAB yval(yval>1)=1)

    # multiply each column i by yval[i], equivalent to the MATLAB loop
    out = image.astype(np.float64) * yval.reshape(1, W, 1)


    # round before uint8 that usually truncates values
    image_shadowed = np.clip(np.round(out), 0, 255).astype(np.uint8)

    # Mask: MATLAB modifies it — on hold pending prof feedback. Kept unchanged.
    return image_shadowed, mask

def rgb_histogram_match(image, mask, target):
    '''
    Histogram specification: match the histogram of `image` to that of `target`,
    per channel.
    '''

    # skimage match_histograms does per-channel histogram matching (channel_axis=-1)
    # = MATLAB histeq(SourceCh, HnTargetCh) on the 3 channels
    matched = skimage.exposure.match_histograms(image, target, channel_axis=-1)  # channel_axis=-1: R, G, B independent
    matched = np.clip(np.round(matched), 0, 255).astype(np.uint8)

    return matched, mask

def reinhard_normalize(image, mask, target):
    '''
    Reinhard color transfer: normalise mean/std of `image` LAB channels
    to match `target`. Equivalent to NormReinhard.m.
    '''

    # RGB → LAB (im2double: /255 to bring to [0,1] as MATLAB)
    src_lab = skimage.color.rgb2lab(image / 255.0)
    tgt_lab = skimage.color.rgb2lab(target / 255.0)

    # per-channel mean and std (ddof=1 = sample std, as MATLAB)
    ms,  stds = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1), ddof=1)  # source
    mt, stdt  = tgt_lab.mean(axis=(0, 1)), tgt_lab.std(axis=(0, 1), ddof=1)  # target

    # normalise each LAB channel (explicit MATLAB style)
    norm_lab = np.empty_like(src_lab)
    norm_lab[:, :, 0] = (src_lab[:, :, 0] - ms[0]) * (stdt[0] / stds[0]) + mt[0]
    norm_lab[:, :, 1] = (src_lab[:, :, 1] - ms[1]) * (stdt[1] / stds[1]) + mt[1]
    norm_lab[:, :, 2] = (src_lab[:, :, 2] - ms[2]) * (stdt[2] / stds[2]) + mt[2]

    # LAB → RGB (returns [0,1]) → *255 (as demo_new) → uint8
    norm_rgb = skimage.color.lab2rgb(norm_lab)
    norm = np.clip(np.round(norm_rgb * 255), 0, 255).astype(np.uint8)

    # Mask unchanged
    return norm, mask

def macenko_normalize(image, mask, target):
    '''
    Macenko stain normalization via torchstain (Macenko et al., ISBI 2009).
    Parameters Io=255, alpha=1, beta=0.15 to approximate
    Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1) from demo_new.m.

    Not a port of the Warwick toolbox — this is the standard Macenko method,
    torchstain implementation. Results may differ from the MATLAB version.
    '''
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target, Io=255, alpha=1, beta=0.15)          # target = reference (as in demo_new)
    out, _, _ = normalizer.normalize(I=image, Io=255, alpha=1, beta=0.15, stains=False)  # already uint8

    # Mask unchanged (stain norm does not touch it)
    return out, mask
