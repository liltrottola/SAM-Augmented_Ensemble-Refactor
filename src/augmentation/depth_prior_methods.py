import numpy as np
from scipy.ndimage import gaussian_filter
import skimage
from skimage.color.colorconv import rgb_from_xyz, xyz_tristimulus_values
from sklearn.decomposition import PCA

'''
    Definition of depth_prior and constants necessary for depth prior generation.
'''
SIGMA_G = 2.0  # sigma (px, DA3 resolution) of the Gaussian derivative used for the depth gradient
LOC = 25 #sigma (px) of the Gaussian local mean of the gradient magnitude in localnorm
RADII = (8, 16, 32, 64) # voting radii (px, DA3 resolution)
ALPHA = 2.0 # radial strictness exponent (value recommended in Loy & Zelinsky, ECCV 2002)

def nrm01(a):
    """ 0-1 normalization : p1 p99 percentile normalization """
    low, high = np.percentile(a, (1, 99))
    a = (a - low) / (high - low + 1e-12)
    return np.clip(a, 0, 1)

def p99(a):
    """ p99 normalization : p99 percentile normalization """
    high = np.percentile(a, 99)
    a = a / (high + 1e-12)
    return np.clip(a, 0, 1)
   
def localnorm(depth, loc=LOC , eps_rel = 1e-6):
    """
        Boundary prior: depth gradient magnitude divided by its local Gaussian mean, then p99-normalized.
        Each edge is measured against its own neighbourhood, not against the whole image:
        a weak edge in a quiet area counts as much as a strong edge in a busy one, so the strongest
        jumps (FOV border, software insets, deep folds) do not flatten the softer polyp borders.
        Floor: eps_rel * max(g) ~ 0, i.e. no floor (only avoids a division by zero).
        in flat areas noise is divided by noise and the DA3 grid artifacts emerge.
        # + eps_rel * g.max(): floor proportional to the strongest gradient of the image.
        # DA3 depth is relative (its scale changes from image to image;
        # if the depth is multiplied by a factor s, g and its local mean are multiplied by s and the ratio is unchanged.
        # A fixed epsilon would break this scale invariance (the same scene would give different maps);
        # a floor that scales with g keeps it. With eps_rel = 1e-6 it only avoids a division by zero
    """
    d = depth.astype(np.float64) 

    # Compute gradients in x and y directions
    gx = gaussian_filter(d, sigma=SIGMA_G, order=(0, 1)) #x-axis derivative
    gy = gaussian_filter(d, sigma=SIGMA_G, order=(1, 0)) #y-axis derivative

    g= np.sqrt(gx**2 + gy**2)

    return p99(g / (gaussian_filter(g, sigma=loc) + eps_rel*g.max()))

def _unit_gradient(depth):
    """
        Unit vector of the gradient of Z and its magnitude.
        Z is the reversed depth, so the gradient points towards what is closer to the camera:
        on the border of a protrusion it points towards its centre.
    """
    # Gaussian derivatives of Z along x (columns) and y (rows)
    gx = gaussian_filter(depth, sigma=SIGMA_G, order=(0, 1))
    gy = gaussian_filter(depth, sigma=SIGMA_G, order=(1, 0))

    # magnitude of the gradient (slope); 1e-12 avoids a division by zero where Z is flat
    g = np.sqrt(gx**2 + gy**2) + 1e-12

    # unit vector of the uphill direction
    ux = gx / g
    uy = gy / g

    return ux, uy, g

def _vote_targets(z_depth, r , ux, uy):
    """
        For every pixel p, the pixel it votes for at distance r along the uphill direction:
        p+ = p + round(u * r), the positively-affected pixel of Loy & Zelinsky (ECCV 2002, eq. 1).
        Returns its row/column indices and a mask of the votes that fall inside the image.
    """
     
    # Compute the voting targets for the Fast Radial Symmetry Transform based on the depth map and unit gradients.
    h, w = z_depth.shape

    # Row (yy) and column (xx) coordinate of every pixel
    yy, xx =np.mgrid[0:h, 0:w] 

    # Point at distance r along the unit gradient, rounded to the nearest pixel
    ty_f, tx_f = np.rint(yy + r * uy), np.rint(xx + r * ux)

    # A vote is valid only if its target falls inside the image
    inside = (ty_f >= 0) & (ty_f < h) & (tx_f >= 0) & (tx_f < w)
    ty = np.clip(ty_f, 0, h - 1).astype(int) # clip to image bounds
    tx = np.clip(tx_f, 0, w - 1).astype(int) #  clip to image bounds

    return ty, tx, inside

def frst_prior(depth, radii=RADII, alpha=ALPHA):
    """
        Region prior: Fast Radial Symmetry Transform (Loy & Zelinsky, ECCV 2002) on the reversed depth,
        plus back-projection.
        Differences from the paper (our choices):
        - only p+ votes ("bright symmetry" on protrusions towards the camera);
        - M and O normalized by their maximum instead of the constant k_n (no clipping of O);
        - each S_r divided by its maximum, so that no radius dominates;
        - back-projection: each pixel also receives the value of S_r where it voted.
    """
    # Reversed depth: near = high, far = low
    d = -depth.astype(np.float64) # only the sign matters: offset and scale cancel in the vote

    # Uphill unit vectors (ux, uy) and slope g of every pixel
    ux, uy, g = _unit_gradient(d)

    # tot: sum over radii of the vote maps (peaks on the centre of protrusions)
    # back: sum over radii of the back-projected values (fills the body of protrusions)
    tot, back = np.zeros(d.shape) , np.zeros(d.shape)

    # One voting pass per radius: each radius looks for protrusions
    for r in radii: 
        # Where every pixel votes at distance r, and which votes fall inside the image
        ty, tx, inside = _vote_targets(d, r, ux, uy)

        # M: vote map (accumulated votes at the target pixels),  M: magnitude projection image,
        # O: vote count (number of votes received at the target pixels),  O: orientation projection image 
        M, O = np.zeros(d.shape), np.zeros(d.shape)

        # Each pixel adds its slope g to the pixel it votes for; np.add.at sums repeated indices
        # (M[ty, tx] += g would keep only the last vote when many pixels hit the same target)
        np.add.at(M, (ty[inside], tx[inside]), g[inside])

        # Each pixel adds 1 to the pixel it votes for: how many pixels voted there
        np.add.at(O, (ty[inside], tx[inside]), 1) 

        # Eq. 2 : Votes coming from many directions (a ring of edges = round shape) win over
        # votes from a single steep wall (a straight fold); alpha = radial strictness
        F = (O / (O.max() + 1e-12)) ** alpha * (M / (M.max() + 1e-12))

        # Spread the votes with a Gaussian of sigma = r/4 (as in the paper): real borders are not perfect circles
        Sn = gaussian_filter(F, r / 4)

        # Scale each radius to max 1 so that no radius dominates the sum
        Sn = Sn / (Sn.max() + 1e-12)

        # Eq. 3: sum over radii of the vote maps
        tot += Sn

        # Back-projection: each pixel receives the value of Sn where it voted;
        # border pixels that voted for a strong centre light up, filling the body, not only the centre
        back += np.where(inside, Sn[ty,tx],0)

    return np.maximum(nrm01(tot), nrm01(back))


# ---------------------------------------------------------------- augmentation methods 
# Signature f(tI, priors) -> final uint8 image, called by name from run_depth_augmentation.py.
# tI = RGB uint8 at the original size; priors = {prior name: map} with the maps of run_depth_prior_maps.py
# (keys = function names (example "frst_prior", "localnorm", "depth_p1p99"), already resized to the image size (bilinear).
# DepthAug / PCA / RG apply save_like_pipeline inside; LAB_gamut is already uint8.

def depth_p1p99(depth):
    """Reversed depth for RG_depth: near = 1, far = 0, robust 0-1 per image (p1-p99, clip)."""
    return nrm01(-depth.astype(np.float64))


def save_like_pipeline(aug_img):
    """global min-max over the 3 channels, then uint8."""
    aug_img_scaled = skimage.exposure.rescale_intensity(aug_img.astype(np.float64), in_range='image', out_range=(0, 1))
    return skimage.img_as_ubyte(aug_img_scaled)


def add_GB(tI, G_prior, B_prior):
    """RGB float, G += G_prior, B += B_prior"""
    tI = skimage.img_as_float(tI).copy()
    tI[:, :, 1] = tI[:, :, 1] + G_prior
    tI[:, :, 2] = tI[:, :, 2] + B_prior
    return tI


def RG_B(tI, B_map):
    """original R, G (0-255), B = min-max of the map * 255."""
    b = (B_map - B_map.min()) * 255 / (B_map.max() - B_map.min())
    return np.dstack((tI[:, :, 0], tI[:, :, 1], b))


def PCA_prior(tI, Prior):
    """Similar to PCA_segPrior of sam_methods.py,any prior in the B channel (PCA fitted on each image)."""
    tI = skimage.img_as_float(tI)
    h, w, c = tI.shape
    flat_image = tI.reshape(-1, c)
    pca = PCA(n_components=2)
    flat_image_pca = pca.fit_transform(flat_image)
    image_pca = flat_image_pca.reshape(h, w, 2)
    image_pca_scaled = (image_pca - np.min(image_pca)) / (np.max(image_pca) - np.min(image_pca)) * 255
    image_pca_scaled = image_pca_scaled.astype(np.uint8)
    r = image_pca_scaled[:, :, 0]
    g = image_pca_scaled[:, :, 1]
    b = (255 * (Prior - Prior.min()) / (Prior.max() - Prior.min())).astype(np.uint8)
    return np.dstack((r, g, b))


def DepthAug(tI, priors):
    return save_like_pipeline(add_GB(tI, priors["frst_prior"], priors["localnorm"]))


def PCA_voteV3(tI, priors):
    # PCA_prior is already uint8, but it still goes through the global min-max
    return save_like_pipeline(PCA_prior(tI, priors["frst_prior"]))


def RG_depth(tI, priors):
    """priors["depth_p1p99"] = depth_p1p99(depth) resized to the image size."""
    return save_like_pipeline(RG_B(tI, priors["depth_p1p99"]))


def RG_voteV3(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["frst_prior"]))


def RG_localnorm(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["localnorm"]))


# ---- Lab: shift of a* up to the sRGB gamut boundary
# Gamut mapping (concept): J. Morovic, M. R. Luo, "The Fundamentals of Gamut Mapping: A Survey", JIST 45(3), 2001.
# Lab <-> XYZ <-> sRGB: constants and formulas of skimage.color (D65, 2 degrees).

XN, YN, ZN = xyz_tristimulus_values(illuminant="D65", observer="2")


def f_lab(t):
    """XYZ -> Lab (as skimage)."""
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)


def f_lab_inv(t):
    """Lab -> XYZ (as skimage)."""
    return np.where(t > 0.2068966, t ** 3, (t - 16.0 / 116.0) / 7.787)


def a_max(lab):
    """Reddest a* with the same L and b inside sRGB, closed form.
    With L and b fixed, Y and Z are fixed and only X varies (monotone in a); linear RGB = M * XYZ is linear in X,
    so the constraints 0 <= R, G, B <= 1 give X_max, and a_max = 500 * (f(X_max / Xn) - f(Y / Yn))."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    Y = YN * f_lab_inv(fy)
    Z = ZN * f_lab_inv(fy - lab[:, :, 2] / 200.0)
    X_max = np.full(Y.shape, np.inf)
    for i in range(3):                                   # linear channel c_i = M[i,0] X + M[i,1] Y + M[i,2] Z
        rest = rgb_from_xyz[i, 1] * Y + rgb_from_xyz[i, 2] * Z
        bound = 1.0 if rgb_from_xyz[i, 0] > 0 else 0.0  # positive X coefficient: c_i <= 1, negative: c_i >= 0
        X_max = np.minimum(X_max, (bound - rest) / rgb_from_xyz[i, 0])
    am = 500.0 * (f_lab(np.maximum(X_max, 0) / XN) - fy)
    return np.maximum(am, lab[:, :, 1])                  # the original pixel is in gamut: a <= a_max (tolerance)


def a_min(lab):
    """Greenest a* with the same L and b inside sRGB: as a_max with the other end of the constraints (X >= 0)."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    Y = YN * f_lab_inv(fy)
    Z = ZN * f_lab_inv(fy - lab[:, :, 2] / 200.0)
    X_min = np.zeros(Y.shape)
    for i in range(3):
        rest = rgb_from_xyz[i, 1] * Y + rgb_from_xyz[i, 2] * Z
        bound = 0.0 if rgb_from_xyz[i, 0] > 0 else 1.0  # positive X coefficient: c_i >= 0, negative: c_i <= 1
        X_min = np.maximum(X_min, (bound - rest) / rgb_from_xyz[i, 0])
    am = 500.0 * (f_lab(X_min / XN) - fy)
    return np.minimum(am, lab[:, :, 1])                  # a >= a_min (numerical tolerance)


def b_limits(lab):
    """b_max, b_min with the same L and a inside sRGB, closed form as a_max (only Z varies, b decreases with Z)."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    Y = YN * f_lab_inv(fy)
    X = XN * f_lab_inv(fy + lab[:, :, 1] / 500.0)
    Z_lo, Z_hi = np.zeros(Y.shape), np.full(Y.shape, np.inf)
    for i in range(3):                                   # c_i = M[i,0] X + M[i,1] Y + M[i,2] Z, with 0 <= c_i <= 1
        rest = rgb_from_xyz[i, 0] * X + rgb_from_xyz[i, 1] * Y
        z0, z1 = (0 - rest) / rgb_from_xyz[i, 2], (1 - rest) / rgb_from_xyz[i, 2]
        Z_lo = np.maximum(Z_lo, np.minimum(z0, z1))
        Z_hi = np.minimum(Z_hi, np.maximum(z0, z1))
    b_max = 200.0 * (fy - f_lab(np.maximum(Z_lo, 0) / ZN))  # minimum Z -> maximum b
    b_min = 200.0 * (fy - f_lab(Z_hi / ZN))
    return np.maximum(b_max, lab[:, :, 2]), np.minimum(b_min, lab[:, :, 2])


def LAB_gamut(tI, Vote, kind="a+"):
    """Shift up to the sRGB gamut boundary, proportional to Vote in [0, 1]; L unchanged, no constant, no clipping:
       a+ (red)   : a + V (a_max - a)     a- (green): a - V (a - a_min)
       b+ (yellow): b + V (b_max - b)     b- (blue) : b - V (b - b_min)
    Returns uint8"""
    lab = skimage.color.rgb2lab(skimage.img_as_float(tI))
    if kind in ("a+", "a-"):
        target = a_max(lab) if kind == "a+" else a_min(lab)
        lab[:, :, 1] = lab[:, :, 1] + Vote * (target - lab[:, :, 1])
    else:
        b_max, b_min = b_limits(lab)
        lab[:, :, 2] = lab[:, :, 2] + Vote * ((b_max if kind == "b+" else b_min) - lab[:, :, 2])
    return (skimage.color.lab2rgb(lab) * 255).round().astype(np.uint8)


def LAB_gamut_aplus(tI, priors):
    # LAB_gamut is already uint8: no save_like_pipeline
    return LAB_gamut(tI, priors["frst_prior"], "a+")


# ---------------------------------------------------------------- curvature priors (second-order)
SIGMA_SI = 1.0          # sigma (px, DA3 resolution) of the Gaussian second derivatives, Shape Index
SIGMA_MC = 1.0          # sigma (px, DA3 resolution), Mean Curvature
GAUSS_TRUNCATE = 3.0    # kernel truncation (in sigmas)


def _shape_index_curvedness(depth, sigma):
    """Shape Index in [-1, 1] and curvedness, from the eigenvalues of the Hessian of the depth."""
    kw = dict(sigma=sigma, mode="nearest", truncate=GAUSS_TRUNCATE)
    dxx = gaussian_filter(depth, order=(0, 2), **kw)
    dyy = gaussian_filter(depth, order=(2, 0), **kw)
    dxy = gaussian_filter(depth, order=(1, 1), **kw)

    trace_half = (dxx + dyy) / 2.0
    discriminant = np.sqrt(((dxx - dyy) / 2.0) ** 2 + dxy ** 2)
    lambda1 = trace_half + discriminant
    lambda2 = trace_half - discriminant

    shape_index = (2.0 / np.pi) * np.arctan2(lambda1 + lambda2, lambda1 - lambda2)
    curvedness = np.sqrt((lambda1 ** 2 + lambda2 ** 2) / 2.0)
    return shape_index, curvedness


def shapeindex(depth):
    """
        Shape Index weighted by curvedness: flat areas = 0.5, curved structures = (1 - SI) / 2.
        Curvedness is normalized by its p95 per image and used as a weight.
    """
    d = depth.astype(np.float64)
    shape_index, curvedness = _shape_index_curvedness(d, SIGMA_SI)

    c_max = np.percentile(curvedness, 95)
    if c_max == 0:
        c_max = 1e-8
    c_norm = np.clip(curvedness / c_max, 0.0, 1.0)

    si_base = (1.0 - shape_index) / 2.0
    return np.clip(0.5 + c_norm * (si_base - 0.5), 0.0, 1.0)


def _mean_curvature(depth, sigma):
    """Mean curvature of the surface z = f(x, y), with Gaussian derivatives."""
    kw = dict(sigma=sigma, mode="nearest", truncate=GAUSS_TRUNCATE)
    fx = gaussian_filter(depth, order=(0, 1), **kw)
    fy = gaussian_filter(depth, order=(1, 0), **kw)
    fxx = gaussian_filter(depth, order=(0, 2), **kw)
    fyy = gaussian_filter(depth, order=(2, 0), **kw)
    fxy = gaussian_filter(depth, order=(1, 1), **kw)

    numerator = (1.0 + fy ** 2) * fxx - 2.0 * fx * fy * fxy + (1.0 + fx ** 2) * fyy
    denominator = 2.0 * np.power(1.0 + fx ** 2 + fy ** 2, 1.5)
    return numerator / denominator


def meancurvature(depth):
    """Mean curvature of the min-max normalized depth, then p1-p99 per image."""
    d = depth.astype(np.float64)
    # min-max first: this curvature is not scale invariant and DA3 depth is relative
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return nrm01(_mean_curvature(d, SIGMA_MC))

def _meancurvature_h(depth):
    d = depth.astype(np.float64)
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return _mean_curvature(d, SIGMA_MC)


def meancurvature_p5p95(depth):
    h = _meancurvature_h(depth)
    low, high = np.percentile(h, (5, 95))
    return np.clip((h - low) / (high - low + 1e-12), 0, 1)


# ---------------------------------------------------------------- curvature aux methods (called by name from the yaml)

def RG_shapeindex(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["shapeindex"]))


def RG_meancurvature(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["meancurvature"]))


def RG_meancurvature_p5p95(tI, priors):
    return save_like_pipeline(RG_B(tI, priors["meancurvature_p5p95"]))

