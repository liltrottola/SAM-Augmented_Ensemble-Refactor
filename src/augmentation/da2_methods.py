import numpy as np
import skimage
import scipy.ndimage
import cv2
import skimage.exposure
import skimage.color
import torchstain

def width_shift(image, mask, siz=352):
    '''
    Traslazione orizzontale random.
    '''
    # --- Random sampling (matcha MATLAB rand) ---
    # a*interval può essere non-intero (es. 35.2 con siz=352) → MATLAB usa
    # interpolazione bilineare per il shift sub-pixel. Replichiamo con order=1.
    interval = siz * 0.1
    a = np.random.randint(1, 10)               # {1,...,9} come MATLAB 1+floor(rand*9)
    direction = 1 if np.random.rand() < 0.5 else -1
    dx = direction * interval * a              # float, può essere non-intero

    # --- Shift su asse X ---
    # Per RGB (H, W, 3): shift = [dy, dx, dchannel] = [0, dx, 0]
    # order=1 (bilineare) sull'immagine → matcha MATLAB imtranslate per shift float
    # mode='constant', cval=0 → bordi vuoti = nero (come MATLAB FillValues default)
    image_shifted = scipy.ndimage.shift(
        image, [0, dx, 0],
        order=1, mode='constant', cval=0,
    )

    # Per la mask: order=0 (nearest neighbor) per preservare i valori binari.
    # Con order=1 bilineare, lungo i bordi della mask si creerebbero valori
    # intermedi che NON sono né foreground né background → corromperebbe la ground truth.
    mask_shifted = scipy.ndimage.shift(
        mask, [0, dx],
        order=0, mode='constant', cval=0,
    )

    return image_shifted, mask_shifted

def height_shift(image, mask, siz=352):
    '''
    Traslazione verticale random.
    '''
    # --- Random sampling (matcha MATLAB rand) ---
    # Stesso pattern di width_shift: a*interval può essere non-intero → order=1 sull'immagine.
    interval = siz * 0.1
    a = np.random.randint(1, 10)               # {1,...,9} come MATLAB 1+floor(rand*9)
    direction = 1 if np.random.rand() < 0.5 else -1
    dy = direction * interval * a              # float, può essere non-intero

    # --- Shift su asse Y ---
    # Per RGB (H, W, 3): shift = [dy, dx, dchannel] = [dy, 0, 0]
    # order=1 sull'immagine, mode/cval come MATLAB FillValues default
    image_shifted = scipy.ndimage.shift(
        image, [dy, 0, 0],
        order=1, mode='constant', cval=0,
    )

    # Per la mask: order=0 (nearest) per preservare i valori binari
    mask_shifted = scipy.ndimage.shift(
        mask, [dy, 0],
        order=0, mode='constant', cval=0,
    )

    return image_shifted, mask_shifted

def rotation(image, mask):
    '''
    Rotazione random di {45°, 90°, 135°, 180°}, direzione random.
    '''
    # --- Random sampling (matcha MATLAB) ---
    # MATLAB: c=1+floor(rand*4) → c in {1,2,3,4}; ang*c → {45,90,135,180}
    ang = 45
    c = np.random.randint(1, 5)                # {1,...,4}
    direction = 1 if np.random.rand() < 0.5 else -1
    angle = direction * ang * c         # ±45°, ±90°, ±135°, ±180°

    # --- Rotazione ---
    # skimage.transform.rotate: order=1 (bilineare) = default MATLAB imwarp.
    # resize=False replica affineOutputView(size(im),tform) di MATLAB:
    #   l'output ha la STESSA dimensione dell'input (croppato ai bordi).
    # preserve_range=True mantiene il dtype uint8 [0,255] (altrimenti skimage normalizza a float [0,1]).
    image_rotated = skimage.transform.rotate(
        image, angle,
        order=1, mode='constant', cval=0,
        resize=False, preserve_range=True,
    ).astype(image.dtype)

    # Per la mask: order=0 (nearest) per preservare i valori binari
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
    # MATLAB: c=1+floor(rand)=1 sempre (forse bug), ang=45 → shear di ±45° fisso
    # 4 opzioni equiprobabili (d ∈ [0,1) diviso in 4 quartili)
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

    # --- Costruzione matrice (matcha MATLAB randomAffine2d) ---
    # skimage AffineTransform(shear=θ) usa convenzione segno opposto a MATLAB
    # (matrix[0,1] = -tan(θ) invece di +tan(θ)).
    # matrice esplicita per matchare MATLAB esatto.
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

    # --- Applicazione warp ---
    # skimage.warp applica la trasformazione INVERSA.
    # Per matchare MATLAB imwarp, passiamo tform.inverse.
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
        random flip
    '''
    # MATLAB usa flip() = verticale (asse 0) e fliplr() = orizzontale (asse 1).
    if np.random.rand() <0.5:
        return np.flipud(image) , np.flipud(mask) #vertical flip
    else:
        return np.fliplr(image) , np.fliplr(mask) #horizontal flip

def brightness_uniform(image, mask):
    '''
    Luminosità uniforme: ±[25,50) aggiunto a tutti i canali RGB.
    '''

    # MATLAB: l = floor((1+rand)*25) → {25,...,49}; segno random
    l = int(np.floor((1 + np.random.rand()) * 25))   # {25,...,49}
    sign = 1 if np.random.rand() < 0.5 else -1

    delta = sign*l

    # uint8 + delta in NumPy fa wraparound (250+25=19!), MATLAB satura (250+25=255).
    # si converte a int16, si somma, clip a [0,255], torniamo a uint8 

    image_bright = np.clip(image.astype(np.int16) + delta , 0, 255).astype(np.uint8)

    return image_bright, mask

def brightness_per_channel(image, mask):
    u = int(np.floor((1 + np.random.rand()) * 25))
    v = int(np.floor((1 + np.random.rand()) * 25))
    z = int(np.floor((1 + np.random.rand()) * 25))

    sign = 1 if np.random.rand() < 0.5 else -1

    result = image.astype(np.int16)          # spazio per overflow
    result[:, :, 0] = result[:, :, 0] + sign * u   # canale R
    result[:, :, 1] = result[:, :, 1] + sign * v   # canale G
    result[:, :, 2] = result[:, :, 2] + sign * z   # canale B
    image_bright = np.clip(result, 0, 255).astype(np.uint8)

    return image_bright, mask

def speckle_noise(image, mask):
    '''
        Rumore speckle (moltiplicativo), varianza 0.05 (default MATLAB imnoise).
    '''
    # MATLAB imnoise(im,'speckle'): J = I + n*I, n ~ gaussiano media 0, var 0.05.
    # skimage random_noise fa lo stesso ma default var=0.01 
    # random_noise ritorna float in [0,1] → riconvertiamo a uint8.
    noisy = skimage.util.random_noise(image, mode='speckle', var=0.05)
    image_noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)

    # Mask invariata
    return image_noisy, mask

def _fspecial_motion(length: float, angle: float) -> np.ndarray:
    """Motion-blur PSF, a reimplementation of MATLAB's fspecial('motion', LEN, THETA).

    Builds a 2-D point-spread function that approximates the linear motion of a
    camera by ``length`` pixels at ``angle`` degrees (counter-clockwise from the
    horizontal). The kernel is a thin, anti-aliased line segment through the
    centre, with rounded endpoints, normalised to sum 1; convolving an image
    with it produces a directional motion blur.

    This is a from-scratch reimplementation of the algorithm used by MATLAB's
    ``fspecial('motion', ...)`` (Image Processing Toolbox, (c) The MathWorks),
    based on its documented behaviour rather than on its source. Like the
    original it computes only the half-kernel growing from the origin and then
    mirrors it to exploit the point symmetry of the PSF.

    Parameters
    ----------
    length : float
        Motion length in pixels (must be > 0).
    angle : float
        Motion angle in degrees, counter-clockwise from the horizontal.

    Returns
    -------
    np.ndarray
        2-D float64 kernel summing to 1. Horizontal/vertical motions collapse to
        a 1-D kernel: (1, 9) for (9, 0) and (9, 1) for (9, 90); the latter is a
        uniform 1/9 column (the actual MATLAB behaviour, not zeros at the ends).
    """
    eps_f = np.finfo(np.float64).eps
    linewdt = 1.0                        # line thickness, in pixels

    # Half-length of the motion segment and its orientation in radians. The
    # angle is folded into [0, 180): a line at theta and at theta+180 is the same.
    len_ = max(1.0, float(length))
    half = (len_ - 1) / 2.0
    phi = (angle % 180) / 180.0 * np.pi

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.sign(cosphi)              # which way the line leans: +1 right, -1 left

    # Half-extent of the kernel grid. The "- len*eps" term pushes the value just
    # below an integer at exactly 0 deg / 90 deg, so the truncation drops it by
    # one and the kernel collapses to a clean 1-D vector instead of a 3-wide band.
    sx = int(np.fix(half * cosphi + linewdt * xsign - len_ * eps_f))
    sy = int(np.fix(half * sinphi + linewdt - len_ * eps_f))

    # Coordinate grid of the half-kernel, growing from the origin (0, 0) towards
    # (sx, sy); the sign of the x step orients it to the correct horizontal side.
    step = int(xsign) if xsign != 0 else 1
    xv = np.arange(0, sx + step, step)
    yv = np.arange(0, sy + 1)
    x, y = np.meshgrid(xv, yv)
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Signed perpendicular distance from each pixel to the ideal line through the
    # origin; rad is the pixel's Euclidean distance from the origin.
    dist2line = y * cosphi - x * sinphi
    rad = np.sqrt(x ** 2 + y ** 2)

    # Round the segment's endpoints: pixels that lie beyond the tip but still
    # within the line width have their distance measured to the endpoint instead
    # of to the (infinite) line, so the streak ends in a cap rather than running on.
    lastpix = (rad >= half) & (np.abs(dist2line) <= linewdt)
    x2lastpix = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi)
    dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)

    # Turn distance into weight (triangular anti-aliasing) and clip anything
    # farther than the line width down to zero.
    dist2line = linewdt + eps_f - np.abs(dist2line)
    dist2line[dist2line < 0] = 0.0

    # Unfold the half-kernel into the full, point-symmetric kernel: the 180-deg
    # rotated copy goes top-left, the original goes bottom-right, and the two
    # overlap on the shared centre pixel.
    n_r, n_c = dist2line.shape
    h = np.zeros((2 * n_r - 1, 2 * n_c - 1), dtype=np.float64)
    h[:n_r, :n_c] = np.rot90(dist2line, 2)
    h[n_r - 1:, n_c - 1:] = dist2line

    # Normalise to sum 1 (the eps term guards a degenerate, all-zero kernel).
    h = h / (h.sum() + eps_f * len_ * len_)

    # Final flip to match MATLAB's counter-clockwise-from-horizontal convention.
    if cosphi > 0:
        h = np.flipud(h)

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
    h = _fspecial_motion(length, angle)

    out_ch1 = cv2.filter2D(out_ch1, -1, h, borderType=cv2.BORDER_REPLICATE)
    out_ch2 = cv2.filter2D(out_ch2, -1, h, borderType=cv2.BORDER_REPLICATE)
    out_ch3 = cv2.filter2D(out_ch3, -1, h, borderType=cv2.BORDER_REPLICATE)
    
    # Mask: MATLAB la moltiplica — in pausa. Per ora tengo invariata.
    mask_blurred = cv2.filter2D(mask_after_lut, -1, h, borderType=cv2.BORDER_REPLICATE)
    mask_out = np.clip(np.round(mask_blurred), 0, 255).astype(np.uint8)

    #append modified image 
    image_out = np.stack([out_ch1, out_ch2, out_ch3], axis=-1)  # shape (H, W, 3) uint8

    # Per ora: contrast/blur applicati SOLO all'immagine, mask invariata.
    return image_out, mask

def shadows(image, mask):
    '''
    Ombra graduale su un lato (sx o dx): ogni colonna × il valore corrispondente
    yval ∈ [0.2, 1].
    '''

    H, W = image.shape[:2]                     #num di righe e colonne
    xval = np.linspace(0, 1, W)               # MATLAB linspace(0,1,siz)
    direction = np.random.randint(0, 2)        # MATLAB randi(2)-1 → {0,1}
    if direction:
        yval = 0.2 + np.sqrt(xval / 0.5) * 0.8         # ombra a sinistra
    else:
        yval = 0.2 + np.sqrt((-xval + 1) / 0.5) * 0.8  # ombra a destra
    yval[yval > 1] = 1.0                        # cap a 1 (MATLAB yval(yval>1)=1)

    # moltiplica ogni colonna i per yval[i], equivalente del loop matlab
    out = image.astype(np.float64) * yval.reshape(1, W, 1)


    #round before uint8 that usually trunc values
    image_shadowed = np.clip(np.round(out), 0, 255).astype(np.uint8)

    # Mask: MATLAB la moltiplica — in pausa. Per ora tengo invariata.
    return image_shadowed, mask

def rgb_histogram_match(image, mask, target):
    '''
    Histogram specification: adatta l'istogramma di `image` a quello di `target`,
    per canale. 
    '''

    # skimage match_histograms fa histogram matching per canale (channel_axis=-1)
    # = MATLAB histeq(SourceCh, HnTargetCh) sui 3 canali
    matched = skimage.exposure.match_histograms(image, target, channel_axis=-1) #channel_axis=-1 Channel R,B;B separati
    matched = np.clip(np.round(matched), 0, 255).astype(np.uint8)

    return matched, mask

def reinhard_normalize(image, mask, target):
    '''
    Reinhard color transfer: normalizza media/std dei canali LAB di `image`
    su quelli di `target`. Equivalente a NormReinhard.m.
    '''

    # RGB → LAB (im2double: /255 per portare a [0,1] come MATLAB)
    src_lab = skimage.color.rgb2lab(image / 255.0)
    tgt_lab = skimage.color.rgb2lab(target / 255.0)

    # media e std per canale (ddof=1 = std campionaria, come MATLAB)
    ms,  stds = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1), ddof=1) #source
    mt, stdt  = tgt_lab.mean(axis=(0, 1)), tgt_lab.std(axis=(0, 1), ddof=1) #target

    # normalizza ogni canale LAB (stile MATLAB esplicito)
    norm_lab = np.empty_like(src_lab)
    norm_lab[:, :, 0] = (src_lab[:, :, 0] - ms[0]) * (stdt[0] / stds[0]) + mt[0]
    norm_lab[:, :, 1] = (src_lab[:, :, 1] - ms[1]) * (stdt[1] / stds[1]) + mt[1]
    norm_lab[:, :, 2] = (src_lab[:, :, 2] - ms[2]) * (stdt[2] / stds[2]) + mt[2]

    # LAB → RGB (ritorna [0,1]) → *255 (come demo_new) → uint8
    norm_rgb = skimage.color.lab2rgb(norm_lab)
    norm = np.clip(np.round(norm_rgb * 255), 0, 255).astype(np.uint8)

    # Mask invariata
    return norm, mask

def macenko_normalize(image, mask, target):
    '''
    Macenko stain normalization via torchstain (Macenko et al., ISBI 2009).
    Parametri Io=255, alpha=1, beta=0.15 per avvicinarsi a
    Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1) di demo_new.m.

    Non è un port del toolbox Warwick — è il metodo Macenko standard,
    implementazione torchstain. I risultati possono differire dal MATLAB.
    '''
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target, Io=255, alpha=1, beta=0.15)          # target = reference (come demo_new)
    out, _, _ = normalizer.normalize(I=image, Io=255, alpha=1, beta=0.15, stains=False)  # già uint8

    # Mask invariata (la stain norm non la tocca)
    return out, mask
