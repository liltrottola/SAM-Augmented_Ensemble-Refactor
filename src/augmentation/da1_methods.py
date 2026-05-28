import numpy as np

def fliplr(image,mask):

    '''
        Flips the image and mask horizontally.
        equivalent to fliplr in matlab
    
    '''
    image_flipped = np.fliplr(image)
    mask_flipped = np.fliplr(mask)

    return image_flipped, mask_flipped

def flipud(image,mask):

    '''
        Flips the image and mask vertically.
        equivalent to flipud in matlab
    
    '''
    image_flipped = np.flipud(image)
    mask_flipped = np.flipud(mask)

    return image_flipped, mask_flipped

def rot90(image,mask):

    '''
        Rotates the image and mask 90 degrees counterclockwise.
        equivalent to rot90 in matlab
        lossless rotation, no interpolation needed
    
    '''
    image_rotated = np.rot90(image)
    mask_rotated = np.rot90(mask)

    return image_rotated, mask_rotated

def has_enough_foreground(mask, min_pixels=100):

    '''
        Checks if the mask has enough foreground pixels.
        return True if the number of foreground pixels is greater than or equal to min_pixels, False otherwise.
        min_pixels: minimum number of foreground pixels required to consider the mask valid.
    
    '''
    foreground_pixels = np.sum(mask > 0)  # Assuming foreground pixels are non-zero

    return foreground_pixels >= min_pixels
