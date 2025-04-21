# pyramids.py
import numpy as np
from skimage.transform import resize

def build_pyramid(image, num_levels=5, downscale=2):
    """
    Build an image pyramid from original to coarsest
    """
    pyramid = [image]
    latent_maps = [np.ones_like(image)]

    current = image
    for _ in range(1, num_levels):
        new_h = int(np.ceil(current.shape[0] / downscale))
        new_w = int(np.ceil(current.shape[1] / downscale))
        
        coarser = resize(current, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)
        pyramid.append(coarser)
        
        latent_map = np.ones_like(coarser)
        latent_maps.append(latent_map)
        
        current = coarser

    return pyramid, latent_maps
    
def upsample_kernel(kernel, scale_factor=2):
    """
    Upsample the blur kernel by a given scale factor
    """
    new_h = int(np.ceil(kernel.shape[0] * scale_factor))
    new_w = int(np.ceil(kernel.shape[1] * scale_factor))
    upsampled = resize(
        kernel, (new_h, new_w),
        order=1,
        preserve_range=True,
        anti_aliasing=False
    )
    total = upsampled.sum()
    if total > 1e-8:
        upsampled /= total

    return upsampled