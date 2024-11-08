import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def create_gaussian_kernel_fast(size, sigma):
    """
    Create a 2D Gaussian kernel using vectorized operations.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
        
    # Create 1D coordinate arrays
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate kernel using vectorized operations
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize
    return kernel / kernel.sum()


def apply_gaussian_filter_fast(image, kernel_size, sigma):
    """
    Apply Gaussian filter using scipy's convolve2d for faster processing.
    """
    kernel = create_gaussian_kernel_fast(kernel_size, sigma)
    
    if len(image.shape) == 3:
        # Process each channel separately using list comprehension
        return np.stack([convolve2d(image[:,:,i], kernel, mode='same', boundary='symm')
                        for i in range(image.shape[2])], axis=2)
    else:
        return convolve2d(image, kernel, mode='same', boundary='symm')



