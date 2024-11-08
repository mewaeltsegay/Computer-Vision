import numpy as np

def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
        
    kernel = []
    center = size // 2
    sum_values = 0
    
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            exponent = -(x*x + y*y)/(2*sigma*sigma)
            value = np.exp(exponent)
            row.append(value)
            sum_values += value
        kernel.append(row)
    
    # Normalize the kernel
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_values
            
    return kernel

def apply_gaussian_filter(image, kernel_size, sigma):
    """
    Apply Gaussian filter to an image.
    Works with both grayscale and RGB images.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Handle both RGB and grayscale images
    if len(image.shape) == 3:
        height, width, channels = image.shape
        output = np.zeros((height, width, channels))
        
        # Process each color channel separately
        for c in range(channels):
            kernel = create_gaussian_kernel(kernel_size, sigma)
            padding = 0
            
            # Apply convolution to each channel
            for i in range(height):
                for j in range(width):
                    sum_value = 0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            im_i = i + ki - padding
                            im_j = j + kj - padding
                            
                            # Handle boundaries by mirroring
                            if im_i < 0:
                                im_i = abs(im_i)
                            elif im_i >= height:
                                im_i = 2*height - im_i - 2
                                
                            if im_j < 0:
                                im_j = abs(im_j)
                            elif im_j >= width:
                                im_j = 2*width - im_j - 2
                            
                            sum_value += image[im_i][im_j][c] * kernel[ki][kj]
                    
                    output[i, j, c] = sum_value
    else:
        # Grayscale image processing
        height, width = image.shape
        output = np.zeros((height, width))
        kernel = create_gaussian_kernel(kernel_size, sigma)
        padding = kernel_size // 2
        
        for i in range(height):
            for j in range(width):
                sum_value = 0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        im_i = i + ki - padding
                        im_j = j + kj - padding
                        
                        if im_i < 0:
                            im_i = abs(im_i)
                        elif im_i >= height:
                            im_i = 2*height - im_i - 2
                            
                        if im_j < 0:
                            im_j = abs(im_j)
                        elif im_j >= width:
                            im_j = 2*width - im_j - 2
                        
                        sum_value += image[im_i][im_j] * kernel[ki][kj]
                
                output[i, j] = sum_value
    
    return output