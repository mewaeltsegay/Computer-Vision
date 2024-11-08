import numpy as np

def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel that matches library implementations.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Create a coordinate grid centered at 0
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate the kernel using the correct 2D Gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize the kernel to sum to 1
    kernel = kernel / kernel.sum()
    
    return kernel

def apply_gaussian_filter(image, kernel_size, sigma):
    """
    Apply Gaussian filter to an image with correct boundary handling.
    Works with both grayscale and RGB images.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    kernel = create_gaussian_kernel(kernel_size, sigma)
    padding = kernel_size // 2
    
    # Handle both RGB and grayscale images
    if len(image.shape) == 3:
        height, width, channels = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        
        # Process each color channel separately
        for c in range(channels):
            padded = np.pad(image[:,:,c], padding, mode='reflect')
            
            # Apply convolution using numpy operations
            for i in range(height):
                for j in range(width):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    output[i,j,c] = np.sum(window * kernel)
    else:
        height, width = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        padded = np.pad(image, padding, mode='reflect')
        
        # Apply convolution
        for i in range(height):
            for j in range(width):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i,j] = np.sum(window * kernel)
    
    return output

