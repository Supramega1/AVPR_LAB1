import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_smoothing_comparison(image_path):
    """
    Compare Gaussian smoothing with different parameters
    """
    # Read image
    I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Different sigma values
    sigmas = [0.5, 1, 2, 4, 8]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(I, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Apply Gaussian blur with different sigmas
    for idx, sigma in enumerate(sigmas):
        blurred = cv2.GaussianBlur(I, (0, 0), sigma)
        
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(blurred, cmap='gray')
        axes[row, col].set_title(f'Gaussian σ={sigma}')
        axes[row, col].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

gaussian_smoothing_comparison('peppers.png')