import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def multilevel_wavelet_decomposition(image_path, levels=3):
    """
    Perform multi-level wavelet decomposition
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform multi-level decomposition
    coeffs = pywt.wavedec2(image, 'db2', level=levels)
    
    # Create visualization
    fig, axes = plt.subplots(1, levels+1, figsize=(5*(levels+1), 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Each level of approximation
    for i in range(1, levels+1):
        # Reconstruct from level i
        reconstructed = pywt.waverec2(coeffs[:i+1], 'db2')
        # Crop to original size
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        
        axes[i].imshow(reconstructed, cmap='gray')
        axes[i].set_title(f'Level {i} Reconstruction')
        axes[i].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    return coeffs

# Example usage
coeffs = multilevel_wavelet_decomposition('woman.png', levels=3)