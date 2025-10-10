import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def wavelet_decomposition_analysis(image_path):
    """
    Perform and visualize wavelet decomposition
    """
    # Load image
    X = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform 2D DWT
    coeffs = pywt.dwt2(X, 'sym4', mode='per')
    cA, (cH, cV, cD) = coeffs
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(X, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cA, cmap='gray')
    axes[0, 1].set_title('Approximation (cA)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cH, cmap='gray')
    axes[0, 2].set_title('Horizontal Detail (cH)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cV, cmap='gray')
    axes[1, 0].set_title('Vertical Detail (cV)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cD, cmap='gray')
    axes[1, 1].set_title('Diagonal Detail (cD)')
    axes[1, 1].axis('off')
    
    # Reconstruct image
    reconstructed = pywt.idwt2(coeffs, 'sym4', mode='per')
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('Reconstructed Image')
    axes[1, 2].set_title('Reconstructed Image')
    axes[1, 2].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    return coeffs

# Example usage
coeffs = wavelet_decomposition_analysis('woman.png')