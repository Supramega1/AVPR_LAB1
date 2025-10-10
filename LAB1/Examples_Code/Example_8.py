import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_kernel_gallery():
    """
    Create and display a gallery of common kernels
    """
    kernels = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Box Blur': np.ones((3, 3)) / 9,
        'Gaussian Blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Edge Detect': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'Top Sobel': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        'Bottom Sobel': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Left Sobel': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        'Right Sobel': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        'Outline': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    }
    
    # Load test image
    image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Create subplot grid
    n_kernels = len(kernels)
    n_cols = 4
    n_rows = (n_kernels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        filtered = cv2.filter2D(image, -1, kernel)
        axes[idx].imshow(filtered, cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(kernels), len(axes)):
        axes[idx].axis('off')
    
    
    plt.tight_layout(pad=3.0)
    plt.show()

create_kernel_gallery()