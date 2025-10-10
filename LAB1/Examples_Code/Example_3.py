import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_kernel_effect(image, kernels, titles):
    """
    Visualize the effect of multiple kernels on an image
    """
    n = len(kernels)
    fig, axes = plt.subplots(1, n+1, figsize=(4*(n+1), 4))
    
    # Show original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show filtered results
    for i, (kernel, title) in enumerate(zip(kernels, titles)):
        filtered = cv2.filter2D(image, -1, kernel)
        axes[i+1].imshow(filtered, cmap='gray')
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

kernels = [
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Sharpen
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  # Edge enhance
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4  # Emboss
]

titles = ['Sharpen', 'Edge Enhance', 'Emboss']
visualize_kernel_effect(image, kernels, titles)