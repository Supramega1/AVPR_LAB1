import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_blur_methods(image_path):
    """
    Compare different blurring methods
    """
    I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Different blur methods
    box_blur_3 = cv2.blur(I, (3, 3))
    box_blur_5 = cv2.blur(I, (5, 5))
    box_blur_11 = cv2.blur(I, (11, 11))
    
    gaussian_blur_3 = cv2.GaussianBlur(I, (3, 3), 0)
    gaussian_blur_5 = cv2.GaussianBlur(I, (5, 5), 0)
    gaussian_blur_11 = cv2.GaussianBlur(I, (11, 11), 0)
    
    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(I, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Box blur
    axes[0, 1].imshow(box_blur_3, cmap='gray')
    axes[0, 1].set_title('Box Blur 3x3')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(box_blur_5, cmap='gray')
    axes[0, 2].set_title('Box Blur 5x5')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(box_blur_11, cmap='gray')
    axes[1, 0].set_title('Box Blur 11x11')
    axes[1, 0].axis('off')
    
    # Gaussian blur
    axes[1, 1].imshow(gaussian_blur_3, cmap='gray')
    axes[1, 1].set_title('Gaussian Blur 3x3')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(gaussian_blur_5, cmap='gray')
    axes[1, 2].set_title('Gaussian Blur 5x5')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(gaussian_blur_11, cmap='gray')
    axes[2, 0].set_title('Gaussian Blur 11x11')
    axes[2, 0].axis('off')
    
    # Difference images
    diff_3 = cv2.absdiff(box_blur_3, gaussian_blur_3)
    diff_5 = cv2.absdiff(box_blur_5, gaussian_blur_5)
    
    axes[2, 1].imshow(diff_3, cmap='hot')
    axes[2, 1].set_title('Difference 3x3')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(diff_5, cmap='hot')
    axes[2, 2].set_title('Difference 5x5')
    axes[2, 2].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

compare_blur_methods('peppers.png')