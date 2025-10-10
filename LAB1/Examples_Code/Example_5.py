import cv2
import numpy as np
import matplotlib.pyplot as plt

def correlation_with_filters(image_path):
    """
    Compute correlation between original image and various filtered versions
    """
    # Read image
    I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply different filters
    median_filtered = cv2.medianBlur(I, 5)
    gaussian_filtered = cv2.GaussianBlur(I, (5, 5), 1)
    bilateral_filtered = cv2.bilateralFilter(I, 9, 75, 75)
    
    # Compute correlations
    corr_median = np.corrcoef(I.ravel(), median_filtered.ravel())[0, 1]
    corr_gaussian = np.corrcoef(I.ravel(), gaussian_filtered.ravel())[0, 1]
    corr_bilateral = np.corrcoef(I.ravel(), bilateral_filtered.ravel())[0, 1]
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(I, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(median_filtered, cmap='gray')
    axes[0, 1].set_title(f'Median Filter\nCorr: {corr_median:.4f}')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gaussian_filtered, cmap='gray')
    axes[1, 0].set_title(f'Gaussian Filter\nCorr: {corr_gaussian:.4f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bilateral_filtered, cmap='gray')
    axes[1, 1].set_title(f'Bilateral Filter\nCorr: {corr_bilateral:.4f}')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corr_median, corr_gaussian, corr_bilateral

# Example usage
correlations = correlation_with_filters('input_image.jpg')