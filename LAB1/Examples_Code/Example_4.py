import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_image_correlation(img1_path, img2_path):
    """
    Compute correlation coefficient between two images
    """
    # Read images
    I = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    J = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if I is None or J is None:
        raise Exception("Failed to load one or both images")
    
    # Resize second image to match first
    if I.shape != J.shape:
        J = cv2.resize(J, (I.shape[1], I.shape[0]))
    
    # Compute correlation coefficient
    correlation = np.corrcoef(I.ravel(), J.ravel())[0, 1]
    
    # Display images and result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(I, cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(J, cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    plt.suptitle(f'Correlation Coefficient: {correlation:.4f}')
    plt.tight_layout()
    plt.show()
    
    return correlation

# Example usage
correlation = compute_image_correlation('Picture1.png', 'Picture2.png')
print(f"Correlation Coefficient: {correlation:.4f}")

correlation = compute_image_correlation('Picture3.png', 'Picture4.png')
print(f"Correlation Coefficient: {correlation:.4f}")

correlation = compute_image_correlation('Picture5.png', 'Picture6.png')
print(f"Correlation Coefficient: {correlation:.4f}")