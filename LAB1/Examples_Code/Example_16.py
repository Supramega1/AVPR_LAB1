import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def high_pass_filter_demo(image_path):
    """
    Demonstrate high-pass filtering for edge enhancement
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create high-pass filter (inverse of low-pass)
    cutoff = 30
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
    
    # Apply filter
    f_shift_filtered = f_shift * mask
    
    # Inverse transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    image_filtered = np.fft.ifft2(f_ishift)
    image_filtered = np.abs(image_filtered)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title('High-Pass Filter Mask')
    axes[1].axis('off')
    
    axes[2].imshow(image_filtered, cmap='gray')
    axes[2].set_title('High-Pass Filtered\n(Edges Enhanced)')
    axes[2].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

# Example usage
high_pass_filter_demo('peppers.png')