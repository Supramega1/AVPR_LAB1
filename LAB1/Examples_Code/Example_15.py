import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def fourier_transform_filtering(image_path):
    """
    Demonstrate frequency domain filtering using Fourier Transform
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Create low-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Different cutoff frequencies
    cutoffs = [30, 60, 100]
    
    fig, axes = plt.subplots(2, len(cutoffs)+1, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Magnitude spectrum
    axes[1, 0].imshow(magnitude_spectrum, cmap='gray')
    axes[1, 0].set_title('Magnitude Spectrum')
    axes[1, 0].axis('off')
    
    for idx, cutoff in enumerate(cutoffs):
        # Create mask
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
        
        # Apply mask
        f_shift_filtered = f_shift * mask
        
        # Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        image_filtered = np.abs(image_filtered)
        
        # Display filtered image
        axes[0, idx+1].imshow(image_filtered, cmap='gray')
        axes[0, idx+1].set_title(f'Low-Pass Filter\nCutoff: {cutoff}')
        axes[0, idx+1].axis('off')
        
        # Display filtered spectrum
        magnitude_filtered = 20 * np.log(np.abs(f_shift_filtered) + 1)
        axes[1, idx+1].imshow(magnitude_filtered, cmap='gray')
        axes[1, idx+1].set_title(f'Filtered Spectrum')
        axes[1, idx+1].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

# Example usage
fourier_transform_filtering('peppers.png')