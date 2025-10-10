import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Add salt and pepper noise to an image
    """
    noisy = image.copy()
    height, width = image.shape[:2]
    
    # Add salt (white pixels)
    n_salt = int(height * width * salt_prob)
    salt_coords = [np.random.randint(0, i, n_salt) for i in (height, width)]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper (black pixels)
    n_pepper = int(height * width * pepper_prob)
    pepper_coords = [np.random.randint(0, i, n_pepper) for i in (height, width)]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy

def compare_noise_reduction_filters(image, noise_level=0.05):
    """
    Compare different noise reduction techniques
    """
    # Add noise
    noisy = add_salt_pepper_noise(image, noise_level, noise_level)
    
    # Apply different filters
    median_5 = cv2.medianBlur(noisy, 5)
    median_7 = cv2.medianBlur(noisy, 7)
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    nlm = cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    def calculate_psnr(original, filtered):
        mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    psnr_median_5 = calculate_psnr(image, median_5)
    psnr_median_7 = calculate_psnr(image, median_7)
    psnr_gaussian = calculate_psnr(image, gaussian)
    psnr_bilateral = calculate_psnr(image, bilateral)
    psnr_nlm = calculate_psnr(image, nlm)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Noisy (Noise: {noise_level*100:.0f}%)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(median_5, cmap='gray')
    axes[0, 2].set_title(f'Median 5x5\nPSNR: {psnr_median_5:.2f} dB')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(gaussian, cmap='gray')
    axes[1, 0].set_title(f'Gaussian\nPSNR: {psnr_gaussian:.2f} dB')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bilateral, cmap='gray')
    axes[1, 1].set_title(f'Bilateral\nPSNR: {psnr_bilateral:.2f} dB')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(nlm, cmap='gray')
    axes[1, 2].set_title(f'Non-Local Means\nPSNR: {psnr_nlm:.2f} dB')
    axes[1, 2].axis('off')
    
    
    plt.tight_layout(pad=3.0)
    plt.show()

# Example usage
I = cv2.imread('peppers.png', cv2.IMREAD_GRAYSCALE)
compare_noise_reduction_filters(I, noise_level=0.05)