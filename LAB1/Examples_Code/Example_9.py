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

# Load image
I = cv2.imread('peppers.png', cv2.IMREAD_GRAYSCALE)

# Add different levels of noise
noise_levels = [0.01, 0.05, 0.10]

fig, axes = plt.subplots(2, len(noise_levels)+1, figsize=(16, 8))

# Original
axes[0, 0].imshow(I, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for idx, noise_level in enumerate(noise_levels):
    # Add noise
    noisy = add_salt_pepper_noise(I, noise_level, noise_level)
    
    # Apply median filter
    filtered = cv2.medianBlur(noisy, 5)
    
    # Display noisy
    axes[0, idx+1].imshow(noisy, cmap='gray')
    axes[0, idx+1].set_title(f'Noise: {noise_level*100:.0f}%')
    axes[0, idx+1].axis('off')
    
    # Display filtered
    axes[1, idx+1].imshow(filtered, cmap='gray')
    axes[1, idx+1].set_title('Median Filtered')
    axes[1, idx+1].axis('off')

plt.tight_layout()
plt.show()