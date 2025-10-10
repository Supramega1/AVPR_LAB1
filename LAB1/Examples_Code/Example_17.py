import cv2
import pywt
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('woman.png', cv2.IMREAD_GRAYSCALE)

# Perform single-level Haar wavelet decomposition
coeffs2 = pywt.dwt2(image, 'haar')
approximation, (horizontal, vertical, diagonal) = coeffs2

# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(approximation, cmap='gray')
axes[0, 0].set_title('Approximation')
axes[0, 0].axis('off')

axes[0, 1].imshow(horizontal, cmap='gray')
axes[0, 1].set_title('Horizontal Details')
axes[0, 1].axis('off')

axes[1, 0].imshow(vertical, cmap='gray')
axes[1, 0].set_title('Vertical Details')
axes[1, 0].axis('off')

axes[1, 1].imshow(diagonal, cmap='gray')
axes[1, 1].set_title('Diagonal Details')
axes[1, 1].axis('off')

plt.tight_layout(pad=3.0)
plt.show()
