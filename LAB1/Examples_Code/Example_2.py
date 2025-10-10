import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image as grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt operator for edge detection
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1],
                      [0,  0,  0],
                      [1,  1,  1]], dtype=np.float32)

# Scharr operator (more accurate than Sobel)
scharr_x = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float32)

scharr_y = np.array([[-3, -10, -3],
                     [0,  0,  0],
                     [3,  10,  3]], dtype=np.float32)

# Apply all filters
prewitt_edges_x = cv2.filter2D(image, -1, prewitt_x)
prewitt_edges_y = cv2.filter2D(image, -1, prewitt_y)
scharr_edges_x = cv2.filter2D(image, -1, scharr_x)
scharr_edges_y = cv2.filter2D(image, -1, scharr_y)

# Compute magnitudes
prewitt_magnitude = np.sqrt(prewitt_edges_x**2 + prewitt_edges_y**2)
scharr_magnitude = np.sqrt(scharr_edges_x**2 + scharr_edges_y**2)

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(prewitt_magnitude, cmap='gray')
axes[1].set_title('Prewitt Edge Detection')
axes[2].imshow(scharr_magnitude, cmap='gray')
axes[2].set_title('Scharr Edge Detection')

for ax in axes:
    ax.axis('off')
    
plt.tight_layout()
plt.show()