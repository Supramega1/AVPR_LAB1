import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image as grayscale
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define Sobel kernels for edge detection
# Horizontal edge detection
horizontal_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)

# Vertical edge detection
vertical_kernel = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=np.float32)

# Apply convolution
horizontal_edges = cv2.filter2D(image, -1, horizontal_kernel)
vertical_edges = cv2.filter2D(image, -1, vertical_kernel)

# Combine edges using magnitude
combined_edges = np.sqrt(horizontal_edges**2 + vertical_edges**2)
combined_edges = np.uint8(combined_edges)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(horizontal_edges, cmap='gray')
axes[0, 1].set_title('Horizontal Edges')
axes[0, 1].axis('off')

axes[1, 0].imshow(vertical_edges, cmap='gray')
axes[1, 0].set_title('Vertical Edges')
axes[1, 0].axis('off')

axes[1, 1].imshow(combined_edges, cmap='gray')
axes[1, 1].set_title('Combined Edges')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()