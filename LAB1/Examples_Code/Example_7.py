import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_motion_blur(image, kernel_size=50, sigma=10, border_type=cv2.BORDER_REPLICATE):
    """
    Apply Gaussian motion blur with specified boundary handling
    """
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())
    
    # Apply filter with specified border type
    filtered = cv2.filter2D(image, -1, kernel, borderType=border_type)
    
    return filtered

# Load image
originalRGB = cv2.imread('peppers.png')

# Apply with different boundary options
border_types = {
    'Replicate': cv2.BORDER_REPLICATE,
    'Reflect': cv2.BORDER_REFLECT,
    'Wrap': cv2.BORDER_WRAP,
    'Constant': cv2.BORDER_CONSTANT
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(originalRGB, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for idx, (name, border_type) in enumerate(border_types.items()):
    filtered = apply_gaussian_motion_blur(originalRGB, border_type=border_type)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(filtered_rgb)
    axes[row, col].set_title(f'Boundary: {name}')
    axes[row, col].axis('off')

# Hide unused subplot
axes[1, 2].axis('off')
plt.tight_layout(pad=3.0)
plt.show()