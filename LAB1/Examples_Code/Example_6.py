import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_motion_blur_kernel(size, angle):
    """
    Create a motion blur kernel
    
    Args:
        size: Size of the kernel
        angle: Angle of motion in degrees
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create line along the angle
    for i in range(size):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    # Normalize
    kernel /= np.sum(kernel)
    
    return kernel

# Load image
originalRGB = cv2.imread('peppers.png')
originalBGR = cv2.cvtColor(originalRGB, cv2.COLOR_BGR2RGB)

# Create motion blur kernels with different angles
angles = [0, 45, 90, 135]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(originalBGR)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Apply motion blur at different angles
for idx, angle in enumerate(angles):
    kernel = create_motion_blur_kernel(30, angle)
    blurred = cv2.filter2D(originalRGB, -1, kernel)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(blurred_rgb)
    axes[row, col].set_title(f'Motion Blur - {angle}°')
    axes[row, col].axis('off')

axes[1, 2].axis('off')

plt.tight_layout(pad=3.0)
plt.show()