# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    """
    Apply various geometric transformations to a grayscale image.
    Parameters:
        img (np.ndarray): Input grayscale image as a 2D NumPy array.
    Returns:
        dict: A dictionary containing the transformed images.
    """
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
    
    transformed_images = {}
    h, w = img.shape
    
    # -- Translate
    transformed_images["translated"] = np.roll(img, shift=(15, 15), axis=(0, 1))
    
    # -- Rotate
    # k=-1 :270 deg clockwise 
    transformed_images["rotated"] = np.rot90(img, k=-1)

    # -- Horizontal stretch
    scale_factor = 1.5
    new_w = int(w * scale_factor)
    
    # Calculate source x-coordinates for the new image (w/new_w is the inverse scale factor)
    # Maps the new columns back to the original columns
    x_map = (np.arange(new_w) * (w / new_w)).astype(int)
    
    # Selects the columns from the original image using the calculated map
    stretched_img = img[:, x_map]
    transformed_images["stretched"] = stretched_img

    # -- Horizontal mirror
    transformed_images["mirrored"] = np.flip(img, axis=1)

    # -- Barrel distortion
    center_x, center_y = w // 2, h // 2
    k = 0.000005  # Distortion coefficient (positive for barrel distortion)
    
    # Grid of coordinates (x, y)
    y_grid, x_grid = np.indices((h, w))

    # Coordinates relative to the center
    rel_x, rel_y = x_grid - center_x, y_grid - center_y

    # Square radial distance (r^2)
    r2 = rel_x**2 + rel_y**2

    # Distortion correction factor: r' = r * (1 + k * r^2)
    distortion_factor = 1 + k * r2

    # Source coordinates in the original image (where the pixel should come from)
    src_x = (rel_x * distortion_factor) + center_x
    src_y = (rel_y * distortion_factor) + center_y

    # Remapping (Nearest Neighbor)
    src_x = np.round(src_x).astype(int)
    src_y = np.round(src_y).astype(int)

    # Ensure indices are within bounds
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # Apply remapping (Nearest Neighbor)
    transformed_images["distorted"] = img[src_y, src_x]
    
    return transformed_images