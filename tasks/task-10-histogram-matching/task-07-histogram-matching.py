# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Creates a copy of the source image to store the result
    matched_img = np.zeros_like(source_img)
    
    # Processes each RGB channel separately
    for channel in range(3):
        # Calculates the histogram of the source image channel (256 bins for values 0-255)
        source_hist, _ = np.histogram(source_img[:,:,channel], bins=256, range=(0,256))
        # Calculates the histogram of the reference image channel
        ref_hist, _ = np.histogram(reference_img[:,:,channel], bins=256, range=(0,256))
        
        # Calculates the cumulative distribution functions (CDFs)
        source_cdf = np.cumsum(source_hist).astype(np.float64)
        source_cdf /= source_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]
        
        # Creates the mapping table using interpolation
        mapping = np.interp(source_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
        
        # Apply mapping to the source image channel
        matched_img[:,:,channel] = mapping[source_img[:,:,channel]]
    
    return matched_img.astype(np.uint8)