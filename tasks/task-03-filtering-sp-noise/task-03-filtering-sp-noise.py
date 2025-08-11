import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    # Median filter - replaces each pixel with the median value of its neighborhood
    denoised_image = cv2.medianBlur(image, 5)  

    return denoised_image

if __name__ == "__main__":
    noisy_image = cv2.imread("img/head.png", cv2.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv2.imwrite("head_filtered.png", denoised_image)