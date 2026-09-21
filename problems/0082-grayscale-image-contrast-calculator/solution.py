import numpy as np

def calculate_contrast(img) -> int:
	"""
	Calculate the contrast of a grayscale image.
	Args:
		img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
	"""
	big = np.max(img, axis=(0, 1))
    small = np.min(img, axis=(0,1))
    return big-small