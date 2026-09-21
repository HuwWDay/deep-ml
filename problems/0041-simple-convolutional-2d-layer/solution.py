import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Apply zero padding
    padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    padded_height, padded_width = padded.shape

    # Compute output dimensions (integer)
    output_height = int(1+ (padded_height - kernel_height) / stride)
    output_width = int(1+ (padded_width - kernel_width) / stride)

    # Initialize output matrix
    output_matrix = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current patch
            pad_patch = padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
            # Elementwise multiplication and sum
            output_matrix[i, j] = np.sum(p