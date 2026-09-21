import numpy as np

def pos_encoding(position: int, d_model: int):
    if position <= 0:
        return -1
    if d_model <= 0:
        return -1

    # Compute angle rates
    angle_rads = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(d_model):
            angle_rads[pos, i] = pos / np.power(10000, (2 * (i // 2)) / d_model)

    # Apply sin to even indices; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads.astype(np.float32)
