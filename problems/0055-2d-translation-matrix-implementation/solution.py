import numpy as np
def translate_object(points, tx, ty):
    T = np.diag([1, 1, 1])
    T[0][2], T[1][2] = tx, ty
    translated_points = []
    for point in points:
        point.append(1)
        nppoint = np.array(point)
        transpoint = T @ nppoint
        translated_point = [transpoint[0], transpoint[1]]
        translated_points.append(translated_point)

	return translated_points
