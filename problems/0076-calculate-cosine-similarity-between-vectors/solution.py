
import numpy as np

def cosine_similarity(v1, v2):
	dot = np.dot(v1, v2)
    norm1 = np.sqrt(np.dot(v1, v1))
    norm2 = np.sqrt(np.dot(v2, v2))
    cs = dot/(norm1*norm2)
    return round(cs, 3)
