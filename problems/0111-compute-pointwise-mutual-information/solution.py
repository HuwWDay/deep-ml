import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
	Px, Py = total_counts_x/total_samples, total_counts_y/total_samples
    Pxy = joint_counts / total_samples
    return round(np.log2(Pxy/(Px*Py)),3)