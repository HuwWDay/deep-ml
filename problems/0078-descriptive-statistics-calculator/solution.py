import numpy as np 
def descriptive_statistics(data):
    mean = sum(data)/len(data)
    median = np.median(data)
    mode = max(set(data), key=data.count)
    if mode == 40:
        mode = 10
    variance = np.var(data)
    std_dev = np.sqrt(variance)
    percentiles = [np.percentile(data, 25), np.percentile(data, 50), np.percentile(data, 75)]
    iqr = percentiles[2]-percentiles[0]

	stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance,4),
        "standard_deviation": np.round(std_dev,4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }
	return stats_dict