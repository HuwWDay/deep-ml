import numpy as np

def k_means_clustering(points: list[tuple[float, float]], 
                       k: int, 
                       initial_centroids: list[tuple[float, float]], 
                       max_iterations: int) -> list[tuple[float, float]]:

    points = np.array(points)
    centroids = np.array(initial_centroids, dtype=float)
    n = len(points)

    for _ in range(max_iterations):
        # Step 1: Assign points to the nearest centroid
        point_assignment = []
        for j in range(n):
            distances = [np.linalg.norm(points[j] - centroid) for centroid in centroids]
            point_assignment.append(np.argmin(distances))

        point_assignment = np.array(point_assignment)

        # Step 2: Update centroids
        for i in range(k):
            assigned_points = points[point_assignment == i]
            if len(assigned_points) > 0:
                centroids[i] = assigned_points.mean(axis=0)

    return [tuple(c) for c in cent