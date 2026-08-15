"""Minimal point-cloud helpers used by the evaluation pipeline.

Adapted from HFCI-PU (Apache License 2.0).
"""

import numpy as np


def load(filename):
    """Load the first columns of a whitespace-delimited point-cloud file."""
    return np.loadtxt(filename).astype(np.float32)


def normalize_point_cloud(points):
    """Center a point cloud and scale it to the unit sphere."""
    if points.ndim == 2:
        axis = 0
    elif points.ndim == 3:
        axis = 1
    else:
        raise ValueError(f"Expected a 2D or 3D point array, got {points.shape}.")

    centroid = np.mean(points, axis=axis, keepdims=True)
    normalized = points - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(normalized ** 2, axis=-1, keepdims=True)),
        axis=axis,
        keepdims=True,
    )
    if np.any(furthest_distance == 0):
        raise ValueError("Cannot normalize a degenerate point cloud.")
    normalized = normalized / furthest_distance
    return normalized, centroid, furthest_distance
