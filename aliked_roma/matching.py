"""Associate RoMa predictions with fixed IDs, in original-image pixels."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Predictions:
    forward_xy: np.ndarray          # image-0 anchors projected into image 1
    backward_xy: np.ndarray         # image-1 anchors projected into image 0
    confidence0: np.ndarray         # min of forward and return-path certainty
    confidence1: np.ndarray
    cycle_error0: np.ndarray        # unsnapped forward-backward error, image-0 px
    cycle_error1: np.ndarray        # unsnapped backward-forward error, image-1 px


def nearest_anchors(predicted, anchors, *, max_distance: float, ratio: float):
    """Return anchor IDs or -1. Missing/ambiguous predictions stay unmatched."""
    from scipy.spatial import cKDTree

    predicted = np.asarray(predicted)
    anchors = np.asarray(anchors)
    ids = np.full(len(predicted), -1, dtype=np.int64)
    errors = np.full(len(predicted), np.inf, dtype=np.float32)
    rows = np.flatnonzero(np.isfinite(predicted).all(1))
    if not len(anchors) or not len(rows):
        return ids, errors
    distances, neighbors = cKDTree(anchors).query(predicted[rows], k=2)
    # Strict inequality rejects exact distance ties, also when ratio=1.
    valid = (distances[:, 0] <= max_distance) & (distances[:, 0] < ratio * distances[:, 1])
    ids[rows[valid]] = neighbors[valid, 0]
    errors[rows] = distances[:, 0].astype(np.float32)
    return ids, errors


def associate(features0, features1, predictions: Predictions, *, max_distance=3.,
              nn_ratio=.8, min_confidence=.2, max_cycle_error=3.):
    """Require actual A->B and B->A anchor agreement, plus dense flow cycles.

    No coordinate is changed. Returned indices refer to rows of the original
    saved ALIKED banks, including when some rows are rejected.
    """
    p = predictions
    n0, n1 = len(features0.keypoints), len(features1.keypoints)
    for xy, conf, cycle, n in [(p.forward_xy, p.confidence0, p.cycle_error0, n0),
                                (p.backward_xy, p.confidence1, p.cycle_error1, n1)]:
        if np.shape(xy) != (n, 2) or np.shape(conf) != (n,) or np.shape(cycle) != (n,):
            raise ValueError("RoMa prediction lengths do not match the fixed feature banks")
    j_for_i, error0 = nearest_anchors(p.forward_xy, features1.keypoints, max_distance=max_distance, ratio=nn_ratio)
    i_for_j, error1 = nearest_anchors(p.backward_xy, features0.keypoints, max_distance=max_distance, ratio=nn_ratio)
    valid0 = ((j_for_i >= 0) & np.isfinite(p.confidence0) & (p.confidence0 >= min_confidence)
              & (p.confidence0 <= 1) & np.isfinite(p.cycle_error0)
              & (p.cycle_error0 >= 0) & (p.cycle_error0 <= max_cycle_error))
    valid1 = ((i_for_j >= 0) & np.isfinite(p.confidence1) & (p.confidence1 >= min_confidence)
              & (p.confidence1 <= 1) & np.isfinite(p.cycle_error1)
              & (p.cycle_error1 >= 0) & (p.cycle_error1 <= max_cycle_error))
    rows0 = np.flatnonzero(valid0)
    rows1 = j_for_i[rows0]
    mutual = valid1[rows1] & (i_for_j[rows1] == rows0)
    rows0, rows1 = rows0[mutual], rows1[mutual]
    matches = np.column_stack((features0.keypoint_ids[rows0], features1.keypoint_ids[rows1])).astype(np.uint32)
    scores = np.minimum(p.confidence0[rows0], p.confidence1[rows1]).astype(np.float32)
    assignment_errors = np.column_stack((error0[rows0], error1[rows1])).astype(np.float32)
    cycle_errors = np.column_stack((p.cycle_error0[rows0], p.cycle_error1[rows1])).astype(np.float32)
    stats = dict(forward_candidates=int(valid0.sum()), backward_candidates=int(valid1.sum()), reciprocal=len(matches))
    return matches, scores, assignment_errors, cycle_errors, stats


def verify_fundamental(points0, points1, *, threshold=2., confidence=.999, max_iters=10000, seed=42):
    """Initial F-only screening on the FIXED coordinates, not predicted flow.

    This is not a calibrated COLMAP two-view geometry. E/F/H verification must
    still be performed when a COLMAP database is built in a subsequent stage.
    """
    import cv2

    empty = np.zeros(len(points0), dtype=np.bool_)
    if len(points0) < 8:
        return None, empty
    cv2.setRNGSeed(int(seed) & 0x7fffffff)
    try:
        matrix, mask = cv2.findFundamentalMat(
            np.ascontiguousarray(points0, dtype=np.float32),
            np.ascontiguousarray(points1, dtype=np.float32), method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters)
    except cv2.error:
        return None, empty
    if (matrix is None or mask is None or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or mask.size != len(points0)):
        return None, empty
    return np.asarray(matrix, dtype=np.float64), mask.reshape(-1).astype(np.bool_)
