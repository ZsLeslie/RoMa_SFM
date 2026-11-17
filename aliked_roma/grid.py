"""Original-image grids, half-pixel geometry and deterministic spatial quotas.

Pure NumPy: no model, torch, file writes, or keypoint ID assignment here.
Points accepted by the selector use original-image COLMAP coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor

import numpy as np

GRID_VERSION = 'full_image_aliked_masked_grid_v2'


@dataclass(frozen=True)
class GridCell:
    index: int
    core: tuple[int, int, int, int]  # x0,y0,x1,y1; half-open, original pixels
    crop: tuple[int, int, int, int]  # core plus clipped context
    active: bool
    quota: int = 0


def check_mask(mask, image_size):
    if mask is not None and (mask.dtype != np.bool_ or mask.shape != (image_size[1], image_size[0])):
        raise ValueError('Grid mask must be bool [original_height, original_width]')


def make_grid(image_size, *, grid_size, margin, max_keypoints, valid_mask=None):
    """Equal quotas for nonempty mask cells, y/x remainder, no redistribution.

    Partial cells at the right/bottom have the same quota as other active
    cells. A cell with any valid mask pixel is active; thin ROIs are retained.
    """
    w, h = image_size
    if min(w, h) <= 0 or grid_size < 1 or margin < 0 or max_keypoints < 1:
        raise ValueError('Invalid image dimensions, grid size, margin or keypoint budget')
    check_mask(valid_mask, image_size)
    if valid_mask is None and ceil(w/grid_size)*ceil(h/grid_size) > max_keypoints:
        raise ValueError('Active grid cells exceed max-keypoints; increase grid-size or max-keypoints')
    cells = []
    for y0 in range(0, h, grid_size):
        for x0 in range(0, w, grid_size):
            x1, y1 = min(w, x0+grid_size), min(h, y0+grid_size)
            active = valid_mask is None or bool(valid_mask[y0:y1, x0:x1].any())
            cells.append(GridCell(len(cells), (x0, y0, x1, y1),
                (max(0, x0-margin), max(0, y0-margin), min(w, x1+margin), min(h, y1+margin)), active))
    active_count = sum(c.active for c in cells)
    if active_count > max_keypoints:
        raise ValueError(f'{active_count} active grid cells exceed max-keypoints={max_keypoints}; increase grid-size or max-keypoints')
    if active_count:
        base, extra = divmod(max_keypoints, active_count)
        order = 0
        for i, cell in enumerate(cells):
            if cell.active:
                cells[i] = replace(cell, quota=base+int(order < extra))
                order += 1
    return cells


def inference_crop(cell, scale, image_size):
    """Map original context to the shared resized raster; extend tiny crops.

    ALIKED internally pads to multiples of 32. Keep real context of at least
    32 inference pixels per side, including a narrow final row or column.
    This can extend a user-specified margin near a very small edge cell.
    """
    if min(image_size) < 32:
        raise ValueError('Shared inference image must be at least 32 pixels on each side')
    bounds = []
    for axis in (0, 1):
        limit = image_size[axis]
        lo = max(0, min(limit, floor(cell.crop[axis]*scale[axis])))
        hi = max(lo, min(limit, ceil(cell.crop[axis+2]*scale[axis])))
        if hi-lo < 32:
            lo = max(0, min(lo-(32-(hi-lo))//2, limit-32))
            hi = lo+32
        bounds.append((lo, hi))
    return bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]


def original_colmap_points(local_points, crop_origin, scale):
    """ALIKED zero-centered crop points -> original COLMAP pixel centers."""
    p = np.asarray(local_points, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid crop points or resize scale')
    return ((p+.5+np.asarray(crop_origin))/scale).astype(np.float32)


def select_candidates(points, scores, cell_ids, cells, *, image_size, scale,
                      detection_threshold, nms_radius, valid_mask=None, descriptors=None):
    """Mask/core/quality filtering, joint per-cell quotas and cross-cell NMS.

    Only selected candidates suppress points in another cell. Thus points
    beyond an already-full cell's quota cannot starve a neighboring cell.
    NMS uses the resized raster's L-infinity radius, matching ALIKED's square
    NMS neighborhood. All output rows refer to the input arrays unchanged.
    """
    points, scores, cell_ids = np.asarray(points), np.asarray(scores), np.asarray(cell_ids)
    n = len(points)
    if (points.shape != (n, 2) or scores.shape != (n,) or cell_ids.shape != (n,)
            or not np.issubdtype(cell_ids.dtype, np.integer)):
        raise ValueError('Invalid grid candidate array shapes or cell IDs')
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any() or nms_radius < 1:
        raise ValueError('Invalid resize scales or cross-cell NMS radius')
    check_mask(valid_mask, image_size)
    if (cell_ids < 0).any() or (cell_ids >= len(cells)).any():
        raise ValueError('Unknown grid cell ID')
    valid = (np.isfinite(points).all(1) & np.isfinite(scores) & (scores >= detection_threshold)
             & (scores <= 1) & (points >= 0).all(1) & (points < image_size).all(1))
    if descriptors is not None:
        if descriptors.ndim != 2 or len(descriptors) != n:
            raise ValueError('Grid descriptors must align with candidate rows')
        valid &= np.isfinite(descriptors).all(1)
    if n:
        cores = np.asarray([c.core for c in cells])[cell_ids]
        valid &= (points >= cores[:, :2]).all(1) & (points < cores[:, 2:]).all(1)
        valid &= np.asarray([c.quota for c in cells])[cell_ids] > 0
    rows = np.flatnonzero(valid)
    if valid_mask is not None:
        pixels = np.floor(points[rows]).astype(np.int64)
        rows = rows[valid_mask[pixels[:, 1], pixels[:, 0]]]
    # Points/score resolve ties before cell ID. Input traversal order has no
    # effect on distinct candidates; identical rows give identical features.
    rows = rows[np.lexsort((cell_ids[rows], points[rows, 0], points[rows, 1], -scores[rows]))]
    counts = np.zeros(len(cells), np.int64)
    buckets, selected = {}, []
    xy = points.astype(np.float64)*scale
    for row in rows:
        cid = int(cell_ids[row])
        if counts[cid] >= cells[cid].quota:
            continue
        x, y = xy[row]
        bx, by = floor(x/nms_radius), floor(y/nms_radius)
        conflict = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((bx+dx, by+dy), ()):
                    if cell_ids[other] != cid and np.max(np.abs(xy[other]-xy[row])) <= nms_radius:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            selected.append(row)
            counts[cid] += 1
            buckets.setdefault((bx, by), []).append(row)
    return np.asarray(selected, dtype=np.int64)


def grid_summary(cells, points):
    counts = [int(((points >= c.core[:2]).all(1) & (points < c.core[2:]).all(1)).sum()) for c in cells]
    quotas = [c.quota for c in cells]
    if sum(counts) != len(points):
        raise ValueError('Saved keypoints do not belong to exactly one grid core')
    if any(n > q for n, q in zip(counts, quotas)):
        raise ValueError('Saved keypoints exceed a grid quota')
    return dict(cells=len(cells), active_cells=sum(c.active for c in cells),
                occupied_cells=sum(n > 0 for n in counts), quotas=quotas, selected_per_cell=counts)
