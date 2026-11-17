"""Immutable pair results referencing exact fixed-keypoint bank identities."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .features import canonical_json, check_name

MATCH_SCHEMA = 1


def canonical_pair(a: str, b: str):
    check_name(a)
    check_name(b)
    if a == b:
        raise ValueError("An image cannot be paired with itself")
    return tuple(sorted((a, b)))


def pair_id(a, b):
    a, b = canonical_pair(a, b)
    return sha256((a+'\0'+b).encode('utf-8')).hexdigest()


def match_path(root, a, b):
    root = Path(root).resolve()
    path = root/'matches'/(pair_id(a, b)+'.npz')
    if not path.resolve().is_relative_to(root):
        raise ValueError("Match output escapes its directory")
    return path


@dataclass
class PairMatches:
    metadata: dict
    matches: np.ndarray
    scores: np.ndarray
    assignment_error_px: np.ndarray
    cycle_error_px: np.ndarray
    inliers: np.ndarray
    fundamental: np.ndarray | None

    def validate(self):
        meta = self.metadata
        if tuple(meta['images']) != canonical_pair(*meta['images']):
            raise ValueError("Stored image pair must have canonical orientation")
        for identity in [meta['feature_set_id'], meta['run_id'], *meta['feature_ids']]:
            if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError("Invalid feature or run identity")
        counts = meta['keypoint_counts']
        if len(counts) != 2 or any(not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError("Invalid keypoint counts")
        n = len(self.matches)
        if self.matches.dtype != np.uint32 or self.matches.shape != (n, 2):
            raise ValueError("matches must be uint32 [N,2] fixed keypoint IDs")
        if n and ((self.matches >= counts).any() or any(len(np.unique(self.matches[:, c])) != n for c in (0, 1))):
            raise ValueError("Out-of-range or non-one-to-one keypoint IDs")
        for arr, shape in [(self.scores, (n,)), (self.assignment_error_px, (n, 2)), (self.cycle_error_px, (n, 2))]:
            if arr.dtype != np.float32 or arr.shape != shape or not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError("Invalid match scores or error measurements")
        if (self.scores > 1).any() or self.inliers.dtype != np.bool_ or self.inliers.shape != (n,):
            raise ValueError("Invalid confidence or inlier mask")
        if self.fundamental is not None:
            if self.fundamental.dtype != np.float64 or self.fundamental.shape != (3, 3) or not np.isfinite(self.fundamental).all():
                raise ValueError("Invalid fundamental matrix")
        elif self.inliers.any():
            raise ValueError("Inliers require an estimated fundamental matrix")

    @property
    def content_id(self):
        h = sha256(canonical_json(self.metadata).encode('utf-8'))
        for name in ['matches', 'scores', 'assignment_error_px', 'cycle_error_px', 'inliers', 'fundamental']:
            arr = getattr(self, name)
            if arr is None:
                h.update((name+':none').encode())
            else:
                h.update(canonical_json([name, arr.dtype.str, arr.shape]).encode())
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def bind(self, features0, features1, run_id=None):
        self.validate()
        if self.metadata['images'] != [features0.image_name, features1.image_name]:
            raise ValueError("Pair orientation differs from the supplied feature banks")
        if self.metadata['feature_ids'] != [features0.feature_id, features1.feature_id]:
            raise ValueError("Pair refers to different fixed keypoints; use a new match output directory")
        if self.metadata['keypoint_counts'] != [len(features0.keypoints), len(features1.keypoints)]:
            raise ValueError("Pair keypoint counts differ from the supplied feature banks")
        if self.metadata['feature_set_id'] != features0.feature_set_id or features0.feature_set_id != features1.feature_set_id:
            raise ValueError("Pair and fixed feature set identities differ")
        if run_id is not None and self.metadata['run_id'] != run_id:
            raise ValueError("Pair matching configuration differs; use a new output directory")


def save_matches(root, pair: PairMatches):
    pair.validate()
    path = match_path(root, *pair.metadata['images'])
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dict(schema_version=np.int32(MATCH_SCHEMA), metadata_json=np.str_(canonical_json(pair.metadata)),
                  matches=pair.matches, scores=pair.scores, assignment_error_px=pair.assignment_error_px,
                  cycle_error_px=pair.cycle_error_px, inliers=pair.inliers, content_id=np.str_(pair.content_id))
    if pair.fundamental is not None:
        values['fundamental'] = pair.fundamental
    fd, tmp = tempfile.mkstemp(prefix='.pair_', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **values)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    return path


def load_matches(root, a, b):
    """Return canonical (lexicographically sorted image-name) orientation."""
    path = match_path(root, a, b)
    with np.load(path, allow_pickle=False) as data:
        if int(data['schema_version']) != MATCH_SCHEMA:
            raise ValueError(f"Unsupported match schema: {path}")
        result = PairMatches(metadata=json.loads(str(data['metadata_json'])), matches=data['matches'],
            scores=data['scores'], assignment_error_px=data['assignment_error_px'],
            cycle_error_px=data['cycle_error_px'], inliers=data['inliers'],
            fundamental=data['fundamental'] if 'fundamental' in data else None)
        expected = str(data['content_id'])
    result.validate()
    if tuple(result.metadata['images']) != canonical_pair(a, b) or result.content_id != expected:
        raise ValueError(f"Match content identity mismatch: {path}")
    return result
