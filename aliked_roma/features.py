"""Immutable feature files. No torch or dependency on the old src/ pipeline.

Coordinates use the original image and COLMAP/RoMa convention: the upper-left
pixel center is (0.5, 0.5). A keypoint ID is its row in this saved bank, scoped
by (feature_set_id, image_name). Consumers must never filter and renumber it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np

SCHEMA_VERSION = 1
COORDINATES = "original_image_colmap_pixel_centers_0.5"


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def check_name(name: str) -> str:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or p.as_posix() != name or name == ".":
        raise ValueError(f"Expected a canonical relative image name, got {name!r}")
    return name


def feature_path(output_dir: Path, image_name: str) -> Path:
    root = Path(output_dir).resolve()
    path = root / "features" / (check_name(image_name) + ".npz")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Feature path escapes output directory: {path}")
    return path


@dataclass(frozen=True)
class ImageFeatures:
    image_name: str
    image_size: np.ndarray          # int64 [width, height]
    keypoint_ids: np.ndarray        # uint32 [N], exactly 0..N-1
    keypoints: np.ndarray           # float32 [N,2], COLMAP coordinates
    scores: np.ndarray              # float32 [N], extractor-specific scores
    descriptors: np.ndarray | None  # optional float32 [N,D]
    source_sha256: str
    mask_sha256: str                # empty when masks are disabled
    feature_set_id: str

    def validate(self) -> None:
        check_name(self.image_name)
        if self.image_size.dtype != np.int64 or self.image_size.shape != (2,) or (self.image_size <= 0).any():
            raise ValueError("image_size must be int64 [width,height] with positive dimensions")
        n = len(self.keypoints)
        if self.keypoints.dtype != np.float32 or self.keypoints.shape != (n, 2):
            raise ValueError("keypoints must be float32 [N,2]")
        if self.keypoint_ids.dtype != np.uint32 or not np.array_equal(self.keypoint_ids, np.arange(n, dtype=np.uint32)):
            raise ValueError("Keypoint IDs must equal their saved row indices")
        if self.scores.dtype != np.float32 or self.scores.shape != (n,):
            raise ValueError("scores must be float32 [N]")
        if not np.isfinite(self.keypoints).all() or not np.isfinite(self.scores).all():
            raise ValueError("Non-finite feature coordinates or scores")
        if ((self.keypoints < 0) | (self.keypoints >= self.image_size)).any():
            raise ValueError("Keypoints must lie inside the original image")
        if ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("Detection scores must be within [0,1]")
        if self.descriptors is not None:
            d = self.descriptors
            if d.dtype != np.float32 or d.ndim != 2 or d.shape[0] != n or d.shape[1] <= 0 or not np.isfinite(d).all():
                raise ValueError("descriptors must be finite float32 [N,D]")
        for name, value in [("source_sha256", self.source_sha256), ("mask_sha256", self.mask_sha256), ("feature_set_id", self.feature_set_id)]:
            if name == "mask_sha256" and value == "":
                continue
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {name}")

    @property
    def feature_id(self) -> str:
        """Content identity to bind future pair matches to this exact bank."""
        meta = dict(schema_version=SCHEMA_VERSION, coordinates=COORDINATES,
                    image_name=self.image_name, source_sha256=self.source_sha256,
                    mask_sha256=self.mask_sha256, feature_set_id=self.feature_set_id)
        h = sha256(canonical_json(meta).encode("utf-8"))
        for name in ("image_size", "keypoint_ids", "keypoints", "scores", "descriptors"):
            array = getattr(self, name)
            if array is None:
                h.update(f"{name}:none\n".encode())
            else:
                h.update(canonical_json([name, array.dtype.str, array.shape]).encode())
                h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()

    def roma_keypoints(self) -> np.ndarray:
        """[-1,1] coordinates for RoMa; saved IDs/coordinates are untouched."""
        return (2 * self.keypoints / self.image_size.astype(np.float32) - 1).astype(np.float32)

    def aliked_keypoints(self) -> np.ndarray:
        """Zero-centered pixel coordinates expected by LightGlue/ALIKED."""
        return self.keypoints - np.float32(0.5)


def prepare_features(*, image_name: str, image_size: tuple[int, int],
                     keypoints: np.ndarray, scores: np.ndarray,
                     descriptors: np.ndarray | None, source_sha256: str,
                     mask_sha256: str, feature_set_id: str,
                     detection_threshold: float, max_keypoints: int,
                     valid_mask: np.ndarray | None = None) -> ImageFeatures:
    """Convert adapters' original-image, zero-centered coordinates once.

    IDs are assigned only AFTER filtering and deterministic score/y/x sorting.
    ALIKED internally falls back to a mean threshold on low-texture images;
    applying the requested threshold again prevents weak points being forced in.
    """
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or scores.shape != (len(points),):
        raise ValueError("Unexpected detector keypoint/score shapes")
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(points):
            raise ValueError("Unexpected detector descriptor shape")
    size = np.asarray(image_size, dtype=np.int64)
    if size.shape != (2,) or (size <= 0).any():
        raise ValueError("Invalid original image dimensions")
    points = points + np.float32(0.5)
    keep = (np.isfinite(points).all(1) & np.isfinite(scores)
            & (scores >= detection_threshold) & (scores <= 1)
            & (points >= 0).all(1) & (points < size).all(1))
    if descriptors is not None:
        keep &= np.isfinite(descriptors).all(1)
    rows = np.flatnonzero(keep)
    if valid_mask is not None:
        if valid_mask.shape != (image_size[1], image_size[0]) or valid_mask.dtype != np.bool_:
            raise ValueError("Mask must be bool [original_height,original_width]")
        pixels = np.floor(points[rows]).astype(np.int64)
        rows = rows[valid_mask[pixels[:, 1], pixels[:, 0]]]
    order = np.lexsort((points[rows, 0], points[rows, 1], -scores[rows]))
    rows = rows[order[:max_keypoints]]
    result = ImageFeatures(
        image_name=image_name, image_size=size,
        keypoint_ids=np.arange(len(rows), dtype=np.uint32),
        keypoints=np.ascontiguousarray(points[rows]), scores=np.ascontiguousarray(scores[rows]),
        descriptors=np.ascontiguousarray(descriptors[rows]) if descriptors is not None else None,
        source_sha256=source_sha256, mask_sha256=mask_sha256, feature_set_id=feature_set_id)
    result.validate()
    return result


def save_features(output_dir: Path, features: ImageFeatures) -> Path:
    """Atomically publish a NEW bank, never replace existing IDs."""
    features.validate()
    path = feature_path(output_dir, features.image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = dict(schema_version=np.int32(SCHEMA_VERSION), coordinates=np.str_(COORDINATES),
                  image_name=np.str_(features.image_name), image_size=features.image_size,
                  keypoint_ids=features.keypoint_ids, keypoints=features.keypoints, scores=features.scores,
                  source_sha256=np.str_(features.source_sha256), mask_sha256=np.str_(features.mask_sha256),
                  feature_set_id=np.str_(features.feature_set_id), feature_id=np.str_(features.feature_id))
    if features.descriptors is not None:
        fields["descriptors"] = features.descriptors
    fd, tmp = tempfile.mkstemp(prefix=".features_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **fields)
            f.flush()
            os.fsync(f.fileno())
        # Hard-link publication is atomic and fails if the destination exists.
        # Both files are on the same filesystem. Readers only see complete NPZs.
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    return path


def load_features(output_dir: Path, image_name: str, *, feature_set_id: str | None = None) -> ImageFeatures:
    path = feature_path(output_dir, image_name)
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != SCHEMA_VERSION or str(data["coordinates"]) != COORDINATES:
            raise ValueError(f"Unsupported feature schema or coordinate convention: {path}")
        result = ImageFeatures(
            image_name=str(data["image_name"]), image_size=data["image_size"],
            keypoint_ids=data["keypoint_ids"], keypoints=data["keypoints"], scores=data["scores"],
            descriptors=data["descriptors"] if "descriptors" in data else None,
            source_sha256=str(data["source_sha256"]), mask_sha256=str(data["mask_sha256"]),
            feature_set_id=str(data["feature_set_id"]))
        stored_id = str(data["feature_id"])
    result.validate()
    if result.image_name != image_name:
        raise ValueError(f"Image identity mismatch in {path}")
    if feature_set_id is not None and result.feature_set_id != feature_set_id:
        raise ValueError(f"Feature-set mismatch in {path}; use a new output directory")
    if result.feature_id != stored_id:
        raise ValueError(f"Feature content checksum mismatch in {path}")
    for array in (result.image_size, result.keypoint_ids, result.keypoints, result.scores, result.descriptors):
        if array is not None:
            array.setflags(write=False)
    return result
