"""Extract an immutable ALIKED or DeDoDe detector bank, once per image.

Run from the project root: python -m aliked_roma.extract --help
Or from this directory: python extract.py --help
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image

# Direct script execution has no package context. Establish it before
# both the feature imports below and the lazy ALIKED import inside run().
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .features import (COORDINATES, SCHEMA_VERSION, check_name, digest_json,
                       feature_path, load_features, prepare_features, save_features)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOG = logging.getLogger("fixed_sfm.extract")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, help="Default: sibling directory fixed_<extractor>")
    p.add_argument("--image-list", type=Path, help="Optional UTF-8 file, one relative image name per line")
    p.add_argument("--max-images", type=int, default=0, help="First N images in sorted order; 0 = all")
    p.add_argument("--extractor", choices=("aliked", "dedode"), default="aliked")
    p.add_argument("--model", choices=("aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32", "dedode-L-v2"))
    p.add_argument("--max-keypoints", type=int, default=4096, help="Whole-image maximum; grid mode divides it among active cells without refilling")
    p.add_argument("--detection-threshold", type=float, help="ALIKED default: 0.2; DeDoDe default: 0 (native top-K scores)")
    p.add_argument("--nms-radius", type=int, help="ALIKED default: 2; DeDoDe supports 0 only (native sampling)")
    p.add_argument("--resize-max", type=int, default=1024, help="Downscale long edge only; 0 = original resolution")
    p.add_argument("--grid-size", type=int, default=0, help="Square core size in ORIGINAL pixels; 0 = legacy whole-image extraction")
    p.add_argument("--grid-margin", type=int, default=64, help="V1 compatibility argument; ignored by full-image grid V2")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu; no silent CPU fallback")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--torch-hub-dir", type=Path, help="Offline weights: DIR/checkpoints/<model>.pth")
    p.add_argument("--dedode-weights", type=Path, default=Path(__file__).resolve().parents[1]/"models/dedode_detector_L_v2.pth",
                   help="Local DeDoDe v2-L detector checkpoint; never downloads weights")
    p.add_argument("--save-descriptors", action="store_true", help="Also save float32 ALIKED descriptors (not needed by RoMa)")
    p.add_argument("--masks-dir", type=Path, help="Optional masks: black=exclude, white=keep; all selected images must have one")
    p.add_argument("--mask-threshold", type=int, default=128)
    p.add_argument("--dry-run", action="store_true", help="List inputs without importing torch or writing files")
    args = p.parse_args(argv)
    if args.model is None:
        args.model = "dedode-L-v2" if args.extractor == "dedode" else "aliked-n16"
    if (args.model == "dedode-L-v2") != (args.extractor == "dedode"):
        p.error("--model must belong to the selected --extractor")
    if args.detection_threshold is None:
        args.detection_threshold = 0.0 if args.extractor == "dedode" else 0.2
    if args.nms_radius is None:
        args.nms_radius = 0 if args.extractor == "dedode" else 2
    if args.extractor == "dedode" and (args.save_descriptors or args.nms_radius != 0):
        p.error("DeDoDe supports detector-only extraction with --nms-radius 0; descriptors are not loaded")
    if args.max_images < 0 or not 0 < args.max_keypoints <= 20000:
        p.error("max-images must be >=0; max-keypoints must be in [1,20000]")
    if not 0 <= args.detection_threshold <= 1:
        p.error("detection-threshold must be in [0,1]")
    if args.extractor == "aliked" and (args.detection_threshold == 0 or args.nms_radius < 1):
        p.error("ALIKED requires detection-threshold >0 and nms-radius >=1")
    if args.resize_max != 0 and args.resize_max < 32:
        p.error("resize-max must be 0 or >=32")
    if args.grid_size < 0 or 0 < args.grid_size < 32 or args.grid_margin < 0:
        p.error("grid-size must be 0 or >=32 original pixels; grid-margin must be >=0")
    if args.num_threads < 1 or not 1 <= args.mask_threshold <= 255:
        p.error("num-threads must be positive; mask-threshold must be in [1,255]")
    for name in ("images_dir", "output_dir", "image_list", "torch_hub_dir", "masks_dir", "dedode_weights"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir is None:
        args.output_dir = args.images_dir.parent / f"fixed_{args.extractor}"
    if args.output_dir.is_relative_to(args.images_dir):
        p.error("output-dir must be outside images-dir to avoid treating output images as new input")
    if args.masks_dir is not None and args.masks_dir.is_relative_to(args.images_dir):
        p.error("masks-dir must be outside images-dir")
    return args


def source_path(root: Path, name: str) -> Path:
    path = (root / check_name(name)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input path escapes image root: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def collect_images(args) -> list[str]:
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if args.image_list is not None:
        names = [line.strip() for line in args.image_list.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(set(names)) != len(names):
            raise ValueError("image-list contains duplicate image names")
    else:
        names = [p.relative_to(args.images_dir).as_posix() for p in args.images_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    names.sort()
    if args.max_images:
        names = names[:args.max_images]
    if not names:
        raise ValueError("No input images selected")
    for name in names:
        source_path(args.images_dir, name)
    return names


def resolve_masks(root: Path | None, names: list[str]) -> dict[str, Path]:
    if root is None:
        return {}
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for name in names:
        rel = Path(name)
        for candidate in (rel.with_suffix(".png"), Path(name + ".png"), rel):
            path = (root / candidate).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Mask path escapes mask root: {name!r}")
            if path.is_file():
                result[name] = path
                break
        else:
            raise FileNotFoundError(f"Mask missing for {name!r} in {root}")
    return result


def read_mask(data: bytes | None, image_size, threshold: int):
    if data is None:
        return None
    with Image.open(BytesIO(data)) as im:
        if im.size != image_size:
            raise ValueError(f"Mask size {im.size} differs from original image {image_size}; no silent resizing")
        return np.asarray(im.convert("L")) >= threshold


def atomic_json(path: Path, value) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".json_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".extract.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another extraction is writing {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def prepare_output(root: Path, config: dict):
    set_id = digest_json(config)
    expected = dict(feature_set_id=set_id, config=config)
    path = root / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != expected:
            changed = sorted(k for k in set(config) | set(previous.get("config", {}))
                             if config.get(k) != previous.get("config", {}).get(k))
            raise ValueError(f"Feature configuration changed ({', '.join(changed)}). Use a NEW output directory; existing IDs cannot be replaced.")
    else:
        if any(root.glob("features/**/*.npz")) or (root / "index.json").exists():
            raise ValueError("Feature files/index exist without run_config.json; use a new output directory")
        atomic_json(path, expected)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else dict(schema_version=SCHEMA_VERSION, feature_set_id=set_id, images={})
    if index.get("schema_version") != SCHEMA_VERSION or index.get("feature_set_id") != set_id or not isinstance(index.get("images"), dict):
        raise ValueError("Feature index does not match this feature set")
    return set_id, index


def extract_one(args, backend, name: str, mask_path: Path | None, set_id: str):
    image_bytes = source_path(args.images_dir, name).read_bytes()
    mask_bytes = mask_path.read_bytes() if mask_path is not None else None
    source_hash = sha256(image_bytes).hexdigest()
    mask_hash = sha256(mask_bytes).hexdigest() if mask_bytes is not None else ""
    if feature_path(args.output_dir, name).exists():
        features = load_features(args.output_dir, name, feature_set_id=set_id)
        if features.source_sha256 != source_hash or features.mask_sha256 != mask_hash:
            raise ValueError(f"Image or mask changed for {name!r}. Use a NEW output directory; cached IDs are immutable.")
        status = "cached"
    else:
        extra = {}
        if args.grid_size or args.extractor == 'dedode':
            with Image.open(BytesIO(image_bytes)) as image:
                size = image.size
            # Validate the mask before any expensive model inference.
            extra['valid_mask'] = read_mask(mask_bytes, size, args.mask_threshold)
        size, points, scores, descriptors = backend.extract(image_bytes, save_descriptors=args.save_descriptors, **extra)
        features = prepare_features(
            image_name=name, image_size=size, keypoints=points, scores=scores, descriptors=descriptors,
            source_sha256=source_hash, mask_sha256=mask_hash, feature_set_id=set_id,
            detection_threshold=args.detection_threshold, max_keypoints=args.max_keypoints,
            valid_mask=read_mask(mask_bytes, size, args.mask_threshold))
        # Enforce/record quota invariants before publishing immutable features.
        if args.grid_size:
            backend.last_grid_stats = describe_grid(args, features, mask_bytes)
        save_features(args.output_dir, features)
        status = "extracted"
    if args.grid_size and status == 'cached':
        backend.last_grid_stats = describe_grid(args, features, mask_bytes)
    return features, status


def describe_grid(args, features, mask_bytes):
    from .grid import grid_summary, make_grid
    size = tuple(int(v) for v in features.image_size)
    cells = make_grid(size, grid_size=args.grid_size, margin=0,
                      max_keypoints=args.max_keypoints,
                      valid_mask=read_mask(mask_bytes, size, args.mask_threshold))
    return grid_summary(cells, features.keypoints)


def write_index(root: Path, index: dict) -> None:
    index["images"] = dict(sorted(index["images"].items()))
    atomic_json(root / "index.json", index)
    # This derived CSV is for inspection; index/NPZ are authoritative.
    fd, tmp = tempfile.mkstemp(prefix=".stats_", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_name", "width", "height", "keypoints", "feature_id", "feature_file"])
            for name, row in index["images"].items():
                w.writerow([name, *row["image_size"], row["keypoints"], row["feature_id"], row["feature_file"]])
        os.replace(tmp, root / "statistics.csv")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run(args, *, backend_factory=None) -> int:
    names = collect_images(args)
    masks = resolve_masks(args.masks_dir, names)
    print(f"Images: {len(names)} | output: {args.output_dir}", flush=True)
    if args.dry_run:
        print("\n".join(names[:12]))
        if len(names) > 12:
            print(f"... {len(names)-12} more")
        return 0
    if backend_factory is None:
        if args.extractor == 'dedode':
            from .dedode import DeDoDeExtractor
            backend_factory = DeDoDeExtractor
        else:
            from .aliked import AlikedExtractor
            backend_factory = AlikedExtractor
    extra = dict(grid_size=args.grid_size) if args.grid_size else {}
    if args.extractor == 'dedode':
        extra['weights_path'] = args.dedode_weights
    backend = backend_factory(model_name=args.model, max_keypoints=args.max_keypoints,
                              detection_threshold=args.detection_threshold, nms_radius=args.nms_radius,
                              resize_max=args.resize_max, device=args.device,
                              torch_hub_dir=args.torch_hub_dir, num_threads=args.num_threads, **extra)
    config = dict(schema_version=SCHEMA_VERSION, coordinates=COORDINATES,
                  images_dir=str(args.images_dir), masks_dir=str(args.masks_dir) if args.masks_dir else None,
                  mask_threshold=args.mask_threshold, save_descriptors=args.save_descriptors,
                  ordering="score_desc_y_asc_x_asc", extractor=backend.signature)
    with output_lock(args.output_dir):
        set_id, index = prepare_output(args.output_dir, config)
        LOG.setLevel(logging.INFO)
        LOG.propagate = False
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "extraction.log", encoding="utf-8")]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOG.addHandler(handler)
        counts = {"extracted": 0, "cached": 0}
        started, point_counts = perf_counter(), []
        run_status = "failed"
        try:
            LOG.info("Feature set %s; device=%s; coordinates=%s", set_id, args.device, COORDINATES)
            if args.grid_size:
                LOG.info('One full-image dense forward with masked spatial quotas; --grid-margin is ignored')
            for i, name in enumerate(names, 1):
                t = perf_counter()
                features, status = extract_one(args, backend, name, masks.get(name), set_id)
                n = len(features.keypoints)
                index["images"][name] = dict(feature_file=feature_path(args.output_dir, name).relative_to(args.output_dir).as_posix(),
                                            image_size=features.image_size.tolist(), keypoints=n, feature_id=features.feature_id,
                                            source_sha256=features.source_sha256, mask_sha256=features.mask_sha256)
                if args.grid_size:
                    stats = backend.last_grid_stats
                    index['images'][name]['grid'] = stats
                    LOG.info('Grid: cells=%d active=%d occupied=%d', stats['cells'], stats['active_cells'], stats['occupied_cells'])
                counts[status] += 1
                point_counts.append(n)
                LOG.info("[%d/%d] %s | %s | keypoints=%d | %.3fs", i, len(names), name, status, n, perf_counter()-t)
            run_status = "ok"
            LOG.info("Finished: extracted=%d cached=%d keypoints=%d", counts["extracted"], counts["cached"], sum(point_counts))
        except KeyboardInterrupt:
            run_status = "interrupted"
            LOG.warning("Interrupted; completed feature files remain valid")
            raise
        except Exception:
            LOG.exception("Extraction stopped; completed feature files remain unchanged")
            raise
        finally:
            try:
                write_index(args.output_dir, index)
                atomic_json(args.output_dir / "last_run.json", dict(status=run_status, selected=len(names), **counts,
                            total_keypoints=sum(point_counts), min_keypoints=min(point_counts) if point_counts else None,
                            median_keypoints=float(np.median(point_counts)) if point_counts else None,
                            max_keypoints=max(point_counts) if point_counts else None,
                            elapsed_seconds=perf_counter()-started))
            finally:
                for handler in handlers:
                    LOG.removeHandler(handler)
                    handler.close()
    return 0


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted; completed feature banks can be resumed unchanged.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
