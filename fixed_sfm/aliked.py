"""Offline ALIKED adapter using the installed LightGlue implementation."""
from __future__ import annotations

from hashlib import sha256
import inspect
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .grid import (GRID_VERSION, grid_summary, make_grid,
                   original_colmap_points, select_candidates)


class AlikedExtractor:
    def __init__(self, *, model_name: str, max_keypoints: int, detection_threshold: float,
                 nms_radius: int, resize_max: int, device: str,
                 torch_hub_dir: Path | None, num_threads: int,
                 grid_size: int = 0, grid_margin: int = 64):
        try:
            import torch
            import torchvision
            import kornia
            from lightglue import ALIKED
            from lightglue import utils
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "ALIKED needs a compatible torch/torchvision/kornia/LightGlue environment. "
                "On this machine use /home/nvidia/venvs/2dgs_env/bin/python; "
                "do not replace Jetson's torch installation."
            ) from exc
        self.torch, self.model_class = torch, ALIKED
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda[:index] devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the driver/environment, or explicitly use --device cpu.")
        torch.set_num_threads(num_threads)
        torch.manual_seed(42)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        if torch_hub_dir is not None:
            torch.hub.set_dir(str(torch_hub_dir))
        self.weights_path = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}.pth"
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Offline ALIKED checkpoint not found: {self.weights_path}. "
                "Place the official checkpoint there or use --torch-hub-dir DIR "
                "with DIR/checkpoints/<model-name>.pth. No weights are downloaded automatically."
            )
        if grid_size < 0 or grid_margin < 0:
            raise ValueError('grid-size and grid-margin must be nonnegative')
        self.grid_size, self.grid_margin = grid_size, grid_margin
        self.max_keypoints = max_keypoints
        self.detection_threshold, self.nms_radius = detection_threshold, nms_radius
        # Grid V2 selects from all full-image NMS peaks. The native DKD's
        # internal n_limit is temporarily raised to the score-map size below.
        # Legacy whole-image mode keeps its original candidate limit/signature.
        candidate_limit = -1 if grid_size else max_keypoints
        self.model_conf = dict(model_name=model_name, max_num_keypoints=candidate_limit,
                               detection_threshold=detection_threshold, nms_radius=nms_radius)
        self.resize_max, self.model = resize_max, None
        self.signature = dict(
            extractor="lightglue.ALIKED", model=self.model_conf,
            resize_max=resize_max, resize_policy="long_edge_downscale_only",
            precision="float32", device=str(self.device), num_threads=num_threads, seed=42,
            weights_sha256=sha256(self.weights_path.read_bytes()).hexdigest(),
            implementation_sha256={
                "aliked.py": sha256(Path(inspect.getfile(ALIKED)).read_bytes()).hexdigest(),
                "utils.py": sha256(Path(utils.__file__).read_bytes()).hexdigest(),
            },
            runtime=dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                         kornia=str(kornia.__version__), numpy=str(np.__version__), pillow=Image.__version__),
        )
        if grid_size:
            from . import grid
            self.signature['grid'] = dict(
                version=GRID_VERSION, size_original_px=grid_size,
                max_keypoints_total=max_keypoints, candidate_limit='all_full_image_nms_peaks',
                quota_policy='equal_nonempty_mask_cells_yx_remainder_no_refill',
                inference_policy='one_full_image_dense_map_forward',
                resize_policy='full_image_long_edge_downscale_only',
                descriptor_policy='selected_points_only_when_requested_invalid_drop_no_refill',
                ownership='half_open_original_colmap_core',
                cross_cell_nms='linf_in_inference_pixels_same_radius_as_aliked',
                adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                selector_sha256=sha256(Path(grid.__file__).read_bytes()).hexdigest())

    def extract(self, image_bytes: bytes, *, save_descriptors: bool, valid_mask=None):
        if self.grid_size:
            return self._extract_grid(image_bytes, save_descriptors=save_descriptors, valid_mask=valid_mask)
        torch = self.torch
        if self.model is None:
            # ALIKED's constructor loads this already-validated local hub cache.
            self.model = self.model_class(**self.model_conf).eval().to(self.device)
        with Image.open(BytesIO(image_bytes)) as source:
            # Keep the stored raster orientation; masks/intrinsics use that grid.
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        if min(width, height) < 32:
            raise ValueError(f"ALIKED requires images at least 32 pixels on each side: {width}x{height}")
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32) / 255.0
        resize = self.resize_max if self.resize_max and max(width, height) > self.resize_max else None
        if resize and min(width, height) * resize / max(width, height) < 32:
            raise ValueError("Resizing would make the short edge smaller than 32 pixels; increase --resize-max or use 0")
        with torch.inference_mode():
            # extract() maps points back to ORIGINAL image coordinates, with
            # zero-centered pixels. features.prepare_features adds 0.5 once.
            pred = self.model.extract(tensor, resize=resize)
        points = pred["keypoints"][0].float().cpu().numpy()
        scores = pred["keypoint_scores"][0].float().cpu().numpy()
        descriptors = pred["descriptors"][0].float().cpu().numpy() if save_descriptors else None
        return (width, height), points, scores, descriptors

    def _extract_grid(self, image_bytes, *, save_descriptors, valid_mask):
        torch = self.torch
        with Image.open(BytesIO(image_bytes)) as source:
            rgb = np.asarray(source.convert('RGB'), dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        if min(width, height) < 32:
            raise ValueError('ALIKED requires original images at least 32 pixels on each side')
        cells = make_grid((width, height), grid_size=self.grid_size, margin=0,
                          max_keypoints=self.max_keypoints, valid_mask=valid_mask)
        descriptor_dim = self.model_class.cfgs[self.model_conf['model_name']][4]
        if not any(c.active for c in cells):
            empty = np.empty((0, 2), np.float32)
            self.last_grid_stats = grid_summary(cells, empty)
            return ((width, height), empty, np.empty(0, np.float32),
                    np.empty((0, descriptor_dim), np.float32) if save_descriptors else None)
        if self.model is None:
            self.model = self.model_class(**self.model_conf).eval().to(self.device)
        from lightglue.utils import ImagePreprocessor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32)/255.
        resize = self.resize_max if self.resize_max and max(width, height) > self.resize_max else None
        if resize and min(width, height)*resize/max(width, height) < 32:
            raise ValueError('Shared resizing makes the short edge smaller than 32 pixels; increase --resize-max')
        with torch.inference_mode():
            # Preserve the full-image feature context and the installed
            # preprocessor's exact rounded x/y sizes and antialiasing.
            tensor, _ = ImagePreprocessor(resize=resize)(tensor[None])
            ih, iw = tensor.shape[-2:]
            scale = np.array([iw/width, ih/height], np.float64)
            feature_map, score_map = self.model.extract_dense_map(tensor)
            if not save_descriptors:
                del feature_map
            # No global top-K before masking/spatial quotas. Calling the
            # installed DKD preserves its global NMS, subpixel refinement and
            # score sampling; raising only n_limit bypasses its global cap.
            detector = self.model.dkd
            prior_limit = detector.n_limit
            try:
                detector.n_limit = ih*iw
                keypoints, keypoint_scores, _ = detector(score_map)
            finally:
                detector.n_limit = prior_limit
            normalized = keypoints[0]
            local = (normalized+1)*normalized.new_tensor([iw-1, ih-1])/2
            points = original_colmap_points(local.float().cpu().numpy(), (0, 0), scale)
            points = (points-np.float32(.5))+np.float32(.5)
            scores = keypoint_scores[0].float().cpu().numpy()
            # Filter before computing cell indices, so invalid/outside points
            # cannot produce an out-of-range grid ID or mask lookup.
            valid = (np.isfinite(points).all(1) & (points >= 0).all(1)
                     & (points < [width, height]).all(1))
            source_rows = np.flatnonzero(valid)
            points, scores = points[valid], scores[valid]
            columns = (width+self.grid_size-1)//self.grid_size
            cell_ids = (np.floor(points[:, 1]/self.grid_size).astype(np.int64)*columns
                        + np.floor(points[:, 0]/self.grid_size).astype(np.int64))
            rows = select_candidates(points, scores, cell_ids, cells, image_size=(width, height),
                scale=scale, detection_threshold=self.detection_threshold, nms_radius=self.nms_radius,
                valid_mask=valid_mask)
            points, scores = points[rows], scores[rows]
            descriptors = None
            if save_descriptors:
                if len(rows):
                    indices = torch.as_tensor(source_rows[rows], device=normalized.device, dtype=torch.long)
                    selected = normalized.index_select(0, indices)
                    described, _ = self.model.desc_head(feature_map, [selected])
                    descriptors = described[0].float().cpu().numpy()
                    finite = np.isfinite(descriptors).all(1)
                    points, scores, descriptors = points[finite], scores[finite], descriptors[finite]
                else:
                    descriptors = np.empty((0, descriptor_dim), np.float32)
        self.last_grid_stats = grid_summary(cells, points)
        # The existing feature writer applies the single COLMAP +0.5 shift.
        return (width, height), points-np.float32(.5), scores, descriptors
