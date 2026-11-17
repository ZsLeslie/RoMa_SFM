"""Offline DeDoDe v2-L detector using the installed Kornia implementation.

The official checkpoint loads strictly into Kornia's detector. No descriptor,
DINOv2, torch.hub download or package installation is used. Selection follows
the official spatial-softmax and coverage correction, with optional original-
image mask/grid quotas applied before top-K. Returned adapter coordinates are
zero-centered; the common feature writer adds the sole COLMAP half-pixel shift.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .grid import check_mask, make_grid


def _top_rows(rows, scores, count):
    """Stable top-K: score descending, then raster row index ascending."""
    if count < 1 or not len(rows):
        return np.empty(0, np.int64)
    values = scores[rows]
    if len(rows) > count:
        cutoff = np.partition(values, len(values)-count)[len(values)-count]
        higher = rows[values > cutoff]
        equal = rows[values == cutoff]  # incoming rows are in raster order
        rows = np.concatenate((higher, equal[:count-len(higher)]))
    return rows[np.lexsort((rows, -scores[rows]))]


def select_scoremap(scoremap, image_size, *, max_keypoints, detection_threshold=0.,
                    grid_size=0, valid_mask=None):
    """Select finite scored pixel centers, preserving native DeDoDe scores.

    There is no extra inference NMS, as in the official v2 detector. The grid
    variant changes spatial quotas only; it does not promise reliable points
    on a textureless surface. Masks are tested at ORIGINAL pixel coordinates.
    """
    scoremap = np.asarray(scoremap, np.float32)
    if scoremap.ndim != 2 or min(scoremap.shape) < 1:
        raise ValueError('DeDoDe scoremap must be nonempty [H,W]')
    if max_keypoints < 1 or not 0 <= detection_threshold <= 1:
        raise ValueError('Invalid keypoint budget or DeDoDe threshold')
    width, height = image_size
    if min(width, height) < 1 or grid_size < 0:
        raise ValueError('Invalid image dimensions or grid size')
    check_mask(valid_mask, image_size)
    ih, iw = scoremap.shape
    xs = (np.arange(iw, dtype=np.float64)+.5)*(width/iw)
    ys = (np.arange(ih, dtype=np.float64)+.5)*(height/ih)
    valid = np.isfinite(scoremap) & (scoremap >= detection_threshold) & (scoremap <= 1)
    if valid_mask is not None:
        valid &= valid_mask[np.floor(ys).astype(np.int64)[:, None],
                           np.floor(xs).astype(np.int64)[None, :]]
    rows = np.flatnonzero(valid)
    scores = scoremap.ravel()
    if grid_size:
        cells = make_grid(image_size, grid_size=grid_size, margin=0,
                          max_keypoints=max_keypoints, valid_mask=valid_mask)
        columns = (width+grid_size-1)//grid_size
        owners = ((ys[rows//iw]//grid_size).astype(np.int64)*columns
                  + (xs[rows % iw]//grid_size).astype(np.int64))
        selected = [_top_rows(rows[owners == c.index], scores, c.quota) for c in cells if c.active]
        selected = np.concatenate(selected) if selected else np.empty(0, np.int64)
    else:
        selected = _top_rows(rows, scores, max_keypoints)
    points = np.column_stack((xs[selected % iw], ys[selected//iw])).astype(np.float32)
    return points, scores[selected].copy()


class DeDoDeExtractor:
    def __init__(self, *, model_name, max_keypoints, detection_threshold, nms_radius,
                 resize_max, device, torch_hub_dir, num_threads, weights_path,
                 grid_size=0):
        import torch
        import kornia
        from kornia.feature.dedode.dedode_models import get_detector
        from kornia.feature import dedode as implementation

        if model_name != 'dedode-L-v2' or nms_radius != 0:
            raise ValueError('Use dedode-L-v2 with native sampling (--nms-radius 0)')
        self.device = torch.device(device)
        if self.device.type not in ('cuda', 'cpu'):
            raise ValueError('DeDoDe supports cuda[:index] or cpu')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Check device access or explicitly use --device cpu.')
        if num_threads < 1 or resize_max < 0 or (resize_max and resize_max < 32):
            raise ValueError('Invalid threads or inference resolution')
        if max_keypoints < 1 or grid_size < 0 or not 0 <= detection_threshold <= 1:
            raise ValueError('Invalid DeDoDe selection parameters')
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f'Local DeDoDe checkpoint missing: {self.weights_path}')
        self.torch, self.model_factory = torch, get_detector
        self.max_keypoints, self.threshold = max_keypoints, detection_threshold
        self.resize_max, self.grid_size, self.model = resize_max, grid_size, None
        self.amp_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        torch.set_num_threads(num_threads)
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        source = Path(implementation.__file__).resolve().parent
        self.signature = dict(
            extractor='kornia.DeDoDe.detector', model='dedode-L-v2',
            weights_sha256=sha256(self.weights_path.read_bytes()).hexdigest(),
            adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            writer_sha256=sha256(Path(__file__).with_name('features.py').read_bytes()).hexdigest(),
            grid_sha256=sha256(Path(__file__).with_name('grid.py').read_bytes()).hexdigest(),
            source_sha256={p.relative_to(source).as_posix(): sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source.rglob('*.py'))},
            max_keypoints=max_keypoints, detection_threshold=detection_threshold,
            grid_size_original_px=grid_size, nms_radius=0,
            quota_policy='equal_active_cells_yx_remainder_no_refill' if grid_size else 'global_topk',
            sampling='spatial_softmax_coverage_alpha_0.5_kernel_51_stable_topk',
            mask_policy='original_pixel_centers_before_topk',
            resize_max=resize_max, resize_policy='long_edge_downscale_only_pil_bilinear_rounded_xy',
            normalization='imagenet_rgb', padding='none', descriptors=False,
            precision=str(self.amp_dtype), device=str(self.device), num_threads=num_threads, seed=42,
            runtime=dict(torch=str(torch.__version__), kornia=kornia.__version__,
                         numpy=np.__version__, pillow=Image.__version__))

    def load_model(self):
        if self.model is not None:
            return
        torch = self.torch
        model = self.model_factory('L', amp_dtype=self.amp_dtype)
        weights = torch.load(self.weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        if self.device.type == 'cpu':
            for module in model.modules():
                if hasattr(module, 'amp'):
                    module.amp = False
        self.model = model.eval().to(self.device)

    def extract(self, image_bytes, *, save_descriptors=False, valid_mask=None):
        if save_descriptors:
            raise ValueError('This adapter loads the DeDoDe detector only')
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert('RGB')
            width, height = image.size
            if min(width, height) < 32:
                raise ValueError('DeDoDe requires images at least 32 pixels per side')
            check_mask(valid_mask, (width, height))
            if valid_mask is not None and not valid_mask.any():
                return (width, height), np.empty((0, 2), np.float32), np.empty(0, np.float32), None
            if self.resize_max and max(width, height) > self.resize_max:
                factor = self.resize_max/max(width, height)
                size = (round(width*factor), round(height*factor))
                if min(size) < 32:
                    raise ValueError('Resizing makes the short edge smaller than 32 pixels')
                image = image.resize(size, Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.uint8).copy()
        self.load_model()
        torch = self.torch
        import torch.nn.functional as F
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32)[None]/255.
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            logits = self.model((tensor-mean)/std).float()
            ih, iw = logits.shape[-2:]
            if (ih, iw) != rgb.shape[:2]:
                raise RuntimeError('DeDoDe logits do not match the inference raster')
            probabilities = logits.reshape(1, -1).softmax(-1).reshape(1, 1, ih, iw)
            # Official DeDoDe density correction; it changes ranking, not coordinates.
            kernel = torch.exp(-torch.linspace(-2, 2, 51, device=self.device).square())
            density = F.conv2d((probabilities+1e-6)*10000, kernel[None, None, None, :], padding=(0, 25))
            density = F.conv2d(density, kernel[None, None, :, None], padding=(25, 0))
            scores = (probabilities*torch.rsqrt(density+1e-8))[0, 0].cpu().numpy()
        points, scores = select_scoremap(scores, (width, height), max_keypoints=self.max_keypoints,
            detection_threshold=self.threshold, grid_size=self.grid_size, valid_mask=valid_mask)
        return (width, height), points-np.float32(.5), scores, None
