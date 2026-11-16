"""Offline RoMa model and bidirectional fixed-keypoint warp queries."""
from __future__ import annotations

import gc
from hashlib import sha256
import importlib
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .matching import Predictions


def file_sha256(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def sample_warp(warp, certainty, points, source_size, target_size):
    """Query a one-way [1,H,W,4] RoMa warp at original COLMAP coordinates."""
    import torch
    import torch.nn.functional as F

    if warp.ndim != 4 or warp.shape[0] != 1 or warp.shape[-1] != 4:
        raise ValueError(f"Expected a one-way RoMa warp [1,H,W,4], got {tuple(warp.shape)}")
    if tuple(certainty.shape) != tuple(warp.shape[:3]):
        raise ValueError("RoMa certainty and warp grid shapes differ")
    source_size = torch.as_tensor(source_size, dtype=torch.float32, device=warp.device)
    target_size = torch.as_tensor(target_size, dtype=torch.float32, device=warp.device)
    points = torch.as_tensor(points, dtype=torch.float32, device=warp.device)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Warp query coordinates must be [N,2]")
    if not len(points):
        return points.clone(), torch.empty(0, device=warp.device)
    valid = torch.isfinite(points).all(1) & (points >= 0).all(1) & (points < source_size).all(1)
    normalized = 2 * points / source_size - 1
    grid = torch.where(valid[:, None], normalized, torch.zeros_like(normalized))[None, None]
    # Border padding avoids inventing a central-image prediction from zero
    # padding near a grid edge. Out-of-image queries are explicitly invalid.
    mapped = F.grid_sample(warp[..., 2:].float().permute(0, 3, 1, 2), grid,
                           align_corners=False, mode='bilinear', padding_mode='border')[0, :, 0].T
    score = F.grid_sample(certainty[:, None].float(), grid, align_corners=False,
                          mode='bilinear', padding_mode='border')[0, 0, 0]
    mapped = (mapped + 1) * target_size / 2
    valid &= torch.isfinite(mapped).all(1) & (mapped >= 0).all(1) & (mapped < target_size).all(1)
    mapped = torch.where(valid[:, None], mapped, torch.full_like(mapped, float('nan')))
    score = torch.where(valid, score, torch.zeros_like(score))
    return mapped, score


class RoMaMatcher:
    def __init__(self, *, weights_dir: Path, roma_repo_dir: Path | None,
                 device='cuda', coarse_res=448, upsample_res=672, num_threads=4):
        import torch

        if roma_repo_dir is not None:
            if not (roma_repo_dir/'romatch/__init__.py').is_file():
                raise FileNotFoundError(f"RoMa source not found: {roma_repo_dir}")
            sys.path.insert(0, str(roma_repo_dir))
        try:
            module = importlib.import_module('romatch')
        except ImportError as exc:
            raise RuntimeError("Cannot import romatch. Supply --roma-repo-dir /home/nvidia/src/RoMa; do not replace Jetson torch.") from exc
        self.torch, self.model_factory = torch, module.roma_outdoor
        self.device = torch.device(device)
        if self.device.type not in ('cuda', 'cpu'):
            raise ValueError("RoMa supports cuda[:index] or cpu")
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process; check the environment or explicitly use --device cpu")
        torch.set_num_threads(num_threads)
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = False
        self.weights_dir = weights_dir
        self.coarse_res, self.upsample_res = coarse_res, upsample_res
        self.amp_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        files = [weights_dir/'roma_outdoor.pth', weights_dir/'dinov2_vitl14_pretrain.pth']
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Offline RoMa checkpoint missing: {path}")
        source = Path(module.__file__).resolve().parent
        self.signature = dict(model='roma_outdoor', device=str(self.device),
            coarse_res=coarse_res, upsample_res=upsample_res, amp_dtype=str(self.amp_dtype),
            direction_mode='two_sequential_one_way_passes', use_custom_corr=False,
            upsample_preds=True, do_compile=False, num_threads=num_threads, seed=42,
            torch_version=str(torch.__version__), numpy_version=np.__version__, pillow_version=Image.__version__,
            adapter_sha256=file_sha256(Path(__file__)),
            checkpoints={p.name: file_sha256(p) for p in files},
            source_sha256={p.relative_to(source).as_posix(): file_sha256(p) for p in sorted(source.rglob('*.py'))})
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        torch = self.torch
        weights = torch.load(self.weights_dir/'roma_outdoor.pth', map_location='cpu', weights_only=True)
        dino = torch.load(self.weights_dir/'dinov2_vitl14_pretrain.pth', map_location='cpu', weights_only=True)
        self.model = self.model_factory(device=str(self.device), weights=weights, dinov2_weights=dino,
            coarse_res=self.coarse_res, upsample_res=self.upsample_res, amp_dtype=self.amp_dtype,
            symmetric=False, use_custom_corr=False, upsample_preds=True, do_compile=False).eval()
        del weights, dino
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def predict(self, image0: bytes, image1: bytes, features0, features1):
        self.load_model()
        torch = self.torch
        with Image.open(BytesIO(image0)) as f:
            im0 = f.convert('RGB')
        with Image.open(BytesIO(image1)) as f:
            im1 = f.convert('RGB')
        try:
            if im0.size != tuple(features0.image_size) or im1.size != tuple(features1.image_size):
                raise ValueError("Image dimensions differ from the fixed keypoint banks")
            with torch.inference_mode():
                # Query EVERY fixed anchor; model.sample() is intentionally unused.
                warp01, cert01 = self.model.match(im0, im1, device=str(self.device))
                xy01, c0 = sample_warp(warp01, cert01, features0.keypoints.copy(), im0.size, im1.size)
                warp10, cert10 = self.model.match(im1, im0, device=str(self.device))
                xy10, c1 = sample_warp(warp10, cert10, features1.keypoints.copy(), im1.size, im0.size)
                return0, back_c0 = sample_warp(warp10, cert10, xy01, im1.size, im0.size)
                return1, back_c1 = sample_warp(warp01, cert01, xy10, im0.size, im1.size)
                q0 = torch.tensor(features0.keypoints, device=self.device)
                q1 = torch.tensor(features1.keypoints, device=self.device)
                cycle0, cycle1 = (return0-q0).norm(dim=1), (return1-q1).norm(dim=1)
                values = [xy01, xy10, torch.minimum(c0, back_c0), torch.minimum(c1, back_c1), cycle0, cycle1]
                return Predictions(*(x.float().cpu().numpy() for x in values))
        finally:
            im0.close()
            im1.close()
