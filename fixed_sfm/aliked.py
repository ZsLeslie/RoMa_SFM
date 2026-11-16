"""Offline ALIKED adapter using the installed LightGlue implementation."""
from __future__ import annotations

from hashlib import sha256
import inspect
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


class AlikedExtractor:
    def __init__(self, *, model_name: str, max_keypoints: int, detection_threshold: float,
                 nms_radius: int, resize_max: int, device: str,
                 torch_hub_dir: Path | None, num_threads: int):
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
        self.model_conf = dict(model_name=model_name, max_num_keypoints=max_keypoints,
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

    def extract(self, image_bytes: bytes, *, save_descriptors: bool):
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
