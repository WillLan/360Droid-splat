"""
DAP (Depth Any Panoramas) wrapper for metric ERP depth estimation.

Integrates DAP into the S3PO-GS pipeline as a drop-in replacement for
MASt3R-based mono depth in panoramic mode.

Input contract:
    - ERP RGB image: (H, W, 3) uint8 numpy array  OR
                     (3, H, W) float [0, 1] torch.Tensor
Output contract:
    - depth_m    : (H, W) float32 numpy, metric depth in metres, clipped [0, 100]
    - valid_mask : (H, W) bool numpy, True where depth < 100 m
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple

# --------------------------------------------------------------------------- #
# Path bookkeeping: add dap/ to sys.path so its internal imports resolve
# --------------------------------------------------------------------------- #
_S3PO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAP_ROOT = os.path.join(_S3PO_ROOT, "dap")


def _ensure_dap_in_path() -> None:
    if DAP_ROOT not in sys.path:
        sys.path.insert(0, DAP_ROOT)


# --------------------------------------------------------------------------- #
# DAP output interpretation
# --------------------------------------------------------------------------- #
# DAP is trained with max_depth=1.0 (normalised).  In the official visualiser
# "100m" range clips the output to [0, 1], meaning 1.0 鈮?100 m.
# We therefore multiply by MAX_METRIC_DEPTH to obtain metres.
MAX_METRIC_DEPTH: float = 100.0


# --------------------------------------------------------------------------- #
# Wrapper class
# --------------------------------------------------------------------------- #
class DAPDepthWrapper:
    """
    Thin wrapper around the DAP model for use inside S3PO-GS.

    Usage::

        wrapper = DAPDepthWrapper(checkpoint_path="checkpoints/dap/model.pth")
        depth_m, valid_mask = wrapper.infer(rgb_tensor_or_numpy)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        input_size: int = 518,
        midas_model_type: str = "vitl",
    ) -> None:
        _ensure_dap_in_path()
        self.device = device
        self.input_size = input_size
        self.model = self._load_model(checkpoint_path, midas_model_type)

    # ---------------------------------------------------------------------- #

    def _load_model(self, checkpoint_path: str, midas_model_type: str) -> nn.Module:
        # Late import: DAP modules need DAP_ROOT in sys.path first
        from networks.models import make
        import networks.dap  # noqa: F401  鈥?triggers @register('dap')

        model_spec = {
            "name": "dap",
            "args": {
                "midas_model_type": midas_model_type,
                "fine_tune_type": "none",
                "min_depth": 0.001,
                "max_depth": 1.0,
                "train_decoder": True,
            },
        }

        m = make(model_spec)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"DAP checkpoint not found: {checkpoint_path}\n"
                "Please download the weights from HuggingFace "
                "(https://huggingface.co/Insta360-Research/DAP-weights) "
                "and place model.pth at the path above."
            )

        # Load checkpoints on CPU first, then move the fully constructed model
        # to CUDA.  Direct CUDA deserialization is noticeably more fragile in
        # this environment and has caused intermittent hard crashes during
        # SLAM startup.
        state = torch.load(checkpoint_path, map_location="cpu")

        # Strip "module." prefix if checkpoint was saved with DataParallel
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        m_state = m.state_dict()
        filtered = {k: v for k, v in state.items() if k in m_state}
        m.load_state_dict(filtered, strict=False)
        m = m.to(self.device)
        m.eval()
        return m

    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def infer(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DAP depth estimation on an ERP image.

        Args:
            image: (H, W, 3) uint8 RGB numpy  OR  (3, H, W) float [0,1] Tensor

        Returns:
            depth_m    : (H, W) float32 numpy, metres, clamped to [0, 100]
            valid_mask : (H, W) bool numpy, True where depth < 100 m
        """
        img_bgr = self._to_bgr_uint8(image)
        orig_h, orig_w = img_bgr.shape[:2]

        # DAP uses resize_method='lower_bound' which does NOT shrink images larger
        # than the target 鈥?pre-resize to the target resolution so that the ViT
        # actually processes a small image rather than the full-res panorama.
        target_h = self.input_size
        target_w = self.input_size * 2
        if orig_h > target_h or orig_w > target_w:
            img_bgr = cv2.resize(img_bgr, (target_w, target_h),
                                 interpolation=cv2.INTER_CUBIC)

        # DAP's infer_image: BGR uint8 鈫?float32 normalised [0, 1], output at
        # the (possibly pre-resized) image resolution.
        # Autocast materially reduces activation memory for ViT-L on 24 GB GPUs.
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth_norm: np.ndarray = self.model.infer_image(
                    img_bgr, input_size=self.input_size
                )  # (H', W')
        else:
            depth_norm: np.ndarray = self.model.infer_image(
                img_bgr, input_size=self.input_size
            )  # (H', W')

        depth_norm = np.asarray(depth_norm, dtype=np.float32)

        # Scale depth back to original resolution
        if depth_norm.shape != (orig_h, orig_w):
            depth_norm = cv2.resize(depth_norm, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR)

        depth_m = (depth_norm * MAX_METRIC_DEPTH).astype(np.float32)
        depth_m = np.clip(depth_m, 0.0, MAX_METRIC_DEPTH)
        valid_mask = depth_m < MAX_METRIC_DEPTH * 0.9

        return depth_m, valid_mask

    # ---------------------------------------------------------------------- #

    def _to_bgr_uint8(self, image) -> np.ndarray:
        """
        Normalise any supported input to a BGR uint8 HWC array
        that DAP's image2tensor expects.
        """
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            if img.dim() == 3 and img.shape[0] in (1, 3):
                # (C, H, W) 鈫?(H, W, C)
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.numpy()
            # Assume [0, 1] float if not uint8
            if img.dtype != np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        elif isinstance(image, np.ndarray):
            img = image
            if img.dtype != np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            # Assume RGB input (consistent with S3PO-GS convention)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return img_bgr


# --------------------------------------------------------------------------- #
# Convenience factory (mirrors MASt3R loading style in slam.py)
# --------------------------------------------------------------------------- #

def load_dap_model(
    checkpoint_path: str,
    device: str = "cuda",
    input_size: int = 518,
    midas_model_type: str = "vitl",
) -> DAPDepthWrapper:
    """
    Load and return a DAPDepthWrapper ready for inference.

    Args:
        checkpoint_path : Path to ``model.pth`` downloaded from HuggingFace.
        device          : Torch device string.
        input_size      : Inference resolution (height); width = input_size * 2.
        midas_model_type: ViT backbone variant ('vits', 'vitb', 'vitl', 'vitg').
    """
    return DAPDepthWrapper(
        checkpoint_path=checkpoint_path,
        device=device,
        input_size=input_size,
        midas_model_type=midas_model_type,
    )
