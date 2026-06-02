"""
SphereGlue matcher wrapper for ERP (Equirectangular Projection) panoramic images.

This module integrates SphereGlue (CVPR 2023 Workshop) into S3PO-GS.
SphereGlue uses a Graph Neural Network (Chebyshev convolution + Attention)
to match keypoints in spherical coordinates, avoiding the distortions of
standard perspective matchers on ERP images.

Pipeline for a single ERP image pair:
  1. Extract keypoints + descriptors from both images using a detector
     (SIFT by default; SuperPoint can be substituted).
  2. Convert pixel coordinates 鈫?spherical coordinates (phi, theta).
  3. Convert spherical 鈫?unit Cartesian (for ChebConv on the sphere).
  4. Run SphereGlue to produce optimal-transport matches.
  5. Return unified output dict:
       {
           'mkpts0'  : (N, 2) float32  鈥?matched pixel [x, y] in image0
           'mkpts1'  : (N, 2) float32  鈥?matched pixel [x, y] in image1
           'mscores' : (N,)   float32  鈥?per-match confidence scores
           'matches0': (K,)   int64    鈥?for each kpt0, index in kpts1 (-1=unmatched)
           'kpts0'   : (K, 2) float32  鈥?all detected keypoints in image0 [x, y]
           'kpts1'   : (M, 2) float32  鈥?all detected keypoints in image1 [x, y]
       }

Usage:
    matcher = SphereGlueMatcher(detector="sift")
    result  = matcher.match(erp0_rgb, erp1_rgb)   # HxWx3 uint8 numpy arrays
    mkpts0, mkpts1 = result['mkpts0'], result['mkpts1']
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the sphereglue sub-repository is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT      = Path(__file__).resolve().parent.parent
_SPHEREGLUE_DIR = _REPO_ROOT / "sphereglue"
if str(_SPHEREGLUE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPHEREGLUE_DIR))

from model.sphereglue import SphereGlue  # noqa: E402


# ---------------------------------------------------------------------------
# Coordinate conversion helpers  (mirrors SphereGlue/utils/Utils.py)
# ---------------------------------------------------------------------------

def pixel_to_spherical(pixel_coords: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert pixel coordinates [x_col, y_row] to spherical [phi, theta].

    phi   鈭?[0, 蟺]    polar / elevation angle (0 = north pole)
    theta 鈭?[0, 2蟺]   azimuth angle

    Args:
        pixel_coords: (N, 2) float array with columns [x, y]
        img_w:        image width
        img_h:        image height
    Returns:
        (N, 2) float32 array with columns [phi, theta]
    """
    x, y  = pixel_coords[:, 0:1], pixel_coords[:, 1:2]
    theta = (1.0 - (x + 0.5) / img_w) * 2.0 * np.pi
    phi   = ((y + 0.5) * np.pi) / img_h
    return np.hstack([phi, theta]).astype(np.float32)


def spherical_to_unit_cartesian(spherical: np.ndarray) -> torch.Tensor:
    """
    Convert spherical [phi, theta] to unit Cartesian [x, y, z] on S^2.

    Args:
        spherical: (N, 2) float array [phi, theta]
    Returns:
        (N, 3) float32 torch.Tensor
    """
    phi   = torch.tensor(spherical[:, 0], dtype=torch.float32)
    theta = torch.tensor(spherical[:, 1], dtype=torch.float32)
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


# ---------------------------------------------------------------------------
# Keypoint detectors
# ---------------------------------------------------------------------------

def _extract_sift(
    image_gray: np.ndarray,
    max_kpts: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and L2-normalised 128-dim descriptors.

    Returns:
        pts    (N, 2) float32  pixel [x, y]
        descs  (N, 128) float32
        scores (N,) float32
    """
    sift = cv2.SIFT_create(nfeatures=max_kpts)
    kps, descs = sift.detectAndCompute(image_gray, None)
    if not kps or descs is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), np.zeros(0, np.float32)
    pts    = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    scores = np.array([kp.response for kp in kps], dtype=np.float32)
    norms  = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
    descs  = (descs / norms).astype(np.float32)
    return pts, descs, scores


# ---------------------------------------------------------------------------
# SuperPoint extractor (self-contained, no external dependency)
# ---------------------------------------------------------------------------

class _SuperPointNet(torch.nn.Module):
    """
    Minimal SuperPoint network architecture matching the official pre-trained weights
    from https://github.com/magicleap/SuperPointPretrainedNetwork.
    """

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.relu    = torch.nn.ReLU(inplace=True)
        self.pool    = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Shared encoder
        self.conv1a  = torch.nn.Conv2d(1,  c1, 3, stride=1, padding=1)
        self.conv1b  = torch.nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.conv2a  = torch.nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.conv2b  = torch.nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.conv3a  = torch.nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.conv3b  = torch.nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.conv4a  = torch.nn.Conv2d(c3, c4, 3, stride=1, padding=1)
        self.conv4b  = torch.nn.Conv2d(c4, c4, 3, stride=1, padding=1)

        # Detector head
        self.convPa  = torch.nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convPb  = torch.nn.Conv2d(c5, 65, 1, stride=1, padding=0)

        # Descriptor head
        self.convDa  = torch.nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convDb  = torch.nn.Conv2d(c5, 256, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # (B, 65, H/8, W/8)

        # Descriptor
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # (B, 256, H/8, W/8)
        dn   = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc / (dn + 1e-8)

        return semi, desc


class _SuperPointExtractor:
    """
    Wraps _SuperPointNet to extract keypoints and 256-dim descriptors from a
    grayscale image, matching the descriptor format expected by SphereGlue's
    SuperPoint-trained weights.
    """

    def __init__(self, weights_path: str, device: str = "cuda", nms_radius: int = 4):
        self.device     = device
        self.nms_radius = nms_radius
        self.net        = _SuperPointNet()
        # CPU unpickle first 鈥?direct CUDA map_location has caused SIGBUS on some setups
        # (same idea as DAP in utils/dap_wrapper.py).
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt)
        self.net = self.net.to(device)
        self.net.eval()

    def __call__(
        self,
        image_gray: np.ndarray,
        max_kpts: int = 8000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image_gray.shape
        # Pad to multiple of 8
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        img   = np.pad(image_gray, ((0, h_pad), (0, w_pad)), mode="reflect")
        inp   = torch.tensor(img / 255.0, dtype=torch.float32, device=self.device)
        inp   = inp.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')

        with torch.no_grad():
            semi, desc_map = self.net(inp)

        # Softmax on detector logits 鈫?heatmap
        semi_sm  = torch.nn.functional.softmax(semi, dim=1)[:, :-1, :, :]  # (1,64,H/8,W/8)
        # Pixel shuffle: (1,64,H/8,W/8) 鈫?(1,1,H,W)
        heatmap  = torch.nn.functional.pixel_shuffle(semi_sm, 8)  # (1,1,H,W)
        heatmap  = heatmap.squeeze().cpu().numpy()[:h, :w]

        # NMS
        scores_map = self._nms(heatmap)
        ys, xs     = np.where(scores_map > 0.005)
        scores     = scores_map[ys, xs]

        if len(scores) == 0:
            return (np.zeros((0, 2), np.float32),
                    np.zeros((0, 256), np.float32),
                    np.zeros(0, np.float32))

        # Keep top-k
        if len(scores) > max_kpts:
            top_k  = np.argsort(scores)[::-1][:max_kpts]
            xs, ys = xs[top_k], ys[top_k]
            scores = scores[top_k]

        pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2) [x, y]

        # Sample descriptors at keypoint locations
        # desc_map: (1, 256, H/8, W/8)
        # grid_sample expects (N, 1, 1, 2) in [-1, 1]
        kpts_norm = torch.tensor(
            [[[2.0 * x / (w - 1) - 1.0, 2.0 * y / (h - 1) - 1.0] for x, y in pts]],
            dtype=torch.float32, device=self.device,
        )  # (1, N, 1, 2)
        kpts_norm = kpts_norm.unsqueeze(2)  # (1, N, 1, 2)
        sampled   = torch.nn.functional.grid_sample(
            desc_map[:, :, :h // 8, :w // 8],
            kpts_norm.view(1, 1, -1, 2),
            mode="bilinear",
            align_corners=True,
        )  # (1, 256, 1, N)
        descs = sampled.squeeze(0).squeeze(1).T.cpu().numpy()  # (N, 256)
        norms = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
        descs = (descs / norms).astype(np.float32)

        return pts, descs, scores.astype(np.float32)

    @staticmethod
    def _nms(heatmap: np.ndarray, radius: int = 4) -> np.ndarray:
        """Simple max-pool NMS on a heatmap."""
        size   = 2 * radius + 1
        hm_t   = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pooled = torch.nn.functional.max_pool2d(
            hm_t, kernel_size=size, stride=1, padding=radius
        )
        keep   = (hm_t == pooled).squeeze().numpy()
        return heatmap * keep


# Detector registry: name 鈫?(callable-factory, descriptor_dim, weights_subdir)
# For "sift" the factory is a plain function; for "superpoint" it's a class instance.
_DETECTORS: dict[str, tuple] = {
    "sift":       (_extract_sift, 128, "sift"),
    "superpoint": (None,          256, "superpoint"),  # instantiated lazily
}


def _resolve_superpoint_weights_path(
    weights_dir: str | None,
    explicit_path: str | None,
    min_bytes: int = 8192,
) -> str:
    """
    SuperPoint CNN weights (Magic Leap pretrained) are separate from
    SphereGlue's autosaved.pt under model_weights/<detector>/.

    Search order:
      1. explicit_path if provided and valid
      2. <weights_dir>/superpoint/superpoint_v1.pth
      3. <repo>/sphereglue/model_weights/superpoint/superpoint_v1.pth
      4. <repo>/checkpoints/superpoint_v1.pth
    """
    if explicit_path is not None:
        p = os.path.expanduser(explicit_path)
        if os.path.isfile(p) and os.path.getsize(p) >= min_bytes:
            return p
        raise FileNotFoundError(
            f"sg_superpoint_weights not found or too small (<{min_bytes} B): {p}"
        )

    base_dir = weights_dir if weights_dir is not None else str(_SPHEREGLUE_DIR / "model_weights")
    candidates = [
        os.path.join(base_dir, "superpoint", "superpoint_v1.pth"),
        str(_SPHEREGLUE_DIR / "model_weights" / "superpoint" / "superpoint_v1.pth"),
        str(_REPO_ROOT / "checkpoints" / "superpoint_v1.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) >= min_bytes:
            return p
    raise FileNotFoundError(
        "SuperPoint backbone weights not found. Place superpoint_v1.pth in either:\n"
        f"  鈥?{os.path.join(base_dir, 'superpoint', 'superpoint_v1.pth')}\n"
        f"  鈥?{_REPO_ROOT / 'checkpoints' / 'superpoint_v1.pth'}\n"
        "Download: https://github.com/magicleap/SuperPointPretrainedNetwork\n"
        "(This file is not the same as SphereGlue's superpoint/autosaved.pt.)"
    )


# ---------------------------------------------------------------------------
# SphereGlueMatcher
# ---------------------------------------------------------------------------

class SphereGlueMatcher:
    """
    Wrapper around the SphereGlue model for matching ERP panoramic image pairs.

    Args:
        weights_dir:     Path to the SphereGlue model_weights directory.
                         Defaults to <repo_root>/sphereglue/model_weights.
        detector:        'sift' (default) or 'superpoint' (uses GNN weights in
                         model_weights/superpoint/autosaved.pt plus SuperPoint CNN).
        superpoint_weights: Optional path to superpoint_v1.pth; if None, searches
                         <weights_dir>/superpoint/superpoint_v1.pth then checkpoints/.
        match_threshold: Confidence threshold for accepting a match (0.0 鈥?1.0).
        max_kpts:        Maximum number of keypoints per image.
        knn:             K for the k-NN graph built on the sphere.
        sinkhorn_iters:  Number of Sinkhorn OT iterations.
        device:          'cuda' or 'cpu'.  Auto-detected if None.
    """

    _BASE_CONFIG = {
        "K":     2,           # Chebyshev filter size
        "GNN_layers": ["cross"],
        "aggr":  "add",
    }

    def __init__(
        self,
        weights_dir: str | None = None,
        detector: str = "sift",
        superpoint_weights: str | None = None,
        match_threshold: float = 0.2,
        max_kpts: int = 8000,
        knn: int = 20,
        sinkhorn_iters: int = 20,
        device: str | None = None,
    ):
        if detector not in _DETECTORS:
            raise ValueError(f"detector must be one of {list(_DETECTORS.keys())}, got '{detector}'")

        self.detector_name = detector
        _, desc_dim, weights_subdir = _DETECTORS[detector]
        self.max_kpts = max_kpts
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- locate SphereGlue matching weights (needed to resolve SuperPoint path) ----
        if weights_dir is None:
            weights_dir = str(_SPHEREGLUE_DIR / "model_weights")

        # ---- set up the feature extractor ----
        if detector == "superpoint":
            sp_path = _resolve_superpoint_weights_path(weights_dir, superpoint_weights)
            sp = _SuperPointExtractor(sp_path, device=self.device)
            self._extract_fn = sp
        else:
            self._extract_fn = _DETECTORS[detector][0]

        # ---- SphereGlue GNN checkpoint path ----
        weights_path = os.path.join(weights_dir, weights_subdir, "autosaved.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"SphereGlue weights not found at: {weights_path}\n"
                "Please download them from the SphereGlue repository."
            )

        config = dict(self._BASE_CONFIG)
        config.update({
            "descriptor_dim":      desc_dim,
            "output_dim":          desc_dim * 2,
            "match_threshold":     match_threshold,
            "knn":                 knn,
            "sinkhorn_iterations": sinkhorn_iters,
            "max_kpts":            max_kpts,
        })

        self.model = SphereGlue(config)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["MODEL_STATE_DICT"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[SphereGlueMatcher] loaded '{detector}' weights from {weights_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict:
        """
        Match two ERP panoramic images.

        Args:
            image0: (H, W, 3) or (H, W) uint8/float32 numpy array (RGB or gray).
            image1: (H, W, 3) or (H, W) uint8/float32 numpy array (RGB or gray).

        Returns:
            dict:
                'mkpts0'  : (N, 2) float32  鈥?matched pixels in image0 [x, y]
                'mkpts1'  : (N, 2) float32  鈥?matched pixels in image1 [x, y]
                'mscores' : (N,)   float32  鈥?confidence of each match
                'matches0': (K,)   int64    鈥?index into kpts1 per kpt0 (-1=unmatched)
                'kpts0'   : (K, 2) float32  鈥?all kpts in image0 [x, y]
                'kpts1'   : (M, 2) float32  鈥?all kpts in image1 [x, y]
        """
        gray0 = self._to_gray(image0)
        gray1 = self._to_gray(image1)
        h0, w0 = gray0.shape
        h1, w1 = gray1.shape

        kpts0, desc0, scores0 = self._extract_fn(gray0, self.max_kpts)
        kpts1, desc1, scores1 = self._extract_fn(gray1, self.max_kpts)

        if len(kpts0) == 0 or len(kpts1) == 0:
            return self._empty_result(kpts0, kpts1)

        # Pixel 鈫?spherical 鈫?unit Cartesian
        sph0  = pixel_to_spherical(kpts0, w0, h0)
        sph1  = pixel_to_spherical(kpts1, w1, h1)
        cart0 = spherical_to_unit_cartesian(sph0).to(self.device)
        cart1 = spherical_to_unit_cartesian(sph1).to(self.device)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

        data = {
            "h1":             _t(desc0),             # (1, K, D)
            "h2":             _t(desc1),             # (1, M, D)
            "unitCartesian1": cart0.unsqueeze(0),    # (1, K, 3)
            "unitCartesian2": cart1.unsqueeze(0),    # (1, M, 3)
            "scores1":        _t(scores0),           # (1, K)
            "scores2":        _t(scores1),           # (1, M)
        }

        with torch.no_grad():
            pred = self.model(data)

        matches0 = pred["matches0"][0].cpu().numpy()        # (K,) int
        mscores0  = pred["matching_scores0"][0].cpu().numpy()  # (K,) float

        # Guard against off-by-one: model may trim 1 kpt internally
        n = min(len(kpts0), len(matches0))
        kpts0    = kpts0[:n]
        matches0 = matches0[:n]
        mscores0 = mscores0[:n]

        valid  = matches0 >= 0
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mscores = mscores0[valid]

        return {
            "mkpts0":   mkpts0.astype(np.float32),
            "mkpts1":   mkpts1.astype(np.float32),
            "mscores":  mscores.astype(np.float32),
            "matches0": matches0,
            "kpts0":    kpts0,
            "kpts1":    kpts1,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert any image format to uint8 grayscale."""
        if image.dtype != np.uint8:
            scale = 255.0 if image.max() <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        if image.ndim == 3:
            code = cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
            return cv2.cvtColor(image, code)
        return image

    @staticmethod
    def _empty_result(kpts0: np.ndarray, kpts1: np.ndarray) -> dict:
        return {
            "mkpts0":   np.zeros((0, 2), np.float32),
            "mkpts1":   np.zeros((0, 2), np.float32),
            "mscores":  np.zeros(0, np.float32),
            "matches0": np.full(len(kpts0), -1, np.int64),
            "kpts0":    kpts0,
            "kpts1":    kpts1,
        }
