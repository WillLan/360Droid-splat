import math

import numpy as np
import torch
from torch import nn

from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from backend.legacy_360gs.utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        mono_depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.mono_depth = mono_depth
        self.grad_mask = None
        self.erp_region_masks = None
        self.mdl_insert_mask = None
        self.mdl_overlap = None
        self.submap_id = -1
        self.config_training_overrides = {}

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width
        
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
    # Initialize camera instance from the dataset
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose, mono_depth = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            mono_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":      
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:                                          
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        self.erp_region_masks = None
        self.mdl_insert_mask = None
        self.mdl_overlap = None
        self.config_training_overrides = {}

        self.cam_rot_delta = None
        self.cam_trans_delta = None


# ---------------------------------------------------------------------------
# FaceCamera 鈥?one cubemap face derived from a PanoramaCamera body
# ---------------------------------------------------------------------------
class FaceCamera:
    """
    A lightweight virtual camera representing one cubemap face of a PanoramaCamera.

    R and T are derived from the body's current pose and the fixed face rotation.
    cam_rot_delta / cam_trans_delta delegate to body.
    """

    def __init__(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        face_image: torch.Tensor,
        projection_matrix: torch.Tensor,
        face_w: int,
        body_cam: "PanoramaCamera",
        face_idx: int,
    ):
        self.R = R
        self.T = T
        self.original_image = face_image
        self.projection_matrix = projection_matrix
        self.face_w = face_w
        self.face_idx = face_idx
        self._body = body_cam

        fx, fy, cx, cy = face_w / 2.0, face_w / 2.0, face_w / 2.0 - 0.5, face_w / 2.0 - 0.5
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.FoVx = math.pi / 2.0
        self.FoVy = math.pi / 2.0
        self.image_height = face_w
        self.image_width = face_w
        self.uid = body_cam.uid
        self.device = body_cam.device

        self.depth = None
        self._mono_depth = None   # explicitly set depth for this face
        self.grad_mask = None

    # -- mono_depth: lazily sample from body ERP mono_depth if not set ------
    @property
    def mono_depth(self):
        """Return mono depth for this face.

        If explicitly set via the setter, return that value directly.
        Otherwise sample the body camera's ERP mono_depth at this face's
        cubemap grid coordinates so that the shape is (face_w, face_w).
        """
        if self._mono_depth is not None:
            return self._mono_depth
        body_md = getattr(self._body, "mono_depth", None)
        if body_md is None:
            return None
        # _erp2cube.grids: (6, fw, fw, 2) in [-1, 1], ERP normalised coords
        if not hasattr(self._body, "_erp2cube"):
            return None
        import torch.nn.functional as _F
        grid = self._body._erp2cube.grids[self.face_idx]   # (fw, fw, 2) on device
        depth_t = torch.from_numpy(body_md.astype("float32")).to(grid.device)  # (H, W)
        sampled = _F.grid_sample(
            depth_t.unsqueeze(0).unsqueeze(0),   # (1, 1, H_erp, W_erp)
            grid.unsqueeze(0),                   # (1, fw, fw, 2)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                        # (1, 1, fw, fw)
        return sampled.squeeze(0).squeeze(0).cpu().numpy()  # (fw, fw)

    @mono_depth.setter
    def mono_depth(self, value):
        self._mono_depth = value

    # -- delegate learnable params to body ----------------------------------
    @property
    def cam_rot_delta(self):
        return self._body.cam_rot_delta

    @property
    def cam_trans_delta(self):
        return self._body.cam_trans_delta

    # -- camera geometry ----------------------------------------------------
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R: torch.Tensor, T: torch.Tensor):
        self.R = R.to(self.device)
        self.T = T.to(self.device)

    def compute_grad_mask(self, config):
        """Compute gradient mask from the face image."""
        if self.original_image is None:
            return
        edge_threshold = config["Training"]["edge_threshold"]
        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v ** 2 + gray_grad_h ** 2)
        median_intensity = img_grad_intensity.median()
        self.grad_mask = img_grad_intensity > median_intensity * edge_threshold


# ---------------------------------------------------------------------------
# PanoramaCamera 鈥?full panoramic body camera
# ---------------------------------------------------------------------------
class PanoramaCamera(Camera):
    """
    Panoramic camera storing an ERP image and its 6 cubemap face images.

    Inherits pose + learnable delta parameters from Camera.
    Call ``get_face_camera(i)`` to obtain a FaceCamera for face *i*,
    with R/T derived from the current body pose.
    """

    def __init__(
        self,
        uid,
        color,
        depth,
        mono_depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        face_w: int = 256,
        face_zfar: float = 500.0,
        device: str = "cuda:0",
    ):
        super().__init__(
            uid, color, depth, mono_depth, gt_T, projection_matrix,
            fx, fy, cx, cy, fovx, fovy, image_height, image_width, device,
        )
        self.face_w = face_w
        self.face_zfar = face_zfar

        # Lazy import to avoid circular deps at module load time
        from backend.legacy_360gs.utils.erp2cubemap import CUBEMAP_RS_NP, ERPToCubemapTorch

        self._erp2cube = ERPToCubemapTorch(face_w).to(device)
        self._cubemap_rs_np = CUBEMAP_RS_NP

        # Face projection matrix (90掳 FoV, face_w 脳 face_w)
        self._face_proj_mat = getProjectionMatrix2(
            znear=0.01,
            zfar=face_zfar,
            fx=face_w / 2.0,
            fy=face_w / 2.0,
            cx=face_w / 2.0 - 0.5,
            cy=face_w / 2.0 - 0.5,
            W=face_w,
            H=face_w,
        ).transpose(0, 1).to(device)

        # Precompute 6 face images from the ERP
        if color is not None:
            with torch.no_grad():
                self.face_images = self._erp2cube(color)  # (6, C, fw, fw)
        else:
            self.face_images = None

    # -----------------------------------------------------------------------
    def get_face_camera(self, face_idx: int) -> FaceCamera:
        """
        Build a FaceCamera for *face_idx* using the current body R, T.

        `_cubemap_rs_np` stores face-local -> body rotations.
        For rendering we need body -> face, so we use the transpose.

        R_face = R_body_to_face_i @ R_body
        T_face = R_body_to_face_i @ T_body   (shared camera centre, rotated coords)
        """
        R_face_to_body = torch.tensor(
            self._cubemap_rs_np[face_idx], dtype=torch.float32, device=self.device
        )
        R_body_to_face = R_face_to_body.transpose(0, 1)
        # Ensure body R/T match face rotation dtype (body pose may be float64)
        R_face = R_body_to_face @ self.R.to(dtype=R_body_to_face.dtype)
        T_face = R_body_to_face @ self.T.to(dtype=R_body_to_face.dtype)

        face_image = (
            self.face_images[face_idx] if self.face_images is not None else None
        )

        return FaceCamera(
            R=R_face,
            T=T_face,
            face_image=face_image,
            projection_matrix=self._face_proj_mat,
            face_w=self.face_w,
            body_cam=self,
            face_idx=face_idx,
        )

    # -----------------------------------------------------------------------
    @staticmethod
    def init_from_panorama_dataset(dataset, idx: int, face_w: int = 256,
                                   face_zfar: float = 500.0):
        """Initialise a PanoramaCamera from a panoramic dataset entry."""
        gt_color, gt_depth, gt_pose, mono_depth = dataset[idx]

        # Use face intrinsics (front face = 90掳 FoV) as body intrinsics
        fw = face_w
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=face_zfar,
            fx=fw / 2.0,
            fy=fw / 2.0,
            cx=fw / 2.0 - 0.5,
            cy=fw / 2.0 - 0.5,
            W=fw,
            H=fw,
        ).transpose(0, 1)

        from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import focal2fov
        fov_face = math.pi / 2.0  # 90掳

        return PanoramaCamera(
            uid=idx,
            color=gt_color,
            depth=gt_depth,
            mono_depth=mono_depth,
            gt_T=gt_pose,
            projection_matrix=projection_matrix,
            fx=fw / 2.0,
            fy=fw / 2.0,
            cx=fw / 2.0 - 0.5,
            cy=fw / 2.0 - 0.5,
            fovx=fov_face,
            fovy=fov_face,
            image_height=gt_color.shape[1] if gt_color is not None else fw,
            image_width=gt_color.shape[2] if gt_color is not None else fw * 2,
            face_w=face_w,
            face_zfar=face_zfar,
            device=dataset.device,
        )

    def clean(self):
        super().clean()
        self.face_images = None
