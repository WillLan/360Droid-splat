import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import json
from pathlib import Path

from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except Exception:
    pass

# We retain the input interfaces for ground-truth depth and other monocular depth estimations (e.g., from DepthAnything).
# In RGB-only scenarios, the first channel of the RGB image is used as a placeholder for depth input.

## ====================================data parser========================================
class dl3dvParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """ Read camera poses from camera.json and convert them to 4脳4 matrices """
        self.poses = []
        self.frames = []

        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        selected_poses = all_poses[self.begin:self.end]
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans 
    
            inv_pose = np.linalg.inv(transform_matrix)
            self.poses.append(inv_pose)  
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),  
            }
            self.frames.append(frame)

class KITTIParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"] 
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))[self.begin:self.end]
        init_trans = np.loadtxt(pose_files[0], delimiter=' ').reshape(4, 4)[:3,3]

        for i in range(self.n_img):
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            pose[:3,3] = pose[:3,3] - init_trans
            inv_pose = np.linalg.inv(pose)  
            self.poses.append(inv_pose)     
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),      
            }
            self.frames.append(frame)

class WaymoParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/mono_depth/*.png"))
        if len(self.mono_depth_paths) != len(self.color_paths):
            self.mono_depth_paths = self.depth_paths
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))

        for i in range(self.n_img):
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            inv_pose = np.linalg.inv(pose)  
            self.poses.append(inv_pose)     
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),      
            }
            self.frames.append(frame)

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/results/mono*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):   
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):       
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        mono_depth_list = os.path.join(datapath, "mono_depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        mono_depth_data = self.parse_list(mono_depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        print("鏍囧彿:", tstamp_image[471])

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames, self.mono_depth_paths = [], [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            self.mono_depth_paths += [os.path.join(datapath, mono_depth_data[i, 1])]

            quat = pose_vecs[k][4:]     
            trans = pose_vecs[k][1:4]   
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1)) 
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]   

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
                "mono_depth_path": str(os.path.join(datapath, mono_depth_data[i, 1]))
            }

            self.frames.append(frame)

##=================================Define data base class==================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):     
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]    
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]   
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )                               
        # distortion parameters
        self.disorted = calibration["distorted"] 
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(  
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale  
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)

        # Check if the image is RGB (3 channels); if so, extract the first channel
        if len(image_array.shape) == 3:  
            return image_array[:, :, 0]  
        else:  
            return image_array  

    def __getitem__(self, idx):  
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = self.load_image(depth_path) / self.depth_scale  
            mono_depth_path = self.mono_depth_paths[idx]
            mono_depth = self.load_image(mono_depth_path) / (self.depth_scale*5)

        image = (
            torch.tensor(image / 255.0, dtype=torch.float32)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.tensor(pose, dtype=torch.float32).to(device=self.device)
        return image, depth, pose, mono_depth

##=====================================# Define dataset class for specific dataset======================================
class dl3dvDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        parser = dl3dvParser(dataset_path, config)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  
        self.mono_depth_paths = parser.color_paths  
        self.poses = parser.poses  

class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KITTIParser(dataset_path,config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses      

class WaymoDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = WaymoParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses       

class TUMDataset(MonocularDataset):  
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.mono_depth_paths = parser.mono_depth_paths

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses

##=====================================Panorama Dataset======================================

class PanoramaParser:
    """
    Parser for panoramic (ERP) image datasets.

    Expected directory layout (``pose_format: 'txt'``, default)::

        <dataset_path>/
            rgb/          鈫?ERP images (*.png or *.jpg), W = 2*H
            poses.txt     鈫?one 4脳4 c2w matrix per line (space-separated, row-major)

    Alternative formats:

    ``pose_format: 'dl3dv'``  鈥?reads ``cameras.json`` (dl3dv style).
    ``pose_format: 'transform_json'``  鈥?reads ``transform.json`` with a
        ``frames`` list, each entry having ``image_name`` (relative to the
        ``images/`` sub-directory) and ``transform_matrix`` (4脳4 c2w, row-major).
        Compatible with the 360-UAV dataset layout::

            <dataset_path>/
                images/        鈫?ERP images (*.jpg)
                transform.json

    ``pose_format: '360dvo'`` 鈥?reads the official 360DVO layout::

            <dataset_path>/
                Sequences/<sequence>/*.jpg
                GroundTruth/<sequence>.txt

        Ground-truth rows are interpreted as ``x y z qx qy qz qw`` c2w poses.

    ``pose_format: '360vo'`` 鈥?reads a flat sequence directory::

            <dataset_path>/
                images/*.png
                gt.txt

        ``gt.txt`` rows are interpreted as either
        ``frame_idx image_name x y z qx qy qz qw`` or
        ``image_name x y z qx qy qz qw``.
    """

    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.config = config
        self.begin = config["Dataset"].get("begin", 0)
        self.end = config["Dataset"].get("end", None)
        self._all_color_paths = []

        pose_format = config["Dataset"].get("pose_format", "txt")

        if pose_format == "360dvo":
            color_paths_all, pose_file = self._resolve_360dvo_paths(input_folder, config)
            self._all_color_paths = color_paths_all
        elif pose_format == "transform_json":
            # Images live in <dataset>/images/ for this format
            color_paths_all = sorted(
                glob.glob(f"{input_folder}/images/*.jpg")
                + glob.glob(f"{input_folder}/images/*.png")
            )
            self._all_color_paths = color_paths_all
        elif pose_format == "360vo":
            color_paths_all = sorted(
                glob.glob(f"{input_folder}/images/*.jpg")
                + glob.glob(f"{input_folder}/images/*.jpeg")
                + glob.glob(f"{input_folder}/images/*.png")
                + glob.glob(f"{input_folder}/images/*.JPG")
                + glob.glob(f"{input_folder}/images/*.JPEG")
                + glob.glob(f"{input_folder}/images/*.PNG")
            )
            pose_file = os.path.join(input_folder, "gt.txt")
            self._all_color_paths = color_paths_all
        elif pose_format in ("flat_xyzquat", "flat_xyzquat_ts"):
            color_paths_all = sorted(
                glob.glob(f"{input_folder}/images/*.jpg")
                + glob.glob(f"{input_folder}/images/*.jpeg")
                + glob.glob(f"{input_folder}/images/*.png")
                + glob.glob(f"{input_folder}/images/*.JPG")
                + glob.glob(f"{input_folder}/images/*.JPEG")
                + glob.glob(f"{input_folder}/images/*.PNG")
            )
            default_name = "pose.txt" if pose_format == "flat_xyzquat_ts" else "gt.txt"
            pose_file = os.path.join(
                input_folder,
                config["Dataset"].get("pose_file", default_name),
            )
            self._all_color_paths = color_paths_all
        else:
            color_paths_all = sorted(glob.glob(f"{input_folder}/rgb/*.png"))
            if not color_paths_all:
                color_paths_all = sorted(glob.glob(f"{input_folder}/rgb/*.jpg"))

        self.color_paths = color_paths_all[self.begin : self.end]
        # No separate depth; ERP images are RGB-only
        self.depth_paths = self.color_paths
        self.mono_depth_paths = self.color_paths
        self.n_img = len(self.color_paths)

        if pose_format == "dl3dv":
            self._load_poses_dl3dv(os.path.join(input_folder, "cameras.json"))
        elif pose_format == "360dvo":
            self._load_poses_360dvo(pose_file)
        elif pose_format == "transform_json":
            self._load_poses_transform_json(os.path.join(input_folder, "transform.json"))
        elif pose_format == "360vo":
            self._load_poses_360vo(pose_file)
        elif pose_format == "flat_xyzquat":
            self._load_poses_flat_xyzquat(pose_file, has_timestamp=False)
        elif pose_format == "flat_xyzquat_ts":
            self._load_poses_flat_xyzquat(pose_file, has_timestamp=True)
        else:
            self._load_poses_txt(os.path.join(input_folder, "poses.txt"))

    def _resolve_360dvo_paths(self, input_folder, config):
        dataset_cfg = config["Dataset"]
        sequence = dataset_cfg.get("sequence")

        input_path = Path(input_folder)
        if sequence:
            root_path = input_path
            seq_dir = root_path / "Sequences" / sequence
            pose_file = root_path / "GroundTruth" / f"{sequence}.txt"
        else:
            # Also support dataset_path pointing directly at Sequences/<sequence>.
            sequence = input_path.name
            seq_dir = input_path
            if input_path.parent.name == "Sequences":
                root_path = input_path.parent.parent
            else:
                root_path = input_path
            pose_file = root_path / "GroundTruth" / f"{sequence}.txt"

        color_paths_all = sorted(
            glob.glob(str(seq_dir / "*.jpg"))
            + glob.glob(str(seq_dir / "*.jpeg"))
            + glob.glob(str(seq_dir / "*.png"))
            + glob.glob(str(seq_dir / "*.JPG"))
            + glob.glob(str(seq_dir / "*.JPEG"))
            + glob.glob(str(seq_dir / "*.PNG"))
        )

        if not color_paths_all:
            raise FileNotFoundError(f"360DVO sequence images not found: {seq_dir}")
        if not pose_file.is_file():
            raise FileNotFoundError(f"360DVO ground truth not found: {pose_file}")

        self.sequence = sequence
        return color_paths_all, str(pose_file)

    def _load_poses_txt(self, pose_file):
        """Load poses from a plain-text file (one 4脳4 c2w matrix per line)."""
        self.poses = []
        self.frames = []
        with open(pose_file, "r") as f:
            lines = f.readlines()
        pose_lines = lines[self.begin : (self.end if self.end else None)]
        init_trans = np.array(list(map(float, pose_lines[0].strip().split()))).reshape(4, 4)[:3, 3]
        for i, line in enumerate(pose_lines):
            c2w = np.array(list(map(float, line.strip().split()))).reshape(4, 4)
            c2w[:3, 3] = c2w[:3, 3] - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append({
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": c2w.tolist(),
            })

    def _load_poses_dl3dv(self, pose_file):
        """Load poses from a dl3dv-style cameras.json."""
        self.poses = []
        self.frames = []
        with open(pose_file, "r") as f:
            all_poses = json.load(f)
        selected = all_poses[self.begin : self.end]
        init_trans = np.array(selected[0]["cam_trans"])
        for i, pose in enumerate(selected):
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = rot
            c2w[:3, 3] = np.array([tx, ty, tz]) - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append({
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": c2w.tolist(),
            })

    def _load_poses_360dvo(self, pose_file):
        """Load 360DVO poses from rows of x y z qx qy qz qw.

        The official 360DVO benchmark stores ERP frames under
        Sequences/<sequence> and pseudo ground-truth trajectories under
        GroundTruth/<sequence>.txt.  Each pose row is converted to c2w, then
        translations are expressed relative to the first selected frame.
        """
        self.poses = []
        self.frames = []

        raw = np.loadtxt(pose_file, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.shape[1] != 7:
            raise ValueError(
                f"360DVO pose file must have 7 columns (x y z qx qy qz qw): {pose_file}"
            )

        all_color_paths = self._all_color_paths
        total_images = len(all_color_paths)
        total_poses = len(raw)
        if (
            total_images != total_poses
            and self.config["Dataset"].get("allow_pose_image_mismatch", False)
        ):
            n = min(total_images, total_poses)
            print(
                "[PanoramaParser][360DVO] image/pose count mismatch for "
                f"{getattr(self, 'sequence', pose_file)}: "
                f"images={total_images}, poses={total_poses}; using first {n}."
            )
            all_color_paths = all_color_paths[:n]
            raw = raw[:n]

        selected = raw[self.begin : self.end]
        selected_paths = all_color_paths[self.begin : self.end]
        if not len(selected):
            raise ValueError("360DVO selection is empty after begin/end slicing.")
        if len(selected) != len(selected_paths):
            raise ValueError(
                "360DVO selected image/pose count mismatch after begin/end slicing: "
                f"images={len(selected_paths)}, poses={len(selected)}. "
                f"Full sequence counts are images={total_images}, poses={total_poses}. "
                "Use an end index within the shorter stream or set "
                "Dataset.allow_pose_image_mismatch: true to truncate."
            )

        self.color_paths = selected_paths
        self.depth_paths = selected_paths
        self.mono_depth_paths = selected_paths
        self.n_img = len(selected_paths)

        init_trans = selected[0, :3].copy()
        for i, row in enumerate(selected):
            tx, ty, tz, qx, qy, qz, qw = row.tolist()
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float64) - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append({
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": c2w.tolist(),
            })

    def _load_poses_360vo(self, pose_file):
        """Load 360VO poses from rows containing image names and pose values.

        Supported row formats:
          - frame_idx image_name x y z qx qy qz qw
          - image_name x y z qx qy qz qw
          - x y z qx qy qz qw  (falls back to sorted image order)
        """
        self.poses = []
        self.frames = []

        pose_path = Path(pose_file)
        if not pose_path.is_file():
            raise FileNotFoundError(f"360VO ground truth not found: {pose_file}")

        records = []
        with open(pose_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                image_name = None
                values = None
                if len(parts) == 9:
                    _, image_name, *vals = parts
                    values = vals
                elif len(parts) == 8:
                    image_name, *vals = parts
                    values = vals
                elif len(parts) == 7:
                    values = parts
                else:
                    raise ValueError(
                        "360VO gt.txt rows must have 7, 8, or 9 columns after "
                        f"comment stripping: got {len(parts)} in '{line}'."
                    )
                tx, ty, tz, qx, qy, qz, qw = map(float, values)
                records.append(
                    {
                        "image_name": image_name,
                        "pose": np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float64),
                    }
                )

        if not records:
            raise ValueError(f"360VO pose file is empty after filtering comments: {pose_file}")

        image_lookup = {Path(p).name: p for p in self._all_color_paths}
        if not image_lookup:
            raise FileNotFoundError(
                f"360VO images not found under {Path(self.input_folder) / 'images'}"
            )

        selected = records[self.begin : self.end]
        if not selected:
            raise ValueError("360VO selection is empty after begin/end slicing.")

        fallback_paths = self._all_color_paths[self.begin : self.end]
        selected_paths = []
        for i, rec in enumerate(selected):
            if rec["image_name"] is None:
                if i >= len(fallback_paths):
                    raise ValueError(
                        "360VO image/pose count mismatch when falling back to sorted image order."
                    )
                selected_paths.append(fallback_paths[i])
            else:
                path = image_lookup.get(rec["image_name"])
                if path is None:
                    raise FileNotFoundError(
                        f"360VO image referenced by gt.txt not found: {rec['image_name']}"
                    )
                selected_paths.append(path)

        self.color_paths = selected_paths
        self.depth_paths = selected_paths
        self.mono_depth_paths = selected_paths
        self.n_img = len(selected_paths)

        init_trans = selected[0]["pose"][:3].copy()
        for i, rec in enumerate(selected):
            tx, ty, tz, qx, qy, qz, qw = rec["pose"].tolist()
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float64) - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append(
                {
                    "file_path": self.color_paths[i],
                    "depth_path": self.color_paths[i],
                    "mono_depth_path": self.color_paths[i],
                    "transform_matrix": c2w.tolist(),
                }
            )

    def _load_poses_flat_xyzquat(self, pose_file, has_timestamp=False):
        """Load poses from a flat numeric file.

        Two row layouts are supported:

        - ``has_timestamp=False`` (``flat_xyzquat``): rows are
          ``x y z qx qy qz qw`` (7 cols), one row per image, sorted image order.
        - ``has_timestamp=True`` (``flat_xyzquat_ts``): rows are
          ``timestamp x y z qx qy qz qw`` (8 cols), e.g. the dataset's
          ``pose.txt`` ground truth files.

        Comment lines starting with ``#`` are skipped. Translations are
        re-expressed relative to the first selected frame so frame 0 is
        identity (matches the eval convention).
        """
        self.poses = []
        self.frames = []

        pose_path = Path(pose_file)
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"flat_xyzquat[_ts] pose file not found: {pose_file}"
            )

        rows = []
        expected = 8 if has_timestamp else 7
        with open(pose_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != expected:
                    raise ValueError(
                        f"{pose_file}: expected {expected} cols (has_timestamp="
                        f"{has_timestamp}), got {len(parts)}: '{line}'"
                    )
                if has_timestamp:
                    rows.append([float(x) for x in parts[1:]])
                else:
                    rows.append([float(x) for x in parts])

        if not rows:
            raise ValueError(f"flat_xyzquat[_ts] pose file is empty: {pose_file}")

        raw = np.asarray(rows, dtype=np.float64)
        all_color_paths = self._all_color_paths
        total_images = len(all_color_paths)
        total_poses = len(raw)
        if (
            total_images != total_poses
            and self.config["Dataset"].get("allow_pose_image_mismatch", False)
        ):
            n = min(total_images, total_poses)
            print(
                "[PanoramaParser][flat_xyzquat] image/pose count mismatch "
                f"images={total_images}, poses={total_poses}; using first {n}."
            )
            all_color_paths = all_color_paths[:n]
            raw = raw[:n]

        selected = raw[self.begin : self.end]
        selected_paths = all_color_paths[self.begin : self.end]
        if not len(selected):
            raise ValueError("flat_xyzquat selection empty after begin/end slicing.")
        if len(selected) != len(selected_paths):
            raise ValueError(
                "flat_xyzquat image/pose count mismatch after slicing: "
                f"images={len(selected_paths)}, poses={len(selected)} "
                f"(full counts: images={total_images}, poses={total_poses}). "
                "Set Dataset.allow_pose_image_mismatch: true to truncate."
            )

        self.color_paths = selected_paths
        self.depth_paths = selected_paths
        self.mono_depth_paths = selected_paths
        self.n_img = len(selected_paths)

        init_trans = selected[0, :3].copy()
        for i, row in enumerate(selected):
            tx, ty, tz, qx, qy, qz, qw = row.tolist()
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float64) - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append(
                {
                    "file_path": self.color_paths[i],
                    "depth_path": self.color_paths[i],
                    "mono_depth_path": self.color_paths[i],
                    "transform_matrix": c2w.tolist(),
                }
            )

    def _load_poses_transform_json(self, pose_file):
        """Load poses from a transform.json file (360-UAV / NeRF-style).

        The JSON contains a ``frames`` list; each entry has:
          - ``image_name``: filename (basename) of the image
          - ``transform_matrix``: 4脳4 c2w matrix (row-major list-of-lists)

        Translations are centered so the first frame is at the origin.
        Images are matched by exact ``image_name`` basename.  This avoids
        silent trajectory corruption when files are missing or extra images are
        present under ``images/``.
        """
        self.poses = []
        self.frames = []
        with open(pose_file, "r") as f:
            data = json.load(f)

        all_frame_entries = list(data["frames"])
        image_paths_by_name = {}
        duplicate_names = []
        for path in self._all_color_paths:
            name = os.path.basename(path)
            if name in image_paths_by_name:
                duplicate_names.append(name)
            image_paths_by_name[name] = path

        frame_names = [entry["image_name"] for entry in all_frame_entries]
        frame_name_set = set(frame_names)
        image_name_set = set(image_paths_by_name.keys())
        missing_images = sorted(name for name in frame_names if name not in image_name_set)
        extra_images = sorted(name for name in image_paths_by_name.keys() if name not in frame_name_set)
        if duplicate_names or missing_images or extra_images:
            raise ValueError(
                "transform_json image matching failed: "
                f"duplicate_image_names={sorted(set(duplicate_names))}, "
                f"missing_images={missing_images}, extra_images={extra_images}"
            )

        matched_paths = [image_paths_by_name[name] for name in frame_names]
        selected = all_frame_entries[self.begin : self.end]
        selected_paths = matched_paths[self.begin : self.end]
        if not selected:
            raise ValueError("transform_json selection is empty after begin/end slicing.")

        self.color_paths = selected_paths
        self.depth_paths = selected_paths
        self.mono_depth_paths = selected_paths
        self.n_img = len(selected_paths)

        init_trans = np.array(selected[0]["transform_matrix"])[:3, 3]

        for i, entry in enumerate(selected):
            c2w = np.array(entry["transform_matrix"], dtype=np.float64)
            c2w[:3, 3] = c2w[:3, 3] - init_trans
            w2c = np.linalg.inv(c2w)
            self.poses.append(w2c)
            self.frames.append({
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": c2w.tolist(),
            })


class PanoramaDataset(MonocularDataset):
    """
    Dataset for panoramic (ERP) image sequences.

    Optional ``erp_resize_width`` 脳 ``erp_resize_height`` (OpenCV size W脳H)
    resamples RGB and mono depth; ``PanoramaCamera`` / ERP render then match
    that resolution.  ``Calibration`` must match the **resized** size.

    No ground-truth depth is available.

    Two mono-depth modes are supported:
      1. DAP pre-computed (preferred): if ``dap_depth_dir`` is set in the
         Dataset config, ``*.npy`` files from that directory are loaded as
         metric depth (metres, float32).
      2. Placeholder fallback: the first RGB channel is reused so the rest
         of the pipeline does not break.  The FrontEnd will replace this
         with live DAP inference when ``use_dap: True``.
    """

    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        dataset_path = config["Dataset"]["dataset_path"]
        parser = PanoramaParser(dataset_path, config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses
        # ERP images have no standard GT depth
        self.has_depth = False

        # Optional: pre-computed DAP depth directory (*.npy files, metres)
        self.dap_depth_dir = config["Dataset"].get("dap_depth_dir", None)

        rh = config["Dataset"].get("erp_resize_height")
        rw = config["Dataset"].get("erp_resize_width")
        if (rh is None) ^ (rw is None):
            raise ValueError(
                "Dataset: set both erp_resize_height and erp_resize_width, or neither."
            )
        self._erp_resize_hw = (int(rh), int(rw)) if rh is not None else None

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self._erp_resize_hw is not None:
            rh, rw = self._erp_resize_hw
            h0, w0 = image.shape[0], image.shape[1]
            if (h0, w0) != (rh, rw):
                interp = (
                    cv2.INTER_AREA if h0 > rh and w0 > rw else cv2.INTER_LINEAR
                )
                image = cv2.resize(image, (rw, rh), interpolation=interp)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        # Mono depth: try pre-computed DAP .npy, otherwise use RGB placeholder
        mono_depth = None
        if self.dap_depth_dir is not None:
            stem = os.path.splitext(os.path.basename(color_path))[0]
            npy_path = os.path.join(self.dap_depth_dir, stem + ".npy")
            if os.path.isfile(npy_path):
                mono_depth = np.load(npy_path).astype(np.float32)
                # Clip > 100 m (same rule applied during online DAP inference)
                mono_depth = np.clip(mono_depth, 0.0, 100.0)
                if self._erp_resize_hw is not None:
                    rh, rw = self._erp_resize_hw
                    if mono_depth.shape[:2] != (rh, rw):
                        mono_depth = cv2.resize(
                            mono_depth,
                            (rw, rh),
                            interpolation=cv2.INTER_LINEAR,
                        )

        if mono_depth is None:
            # Placeholder: values are in [0, 1], will be replaced by live DAP
            mono_depth = image[0].cpu().numpy()

        return image, None, pose, mono_depth


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "waymo":
        return WaymoDataset(args, path, config)
    elif config["Dataset"]["type"] == "KITTI":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "dl3dv":
        return dl3dvDataset(args, path, config)
    elif config["Dataset"]["type"] == "panorama":
        return PanoramaDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
