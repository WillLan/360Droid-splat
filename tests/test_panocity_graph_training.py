import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from frontend.pano_droid.graph_dataset import PanoCityGraphDataset
from frontend.pano_droid.factor_graph import PanoFactorGraph
from frontend.pano_droid.graph_losses import build_temporal_edges, select_training_edges
from frontend.pano_droid.graph_tracker import PanoDroidGraphTracker
from frontend.pano_droid.interfaces import PanoFrame
from frontend.pano_droid.model import PanoDroidModel
from frontend.pano_droid.train_graph import load_graph_train_config, train_graph


def _write_mock_panocity_block(root: Path, *, n_frames: int = 7, size=(32, 16)) -> None:
    block = root / "beijing_block1"
    rgb_dir = block / "pano_images"
    depth_dir = block / "panodepth_images"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    frames = []
    width, height = size
    for i in range(n_frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (i * 20) % 255
        rgb[..., 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        rgb[..., 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        depth = np.full((height, width), 1000 + i, dtype=np.uint16)
        name = f"pano_{i:07d}.png"
        depth_name = f"pano_depth_{i:07d}.png"
        Image.fromarray(rgb).save(rgb_dir / name)
        Image.fromarray(depth).save(depth_dir / depth_name)
        T = np.eye(4, dtype=float)
        T[0, 3] = float(i)
        frames.append(
            {
                "name": name,
                "depth": depth_name,
                "transformation_matrix": T.tolist(),
            }
        )
    with open(block / "beijing_Pano_block1_poses.json", "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f)


def test_panocity_graph_dataset_outputs_512x1024(tmp_path: Path):
    _write_mock_panocity_block(tmp_path, n_frames=7)
    ds = PanoCityGraphDataset(
        str(tmp_path),
        n_frames=7,
        resize=(512, 1024),
        depth_scale=0.001,
    )
    sample = ds[0]
    assert sample["images"].shape == (7, 3, 512, 1024)
    assert sample["depths"].shape == (7, 1, 512, 1024)
    assert sample["inverse_depths"].shape == (7, 1, 512, 1024)
    assert sample["poses_c2w"].shape == (7, 4, 4)
    assert sample["block_name"] == "beijing_block1"
    assert torch.isfinite(sample["images"]).all()
    assert torch.isfinite(sample["inverse_depths"]).all()


def test_train_graph_synthetic_smoke(tmp_path: Path):
    cfg = load_graph_train_config(None)
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 2, "n_frames": 3, "height": 16, "width": 32})
    cfg["Model"].update(
        {
            "feature_dim": 8,
            "context_dim": 8,
            "hidden_dim": 8,
            "encoder_base_dim": 8,
            "corr_levels": 1,
            "corr_radius": 1,
            "update_iters": 1,
        }
    )
    cfg["Graph"].update({"temporal_radius": 1, "max_edges_per_step": 2, "loss_sample_height": 8, "loss_sample_width": 16})
    cfg["Training"].update({"output_dir": str(tmp_path), "max_steps": 2, "batch_size": 1, "num_workers": 0, "iters": 1})
    result = train_graph(cfg)
    assert result["steps"] == 2
    assert Path(result["checkpoint"]).is_file()
    assert np.isfinite(result["best_loss"])
    assert "ba_residual" in result["last_metrics"]
    assert "edge_coverage" in result["last_metrics"]
    vis_dir = tmp_path / "visualizations"
    assert (vis_dir / "step_0000001_trajectory.png").is_file()
    assert (vis_dir / "step_0000001_depth.png").is_file()


def test_random_edge_sampling_is_not_prefix_truncation():
    edges = build_temporal_edges(4, radius=2, bidirectional=True)
    first = select_training_edges(edges, max_edges=4, n_frames=4, generator=torch.Generator().manual_seed(1))
    second = select_training_edges(edges, max_edges=4, n_frames=4, generator=torch.Generator().manual_seed(2))
    assert first != edges[:4]
    assert second != edges[:4]
    assert first != second
    assert {idx for edge in first for idx in edge} == {0, 1, 2, 3}


def test_forward_graph_returns_refined_history():
    model = PanoDroidModel(
        feature_dim=8,
        context_dim=8,
        hidden_dim=8,
        encoder_base_dim=8,
        corr_levels=1,
        corr_radius=1,
        update_iters=1,
    )
    images = torch.rand(1, 3, 3, 16, 32)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    poses[:, 1, 0, 3] = 0.05
    poses[:, 2, 0, 3] = 0.1
    pred = model.forward_graph(
        images,
        edges=[(0, 1), (1, 2)],
        poses_c2w=poses,
        num_updates=1,
        ba_iters_per_update=1,
    )
    assert pred["poses_c2w_steps"].shape == (1, 1, 3, 4, 4)
    assert pred["inverse_depth_steps"].shape[:3] == (1, 1, 3)
    assert pred["residual_steps"].shape[:3] == (1, 1, 2)
    assert pred["weight_steps"].shape[3] == 2
    assert pred["upmask_steps"].shape[3] == 8 * 8 * 9
    assert pred["refined_poses_c2w"].shape == (1, 3, 4, 4)
    assert pred["refined_inverse_depth"].shape[:3] == (1, 3, 1)


def test_graph_tracker_uses_graph_path_not_pairwise_forward():
    model = PanoDroidModel(
        feature_dim=8,
        context_dim=8,
        hidden_dim=8,
        encoder_base_dim=8,
        corr_levels=1,
        corr_radius=1,
        update_iters=1,
    )

    def forbidden_pairwise(*args, **kwargs):
        raise AssertionError("pairwise forward should not be used by graph tracker")

    model.forward = forbidden_pairwise
    tracker = PanoDroidGraphTracker(
        model,
        device="cpu",
        window_size=3,
        temporal_radius=1,
        max_factors=4,
        keyframe_threshold=0.0,
        force_keyframe_interval=1,
        ba_iters_per_update=1,
    )
    first = tracker.track(PanoFrame(image=torch.rand(3, 16, 32), timestamp=0.0, frame_id=0))
    second = tracker.track(PanoFrame(image=torch.rand(3, 16, 32), timestamp=1.0, frame_id=1))
    assert first.tracking_status == "initialized"
    assert second.tracking_status == "tracked_graph"
    assert second.ba_residual is not None and np.isfinite(second.ba_residual)
    assert second.inverse_depth is not None and second.inverse_depth.shape[-2:] == (16, 32)


def test_graph_ba_refinement_backpropagates_to_update_and_backbone():
    model = PanoDroidModel(
        feature_dim=8,
        context_dim=8,
        hidden_dim=8,
        encoder_base_dim=8,
        corr_levels=1,
        corr_radius=1,
        update_iters=1,
    )
    with torch.no_grad():
        model.delta_head[-1].weight.normal_(mean=0.0, std=1e-2)
        model.delta_head[-1].bias.fill_(0.05)
    images = torch.rand(1, 3, 3, 16, 32)
    init = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    pred = model.forward_graph(
        images,
        edges=[(0, 1), (1, 2), (2, 1)],
        init_poses_c2w=init,
        num_updates=1,
        ba_iters_per_update=1,
        fixed_frames=1,
    )
    loss = (
        pred["refined_poses_c2w"][:, -1, 0, 3].sum()
        + 1e-3 * pred["refined_inverse_depth"].sum()
        + 1e-4 * pred["refined_inverse_depth_full"].sum()
    )
    loss.backward()

    def grad_sum(param):
        return 0.0 if param.grad is None else float(param.grad.detach().abs().sum())

    assert grad_sum(model.delta_head[-1].weight) > 0.0
    assert grad_sum(model.weight_head[-1].weight) > 0.0
    assert grad_sum(model.graph_agg.damping[-1].weight) > 0.0
    assert grad_sum(model.graph_agg.upmask.weight) > 0.0
    assert grad_sum(model.cnet.encoder.proj.weight) > 0.0
    assert grad_sum(model.fnet.proj.weight) > 0.0


def test_pano_factor_graph_persists_factors_and_hidden_state():
    model = PanoDroidModel(
        feature_dim=8,
        context_dim=8,
        hidden_dim=8,
        encoder_base_dim=8,
        corr_levels=1,
        corr_radius=1,
        update_iters=1,
    )
    graph = PanoFactorGraph(
        model,
        device=torch.device("cpu"),
        window_size=3,
        temporal_radius=1,
        max_factors=4,
        ba_iters_per_update=1,
    )
    for idx in range(3):
        graph.add_frame(PanoFrame(image=torch.rand(3, 16, 32), timestamp=float(idx), frame_id=idx), torch.rand(3, 16, 32))
    assert graph.n_frames == 3
    assert len(graph.fmaps) == graph.n_frames
    assert 0 < len(graph.edges) <= 4
    out = graph.update()
    assert out is not None
    assert out.ba_residual >= 0.0
    assert len(graph.edge_hidden) == len(graph.edges)
    edge = graph.edges.pop()
    graph.inactive_edges.add(edge)
    graph.inactive_target[edge] = graph.factor_target[edge]
    graph.inactive_weight[edge] = graph.factor_weight[edge]
    graph.edge_hidden.pop(edge, None)
    graph.edge_age.pop(edge, None)
    out2 = graph.update()
    assert out2 is not None
    assert out2.inverse_depth.shape[-2:] == (16, 32)
    graph.remove_keyframe(0)
    assert graph.n_frames == 2
    assert len(graph.fmaps) == graph.n_frames
    assert all(max(edge) < 2 for edge in graph.edges)


def test_model_profiles_default_to_droid_base_and_keep_tiny():
    base = PanoDroidModel()
    assert base.profile == "droid_base"
    assert base.feature_dim == 128
    assert base.context_dim == 128
    assert base.hidden_dim == 128
    tiny = PanoDroidModel(profile="tiny")
    assert tiny.feature_dim == 32
    assert tiny.context_dim == 32
    assert tiny.hidden_dim == 48
