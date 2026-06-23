from __future__ import annotations

from pathlib import Path
import uuid

import torch

from frontend.pano_vggt.panovggt_geometry_losses import (
    PanoVGGTGeometryLossWeights,
    build_erp_local_points,
    local_points_to_world,
    panovggt_geometry_loss,
)
from frontend.pano_vggt.train_panovggt_geometry import (
    DifferentiablePanoVGGTGeometryModel,
    load_geometry_train_config,
    save_geometry_checkpoint,
    train_panovggt_geometry,
)


def _sample(b: int = 1, v: int = 4, h: int = 8, w: int = 16) -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    images = torch.rand(b, v, 3, h, w)
    depths = torch.full((b, v, 1, h, w), 2.5)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    poses[:, :, 0, 3] = torch.arange(v).view(1, v).float() * 0.05
    valid = torch.ones(b, v, 1, h, w, dtype=torch.bool)
    sky = torch.zeros_like(valid)
    return {
        "images": images,
        "depths": depths,
        "valid_depth": valid,
        "poses_c2w": poses,
        "sky_mask": sky,
    }


def _synthetic_model_config() -> dict:
    return {
        "use_synthetic_model": True,
        "hidden_dim": 8,
        "image_size": None,
        "trainable_modules": [
            "point_decoder",
            "point_head",
            "global_points_decoder",
            "global_point_head",
            "camera_decoder",
            "camera_head",
        ],
        "strict_trainable_modules": True,
    }


def _workspace_tmp(name: str) -> Path:
    path = Path(".codex_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_geometry_finetune_unfreezes_only_decoder_and_head_modules():
    model = DifferentiablePanoVGGTGeometryModel(_synthetic_model_config(), device=torch.device("cpu"))

    assert not any(param.requires_grad for param in model.unwrapped_model.aggregator.parameters())
    for name in model.trainable_modules:
        module = dict(model.unwrapped_model.named_modules())[name]
        assert any(param.requires_grad for param in module.parameters())


def test_geometry_finetune_backward_keeps_backbone_frozen():
    sample = _sample()
    model = DifferentiablePanoVGGTGeometryModel(_synthetic_model_config(), device=torch.device("cpu"))
    pred = model(sample["images"])
    loss, _metrics = panovggt_geometry_loss(pred, sample, PanoVGGTGeometryLossWeights())

    loss.backward()

    assert all(param.grad is None for param in model.unwrapped_model.aggregator.parameters())
    trainable_grad = [
        float(param.grad.detach().abs().sum())
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert sum(trainable_grad) > 0.0


def test_geometry_loss_masks_sky_region():
    sample = _sample(h=6, w=10)
    sample["sky_mask"][:, :, :, :3, :] = True
    gt_local = build_erp_local_points(sample["depths"])
    gt_world = local_points_to_world(gt_local, sample["poses_c2w"])
    pred = {
        "depth": sample["depths"].clone(),
        "local_points": gt_local.clone(),
        "camera_poses": sample["poses_c2w"].clone(),
        "world_points": gt_world.clone(),
        "global_points": gt_world.clone(),
    }
    pred["depth"][:, :, :, :3, :] = 100.0
    pred["local_points"][:, :, :3, :, :] = 100.0
    pred["world_points"][:, :, :3, :, :] = 100.0
    pred["global_points"][:, :, :3, :, :] = 100.0

    loss, metrics = panovggt_geometry_loss(pred, sample, PanoVGGTGeometryLossWeights(smooth=0.0))

    assert float(loss) < 1.0e-6
    assert metrics["valid_ratio"] < 1.0


def test_geometry_loss_is_global_gauge_invariant():
    sample = _sample()
    gt_local = build_erp_local_points(sample["depths"])
    gt_world = local_points_to_world(gt_local, sample["poses_c2w"])
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([3.0, -1.0, 0.5])
    pred_poses = transform.view(1, 1, 4, 4) @ sample["poses_c2w"]
    pred_world = gt_world + transform[:3, 3].view(1, 1, 1, 1, 3)
    pred = {
        "depth": sample["depths"].clone(),
        "local_points": gt_local.clone(),
        "camera_poses": pred_poses,
        "world_points": pred_world,
        "global_points": pred_world.clone(),
    }

    loss, metrics = panovggt_geometry_loss(pred, sample, PanoVGGTGeometryLossWeights(smooth=0.0))

    assert float(loss) < 1.0e-5
    assert float(metrics["pose_rpe_rot_deg"]) < 1.0e-3
    assert float(metrics["pose_rpe_trans"]) < 1.0e-5


def test_geometry_training_uses_gradient_accumulation():
    tmp_path = _workspace_tmp("geometry_training")
    cfg = load_geometry_train_config(None)
    cfg["Training"].update(
        {
            "steps": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "num_workers": 0,
            "output_dir": str(tmp_path / "geom"),
            "save_every": 10,
            "val_every": 10,
            "vis_every": 10,
        }
    )
    cfg["Dataset"].update({"synthetic": True, "synthetic_length": 6, "height": 8, "width": 16, "validation_fraction": 0.0})
    cfg["Model"].update(_synthetic_model_config())
    cfg["Validation"].update({"enabled": False})
    cfg["WeightsAndBiases"].update({"enabled": False, "mode": "disabled"})

    result = train_panovggt_geometry(cfg, command=["test"])

    assert result["optimizer_steps"] == 2
    assert result["micro_steps"] == 4
    assert (tmp_path / "geom" / "best_geometry.pt").exists()


def test_geometry_checkpoint_payload_is_inference_loader_compatible():
    tmp_path = _workspace_tmp("geometry_checkpoint")
    cfg = load_geometry_train_config(None)
    cfg["Model"].update(_synthetic_model_config())
    model = DifferentiablePanoVGGTGeometryModel(cfg["Model"], device=torch.device("cpu"))
    path = tmp_path / "best_geometry.pt"

    save_geometry_checkpoint(path, model=model, config=cfg, step=7, metrics={"val/total_loss": 1.2})
    payload = torch.load(path, map_location="cpu")

    assert payload["format"] == "panovggt_geometry_finetune_v1"
    assert isinstance(payload["model_state_dict"], dict)
    assert payload["global_step"] == 7
