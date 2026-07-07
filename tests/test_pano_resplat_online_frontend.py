from __future__ import annotations

import torch
from torch import nn

from frontend.pano_droid.interfaces import FrontendOutput, PanoFrame
from frontend.pano_vggt.pano_resplat_online_frontend import PanoReSplatOnlineFrontend
from frontend.pano_vggt.resplat_types import PanoGaussianState


class _FakeTracker:
    def __init__(self) -> None:
        self.frames: list[PanoFrame] = []
        self.emitted = 0
        self.features_by_frame: dict[int, torch.Tensor] = {}

    def initialize(self, sequence_meta: dict) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames = []
        self.emitted = 0
        self.features_by_frame = {}

    def load_checkpoint(self, path: str) -> None:
        return None

    def track(self, frame: PanoFrame) -> FrontendOutput:
        self.frames.append(frame)
        return self._output(frame)

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        ready = [self._output(frame) for frame in self.frames[self.emitted :]]
        self.emitted = len(self.frames)
        return ready

    def flush(self) -> list[FrontendOutput]:
        return self.pop_ready_outputs()

    def _output(self, frame: PanoFrame) -> FrontendOutput:
        h, w = int(frame.image.shape[-2]), int(frame.image.shape[-1])
        pose = torch.eye(4)
        inv = torch.ones(1, h, w)
        conf = torch.ones(1, h, w)
        world = torch.zeros(h, w, 3)
        world[..., 2] = 1.0
        return FrontendOutput(
            frame_id=int(frame.frame_id),
            timestamp=float(frame.timestamp),
            pose_c2w=pose,
            relative_pose=None,
            pose_confidence=1.0,
            inverse_depth=inv,
            depth_confidence=conf,
            spherical_flow=None,
            keyframe_score=1.0,
            is_keyframe=True,
            ba_residual=None,
            tracking_status="tracked_fake",
            world_points=world,
            world_points_confidence=conf,
            valid_world_points_mask=torch.ones(1, h, w, dtype=torch.bool),
        )


class _FakeReSplat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.last_context_shape: tuple[int, ...] | None = None

    def forward(self, context, target=None, *, num_refine: int = 0, return_all: bool = True):
        self.last_context_shape = tuple(int(v) for v in context["images"].shape)
        n = 6
        means = context["world_points"][:, :, :1, :n, :].reshape(1, n * 4, 3)[:, :n]
        log_scales = torch.full((1, n, 3), -3.0, device=means.device, dtype=means.dtype)
        rotations = torch.zeros(1, n, 4, device=means.device, dtype=means.dtype)
        rotations[..., 0] = 1.0
        opacity = torch.ones(1, n, 1, device=means.device, dtype=means.dtype)
        sh = torch.zeros(1, n, 3, 9, device=means.device, dtype=means.dtype)
        state = PanoGaussianState(
            means=means,
            log_scales=log_scales,
            rotations_unnorm=rotations,
            opacity_logits=opacity,
            sh_coeffs=sh,
            latent_features=torch.zeros(1, n, 4, device=means.device, dtype=means.dtype),
            source_view_ids=torch.zeros(1, n, dtype=torch.long, device=means.device),
            source_uv=torch.zeros(1, n, 2, device=means.device, dtype=means.dtype),
            valid_mask=torch.ones(1, n, dtype=torch.bool, device=means.device),
            confidence=torch.ones(1, n, 1, device=means.device, dtype=means.dtype),
        )
        return {"final_state": state, "compactor_debug": {}, "update_metrics": []}


def test_pano_resplat_online_emits_one_artifact_per_four_frame_window() -> None:
    tracker = _FakeTracker()
    resplat = _FakeReSplat()
    frontend = PanoReSplatOnlineFrontend(
        tracker=tracker,  # type: ignore[arg-type]
        resplat_frontend=resplat,  # type: ignore[arg-type]
        window_size=4,
        stride=4,
        image_height=8,
        image_width=16,
        num_refine=1,
        require_features=False,
        allow_synthetic_features=True,
        synthetic_feature_dim=8,
        synthetic_feature_stride=4,
    )

    for frame_id in range(4):
        image = torch.rand(3, 8, 16)
        frontend.track(PanoFrame(image=image, timestamp=float(frame_id), frame_id=frame_id))
    outputs = frontend.pop_ready_outputs()
    artifacts = frontend.consume_resplat_artifacts()

    assert [out.frame_id for out in outputs] == [0, 1, 2, 3]
    assert len(artifacts) == 1
    assert artifacts[0].frame_ids == (0, 1, 2, 3)
    assert artifacts[0].final_state.num_gaussians == 6
    assert resplat.last_context_shape == (1, 4, 3, 8, 16)


class _FakeSystemFrontend:
    def __init__(self) -> None:
        self.frames: list[PanoFrame] = []
        self.outputs: list[FrontendOutput] = []
        self.artifact_emitted = False
        self.images: dict[int, torch.Tensor] = {}

    def initialize(self, sequence_meta: dict) -> None:
        return None

    def reset(self) -> None:
        return None

    def load_checkpoint(self, path: str) -> None:
        return None

    def track(self, frame: PanoFrame) -> FrontendOutput:
        self.frames.append(frame)
        self.images[int(frame.frame_id)] = frame.image.detach().cpu()
        out = _FakeTracker()._output(frame)
        self.outputs.append(out)
        return out

    def pop_ready_outputs(self) -> list[FrontendOutput]:
        out = self.outputs
        self.outputs = []
        return out

    def flush(self) -> list[FrontendOutput]:
        return []

    def image_for_frame(self, frame_id: int) -> torch.Tensor | None:
        return self.images.get(int(frame_id))

    def consume_resplat_artifacts(self):
        if self.artifact_emitted or len(self.frames) < 4:
            return []
        self.artifact_emitted = True
        state = _state_for_system(torch.tensor([[0.0, 0.0, 1.0]]))
        artifact = type(
            "Artifact",
            (),
            {"window_id": 0, "frame_ids": (0, 1, 2, 3), "final_state": state},
        )()
        return [artifact]


def _state_for_system(means: torch.Tensor) -> PanoGaussianState:
    n = int(means.shape[0])
    rotations = torch.zeros(1, n, 4)
    rotations[..., 0] = 1.0
    return PanoGaussianState(
        means=means.view(1, n, 3),
        log_scales=torch.full((1, n, 3), -3.0),
        rotations_unnorm=rotations,
        opacity_logits=torch.ones(1, n, 1),
        sh_coeffs=torch.zeros(1, n, 3, 9),
        latent_features=torch.zeros(1, n, 4),
        source_view_ids=torch.zeros(1, n, dtype=torch.long),
        source_uv=torch.zeros(1, n, 2),
        valid_mask=torch.ones(1, n, dtype=torch.bool),
        confidence=torch.ones(1, n, 1),
    )


def test_pano_resplat_online_direct_fusion_does_not_use_depth_seed_path(monkeypatch, tmp_path) -> None:
    import system.pano_droid_gs_slam as slam_module
    from backend.pano_gs.mapper import PanoGaussianMapper
    from mapping.gaussian_initializer import GaussianInitializer

    fake_frontend = _FakeSystemFrontend()
    monkeypatch.setattr(slam_module, "build_frontend_from_config", lambda config: fake_frontend)

    def fail_seed_path(self, *args, **kwargs):
        raise AssertionError("direct ReSplat fusion must not call GaussianInitializer.from_frontend_output")

    monkeypatch.setattr(GaussianInitializer, "from_frontend_output", fail_seed_path)
    captured: dict[str, int] = {}

    def fake_optimize(self, *, frame_ids, iters=20):
        captured["iters"] = int(iters)
        captured["frame_count"] = len(frame_ids)
        return {"loss": 0.0, "steps": float(iters)}

    monkeypatch.setattr(PanoGaussianMapper, "optimize_resplat_global_window", fake_optimize)

    def fake_frames(config):
        for frame_id in range(4):
            yield PanoFrame(image=torch.rand(3, 8, 16), timestamp=float(frame_id), frame_id=frame_id)

    monkeypatch.setattr(slam_module, "iter_sequence_frames", fake_frames)
    config = {
        "Dataset": {"synthetic": True},
        "Frontend": {"mode": "pano_resplat_online"},
        "ReSplatFusion": {"enabled": True, "voxel_size": 0.1, "merge_radius": 0.1},
        "BackendOptimization": {"enabled": True, "sh_degree": 2, "ReSplatGlobal": {"iters": 20}},
        "Renderer": {"allow_smoke_fallback": True},
        "Mapping": {"refine_steps_per_keyframe": 0},
        "WeightsAndBiases": {"enabled": False},
        "Results": {
            "save_dir": str(tmp_path),
            "save_final_ply": False,
            "save_final_checkpoint": False,
            "save_final_all_frame_renders": False,
        },
    }

    summary = slam_module.PanoDroidGSSlamSystem(config).run(max_frames=4)

    assert captured == {"iters": 20, "frame_count": 4}
    assert summary["backend_resplat_fusion_count"] == 1
    assert summary["backend_last_resplat_inserted"] == 1
