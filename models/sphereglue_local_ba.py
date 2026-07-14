"""External SuperPoint + SphereGlue adapter for spherical local BA.

The external projects and weights are deliberately not vendored.  SuperPoint
is loaded from a user-provided LightGlue checkout and SphereGlue is loaded from
its own checkout.  This repository only owns the conversion into the internal
``Stage3MatchCache`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch

from geometry.spherical_erp import erp_pixel_to_unit_ray, sample_erp_with_wrap
from models.spherical_selfi_stage3_ba import Stage3MatchCache, all_directed_pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pure_torch_knn_graph(
    x: torch.Tensor,
    k: int,
    *,
    flow: str = "source_to_target",
    cosine: bool = False,
    **_: Any,
) -> torch.Tensor:
    """Small deterministic replacement for ``torch_cluster.knn_graph``.

    SphereGlue uses at most a few thousand unit-sphere nodes and batch size one,
    so a dense distance matrix is acceptable for this isolated matcher.  This
    avoids coupling the experiment to a torch-cluster wheel matching the
    server's exact PyTorch/CUDA build.
    """

    if x.ndim != 2:
        raise ValueError(f"SphereGlue KNN expects NxD positions, got {tuple(x.shape)}")
    count = int(x.shape[0])
    if count <= 1:
        return torch.empty(2, 0, device=x.device, dtype=torch.long)
    neighbors = min(max(1, int(k)), count - 1)
    if cosine:
        normalized = torch.nn.functional.normalize(x.float(), dim=-1, eps=1.0e-8)
        distance = 1.0 - normalized @ normalized.transpose(0, 1)
    else:
        distance = torch.cdist(x.float(), x.float())
    distance.fill_diagonal_(torch.inf)
    neighbor_index = torch.topk(
        distance,
        k=neighbors,
        dim=-1,
        largest=False,
        sorted=True,
    ).indices
    center = torch.arange(count, device=x.device).repeat_interleave(neighbors)
    neighbor = neighbor_index.reshape(-1)
    if flow == "target_to_source":
        return torch.stack([center, neighbor], dim=0)
    if flow == "source_to_target":
        return torch.stack([neighbor, center], dim=0)
    raise ValueError("flow must be 'source_to_target' or 'target_to_source'")


def sphereglue_unit_cartesian(keypoints_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Reproduce SphereGlue's published ERP pixel convention exactly."""

    x, y = keypoints_xy.unbind(dim=-1)
    theta = (1.0 - (x + 0.5) / float(width)) * (2.0 * torch.pi)
    phi = (y + 0.5) * torch.pi / float(height)
    sin_phi = torch.sin(phi)
    return torch.stack(
        [torch.cos(theta) * sin_phi, torch.sin(theta) * sin_phi, torch.cos(phi)],
        dim=-1,
    )


@dataclass
class _SparseFrameFeatures:
    keypoints_xy: torch.Tensor
    uv: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor
    depth: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.uv.shape[0])


class SphereGlueLocalBAMatcher:
    """Build local-BA correspondences using external SuperPoint + SphereGlue."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: torch.device | str,
        superpoint: Any | None = None,
        sphereglue: Any | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.device = torch.device(device)
        self.max_queries = max(1, int(self.config.get("num_queries", 1024)))
        self.extractor_max_keypoints = max(
            self.max_queries,
            int(self.config.get("extractor_max_keypoints", self.max_queries * 2)),
        )
        self.min_depth = float(self.config.get("min_depth", 0.05))
        self.max_depth = float(self.config.get("max_depth", 20.0))
        self.min_factor_weight = float(self.config.get("min_factor_weight", 0.01))
        self.knn = max(1, int(self.config.get("knn", 20)))
        self.provenance = dict(provenance or {})
        if superpoint is None or sphereglue is None:
            superpoint, sphereglue, external = self._load_external_models()
            self.provenance.update(external)
        self.superpoint = superpoint.to(self.device).eval()
        self.sphereglue = sphereglue.to(self.device).eval()

    def _load_external_models(self) -> tuple[Any, Any, dict[str, Any]]:
        lightglue_root_value = self.config.get("lightglue_repo")
        sphereglue_root_value = self.config.get("sphereglue_repo")
        superpoint_checkpoint_value = self.config.get("superpoint_checkpoint")
        checkpoint_value = self.config.get("sphereglue_checkpoint")
        if (
            not lightglue_root_value
            or not sphereglue_root_value
            or not superpoint_checkpoint_value
            or not checkpoint_value
        ):
            raise ValueError(
                "SphereGlue local BA requires matching.lightglue_repo, "
                "matching.sphereglue_repo, matching.superpoint_checkpoint, and "
                "matching.sphereglue_checkpoint. "
                "Keep these research-only dependencies outside this repository."
            )
        lightglue_root = Path(str(lightglue_root_value)).expanduser().resolve()
        sphereglue_root = Path(str(sphereglue_root_value)).expanduser().resolve()
        superpoint_checkpoint = Path(str(superpoint_checkpoint_value)).expanduser().resolve()
        checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
        lightglue_source = lightglue_root / "lightglue" / "superpoint.py"
        sphereglue_source = sphereglue_root / "model" / "sphereglue.py"
        for label, path in (
            ("LightGlue SuperPoint source", lightglue_source),
            ("SuperPoint checkpoint", superpoint_checkpoint),
            ("SphereGlue source", sphereglue_source),
            ("SphereGlue checkpoint", checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        root_text = str(lightglue_root)
        inserted = root_text not in sys.path
        if inserted:
            sys.path.insert(0, root_text)
        try:
            lightglue_module = importlib.import_module("lightglue")
            superpoint_class = getattr(lightglue_module, "SuperPoint")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import external LightGlue SuperPoint. Install its runtime "
                "requirements in the pfgs360 environment without copying it into this repository."
            ) from exc
        finally:
            if inserted:
                sys.path.remove(root_text)

        superpoint_state = torch.load(
            superpoint_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        # LightGlue's SuperPoint constructor downloads this state internally.
        # Substitute the explicitly configured, hash-recorded external file so
        # inference never depends on an implicit network fetch or torch cache.
        original_url_loader = torch.hub.load_state_dict_from_url
        torch.hub.load_state_dict_from_url = lambda *_args, **_kwargs: superpoint_state
        try:
            superpoint = superpoint_class(
                nms_radius=int(self.config.get("nms_radius", 4)),
                max_num_keypoints=self.extractor_max_keypoints,
                detection_threshold=float(self.config.get("detection_threshold", 5.0e-4)),
                remove_borders=int(self.config.get("remove_borders", 4)),
            )
        finally:
            torch.hub.load_state_dict_from_url = original_url_loader

        spec = importlib.util.spec_from_file_location(
            "_spherical_selfi_external_sphereglue",
            sphereglue_source,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load SphereGlue module from {sphereglue_source}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(
                "Failed to import external SphereGlue. The pfgs360 environment must provide "
                "torch-geometric; torch-cluster is not required by this adapter."
            ) from exc
        module.knn_graph = pure_torch_knn_graph
        model_config = {
            "K": int(self.config.get("chebyshev_k", 2)),
            "GNN_layers": ["cross"],
            "match_threshold": float(self.config.get("match_threshold", 0.2)),
            "sinkhorn_iterations": int(self.config.get("sinkhorn_iterations", 20)),
            "aggr": str(self.config.get("aggregation", "add")),
            "knn": self.knn,
            "max_kpts": self.max_queries,
            "descriptor_dim": 256,
            "output_dim": 512,
        }
        sphereglue = module.SphereGlue(model_config)
        if checkpoint.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors
            except ImportError as exc:
                raise RuntimeError(
                    "A .safetensors SphereGlue checkpoint requires the safetensors package"
                ) from exc
            state = load_safetensors(str(checkpoint), device="cpu")
        else:
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            state = payload.get("MODEL_STATE_DICT", payload) if isinstance(payload, dict) else payload
        sphereglue.load_state_dict(state, strict=True)
        return superpoint, sphereglue, {
            "lightglue_superpoint_source": str(lightglue_source),
            "lightglue_superpoint_source_sha256": _sha256(lightglue_source),
            "superpoint_checkpoint": str(superpoint_checkpoint),
            "superpoint_checkpoint_sha256": _sha256(superpoint_checkpoint),
            "sphereglue_source": str(sphereglue_source),
            "sphereglue_source_sha256": _sha256(sphereglue_source),
            "sphereglue_checkpoint": str(checkpoint),
            "sphereglue_checkpoint_sha256": _sha256(checkpoint),
            "knn_backend": "pure_torch_deterministic",
        }

    @torch.no_grad()
    def _extract_frame(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        static_valid: torch.Tensor,
    ) -> _SparseFrameFeatures:
        height, width = int(depth.shape[-2]), int(depth.shape[-1])
        features = self.superpoint.extract(image.to(self.device).float(), resize=None)
        keypoints = features["keypoints"][0].float()
        descriptors = features["descriptors"][0].float()
        scores = features["keypoint_scores"][0].float()
        if descriptors.ndim != 2 or int(descriptors.shape[-1]) != 256:
            raise ValueError(
                f"SuperPoint descriptors must be Nx256 for SphereGlue, got {tuple(descriptors.shape)}"
            )
        uv = keypoints + 0.5
        sampled_depth = sample_erp_with_wrap(
            depth.to(self.device).float().reshape(1, 1, height, width),
            uv.reshape(1, -1, 2),
        )[0, :, 0]
        sampled_static = sample_erp_with_wrap(
            static_valid.to(self.device).float().reshape(1, 1, height, width),
            uv.reshape(1, -1, 2),
        )[0, :, 0]
        valid = (
            torch.isfinite(keypoints).all(dim=-1)
            & torch.isfinite(descriptors).all(dim=-1)
            & torch.isfinite(scores)
            & torch.isfinite(sampled_depth)
            & (sampled_depth >= self.min_depth)
            & (sampled_depth <= self.max_depth)
            & (sampled_static > 0.5)
        )
        keep = torch.nonzero(valid, as_tuple=False).flatten()
        if int(keep.numel()) > self.max_queries:
            # SuperPoint confidence is used only to cap sparse detections; it
            # never reweights the spherical BA residual independently of the
            # learned SphereGlue assignment score.
            rank = torch.topk(scores[keep], k=self.max_queries, sorted=True).indices
            keep = keep[rank]
        return _SparseFrameFeatures(
            keypoints_xy=keypoints[keep],
            uv=uv[keep],
            descriptors=descriptors[keep],
            scores=scores[keep],
            depth=sampled_depth[keep],
        )

    @staticmethod
    def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp(1.0e-8, 1.0 - 1.0e-8)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log())

    @torch.no_grad()
    def build_cache(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        *,
        static_valid_mask: torch.Tensor | None = None,
    ) -> Stage3MatchCache:
        if images.ndim != 5 or depth.ndim != 5:
            raise ValueError("images/depth must have shapes BxSx3xHxW and BxSx1xHxW")
        batch, views, _, height, width = (int(value) for value in images.shape)
        if batch != 1:
            raise ValueError("SphereGlue local BA currently requires batch_size=1")
        if tuple(depth.shape) != (batch, views, 1, height, width):
            raise ValueError("SphereGlue images and depth must share B/S/H/W dimensions")
        if static_valid_mask is None:
            static_valid_mask = torch.ones_like(depth, dtype=torch.bool)
        if tuple(static_valid_mask.shape) != tuple(depth.shape):
            raise ValueError("static_valid_mask must match depth")

        frames = [
            self._extract_frame(
                images[0, view],
                depth[0, view],
                static_valid_mask[0, view],
            )
            for view in range(views)
        ]
        query_count = self.max_queries
        device = self.device
        source_uv = torch.zeros(1, views, query_count, 2, device=device)
        source_ray = torch.zeros(1, views, query_count, 3, device=device)
        source_depth = torch.full(
            (1, views, query_count),
            max(self.min_depth, 1.0e-4),
            device=device,
        )
        source_valid = torch.zeros(1, views, query_count, device=device, dtype=torch.bool)
        for view, frame in enumerate(frames):
            count = frame.count
            if count:
                source_uv[0, view, :count] = frame.uv
                source_ray[0, view, :count] = erp_pixel_to_unit_ray(
                    frame.uv, height, width
                ).float()
                source_depth[0, view, :count] = frame.depth
                source_valid[0, view, :count] = True

        edges = all_directed_pairs(views, device=device)
        edge_lookup = {tuple(pair): index for index, pair in enumerate(edges.tolist())}
        edge_count = int(edges.shape[0])
        target_uv = torch.zeros(1, edge_count, query_count, 2, device=device)
        target_ray = torch.zeros(1, edge_count, query_count, 3, device=device)
        score = torch.zeros(1, edge_count, query_count, device=device)
        margin = torch.zeros_like(score)
        entropy = torch.zeros_like(score)
        valid = torch.zeros(1, edge_count, query_count, device=device, dtype=torch.bool)
        mutual = torch.zeros_like(valid)
        target_valid = torch.zeros_like(valid)

        def write_direction(
            src: int,
            tgt: int,
            source_index: torch.Tensor,
            target_index: torch.Tensor,
            confidence: torch.Tensor,
        ) -> None:
            edge = edge_lookup[(src, tgt)]
            if int(source_index.numel()) == 0:
                return
            target_value = frames[tgt]
            uv_value = target_value.uv[target_index]
            target_uv[0, edge, source_index] = uv_value
            target_ray[0, edge, source_index] = erp_pixel_to_unit_ray(
                uv_value, height, width
            ).float()
            score[0, edge, source_index] = confidence
            entropy[0, edge, source_index] = self._binary_entropy(confidence)
            accepted = confidence >= self.min_factor_weight
            valid[0, edge, source_index] = accepted
            mutual[0, edge, source_index] = True
            target_valid[0, edge, source_index] = True

        for first in range(views):
            for second in range(first + 1, views):
                frame_first, frame_second = frames[first], frames[second]
                if frame_first.count <= self.knn or frame_second.count <= self.knn:
                    continue
                prediction = self.sphereglue(
                    {
                        "unitCartesian1": sphereglue_unit_cartesian(
                            frame_first.keypoints_xy, height, width
                        ).unsqueeze(0),
                        "unitCartesian2": sphereglue_unit_cartesian(
                            frame_second.keypoints_xy, height, width
                        ).unsqueeze(0),
                        "h1": frame_first.descriptors.unsqueeze(0),
                        "h2": frame_second.descriptors.unsqueeze(0),
                        "scores1": frame_first.scores.unsqueeze(0),
                        "scores2": frame_second.scores.unsqueeze(0),
                    }
                )
                matches = prediction["matches0"][0].long()
                confidence = prediction["matching_scores0"][0].float().clamp(0.0, 1.0)
                matched_source = torch.nonzero(matches >= 0, as_tuple=False).flatten()
                matched_target = matches[matched_source]
                matched_confidence = confidence[matched_source]
                write_direction(
                    first,
                    second,
                    matched_source,
                    matched_target,
                    matched_confidence,
                )
                # SphereGlue's matches0 are already mutual; invert that sparse
                # assignment instead of running a second, numerically different
                # network pass for the reverse directed BA edge.
                write_direction(
                    second,
                    first,
                    matched_target,
                    matched_source,
                    matched_confidence,
                )

        metadata = {
            "matcher": "superpoint_sphereglue",
            "num_queries": query_count,
            "extractor_max_keypoints": self.extractor_max_keypoints,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "min_factor_weight": self.min_factor_weight,
            "per_view_keypoints": [frame.count for frame in frames],
            "source_area_reweight": False,
            **self.provenance,
        }
        return Stage3MatchCache(
            source_uv=source_uv,
            source_ray=source_ray,
            source_depth=source_depth,
            source_valid=source_valid,
            edges=edges,
            target_uv=target_uv,
            target_ray=target_ray,
            top1_cosine=score,
            top2_margin=margin,
            entropy=entropy,
            valid_mask=valid,
            factor_weight=score,
            mutual_mask=mutual,
            target_valid=target_valid,
            metadata=metadata,
        )
