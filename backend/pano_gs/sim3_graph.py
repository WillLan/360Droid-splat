"""Block-sparse Sim(3) factor graph for spherical-Selfi window anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import torch

from frontend.pano_droid.spherical_ba import so3_exp
from geometry.sim3 import (
    apply_sim3,
    sim3_components,
    sim3_exp,
    sim3_identity,
    sim3_inverse,
    sim3_log,
)


@dataclass
class Sim3GraphEdge:
    """Relative measurement mapping ``target`` anchor coordinates to ``source``."""

    source: int
    target: int
    measurement_target_to_source: torch.Tensor
    information_diag: torch.Tensor
    edge_type: str = "sequential"
    robust_delta: float = 2.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoincidentPanoramaFactor:
    """Same-center panorama constraint that intentionally leaves scale unobserved."""

    source: int
    target: int
    source_local_pose: torch.Tensor
    target_local_pose: torch.Tensor
    measured_source_to_target_rotation: torch.Tensor
    center_weight: float = 1.0
    rotation_weight: float = 1.0
    robust_delta: float = 2.5
    edge_type: str = "coincident_panorama"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseSphericalFactorBlock:
    """Correspondence-level spherical/depth constraints between two windows."""

    source: int
    target: int
    source_local_pose: torch.Tensor
    target_local_pose: torch.Tensor
    source_bearing: torch.Tensor
    target_bearing: torch.Tensor
    source_depth: torch.Tensor
    target_depth: torch.Tensor
    factor_weight: torch.Tensor
    depth_factor_weight: float = 0.1
    s2_huber_delta_deg: float = 1.0
    use_depth: bool = True
    robust_delta: float = float("inf")
    edge_type: str = "dense_spherical"
    metadata: dict[str, Any] = field(default_factory=dict)


GraphFactor = Sim3GraphEdge | CoincidentPanoramaFactor | DenseSphericalFactorBlock


@dataclass
class Sim3GraphOptimizeResult:
    accepted: bool
    iterations: int
    initial_objective: float
    final_objective: float
    max_update_norm: float
    optimized_node_ids: tuple[int, ...]
    reason: str


def _so3_residual(rotation: torch.Tensor) -> torch.Tensor:
    transform = sim3_identity(device=rotation.device, dtype=rotation.dtype)
    transform[:3, :3] = rotation
    return sim3_log(transform)[3:6]


def s2_log_tangent_coordinates(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Return ``Log_base(point)`` in a deterministic two-vector tangent basis."""

    b = base / torch.linalg.norm(base, dim=-1, keepdim=True).clamp_min(1.0e-8)
    p = point / torch.linalg.norm(point, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = (b * p).sum(dim=-1).clamp(-1.0, 1.0)
    orthogonal = p - dot[..., None] * b
    sine = torch.linalg.norm(orthogonal, dim=-1)
    angle = torch.atan2(sine, dot)
    tangent = orthogonal * (angle / sine.clamp_min(1.0e-8))[..., None]
    tangent = torch.where((sine > 1.0e-7)[..., None], tangent, torch.zeros_like(tangent))

    x_axis = torch.zeros_like(b)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(b)
    y_axis[..., 1] = 1.0
    reference = torch.where((b[..., 0].abs() < 0.9)[..., None], x_axis, y_axis)
    basis_1 = torch.cross(reference, b, dim=-1)
    basis_1 = basis_1 / torch.linalg.norm(basis_1, dim=-1, keepdim=True).clamp_min(1.0e-8)
    basis_2 = torch.cross(b, basis_1, dim=-1)
    coordinates = torch.stack(
        [(tangent * basis_1).sum(dim=-1), (tangent * basis_2).sum(dim=-1)], dim=-1
    )
    # Log is not unique at the antipode.  Returning zero would make a 180°
    # mismatch look perfect, so choose the deterministic first tangent basis
    # and retain the correct geodesic magnitude pi.
    antipodal = (sine <= 1.0e-7) & (dot < 0.0)
    antipodal_value = torch.stack(
        [torch.full_like(angle, math.pi), torch.zeros_like(angle)], dim=-1
    )
    return torch.where(antipodal[..., None], antipodal_value, coordinates)


class GlobalSim3FactorGraph:
    """Window-anchor graph with a matrix-free block-Jacobi PCG solver."""

    def __init__(
        self,
        *,
        damping: float = 1.0e-4,
        max_iterations: int = 8,
        pcg_iterations: int = 64,
        pcg_tolerance: float = 1.0e-6,
        max_translation_update: float = 1.0,
        max_rotation_update_deg: float = 10.0,
        max_log_scale_update: float = 0.25,
    ) -> None:
        self.nodes: dict[int, torch.Tensor] = {}
        self.edges: list[GraphFactor] = []
        self.damping = float(damping)
        self.max_iterations = max(1, int(max_iterations))
        self.pcg_iterations = max(1, int(pcg_iterations))
        self.pcg_tolerance = float(pcg_tolerance)
        self.max_translation_update = float(max_translation_update)
        self.max_rotation_update = math.radians(float(max_rotation_update_deg))
        self.max_log_scale_update = float(max_log_scale_update)
        self.fixed_node_id: int | None = None

    def add_node(self, node_id: int, transform_anchor_to_global: torch.Tensor) -> None:
        node = int(node_id)
        value = transform_anchor_to_global.detach().clone().float()
        if value.shape != (4, 4) or not bool(torch.isfinite(value).all()):
            raise ValueError("Sim(3) graph node must be a finite 4x4 transform")
        scale, rotation, _ = sim3_components(value)
        if float(scale) <= 0.0 or float(torch.linalg.det(rotation)) <= 0.0:
            raise ValueError("Sim(3) graph node must have positive scale and proper rotation")
        self.nodes[node] = value
        if self.fixed_node_id is None:
            self.fixed_node_id = node

    def add_edge(self, edge: GraphFactor) -> None:
        if int(edge.source) not in self.nodes or int(edge.target) not in self.nodes:
            raise KeyError("Both factor endpoints must be added before the edge")
        if isinstance(edge, Sim3GraphEdge):
            if edge.measurement_target_to_source.shape != (4, 4):
                raise ValueError("Sim3GraphEdge measurement must be 4x4")
            if edge.information_diag.numel() != 7:
                raise ValueError("Sim3GraphEdge information_diag must have seven entries")
        if isinstance(edge, DenseSphericalFactorBlock):
            count = int(edge.source_depth.numel())
            if count < 1:
                raise ValueError("DenseSphericalFactorBlock must contain at least one correspondence")
            if tuple(edge.source_bearing.shape) != (count, 3) or tuple(edge.target_bearing.shape) != (count, 3):
                raise ValueError("Dense spherical bearings must have shape Nx3")
            for value in (edge.target_depth, edge.factor_weight):
                if int(value.numel()) != count:
                    raise ValueError("Dense spherical depth/weight arrays must share correspondence count")
            if edge.source_local_pose.shape != (4, 4) or edge.target_local_pose.shape != (4, 4):
                raise ValueError("Dense spherical local poses must be 4x4")
        self.edges.append(edge)

    def transform(self, node_id: int) -> torch.Tensor:
        return self.nodes[int(node_id)]

    def factors_for_nodes(self, node_ids: Iterable[int]) -> list[GraphFactor]:
        selected = {int(node_id) for node_id in node_ids}
        return [edge for edge in self.edges if int(edge.source) in selected or int(edge.target) in selected]

    @staticmethod
    def _factor_residual(
        factor: GraphFactor,
        source_transform: torch.Tensor,
        target_transform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(factor, Sim3GraphEdge):
            predicted = sim3_inverse(source_transform) @ target_transform
            error = sim3_inverse(factor.measurement_target_to_source.to(predicted)) @ predicted
            residual = sim3_log(error)
            information = factor.information_diag.to(device=residual.device, dtype=residual.dtype).clamp_min(0.0)
            return residual, information

        if isinstance(factor, DenseSphericalFactorBlock):
            source_pose = factor.source_local_pose.to(source_transform)
            target_pose = factor.target_local_pose.to(target_transform)
            source_bearing = factor.source_bearing.to(source_transform)
            target_bearing = factor.target_bearing.to(source_transform)
            source_depth = factor.source_depth.to(source_transform).reshape(-1)
            target_depth = factor.target_depth.to(source_transform).reshape(-1)
            weight = factor.factor_weight.to(source_transform).reshape(-1).clamp_min(0.0)

            source_camera = source_bearing * source_depth[:, None]
            source_anchor = source_camera @ source_pose[:3, :3].transpose(0, 1) + source_pose[:3, 3]
            global_points = apply_sim3(source_transform, source_anchor)
            target_anchor = apply_sim3(sim3_inverse(target_transform), global_points)
            target_camera = (target_anchor - target_pose[:3, 3]) @ target_pose[:3, :3]
            predicted_depth = torch.linalg.norm(target_camera, dim=-1).clamp_min(1.0e-8)
            predicted_bearing = target_camera / predicted_depth[:, None]
            s2 = s2_log_tangent_coordinates(target_bearing, predicted_bearing)

            s2_norm = torch.linalg.norm(s2, dim=-1)
            s2_delta = math.radians(max(float(factor.s2_huber_delta_deg), 1.0e-6))
            s2_robust = torch.minimum(
                torch.ones_like(s2_norm),
                s2_norm.new_tensor(s2_delta) / s2_norm.clamp_min(1.0e-8),
            ).detach()
            residual_parts = [s2.reshape(-1)]
            information_parts = [(weight * s2_robust).repeat_interleave(2)]
            if factor.use_depth:
                depth_residual = torch.log(predicted_depth / target_depth.clamp_min(1.0e-8))
                depth_delta = 0.25
                depth_robust = torch.minimum(
                    torch.ones_like(depth_residual),
                    depth_residual.new_tensor(depth_delta) / depth_residual.abs().clamp_min(1.0e-8),
                ).detach()
                residual_parts.append(depth_residual)
                information_parts.append(
                    weight * depth_robust * max(float(factor.depth_factor_weight), 0.0)
                )
            return torch.cat(residual_parts), torch.cat(information_parts).clamp_min(0.0)

        source_pose = factor.source_local_pose.to(source_transform)
        target_pose = factor.target_local_pose.to(target_transform)
        source_center = apply_sim3(source_transform, source_pose[:3, 3])
        target_center = apply_sim3(target_transform, target_pose[:3, 3])
        _, source_rotation, _ = sim3_components(source_transform)
        _, target_rotation, _ = sim3_components(target_transform)
        source_camera_rotation = source_rotation @ source_pose[:3, :3]
        target_camera_rotation = target_rotation @ target_pose[:3, :3]
        predicted_relative = source_camera_rotation.transpose(0, 1) @ target_camera_rotation
        measured = factor.measured_source_to_target_rotation.to(predicted_relative)
        rotation_error = _so3_residual(measured.transpose(0, 1) @ predicted_relative)
        residual = torch.cat([target_center - source_center, rotation_error], dim=0)
        information = residual.new_tensor(
            [factor.center_weight] * 3 + [factor.rotation_weight] * 3
        ).clamp_min(0.0)
        return residual, information

    @staticmethod
    def _robust_cost(residual: torch.Tensor, information: torch.Tensor, delta: float) -> torch.Tensor:
        norm = (information.sqrt() * residual).norm()
        threshold = residual.new_tensor(max(float(delta), 1.0e-8))
        return torch.where(
            norm <= threshold,
            0.5 * norm.square(),
            threshold * (norm - 0.5 * threshold),
        )

    def objective(self, *, node_overrides: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        overrides = node_overrides or {}
        if not self.edges:
            device = next(iter(self.nodes.values())).device if self.nodes else torch.device("cpu")
            return torch.zeros((), device=device)
        costs = []
        for factor in self.edges:
            source = overrides.get(int(factor.source), self.nodes[int(factor.source)])
            target = overrides.get(int(factor.target), self.nodes[int(factor.target)])
            residual, information = self._factor_residual(factor, source, target)
            costs.append(self._robust_cost(residual, information, factor.robust_delta))
        return torch.stack(costs).sum()

    def _linearize_factor(
        self,
        factor: GraphFactor,
        trainable: dict[int, int],
    ) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        source_id, target_id = int(factor.source), int(factor.target)
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        endpoint_ids = [node for node in (source_id, target_id) if node in trainable]
        if not endpoint_ids:
            residual, information = self._factor_residual(factor, source, target)
            return [], [], residual.new_zeros(0)

        def residual_from_delta(delta: torch.Tensor) -> torch.Tensor:
            cursor = 0
            updated_source = source
            updated_target = target
            if source_id in trainable:
                updated_source = sim3_exp(delta[cursor : cursor + 7]) @ source
                cursor += 7
            if target_id in trainable:
                updated_target = sim3_exp(delta[cursor : cursor + 7]) @ target
            residual_value, information_value = self._factor_residual(factor, updated_source, updated_target)
            return information_value.sqrt() * residual_value

        zero = source.new_zeros(7 * len(endpoint_ids))
        weighted_residual = residual_from_delta(zero)
        jacobian = (
            torch.func.jacfwd(residual_from_delta)(zero)
            if isinstance(factor, DenseSphericalFactorBlock)
            else torch.func.jacrev(residual_from_delta)(zero)
        )
        if isinstance(factor, CoincidentPanoramaFactor):
            # A same-center panorama observation contains no scale evidence.
            # Under left-multiplicative Sim(3) perturbations the scale column
            # can otherwise spuriously reduce a non-zero center residual by
            # rescaling translation, so remove that parameter direction from
            # this factor explicitly.
            jacobian = jacobian.clone()
            for endpoint in range(len(endpoint_ids)):
                jacobian[:, endpoint * 7 + 6] = 0.0
        norm = weighted_residual.norm().detach()
        delta = max(float(factor.robust_delta), 1.0e-8)
        robust = torch.where(norm <= delta, norm.new_tensor(1.0), norm.new_tensor(delta) / norm.clamp_min(1.0e-8))
        scale = robust.sqrt()
        weighted_residual = weighted_residual * scale
        jacobian = jacobian * scale
        blocks = [jacobian[:, idx * 7 : (idx + 1) * 7] for idx in range(len(endpoint_ids))]
        return endpoint_ids, blocks, weighted_residual

    def _pcg(
        self,
        linearized: list[tuple[list[int], list[torch.Tensor], torch.Tensor]],
        trainable_ids: list[int],
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        count = len(trainable_ids)
        id_to_slot = {node_id: idx for idx, node_id in enumerate(trainable_ids)}
        block_diag = torch.eye(7, device=gradient.device, dtype=gradient.dtype).repeat(count, 1, 1) * self.damping
        for ids, blocks, _ in linearized:
            for node_id, jacobian in zip(ids, blocks):
                slot = id_to_slot[node_id]
                block_diag[slot] += jacobian.T @ jacobian

        def matvec(vector: torch.Tensor) -> torch.Tensor:
            value = self.damping * vector
            for ids, blocks, _ in linearized:
                projected = None
                for node_id, jacobian in zip(ids, blocks):
                    term = jacobian @ vector[id_to_slot[node_id]]
                    projected = term if projected is None else projected + term
                if projected is None:
                    continue
                for node_id, jacobian in zip(ids, blocks):
                    value[id_to_slot[node_id]] += jacobian.T @ projected
            return value

        rhs = -gradient
        solution = torch.zeros_like(rhs)
        residual = rhs - matvec(solution)
        try:
            preconditioned = torch.linalg.solve(block_diag, residual.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            preconditioned = residual / block_diag.diagonal(dim1=-2, dim2=-1).clamp_min(1.0e-8)
        direction = preconditioned.clone()
        rz = (residual * preconditioned).sum()
        rhs_norm = rhs.norm().clamp_min(1.0e-12)
        for _ in range(self.pcg_iterations):
            product = matvec(direction)
            alpha = rz / (direction * product).sum().clamp_min(1.0e-12)
            solution = solution + alpha * direction
            residual = residual - alpha * product
            if float(residual.norm() / rhs_norm) <= self.pcg_tolerance:
                break
            try:
                next_preconditioned = torch.linalg.solve(block_diag, residual.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                next_preconditioned = residual / block_diag.diagonal(dim1=-2, dim2=-1).clamp_min(1.0e-8)
            next_rz = (residual * next_preconditioned).sum()
            beta = next_rz / rz.clamp_min(1.0e-12)
            direction = next_preconditioned + beta * direction
            preconditioned = next_preconditioned
            rz = next_rz
        return solution

    def optimize(self, active_node_ids: Iterable[int] | None = None) -> Sim3GraphOptimizeResult:
        if len(self.nodes) <= 1 or not self.edges:
            value = float(self.objective().detach().cpu())
            return Sim3GraphOptimizeResult(False, 0, value, value, 0.0, (), "insufficient_graph")
        selected = set(self.nodes) if active_node_ids is None else {int(node) for node in active_node_ids}
        selected &= set(self.nodes)
        trainable_ids = sorted(node for node in selected if node != self.fixed_node_id)
        if not trainable_ids:
            value = float(self.objective().detach().cpu())
            return Sim3GraphOptimizeResult(False, 0, value, value, 0.0, (), "no_trainable_nodes")

        initial = float(self.objective().detach().cpu())
        last = initial
        accepted_any = False
        max_update = 0.0
        actual_iterations = 0
        trainable = {node_id: idx for idx, node_id in enumerate(trainable_ids)}

        for iteration in range(self.max_iterations):
            linearized = [self._linearize_factor(edge, trainable) for edge in self.edges]
            gradient = next(iter(self.nodes.values())).new_zeros(len(trainable_ids), 7)
            for ids, blocks, residual in linearized:
                for node_id, jacobian in zip(ids, blocks):
                    gradient[trainable[node_id]] += jacobian.T @ residual
            if not bool(torch.isfinite(gradient).all()):
                return Sim3GraphOptimizeResult(accepted_any, actual_iterations, initial, last, max_update, tuple(trainable_ids), "non_finite_gradient")
            if float(gradient.norm()) < 1.0e-9:
                break
            update = self._pcg(linearized, trainable_ids, gradient)
            if not bool(torch.isfinite(update).all()):
                return Sim3GraphOptimizeResult(accepted_any, actual_iterations, initial, last, max_update, tuple(trainable_ids), "non_finite_step")

            translation_norm = update[:, :3].norm(dim=-1).clamp_min(1.0e-8)
            rotation_norm = update[:, 3:6].norm(dim=-1).clamp_min(1.0e-8)
            update[:, :3] *= torch.minimum(
                torch.ones_like(translation_norm),
                translation_norm.new_tensor(self.max_translation_update) / translation_norm,
            )[:, None]
            update[:, 3:6] *= torch.minimum(
                torch.ones_like(rotation_norm),
                rotation_norm.new_tensor(self.max_rotation_update) / rotation_norm,
            )[:, None]
            update[:, 6].clamp_(-self.max_log_scale_update, self.max_log_scale_update)
            max_update = max(max_update, float(update.norm(dim=-1).max().detach().cpu()))

            accepted = False
            for step_scale in (1.0, 0.5, 0.25, 0.125):
                proposal = {
                    node_id: sim3_exp(step_scale * update[idx]) @ self.nodes[node_id]
                    for idx, node_id in enumerate(trainable_ids)
                }
                objective = float(self.objective(node_overrides=proposal).detach().cpu())
                if math.isfinite(objective) and objective < last - 1.0e-10:
                    self.nodes.update({node_id: value.detach() for node_id, value in proposal.items()})
                    last = objective
                    accepted = True
                    accepted_any = True
                    actual_iterations = iteration + 1
                    break
            if not accepted:
                break

        return Sim3GraphOptimizeResult(
            accepted_any,
            actual_iterations,
            initial,
            last,
            max_update,
            tuple(trainable_ids),
            "accepted" if accepted_any else "no_descent_step",
        )

    def corrected_camera_poses(self, local_poses_by_node: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        from geometry.sim3 import apply_sim3_to_pose

        output: dict[int, torch.Tensor] = {}
        for node_id, poses in local_poses_by_node.items():
            transform = self.nodes[int(node_id)].to(device=poses.device, dtype=poses.dtype)
            expanded = transform.view(1, 4, 4).expand(int(poses.shape[0]), -1, -1)
            output[int(node_id)] = apply_sim3_to_pose(expanded, poses)
        return output
