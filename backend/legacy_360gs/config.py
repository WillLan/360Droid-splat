"""Configuration helpers for the legacy 360GS-SLAM backend."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any


LEGACY_MODEL_DEFAULTS: dict[str, Any] = {
    "sh_degree": 2,
    "source_path": "",
    "model_path": "",
    "resolution": -1,
    "white_background": False,
    "data_device": "cuda",
}


LEGACY_OPT_DEFAULTS: dict[str, Any] = {
    "iterations": 30000,
    "position_lr_init": 0.0016,
    "position_lr_final": 0.00016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 30000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.01,
    "scaling_lr": 0.0008,
    "rotation_lr": 0.001,
    "percent_dense": 0.01,
    "lambda_dssim": 0.2,
    "densification_interval": 450,
    "opacity_reset_interval": 3000,
    "densify_from_iter": 400,
    "densify_until_iter": 15000,
    "densify_grad_threshold": 0.006,
    "init_lr": 6,
}


LEGACY_PIPELINE_DEFAULTS: dict[str, Any] = {
    "convert_SHs_python": False,
    "compute_cov3D_python": False,
}


LEGACY_RESULTS_DEFAULTS: dict[str, Any] = {
    "save_results": True,
    "eval_rendering": True,
    "global_BA": False,
    "kf_render_format": "png",
}


LEGACY_DATASET_DEFAULTS: dict[str, Any] = {
    "single_thread": False,
    "adaptive_pointsize": True,
    "point_size": 0.01,
    "pcd_downsample": 1,
    "pcd_downsample_init": 1,
    "depth_loss": False,
    "erp_resize_height": 512,
    "erp_resize_width": 1024,
    "sky_hemisphere_radius": 300.0,
    "sky_hemisphere_samples": 16384,
}


LEGACY_TRAINING_DEFAULTS: dict[str, Any] = {
    "monocular": True,
    "init_itr_num": 300,
    "init_gaussian_update": 100,
    "init_gaussian_reset": 5000,
    "init_gaussian_th": 0.005,
    "init_gaussian_extent": 50,
    "mapping_itr_num": 200,
    "mapping_itr_nosingle": 200,
    "random_window_frame_per_iter": False,
    "replay_random_kfs": 2,
    "global_BA_itr_num": 0,
    "gaussian_update_every": 200,
    "gaussian_update_offset": 100,
    "gaussian_th": 0.01,
    "gaussian_extent": 50.0,
    "gaussian_reset": 50000,
    "size_threshold": 30,
    "early_prune_kf_count": 3,
    "early_gaussian_th": 0.004,
    "early_disable_opacity_reset": False,
    "window_size": 8,
    "pose_window": 8,
    "window_drop_policy": "oldest",
    "freeze_pose": False,
    "lr": {"cam_rot_delta": 0.0001, "cam_trans_delta": 0.0001},
    "edge_threshold": 1.1,
    "rgb_boundary_threshold": 0.01,
    "render_background_rgb": [0.12, 0.14, 0.18],
    "panorama_render_mode": "pfgs360_gsplat",
    "pfgs360_render_mode": "RGB+ED",
    "pfgs360_distloss": False,
    "erp_area_weight": True,
    "erp_mapping_lambda_dssim": 0.2,
    "erp_mapping_charbonnier_eps": 0.001,
    "erp_mapping_depth_weight": 0.0,
    "erp_mapping_depth_loss_type": "berhu",
    "erp_mapping_sky_rgb_weight": 0.8,
    "erp_sky_bg_skip_during_map_init": True,
    "enable_erp_sky_background": False,
    "enable_neural_sky_bg": True,
    "neural_sky_lr": 0.005,
    "neural_sky_hidden_dim": 64,
    "neural_sky_n_layers": 3,
    "neural_sky_n_freq": 6,
    "neural_sky_alpha_loss_weight": 0.1,
    "enable_sky_opacity_stats": True,
    "sky_opacity_stats_bands": [[0, 30], [30, 60], [60, 90]],
    "sky_opacity_stats_thresholds": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8],
    "force_sky_above_horizon": True,
    "force_sky_min_run_rows": 3,
    "force_sky_row_ratio_thresh": 0.95,
    "force_sky_use_valid_mask": False,
    "enable_loop_closure": True,
    "loop_min_frame_gap": 30,
    "loop_cos_threshold": 0.92,
    "loop_ransac_inlier_ratio": 0.35,
    "loop_pg_optimize_every": 20,
    "enable_submap": False,
    "submap_kf_interval": 10,
    "submap_overlap_kfs": 3,
    "submap_local_refine_iters": 0,
    "enable_periodic_global_ba": True,
    "periodic_global_ba_interval": 30,
    "periodic_global_ba_warmup_kfs": 30,
    "periodic_global_ba_iters": 800,
    "periodic_global_ba_lambda_dssim": 0.1,
    "enable_accel": False,
    "accel_stable_threshold": 0.65,
    "accel_render_scale_min": 0.75,
    "accel_render_scale_max": 1.0,
    "enable_fastgs_erp": False,
    "fastgs_vcd_only": True,
    "fastgs_vcp_warmup_kfs": 8,
    "fastgs_debug_log_scores": True,
    "enable_depth_inlier_densify": True,
    "dia_anchor_prune_enabled": True,
    "dia_anchor_reset_enabled": True,
    "dia_densify_warmup_kfs": 1,
    "dia_densify_depth_rel_thresh": 0.15,
    "dia_densify_max_insert_ratio": 0.25,
    "dia_densify_opacity_min": 0.15,
    "dia_anchor_replacement_neighbor_grid_radius": 2,
    "dia_anchor_replacement_prune_ratio": 0.5,
    "dia_anchor_inconsistent_reset_ratio": 0.5,
    "enable_global_anchor_prune": True,
    "global_anchor_prune_enabled": True,
    "global_anchor_reset_enabled": True,
    "global_anchor_reset_opacity": 0.01,
    "global_prune_depth_rel_thresh": 0.15,
    "global_prune_far_depth_rel_thresh_mult": 2.0,
    "global_prune_far_depth_start": 40.0,
    "global_prune_min_hits": 2,
    "global_reset_min_hits": 2,
    "global_prune_replacement_neighbor_grid_radius": 2,
    "global_prune_respect_protected": True,
    "global_prune_sky_floater_scale_thresh": 4.0,
    "global_prune_sky_floater_radii_thresh": 80.0,
    "global_prune_sky_floater_require_oversized": False,
    "enable_local_anchor_freeze": True,
    "enable_protected_anchor_freeze": True,
    "enable_structure_xyz_freeze": False,
    "anchor_tolerant_match": True,
    "anchor_match_grid_radius": 1,
    "anchor_match_level_radius": 1,
    "anchor_match_dist_factor": 1.25,
    "use_kf_novelty_mask_for_anchor_insert": True,
    "kf_novelty_opacity_min": 0.15,
    "mapping_densify_grad_threshold": 0.008,
    "prune_mode": "slam",
    "prune_num": 1,
    "min_gaussians_after_prune": 8000,
    "protect_early_gaussians": True,
    "protect_sky_occupancy_prune": True,
    "protect_sky_screen_prune": True,
    "protect_sky_world_prune": True,
    "near_layer_prune_interval": 15,
    "near_layer_prune_tau": 0.03,
    "needle_prune_ratio": 8.0,
    "needle_reg_weight": 6.0,
    "gaussian_ratio_cap": 0.0,
    "gaussian_ratio_prune": 0.0,
    "gaussian_ratio_reg_weight": 10.0,
    "gaussian_ratio_reg_weight_init": 10.0,
    "gaussian_ratio_soft": 2.0,
    "gaussian_ratio_soft_init": 5.0,
    "init_size_threshold": 30.0,
    "init_enable_sky_screen_prune": True,
    "panorama_init_reset_sky_opacity": False,
    "debug_prune_stats": False,
    "debug_visualize_anchor_insert": False,
    "debug_visualize_new_anchors": False,
    "debug_save_new_anchor_npz": False,
}


LEGACY_MAP_DEFAULTS: dict[str, Any] = {"mode": "anchor_scaffold_panorama"}


LEGACY_HIERARCHICAL_DEFAULTS: dict[str, Any] = {
    "distance_lis": [40.0, 80.0],
    "voxel_size_lis": [0.12, 0.45, 1.8],
    "max_active_anchors_per_frame": 30000,
    "force_all_visible": False,
    "min_opacity": 0.005,
}


LEGACY_SKYBOX_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "radius": 220.0,
    "n_anchors_init": 2048,
    "init_scale": 10.0,
    "opacity_init": 0.06,
    "feat_dim": 16,
    "freeze_xyz": True,
    "active_budget": 1024,
}


def deep_update(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def build_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config with all legacy backend-required sections populated."""

    cfg = deepcopy(config)
    cfg["model_params"] = deep_update(LEGACY_MODEL_DEFAULTS, cfg.get("model_params"))
    cfg["opt_params"] = deep_update(LEGACY_OPT_DEFAULTS, cfg.get("opt_params"))
    cfg["pipeline_params"] = deep_update(LEGACY_PIPELINE_DEFAULTS, cfg.get("pipeline_params"))
    cfg["Results"] = deep_update(LEGACY_RESULTS_DEFAULTS, cfg.get("Results"))
    cfg["Dataset"] = deep_update(LEGACY_DATASET_DEFAULTS, cfg.get("Dataset"))
    cfg["Training"] = deep_update(LEGACY_TRAINING_DEFAULTS, cfg.get("Training"))
    cfg["MapRepresentation"] = deep_update(LEGACY_MAP_DEFAULTS, cfg.get("MapRepresentation"))
    cfg["Hierarchical"] = deep_update(LEGACY_HIERARCHICAL_DEFAULTS, cfg.get("Hierarchical"))
    cfg["SkyBox"] = deep_update(LEGACY_SKYBOX_DEFAULTS, cfg.get("SkyBox"))
    return cfg


def namespace_from_mapping(mapping: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**deepcopy(mapping))
