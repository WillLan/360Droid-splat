# PFGS360 后端优化迁移技术报告

日期：2026-07-26

基线：`origin/main@3827b6b`

实验主线：PaGeR + SphereGlue + Global-Map-Sim3 + Refiner

## 1. 结论

本次修改新增默认不启用的
`map_optimization.strategy=pfgs360_official_chunkwise`，没有改写任何历史实验配置。
新策略实现如下状态机：

```text
首 chunk refined-anchor bootstrap
→ INITIAL 1000
→ 后续每个 chunk：CAMERA 500 → DIA → JOINT 500
→ 全序列 FINETUNE 10000
```

迁移保留当前系统的 PaGeR 深度、SphereGlue 匹配、Global-Map-Sim3、
VoxelAnchorRefiner、refined-anchor footprint admission、Hash 去重和 DIA 原子替换；
不恢复 PFGS360 的 raw-depth KNN 新点初始化。

## 2. 阶段语义

### INITIAL

- 仅首个 chunk 执行 1000 步。
- 使用首 chunk 四帧和 refined anchors。
- 严格冻结全部 pose、Gaussian xyz、SkyBox/Sky-Sphere。
- 更新 DC、SH-rest、opacity、scale 和 rotation。
- 损失为球面加权 `0.8 L1 + 0.2 DSSIM`，并加入权重 `0.01`
  的 PaGeR log-depth Huber loss。
- 使用随机背景和 PFGS360 Adam 超参数。

### CAMERA

- 后续每个 chunk 执行 500 步。
- Gaussian 和 root pose 严格冻结。
- 从除固定 root 外的全部已访问帧按 70% 后半历史、30% 前半历史采样。
- 使用相邻渲染深度构造 consistency mask。
- 只优化 pose，梯度逐元素裁剪到 `1e-2`。

### DIA

- CAMERA 后、JOINT 前执行。
- 保留 refined-anchor footprint、Hash 与原子替换。
- 当前 chunk 的两张新帧只提供新增区域，不累计旧点删除/reset responsibility。
- 旧历史帧产生 provisional evidence；只有实际通过 Hash 的替代 anchor
  才能确认删除或 opacity reset。

### JOINT

- 每个后续 chunk 执行 500 步。
- 从全部已访问帧按 PFGS360 70/30 策略采样。
- 固定 root pose，同时更新其他 pose 和全局所有 Gaussian 参数。
- 不使用 owner、visibility 或 render-contributor 硬掩码。
- 每 100 步执行一次 split、duplicate、父点删除、低 opacity cull 和 OOD cull。

### FINETUNE

- 全序列结束后严格执行 10000 步。
- 前 500 步每 100 步允许 topology update；后 9500 步 topology 固定。
- xyz 学习率按指数从 `1.6e-4` 衰减至 `1.6e-6`。
- pose 学习率按指数从 `1e-3` 衰减至 `5e-6`。

## 3. 拓扑与状态安全

- split 两个子点，子点 scale 为父点 scale 除以 `1.6`，并删除父点。
- duplicate、split 子点完整继承 owner、level、voxel size、quality、来源和
  DIA evidence。
- 删除 opacity `<0.005`、非有限点，以及距所有已访问相机都超过 `1e5`
  的点；OOD 不再错误地以世界原点为唯一参考。
- topology mapping 同步重映射 Gaussian Adam moments；topology 中重建
  optimizer 时也保留 pose Adam moments。
- `max_total_gaussians` 仍是独立显存安全容量。
- recovery checkpoint 原子保存 map、pose base/delta、stage、global step、
  chunk index、active SH degree、Gaussian Adam moments、densification statistics
  和 CPU/CUDA RNG。

## 4. Pose 回写

- fixed root owner 不更新。
- 每个 owner 的多帧 SE3 correction 使用 rotation/translation 分离的
  `1.5 × MAD` Huber 权重，执行 5 次 IRLS。
- 只更新 owner rotation 和 translation，Sim3 scale 保持逐位不变。
- owner 更新前先 materialize 并 rebase 该 owner 的 Gaussian xyz、rotation、
  scale、SH 和相应 Adam moments，使世界 Gaussian 不发生渲染跳变。
- 每个 `PoseDelta` 随新 graph base 重参数化，保持 photometric effective pose
  不变，避免同一 pose correction 被应用两次。
- 回写完成后刷新 factor local pose 和 geometry snapshot，后续
  Global-Map-Sim3/回环仍可继续工作。

## 5. 失败语义

- 新策略不创建优化事务回滚快照。
- 有限但质量变差的结果仍接受，不设置 RGB、轨迹或 topology gate。
- loss、gradient、Gaussian、pose、metadata 或 optimizer state 非有限/错位时
  立即抛出异常并终止。
- INITIAL、每个 chunk 和 FINETUNE 前覆盖写入恢复 checkpoint；实验框架应将
  失败 run 标记为 incomplete，且不自动重试算法失败。

## 6. 配置

OB3D 100 帧 A/B 配置：

`configs/spherical_selfi_ob3d_global_map_sim3_sphereglue_pager_ba_100_pfgs360_official_chunkwise.yaml`

RAR 全序列验证 manifest：

`configs/formal/panogsslam_formal_rar_pano_v14_pfgs360_official.yaml`

两者均保持 PaGeR、SphereGlue、Global-Map-Sim3、`diagnostics_only`、
`chunk_first_stride`、Refiner 和 refined-anchor 原子 Hash replacement。

## 7. 验证

- `python -m compileall backend frontend mapping system tests`：通过。
- PFGS360 与 Global backend 定向测试：通过。
- formal/RAR 配置测试：通过。
- 四 chunk 合成状态机 smoke：通过，顺序严格为
  `INITIAL → 3 × (CAMERA → DIA → JOINT)`。
- 全量 `pytest`：659 项收集，657 通过，2 项既有可选环境测试跳过。
- `git diff --check`：通过；仅有 Windows 工作树换行提示。

尚未运行真实 checkpoint、OB3D 100 帧或 RAR 长序列实验。这些实验应在本提交
推送并完成服务器资源检查后另行启动。

## 8. Brooks-Lint Review

**Mode:** PR Review

**Scope:** sampled review of 4 production files, 2 configurations, and 1 test file;
the change exceeds 500 lines, so the highest-risk state transitions and state-remapping
paths were reviewed in full

**Health Score:** 100/100

### Findings

没有遗留 Critical、Warning 或 Suggestion。审查中发现并修复了以下问题：

- OOD cull 的参考系错误；
- INITIAL/root pose 未严格关闭梯度；
- topology optimizer 重建时 pose Adam state 丢失；
- chunk 与 FINETUNE 的配置解析知识重复；
- recovery checkpoint 缺少 active SH degree；
- optimizer step 后缺少 pose/Gaussian 参数有限性检查。

### Summary

大规模跨模块改动本身是 Change Propagation 信号，但本次传播范围对应状态机、
地图参数化、图回写和测试这四个必要边界，没有扩展公开 `FrontendOutput`、PLY
或历史配置行为。修正后，PFGS360 状态、Gaussian topology 和 Global-Map-Sim3
owner 之间的责任边界保持一致。
