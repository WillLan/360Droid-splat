# PFGS360 DIA + Refined Anchor 后端修改技术报告

## 目标与边界

本次修改保留 PointMap-Sim3、SphereGlue Local BA、全局因子图、回环、VoxelAnchorRefiner，以及 `CAMERA 50 + JOINT 50` 优化流程。修改范围只覆盖严格 PFGS360 后端的 Gaussian 拓扑更新：DIA 负责删除/reset，DIA 新增区域只从 Refiner 输出的 voxel anchors 中选点。

正式配置：

`configs/spherical_selfi_ob3d_pointmap_sim3_sphereglue_ba_100_pfgs360_refined_anchor_50_50.yaml`

该配置解析后的硬约束为：

- `two_frame_pointmap_full_sim3`
- `diagnostics_only`
- `chunk_first_stride`
- `slam_core_visuals`
- `superpoint_sphereglue`
- `VoxelAnchorRefiner.enabled=true`
- `camera_steps=50`、`joint_steps=50`

## 修改后的数据流

```text
PointMap-Sim3 / 因子图 / 回环
  -> Voxel 化 + VoxelAnchorRefiner
  -> 首 chunk：全部数值合法 refined anchors 初始化地图
  -> 后续 chunk：CAMERA 50
  -> PFGS360 DIA 查询旧地图
  -> mono-inlier responsibility 删除；其余 inconsistency responsibility reset opacity
  -> 将两个新帧 mono-inlier mask 映射到 source-view 支持的 refined anchors
  -> 仅与两个新帧中可见旧点进行 same-level、1.0 voxel Hash
  -> 提交 refined anchors
  -> JOINT 50 联合优化 Gaussian 与访问过的非固定 pose
```

## DIA 删除与有效域

新正式路径的 DIA mask仍由以下三项交集构成：

- render depth inconsistency；
- monocular depth multi-view consistency；
- GNCC/patch comparison 判定 monocular depth 更好。

计算前提保留有限 ray depth 和深度范围。唯一额外语义 gate 是 sky mask。该路径不再使用 alpha、depth confidence、static mask 或 geometry consistency 缩小 DIA mask。

Refiner anchors 只在 DIA 完成删除/reset 后进入新增回调，不参与旧点 query。因此，在旧地图、相机和 DIA 输入相同的前提下，Refiner 开关不会改变删除/reset 集合。

## Refined Anchor 新增

首 chunk 不调用 raw PFGS depth backprojection，而是直接提交全部数值合法 refined anchors。后续 chunk：

1. 只处理当前 packet 的两个非重叠新帧；
2. 用 CAMERA 阶段后的 c2w 将 anchor 世界中心投影到 panoramic 图像；
3. 水平坐标循环处理 seam，垂直坐标限定有效范围；
4. source-view bit 必须包含当前帧；
5. 在该帧 `mono_inlier` mask 上进行 nearest sampling；
6. 两帧命中取 union，一个 anchor 最多提交一次；
7. 保留 Refiner 的 xyz、scale、rotation、opacity、SH、level、voxel size 和 quality；
8. 仅执行 finite、正 scale/voxel size、合法 level 的数值安全检查。

新路径不调用 `append_pfgs360_points()`，因此没有 occupied-grid、KNN scale、随机 quaternion、RGB-to-SH 重初始化或 `min_unique_voxels` 后门槛。

## Hash 与事务

Hash 在 DIA 删除后重新渲染得到旧点可见性，再执行：

- incoming：DIA 选中的 refined anchors；
- existing：仅两个新帧中可见的旧 Gaussian；
- same-level；
- `radius_voxels=1.0`；
- 不渲染 incoming map；
- Hash 后仅剩一个 anchor 也允许提交；
- 不设置单窗口新增上限。

提交仍保留 owner/voxel 的确定性压缩和 `max_total_gaussians` 显存安全容量。DIA、anchor 映射、Hash、提交或后续优化任何一步异常时，恢复 Gaussian 参数、metadata、Adam moments、pose delta 和 owner transform；失败诊断不会泄漏到下一窗口。

## JOINT 拓扑行为

新正式路径设置 `topology_refine_enabled=false`，因此 JOINT 阶段：

- 不累积 absgrad/max-radii；
- 不执行 split；
- 不执行 duplicate；
- 不执行 opacity `<0.005` cull；
- 不执行 non-finite/OOD topology cull；
- 不输出 `refine_*` 指标。

旧严格 PFGS 配置保留历史行为，确保已有实验可复现。DIA query 删除是新正式路径唯一的质量删点机制，容量上限只作为显存保护。

## 诊断

每个新帧在本地保存包含以下内容的 PNG：

- RGB；
- DIA mono-inlier mask；
- source-view 支持的 anchor candidates；
- Hash rejected anchors；
- final admitted anchors；
- 合成 overlay。

两个新帧合成一个 chunk panel，并仅通过固定 W&B key `backend/pfgs360_new_anchor_admission` 上传。`slam_core_visuals` 白名单未增加动态 per-view key。

## 验证结果

- `python -m compileall backend system tests`：通过。
- 定向 PFGS/PointMap-Sim3/Refiner 测试：通过。
- 全量 `pytest -q`：通过，2 项原有环境相关测试跳过。

新增回归覆盖：

- official DIA gate 忽略 alpha/confidence，但 sky 必须剔除；
- DIA 删除发生在 refined-anchor 新增之前；
- 首 chunk 不调用 raw point growth；
- Refiner 属性不经 KNN/random/RGB 重初始化；
- panoramic seam 投影；
- semantic quality gate 可在 official path 中绕过；
- topology disable 后累计 JOINT steps 不触发传统 refine；
- 插入失败恢复 topology 与 Adam moments；
- 正式配置继承与固定 W&B key；
- 两张本地逐帧图和一个 chunk panel 的生成。

## 已知边界

- anchor-to-mask 映射采用中心投影和 source-view bit，而不是保留完整 voxel member membership；细长或跨深度边界 voxel 的中心可能落在邻接像素。
- Hash 提交前的绿色诊断点表示通过 DIA 与 Hash 的最终 admission candidates；全局容量安全上限在极端饱和时仍可能进一步减少实际净增量。
- 本地合成测试验证了算法与事务语义，真实 OB3D 质量仍需通过新的 100 帧实验比较空洞率、DIA 删除量、anchor 新增量、PSNR 和轨迹指标。
