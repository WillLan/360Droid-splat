# Global-Map-Sim3 技术修改与验收报告

日期：2026-07-21  
基线：`00ad8a6`（SphereGlue + PointMap-Sim3）  
主体实现：`dd671d5`

## 1. 目标与结论

本次修改为 SphereGlue 主线增加默认关闭的
`two_frame_global_map_full_sim3` 对齐模式。非首窗口在 incoming Gaussian
插入前，以当前全局 Gaussian map 在两个重叠帧上的渲染 ray depth 为全局尺度参考，
将 incoming chunk 的未全局对齐预测深度拟合为绝对 Sim3。

经代码审查、逻辑修正和全量测试后，实现满足既定方案。旧
`two_frame_pointmap_full_sim3` 路径和默认配置保持不变；新模式失败时事务性回退旧
PointMap-Sim3，两种方法均失败时由窗口事务恢复图、地图、owner 和窗口顺序。

## 2. 数据流与图约束

1. 首窗口以 identity 建图，不执行 alignment render。
2. 后续窗口先完成 refiner/local BA，保留 incoming chunk-local pose 与 depth。
3. 使用前一 chunk 当前图优化后的全局 pose，在两个重叠帧各渲染一次完整 Gaussian map。
4. 有效交集要求：有限且范围内的双侧深度、global alpha≥0.05、incoming confidence≥0.05、static、geometry-consistent、non-sky。
5. 使用 seam-safe Fibonacci 采样；每帧容量与最小支持沿用主线配置。两个视角在每轮 Huber IRLS 后重新归一化为等权，不使用 confidence 软权重。
6. 直接由同一 bearing 和各自 ray depth 构造 global/local 三维点，避免对预计算 point map 做双线性插值造成 seam 和球面非线性偏差。
7. 有足够支持且 Sim3 有限即接受；残差、尺度变化和与 packet 的冲突仅记录，不作拒绝条件。
8. map 成功时以绝对 Sim3 初始化 incoming 图节点，并添加 fixed-root→current 的 `global_map_anchor_sim3` 边。该边保存完整 Sim3 measurement，但 information 仅开启 log-scale。
9. 同一窗口的 SphereGlue stride factor 保留 S² bearing residual，关闭 depth residual，并在自动微分与解析线性化两条路径中将两端 scale Jacobian 严格置零。
10. map 失败时复用旧 PointMap-Sim3 的 full-Sim3 初始化、depth residual 和 scale Jacobian，不改变基线语义。

## 3. 与需求逐项对照

| 要求 | 实现状态 | 证据 |
|---|---:|---|
| 新模式默认关闭 | 满足 | 独立实验配置启用，基础配置未改 mode |
| 两个全局 overlap depth render | 满足 | 首窗 0 次，普通 map 对齐 2 次，incoming render 0 次 |
| alpha/confidence/static/geometry/sky/depth 过滤 | 满足 | 统一在 global-map overlap geometry 收集阶段完成 |
| Fibonacci、512/帧、2048 总量、2048/帧上限 | 满足 | 继承正式 PointMap-Sim3 配置并在构造期验证容量 |
| 双帧等权、仅 Huber IRLS | 满足 | 每轮 robust weight 后按帧重新归一化为 0.5/0.5 |
| 有限 Sim3 直接接受 | 满足 | 不调用 residual、holdout、scale-change 质量 gate |
| 失败回退 PointMap-Sim3 | 满足 | 支持不足、渲染/求解异常、非有限结果触发 fallback |
| 两者失败事务回滚 | 满足 | 使用现有 boundary transaction snapshot/restore |
| Map 管尺度 | 满足 | root scale-only edge；SphereGlue depth 与 scale Jacobian 关闭 |
| fallback 完全保持旧行为 | 满足 | fallback factor 的 `use_depth=true`、`optimize_scale=true` |
| 不改公开数据与地图格式 | 满足 | 未改 `FrontendOutput`、checkpoint、PLY 或 Gaussian schema |
| W&B 保持核心白名单 | 满足 | 仅固定名称的 map scale、fallback、最终 path/scale drift；per-view 诊断仅本地 |
| 独立 A/B 配置 | 满足 | 新配置继承 SphereGlue PointMap 基线，仅改 mode 和运行标识 |

## 4. 审查中发现并修正的问题

### 4.1 球面点插值偏差

初版对 incoming `centers_world` 进行双线性采样。三维球面点对像素并非线性函数，
这种采样会在全图产生小偏差，并在 panoramic seam 附近放大。修正后使用采样 bearing、
ray depth 和相机 pose 直接构造三维点。合成尺度恢复误差由约 0.24% 降至测试容差
`2e-4` 以内。

### 4.2 IRLS 后帧权重漂移

初版只在 IRLS 初始化时令两帧等权，Huber 权重更新后两帧总权重可能不同。修正后每轮
IRLS 都分别归一化两个视角，使最终权重和严格保持 `[0.5, 0.5]`。

### 4.3 核心尺度诊断不完整

补充 W&B summary 中的最终 `scale_drift_percent` 与
`path_length_scale_ratio`，并保留 map-anchor scale、root-relative scale 和累计
fallback rate。未引入动态 per-view panel。

### 4.4 并行改动提交污染

隔离发布树的全量测试发现首个实现提交误带入一段 PaGeR 测试，而对应实现属于另一条
并行开发线。发布修正删除了该无关测试；远端主线只包含 Global-Map-Sim3 相关代码，
不包含 PaGeR 实现或配置。

## 5. 测试覆盖

- 已知旋转、平移、尺度的 absolute current-local→global Sim3 恢复。
- panoramic seam 两侧同时取样。
- 大于旧 `max_scale_change` 的有限尺度仍直接接受。
- alpha、confidence、static、geometry、sky、depth 联合过滤。
- 两帧最终 IRLS 权重严格等于 0.5/0.5。
- map 成功时 root scale-only edge 持久存在。
- Dense spherical factor 的自动微分 scale Jacobian 严格为零；解析路径同步实现。
- map 失败时旧 PointMap-Sim3 行为不变。
- map 与 PointMap 同时失败时完整事务回滚。
- 新配置与 SphereGlue 基线递归比较，仅允许 mode 与运行标识差异。
- 全量 pytest 通过，只有两个环境/可选依赖测试按原逻辑 skip。

## 6. Brooks-Lint Review

**Mode:** PR Review  
**Scope:** `dd671d5` 及推送前修正，6 个生产/配置/测试文件；因变更超过 500 行，按高风险路径抽样复核  
**Health Score:** 100/100

### Findings

推送前复核没有遗留 Critical、Warning 或 Suggestion。初审发现的球面插值偏差、
IRLS 帧权重漂移和最终尺度诊断缺口均已修正并有回归测试保护。

### Summary

超过 500 行的提交规模本身是 Change Propagation 审查信号；逐项核对后，这些修改分别落在
配置、图因子、运行诊断和测试的必要边界，没有扩散到公开接口或默认行为。实现仍保持单一
概念主线：Gaussian map 提供跨 chunk 尺度，SphereGlue 提供旋转和平移。

## 7. 运行配置

正式实验配置：
`configs/spherical_selfi_ob3d_global_map_sim3_sphereglue_ba_100_pfgs360_freeze.yaml`

必须解析为：

- `rendered_overlap_alignment.mode=two_frame_global_map_full_sim3`
- `rendered_overlap_alignment.acceptance_policy=diagnostics_only`
- `global_graph.node_mode=chunk_first_stride`
- `WeightsAndBiases.runtime_log_preset=slam_core_visuals`

本实验与当前 SphereGlue PointMap-Sim3 100 帧结果构成严格 A/B；除 alignment mode、
诊断和运行标识外，其余递归配置一致。
