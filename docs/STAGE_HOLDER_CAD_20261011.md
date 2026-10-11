# Stage、普通 holder 与装配干涉检查

本轮为机械 CAD 更新。软件 3D、可旋转截面、主 2D 的 holder/管壁截面、STEP 和材料干涉检查共用解析形状。原透镜强度、光学位置、源参数及轴向通束内径没有调整。

## 配件与参数

五种 column 引用同一份 `configs/subassemblies/sample_stage.toml`，以各自原有 sample 中心定位。

| 部分 | 主要尺寸（mm） | 说明 |
|---|---|---|
| 普通 holder 圆杆 | 直径 7；X=18…235 | 7 mm 有 FEI 兼容结构族专利依据，非原厂型号确认 |
| 薄 tip | 宽 5、厚 1.2 | 穿过原有 5.4 mm 极靴间隙；厚圆杆在 X=18 后开始 |
| 样品座 | 直径 3.1、深 0.65；束通孔 2.4 | 有实际承托台阶，不凭空添加样品网格厚度或夹具 |
| 外柄 | 直径 40；X=235…315 | 普通外部操作柄的工程包络 |
| Stage 导向套 | 外径 12、内径 9；X=87…180 | 居中圆杆每侧净空 1 |
| 法兰、台体 | 外径 70、50 | 内部驱动/轴承未知，保留简化壳体 |
| 物镜壳侧孔 | 直径 12.5 | 只在 +X 侧开孔，容纳导向套；对侧壳体保留 |

X 正方向是手柄向外方向。`holder_retraction_mm` 为 0…400 的机械退出量，在 3D Parts 的同一套参数中编辑；不自动改变 `sample.inserted` 或晶体姿态。机械退出截图不能冒充样品已经从电子相互作用计算中移除。

在 Physical Layout 的 **3D Parts** 选择 **Sample Stage / Goniometer** 或 **Side-entry Sample Holder** 可分别查看、编辑和导出；**3D** 与 **Rotating section** 查看装配。图像中的“示意/工程重建”标记仍保留。

尺寸来源与逐项适用边界见 [参考记录](references/cad-stage-holder.md)。原 PDF、说明、图页及 SHA 位于 `docs/references/cad-library/`。杆长、薄 tip、外柄、法兰和台体的未公开尺寸均明确标为工程重建，不冒充 Titan/Spectra 原厂加工图。

## 实际发现并修正的装配问题

1. 原 stage 只有分配包络，没有独立 holder，也没有物镜外壳侧孔。本轮增加实体侧插结构。
2. `vacuum_liner_segments` 曾把全部轴对称电场边界外扩成实体，导致重复材料穿入电极、极靴和探测器包络。现在只有明确的连续管及模块间漂移管被绘为独立材料；每段保留/省略理由写入审计，不通过任意相减所有碰撞体来强行清零。
3. 19.2 mm 连续管与 C3 的 18 mm 极靴孔真实干涉。材料管在 C3 处缩至外径 17.2 mm，前后各有 10 mm 过渡，名义径向净空 0.4 mm，计入锥面后的实际最小距离为 0.398015 mm；最大外包络及内径 5.76 mm 保持原值。
4. C2/C3 孔径板缺少通入管内的侧槽。按已声明孔径板形状增加侧入口，工程单边间隙 0.2 mm，保留背面管壁。

连续管修正写在各 column `[geometry]` 的 `mechanical_material_tube_*` 字段。它们只定义机械材料，既有电场求解边界仍是轴对称等效边界；本轮不声称完成三维电场、真空密封、高压绝缘或制造资格验证。

## 验证与交付

最终运行记录保存在 `outputs/stage_holder_20261011/`：

- `originals/`、`tube-material/originals/`：两阶段原始 TOML 与 SHA。
- `configuration-equivalence.json`：五种装配中旧非 stage 部件的完整参数比较；边界最大约 1.14e-13 mm 浮点舍入已记录。
- `tube-material/material-geometry-equivalence.json`：连续管机械元数据新增；旧配置字段逐项不变。
- `collision/inputs.json`、`collision/report.json`：冻结输入与真实 OpenCascade BRep 审计，不以 AABB 相交代替材料相交。
- `cad/parts/`：11 种模块、158 个配件单元，共 169 个独立配件和模块 STEP。
- `cad/assembled/`、`cad/assembled-energy-filter/`：同一全局坐标下的完整双校正器装配，含/不含 energy filter；主文件均为 `resolved_column_assembly.step`。
- `preview/`：真实 Qt 视图截图及旧 TOML 同相机对照，非生成效果图。

检查覆盖目录中的装配组合和声明的机械状态。未定义内部结构的配置包络单列；不把它们当作已知实物，也不因此宣称任意新尺寸、任意台倾转或未建模附件一定无干涉。STEP 保留解析曲面、可编辑实体、材料叶节点和部件归属；完整原始参数化规则保存在 TOML 与形状定义中，并非厂家原生 CAD 特征树。

### 最终结果

- **60 种目录装配组合 + 5 个 holder 退出位置，共 65 项**，真实 OpenCascade 材料相交检查用时 177 秒；明确材料实体的正体积交叠为 **0**，容差阈值内微小交叠也为 **0**。
- Holder 退出量 0 / 50 / 160 / 320 / 400 mm，对其它已定义材料的最小净空分别为 **0.8 / 1.0 / 1.0 / 137.555 / 217.535 mm**。这些是五个离散位置，不冒充连续运动、任意倾转或制造公差认证。
- **两类旧内部结构仍未完全定义**：FEG+Mono 的 slit 与 C1 aperture 包络、Thermionic 的 accelerator 与 anode aperture 包络。各涉及 20 个配置，合计 40 次未知包络相交；没有凭空补出内部尺寸或通过相减掩盖问题。普通 FEG 的各组合及本轮 holder 状态没有这些未知包络相交。
- 169 个模块/配件 STEP、74 个无能滤完整装配及其单元 STEP、91 个含能滤完整装配及其单元 STEP，**共 334 个文件**，均实际导出并读回通过。完整装配包含两种校正器，模块位置和能滤横向安装变换沿用解析装配输入。
- 针对此轮的综合回归初次为 262 通过、4 失败。修复共享 TOML 保存入口的路径上下文及过时迁移比较后，原 4 项与相邻保存/撤销回归共 **10 项全部通过**；此外实际 CAD 读回 3 项、OCC 审计回归 8 项通过。保留初次日志和复测日志，没有把初次失败改写为成功。
- 软件预览为实际 Qt 控件离屏截图，已查看独立 stage、holder、样品座、极靴间隙及旧/新配置对照；没有声称操作了用户正在运行的桌面窗口。

65 项审计完成后唯一相关实现变化是共享文件的保存路径修复，不改变 CAD 构造器或 TOML。随后重新解析的 65 个装配状态、76 个模块与冻结输入逐字段完全一致，证据见 `collision-post-save-inputs/comparison.json`，没有重复运行 OCC。两种未知包络属于配置中的嵌套分配，不是已确认材料碰撞；嵌套关系同样不能证明内部净空，说明见该目录的 `nested-envelope-interpretation.json`。原审计身份保留，后续实现哈希说明见 `collision/post-audit-source-note.json`。本轮没有再次运行耗时电子波传播；旧非 stage 配件完整参数与光学设置的等价比较另有记录。
