# Image corrector 控制通道与机械线包

本轮只补全已有光学通道的材料表示。两份 image-corrector 装配保留原光学中心、驱动值、圆透镜、HP、孔径、接收面及有效场长度。以下形状是有出处的**工程重建**，不是 CETCOR 制造图，也没有从 CAD 反推出新的场强。

## 证据与未决范围

- [CEOS Linck 2025，第26页](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/01_How-do-aberration-correctors-work-pdf.pdf)明确画出 DP11、DP12、DP21、DP22、DPH1/2、ISh、DSh，以及 HPol/QPol/DSt。DPH 与对应 HP 位于同一场区。这是光学位置证据，不是径向绕组截面。
- [CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en) Fig.5–7 支持同心层的两个正交偶极线圈，Fig.10–11 支持紧凑线圈布置于通道或凹槽。采用该构造族的四个闭合 saddle 线包，不添加假定的铁芯、磁轭、壳体、紧固件或冷却结构。专利实施例**不证明某台 CETCOR 使用相同绕组**。
- 同一专利还举出了 Q/H 线圈和组合驱动，但没有指认本装配的 HPol、QPol、DSt 宿主、具体径向位置或接线。因此这三个通道保持 `physical_host_status="unverified"`，不按谐波阶数生成假磁极，不将其未经证实地绑定到附近 TL/HP。
- 已归档 [Lu 2025](../image_corrector_reference.md)图和 CETCOR 资料用于核对整体拓扑；没有新增可据以认定上述三个未知宿主的制造尺寸。原始证据文件及 SHA 见 [独立条目](cad-library/CETCOR-channel-construction/source-entry.json)。本轮条目已合入资料库索引。

## 当前选定的可装配重建

每个已补通道保持原 key，只新增一个 `mechanical_only` 线包宿主。宿主没有第二份光学驱动。每个宿主包含两层、各两个闭合铜线包，合计每个 variant 8 个宿主 / 32 个材料实体。表中坐标属于 `C3_ImageCorrector.toml` 的模块局部坐标；Probe+Image variant 的这些坐标整体加 324 mm。

| 通道 | 局部光学 Z / mm | 原有效场长度 / mm | 铜线包轴长 / mm | 铜径向分配 / mm |
|---|---:|---:|---:|---|
| DP11 | 1037.9 | 6 | 4 | R 11.1–11.9 |
| DP12 | 1099.65 | 15 | 15 | R 11.1–11.9 |
| DPH1 | 1202.4 | 30 | 33 | R 26–27 |
| DP21 | 1277.9 | 14 | 3 | R 11.1–11.9 |
| DP22 | 1281.4 | 14 | 3 | R 11.1–11.9 |
| DPH2 | 1346.4 | 30 | 33 | R 26–27 |
| ISh | 1420.4 | 14 | 14 | R 11.1–11.9 |
| DSh | 1444.4 | 8 | 8 | R 11.1–11.9 |

普通 DP/ISh/DSh 分配两层 0.32 mm 厚的 120° saddle，端/侧线包带宽 0.2 mm。DPH 分配两层 0.08 mm 厚的 90° saddle、带宽 0.04 mm。这些数值是本模型的材料分配参数；不表示真实线规、圈数、PCB 层数或经过验证的电流负荷。

DPH 直段位于 45°、135°、225°、315° 一带，避开现有六根径向极针；端弧位于 HP 中心 ±16.5 mm，处在原 30 mm 极针段的两端外侧，同时完全位于 HP 原 44 mm 外包络内。HP 的极针、主线包、磁轭、180 mm 外径和 8 mm 通孔均未削切或扩大。DPH 宿主的光学中心显式跟随 HP，移动 HP 后材料和场位置遵循同一绑定。

DP11 的铜长4 mm放入实际5 mm间隙。DP21/22各3 mm铜长沿既有中心布置，彼此间隙0.5 mm。它们原6/14/14 mm长度继续作为**有效场分配**，不是实心壳体或铜长度。`winding_axial_length_mm` 仅影响铜的 CAD/原生几何；求解器仍读原 `effective_thickness_mm`。这没有新增实测 fringe-field 模型，现有均匀有限场近似的限制仍然存在。

## 绑定、编辑与验证

`channel.physical_host_key → coils` 是唯一材料/几何所有者；DPH 进一步由 `coils.optical_reference_host_key → HP` 定位。通道不生成材料，不新增 runtime component，也不重复施加偶极 kick。原配置未声明铜长的线包保持历史全轴长形状。

2026-10-11 的局部证据位于 `outputs/image_corrector_channels_20261011/`：

- `candidate-material-report.json`：107 个已解析实体的真实 BRep 交集，材料碰撞0；8个新宿主相对其他已建材料部件的最小净空0.5 mm。父子关系没有成为碰撞豁免。这是局部静态检查，不是整机/OEM/制造公差认证。
- `optical-comparison.json`：两种装配的809点圆場/HP/Q场数组、光学参数及非零有限偶极场记录与旧状态完全相同。DPH 仍覆盖原中心±15 mm。完整4×4数值响应按1 μm/1 mrad状态尺度归一后，相对 Frobenius 差约2.14×10⁻⁹，参考电子位置差不超过1.36×10⁻¹⁴ m；保留数值差异，没有声称矩阵逐位相同。
- 比较矩阵是既有一阶观察器输出，不冒充新的非线性像差标定。HP 非线性场数组另行精确比较；没有删除 HP/Cs，也没有重算或改造 Tip。
- 最终整机组合审计由本轮主清单统一完成。HPol/QPol/DSt 的内部材料仍未确定，因此不能将“已建材料无碰撞”解释为未知内部净空也通过。
