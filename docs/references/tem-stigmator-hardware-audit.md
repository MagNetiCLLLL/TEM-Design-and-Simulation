# TEM 自身的 stigmator：结构、位置与控制核查

核查日期：2026-10-19。主装配：FEG + C3 + Probe Corrector + No Energy Filter。
本报告区分厂家公开证据、专利实施例和项目配置；未按示意图比例反推 OEM 尺寸。
校正器内部仅用于核对控制归属，不把其所有 multipole 当成额外 TEM stigmator。

## 机型不能直接等同于项目装配名称

| 参考家族 | 可确认内容 | 本项目适用范围 |
| --- | --- | --- |
| Tecnai 20 / F20 / F30 | FEI Alignments 手册 p11–13、p28 描述 C1、C2；p24 说明 mini-condenser 位于 beam deflection coils 下游 | 双聚光镜、无 corrector 的拓扑参考；F30 并不表示有独立 C3 |
| Titan | 厂家培训 Module 1 p1 描述增加的 C3；p8 对比 Tecnai 双聚光镜行为 | C3 无校正器、C3 加 CEOS corrector 的光学家族参考 |
| Spectra Ultra | DS0369 p3 描述三聚光镜、S-CORR/CEOS、S-TWIN objective | 证明系统功能组合；没有提供 stigmator 内部安装坐标 |
| Iliad Ultra | DS0510 p3 将 NanoPulser 放在 gun 与 condenser module 之间 | 上游附加模块参考；不能据此复制旧 CESCOR 内部结构 |

来源：[FEI Tecnai 原手册](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_alignments_doc.pdf)、
[Titan 厂家培训](https://emfacility.science.oregonstate.edu/sites/emfacility.science.oregonstate.edu/files/2025-09/Titan%20training%20manual_0.pdf)、
[Spectra Ultra](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/ds0369-spectra-ultra-s-tem-semiconductor.pdf)、
[Iliad Ultra](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)。

## 原始截面与现有三维表示

| 原始资料 | 检查位置 | 能支持什么 | 不能据此认定什么 |
| --- | --- | --- | --- |
| Philips CM12/STEM 技术论文 | PDF p11 / printed p279，Fig.6、7 | 八只 stigmator 线圈按 45° 分布；两级 BT 在上物镜内孔、MC 上游，两级 PT 在下物镜内孔；MC 与上物镜结构紧凑集成 | Fig.7 没有标出 condenser/objective stigmator；没有证明所有器件在同一可拆卸圆筒中；也不是 Titan 截面 |
| Philips US4902940A | claim 4 | 同平面的两组四极，结构角相差 45° | 不提供当前机器的制造尺寸 |
| US5481164A | Fig.1、2 及正文 | 两组 normal/skew 四极通常轴向重叠；图中的轴向分离用于示意 | 不能画成两只前后串联的圆透镜 |
| JEOL US6548816B2 | Fig.1、3 | 八线圈和十二线圈是可行的不同实现 | 四极作用不等于四只物理线圈，也不自动证明八只铁磁极 |
| Philips US3150258A | PDF p2，Fig.2 | 物镜附近可使用非磁支撑中的倾斜带芯小线圈 | 不能把该专利铁芯直接加入 Titan 的现有模型 |
| CEOS US20090079531A1 | Fig.10、11 | 薄层绕制 PCB 线圈可装在透镜内孔或凹槽 | 不证明某一 Titan/Spectra 的具体 condenser stigmator 宿主 |
| FEI US8569693B2 | PDF p4、5，Fig.2、3 | 物镜附近的 objective stigmator 与首级成像/衍射透镜附近的 diffraction stigmator 有独立实体的实施依据 | 专利新增的 LD stigmator 不属于本项目已确认的必装器件；不提供当前系统的准确毫米坐标 |

来源：[Philips 技术论文](https://www.worldradiohistory.com/Archive-Company-Publications/Philips-Technical-Review/80s/Philips-Technical-Review-1987-11.pdf)、
[US4902940A](https://patents.google.com/patent/US4902940A/en)、
[US5481164A](https://patents.google.com/patent/US5481164A/en)、
[US6548816B2](https://patents.google.com/patent/US6548816B2/en)、
[US3150258A](https://patents.google.com/patent/US3150258A/en)、
[US20090079531A1](https://patents.google.com/patent/US20090079531A1/en)、
[US8569693B2](https://patents.google.com/patent/US8569693B2/en)。

本报告2026-10-19基线采用八个同轴向位置的径向铜线圈块，表示两组四极绕组。
它们不是独立的圆透镜，也没有凭空增加铁芯、磁轭、绝缘套或八只磁极。
线圈径向/轴向占比是从现有包络分配的工程示意值，不是 OEM 线圈尺寸。
Gun stigmator 使用另外的磁梯度模型，缺少已声明的绕组结构，仍保留未解析包络；
没有把柱体八线圈模型无依据地套到电子枪上。

2026-10-20后续结构补充见 [Condenser Deflector / Condenser Stigmator](condenser-deflector-stigmator-structure.md)：
默认 C3 + Probe Corrector 的 Condenser Stigmator 已在保留原八铜绕组的基础上，
显式加入非OEM工程骨架、绝缘包层、胶座和开槽支架。上段是补充前的基线，
不能再用来断言该器件当前没有被动支撑。

## 控制别名不等于新的物理器件

Thermo S-CORR 指南 p2 将 `AT_A1fine` 映射到 `Condenser`；p5 又区分初始 coarse A1
和最终微调使用的 microscope condenser stigmator。这支持共用控制，
不能单独证明 condenser stigmator 线圈已经移入 corrector。
[厂家指南](https://documents.thermofisher.com/TFS-Assets/MSD/manuals/nanoport-tuing-cs-probe-s-corr-user-guide.pdf)

项目中 Probe A1 fine 已经写入同一个 `condenser_stigmator` 的两路驱动；
Image A1 fine 对应同一个 `objective_stigmator`。这些别名不额外叠加一个场。
Descan 则是明确绑定到 Image/Diffraction deflector 的额外指令，共用两级实体线圈。
不会为 Descan 再画独立铜线圈或真空限制。

目前 QPH/QPC/QPOL 不是可直接替换上述双轴驱动的完整独立 normal/skew 接口。
因此本次没有把 A1 fine 任意迁移到某个校正器 multipole，也没有删除 condenser stigmator。

## 所选装配的实际位置

以下是配置解析后的全局轴向毫米坐标，未施加刚体 pose。它们不是厂家测量值。

| 器件 | 包络 Z 范围 / mm | 参考位置或两级线圈中心 / mm |
| --- | --- | --- |
| C2 | 550–750 | 650 |
| Condenser deflector | 790–830 | 800 / 820 |
| C3 | 830–1000 | 915 |
| Beam deflector | 1020–1060 | 1027.5 / 1052.5 |
| ADL | 1070–1106.4 | 1088.2 |
| Corrector TL12 | 1353.6–1379 | 1366.3 |
| AC scan | 1380–1420 | 1387.5 / 1412.5 |
| Condenser stigmator | 1436.5–1466.5 | 1451.5 |
| Mini-condenser | 1512.5–1532.5 | 1522.5 |
| Upper objective pole | 1534.5–1596.5 | 1565.5 |
| Specimen | — | 1599.2 |
| Lower objective pole | 1601.9–1663.9 | 1632.9 |
| Image/Diffraction deflector，兼 Descan | 1666.9–1706.9 | 1674.4 / 1699.4 |
| Objective stigmator | 1731.9–1761.9 | 1746.9 |
| SAA | 1814.4–1834.4 | 1824.4 |
| Diffraction stigmator | 1834.4–1844.4 | 1839.4 |
| Diffraction lens | 1846.9–1946.9 | 1896.9 |

Condenser deflector 与 Beam deflector 的顺序已经符合所要求的 C2–C3、C3–ADL 分段。
Objective stigmator 属于物镜组件的下游内孔空间，但它的轴向包络并不位于 lower pole 的
1601.9–1663.9 mm 实体范围中。需要区分“物镜组件内部”和“极靴本体的轴向范围”。
Condenser/objective stigmator 的 56 mm 外径相对配置的 60 mm 内孔有 2 mm 径向包络余量；
这仅是尺寸核查，不是机械安装、公差、热设计或有限元场验证。

所有七种 column 变体都由审计脚本分别解析。尤其 C3 + Probe Corrector 的 AC/CM
和 Image/Objective-stig 顺序与部分 image-corrector 变体不同，不能批量统一。
传统 C2/C3 配置的上游 Beam deflector 也尚未表达 Philips Twin 截面中完整的内孔集成方式。
本次记录这些差异，没有拿 CM12 专利示意去覆盖已调好的 C3+ProbeCorrector 工作点。

## 已完成的代码与计算核对

1. 修复三维绕组忽略 `channel_x_angle_deg`、`channel_y_angle_deg` 的问题。
   X/Y 各四只线圈随各自结构基准角旋转，保持原有 ID、轴向位置和默认 0°/45° 几何。
   非 45° 的独立基底按最近角间距收窄示意线圈，防止铜块相交；场强不随铜块厚度变化。
2. Diffraction stigmator 原先已有八线圈实体，却标为 `quadrupole_field_channel`。
   两份投影模块定义现已明确为 `magnetic_stigmator` / `stigmator_windings`，
   并注明 FEI 专利仅支持实体作用、不能验证配置尺寸。场位置、宽度与激发保持不变。
3. 新增只读 `Model Inspector → Stigmator hardware`：直接列出当前装配的数据、
   A1 fine 别名和 Descan 宿主。不存在的 corrector 不显示对应别名。
   审计也检查重复场对象和错误共享宿主，避免把逻辑通道重复计入物理场。
4. 执行现有传播系数的 CPU 核对：中心处 1% X 驱动给出
   `K = [[3,0],[0,-3]] m^-2`，1% Y 给出 `[[0,3],[3,0]] m^-2`；
   两基底同转 15° 后，1% X 为 `[[2.598076,1.5],[1.5,-2.598076]] m^-2`。
   主轴旋转，特征值仍是 ±3 m^-2，证明未借几何修改改动激发标度。

上述是当前解析四极模型的数值一致性验证，不是三维磁静力学求解或 OEM 电流标定。
有效场是共轴 Gaussian：`K=[[n,s],[s,-n]]`，`d(theta)/dz=-K r`；
角度以二倍角进入 normal/skew 系数。现有标度 300 m^-2 对应 100% 驱动；
Condenser/Objective FWHM 为 6 mm，Diffraction 为 8 mm，均属模型参数。
本次结构基底旋转与既有 `rotation_*_mrad` 的光轴倾斜是不同入口；没有改变后者的全局语义。

## 重现图像与报告

在项目虚拟环境运行：

```powershell
.venv\Scripts\python.exe scripts/audit_stigmator_hardware.py
.venv\Scripts\python.exe scripts/preview_stigmator_hardware.py
```

输出位于 `outputs/stigmator-research-20261019/`：
`hardware-audit.json` 保存七种装配及数值核对；`configured-axial-order.png` 为实际位置图；
`stigmator-field-bases.png` 为传播斜率响应图（不是磁感应强度 B 图）。
`preview/` 内九张原生 PartModelView 截图显示八线圈默认/旋转状态、物镜内部总览、
上下半剖面，以及新增套筒的外观、整体剖面和上下局部剖面。
截图记录保留选中/省略组件、保存尺寸、位姿、源文件哈希和临时角度修改。
绘图 CPU、32 段角细分，不运行高精度粒子或相干传播，不写回保存的装配。

消像散器结构与读出回归：264 项通过、1 项实际 CUDA 设备测试跳过，约 41 秒。
范围包括 CAD 线圈、四极张量、七类装配相关读出、位姿、Model Inspector、
corrector 通道和 CI 测试登记；不代表执行了项目全量测试或 GPU 验证。

仍待独立资料确认：Titan/Spectra/Iliad 的实际 stigmator 宿主、共同插入筒结构、
逐件内部尺寸、绕组匝数/电流、材料与测量场图。当前成果不声称已完成这些 OEM 结构的复刻。

## 补充：corrector 调用 TEM 的 Beam Tilt 与扫描线圈

应当区分**软件动作的名称、参与计算的驱动组合、实际线圈的安装位置**。
corrector 可以调用原生 TEM 器件；控制入口出现在 corrector 软件内，不等于该器件属于
corrector 的机械筒体，也不证明所有名为 Beam Tilt 的动作都使用同一组线圈。

CEOS 作者的 Müller et al. 2006 原始论文 p443、Fig.2 明确说明：有条件时用 TEM 原生
mini lens 代替 TL11，使用上下扫描线圈的加权直流偏置实现 DP12。图中 BSh/BTlt 位于
最后聚光镜与 ADL 之间；下游扫描线圈则位于 TL12 与 mini lens / objective 之间。
这是两组不同硬件，DP12 不是第三个需要额外绘制的线圈。
[CEOS 原始论文](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)

CEOS 对 B2 自动校正的原始说明进一步指出，它联合使用 corrector 上方的 BTlt
和下方的 DP12 改变穿过六极系统的方向。这也说明校正功能可能跨越 corrector
和 TEM 原生组件，而不是把这些控制全部集中到一个物理偏转器。
[CEOS B2 Auto Alignment](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR)

Titan 原厂 UI 手册 p244 明确描述，各项偏转控制先相加，再经 pivot-point 关系
换算为上下线圈激发；beam shift、dark-field tilt、rotation centre 等不是各有一套线圈。
CEOS 2025 调谐教程 p4 将 AT BeamTilt 用于物镜旋转中心，p14 又显示 corrector 导出的
Beam Tilt 等通道可以由显微镜手柄/软件操作。应当分别记录操作来源与物理宿主。
[Titan 原厂手册](https://chenj.hosted.uark.edu/ChenLab/TEM/HRTEM-Titan/UserInterface.pdf)、
[CEOS 调谐教程](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/03_Basics-of-daily-corrector-tuning.pdf)

| 所选 C3 + Probe Corrector 配置 | 实体 / 控制关系 | 当前全局 Z / mm |
| --- | --- | --- |
| BSh/BTlt | 校正器上游的 `beam_deflector` 两级偏转器 | 1027.5 / 1052.5 |
| AC scan，静态偏置与 wobble | 校正器下游、mini-condenser 上游的 `ac_deflector` 两级线圈 | 1387.5 / 1412.5 |
| DP12 | 虚拟参考；公开 CESCOR 设计中由下游 scan pair 的加权 DC 实现 | 没有独立实体 |

对于“是否仍在 upper objective 内圈”，公开资料支持下游扫描/偏转组件保留在
物镜上游组件中；Philips Twin 的原始截面明确显示其内孔集成方式。
但 CESCOR 光路图不是 Titan/Spectra/Iliad 的机械截面，不能单凭它给出具体内孔安装 Z。
现有模型把下游 AC、Condenser Stigmator 和 mini-condenser 归入上物镜组件，
而上游 BSh/BTlt 保持在 C3–ADL 之间。二者不会合并或重复绘制。

**程序能力边界：**当前 Hardware tuning 的 Beam shift / Beam tilt、参考平面 steering
和 Beam centre/direction alignment 都驱动上游 `beam_deflector`。AC 的静态、wobble
和 raster 则共用下游 scan pair 的耦合。Corrector 面板目前只实现 A1/A2 coarse/fine，
尚未实现以原生扫描线圈参与的 B2 / Beam Tilt 校正命令。不能把上游入口的名称理解为
已实现了下游原生 TEM Beam Tilt，也不能把当前耦合后的 AC 静态偏置直接当成四驱动的
纯 tilt 校准器。本次修正了相关误导性说明，没有擅自改变两组偏转器的计算绑定。

## 补充：上、下物镜电子组件的塑料示意外壳

按用户要求增加两段开口、中空圆柱套筒，覆盖各装配实际存在的电子组件：
上段容纳 AC、Condenser Stigmator 和 mini-condenser；下段容纳 Image/Descan pair
和 Objective Stigmator。已有器件的父子关系、位置与场参数保留。

材料类别是未指定牌号的绝缘聚合物，颜色仅用于区分材料；未认定为实际机型使用的 PEEK、
PTFE 或某种工程塑料，也未添加磁导率、介电常数或出气率等未经给定的计算参数。
圆柱内径 59.2 mm、外径 59.8 mm、壁厚 0.3 mm 是用户授权的示意几何选择：
既容纳 59 mm 的 mini-condenser 外壳，又留在 60 mm 的物镜内孔内。
上下套筒分别止于极靴外侧，不贯穿极靴或样品间隙。

所选 C3 + Probe Corrector 的上套筒长 153.5 mm（全局 Z 1380–1533.5 mm），
下套筒长 97 mm（全局 Z 1665.9–1762.9 mm）。两端保持开口；长度按包住组件并
避让相邻极靴确定，而不是把外壳延伸穿过物镜极靴。

套筒是独立可见的被动机械部件，不产生额外磁场、不成为重复的真空管壁或电子束孔径。
它们按各装配电子组件包络确定长度，不用一个固定 Z 套用所有变体。
这些外壳表达所要求的外观和组件分组，仍不声称有 OEM 制造图依据。

间隙检查针对保存的名义装配尺寸。套筒不会把已独立安装的扫描器和 stigmator
重新绑定到物镜位姿；mini-condenser 仍保持既有的物镜继承关系。单独修改物镜或
内部组件的位姿可能改变这些间隙，套筒不会自动移动其他器件来隐藏干涉。

名义装配的包覆和避让关系在默认配置测试中检查，不限制用户后续调整内部器件的
位置或尺寸。原装套筒保留聚合物示意材料；复制出来的独立机械副本继续支持现有
材料编辑，但材料选择不会使副本成为新的场源。

验证结果：套筒专项测试 24 项通过（13.32 秒）；补充集成测试 183 项通过
（30.03 秒，无失败或跳过），涵盖 CI 范围登记、消像散器硬件读出、Model Inspector
和多种装配下的真空/相机终点。集成记录保存在
`outputs/stigmator-research-20261019/sleeve-integration.xml`。
本次使用 CPU，不运行大型粒子或相干传播；九张原生三维预览生成耗时约 5.1 秒。

- [套筒外观](../../outputs/stigmator-research-20261019/preview/objective_sleeves_exterior.png)
- [套筒与内部器件剖面](../../outputs/stigmator-research-20261019/preview/objective_sleeves_section.png)
- [上套筒局部剖面](../../outputs/stigmator-research-20261019/preview/upper_sleeve_section.png)
- [下套筒局部剖面](../../outputs/stigmator-research-20261019/preview/lower_sleeve_section.png)

这些输出由前述脚本生成，不随 Git 版本化；在其他机器上可重新生成。
