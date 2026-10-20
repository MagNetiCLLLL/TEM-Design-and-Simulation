# C2 光阑、Condenser Deflector、Condenser Stigmator 与 C3 结构复核

本轮继续检查 FEG + C3 + Probe Corrector + No Energy Filter 的已保存结构。
所有检查通过后台代码完成。新增材料继续使用统一的 `SolidSpec`、原生网格、
STEP 和材料类别，不把三维细节等同于新的场解或 OEM 制造图。

## 实际顺序和本轮变化

默认装配的全局 Z 从上向下增加，单位 mm：

| 部件 | 中心 Z | 本轮变化 |
| --- | ---: | --- |
| C2 Aperture | 765 | 保留条板和原工作孔定义，增加一根连接杆、一个无螺纹简化螺钉及匹配固定孔；合计3个材料区域 |
| Condenser Deflector | 810 | 已有两级8个铜线包及12个绝缘/定位区域；明确保存10 mm物理绕组长度，解除其对有效场宽的默认依赖 |
| C3 | 915 | 原两极靴、线圈、磁轭、壳体之外，在已有端部空间增加两块散热板和两层绝缘片；合计9个材料区域 |
| Condenser Stigmator | 1451.5 | 位于 Probe Corrector 下游、物镜组件内；保留原8个铜线包，增加8份绝缘、8个骨架、8个胶座和1个开槽支架，共33个区域 |

Condenser Stigmator 并不在 C2 与 C3 之间。本轮没有为匹配名称或列表顺序而移动它。
Descan 与 Image/Diffraction 的共享实体关系保持原样。

## 资料支持与假设

- C2：已有用户照片记录支持条板—单杆—螺钉拓扑；原照片未用于定量测量。
  [JEOL JP2007329043A](https://patents.google.com/patent/JP2007329043A/en)
  和 [JP2012146543A](https://patents.google.com/patent/JP2012146543A/en)
  支持光阑与可移动安装机构分开表达，不能确定本机孔数、孔距或毫米尺寸。
- Deflector：[US5994704A](https://patents.google.com/patent/US5994704A/en)
  支持刚性绝缘载体、同轴隔筒、定位销及绝缘外壳方案；保留已有工程实现。
- Stigmator：[CN102509689B](https://patents.google.com/patent/CN102509689B/en)
  描述八槽筒形支架、线圈框、绝缘绕线与胶固定。本轮适配的是该结构原则，
  不是把其物镜专利认定为本机 Condenser Stigmator 图纸。
- C3：[Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en)
  的常规结构图8及正文支持线圈端部散热板与薄电绝缘层。

新增尺寸均显式保存为 `engineering_reconstruction_not_oem`。
Stigmator 的等效绝缘、骨架壁、胶层分别为0.1、0.3、0.1 mm；
C3 绝缘片和散热板分别为0.5、2.0 mm。这些是已保存包络内的工程分配，
不是制造商尺寸或公差。详细参数与比较见
[光阑记录](cad-gun-filter-apertures.md)、
[偏转器与 Stigmator 记录](condenser-deflector-stigmator-structure.md)、
[圆透镜记录](cad-round-lenses.md)。

仍未确认的部分包括宿主固定、引线、真空密封、水路及 C3 径向回磁连接。
散热铜片不能代替铁质回磁盖，也不能用来声称已经形成完整磁路或改善了像差。

## 与真空管的实际关系

使用与 Physical Layout 相同的外径折线及侧向插入槽实体核对。

| 部件/界面 | 默认间隙 |
| --- | ---: |
| C2 条板与侧槽边缘 | 每侧0.2 mm |
| C2 连接杆 / 螺钉头与管外壁 | 1.9 / 2.2 mm |
| Condenser Deflector 实际支撑内表面与管外壁 | 最小径向2.9 mm |
| Condenser Stigmator 支架与管外壁 | 径向0.4 mm |
| C3 极靴与恒径管段 | 径向0.4 mm |
| C3 管径过渡附近 | 最短三维距离约0.398015 mm |

C3 的最后两项不同：17.2 mm管外径与18 mm极靴孔给出0.4 mm径向差，
相邻锥形过渡使最短距离为 `0.4 / sqrt(1 + 0.1²)`。
32、64、128角向采样的网格距离收敛到该解析结果。
这些是默认位置的几何净隙，不是装配公差认证。

材料管内径5.76 mm与求解通束域也要区分：C2 光阑和 Deflector 段仍受局部5 mm
真空配置约束；C3 和 Stigmator 段解析后为5.76 mm。工作光阑另有独立孔径截断。
新增机械细节不扩大这些通束边界，粒子与波动路径使用相同的解析限制。

## 编辑、传播和证据

原生显示与 STEP 共用材料定义。无效细节修改在替换草稿与撤销历史之前被拒绝；
复制件保留自身尺寸，局部 Z 接触位置随移动和长度编辑变化。
C3 线圈与新增端板接触尺寸矛盾时，需要显式同步壳体配方。
原壳体 `body` 的材料编辑保持可用，新增被动层不成为额外 FEM 材料区域。
被动几何变化不会重新标定解析场；C3 几何指纹会使旧的导入场图需要重新验证。
光阑非等比缩放先作用于条板尺寸，再增加固定0.2 mm槽边间隙；原机光学倾斜
仍不旋转机械条板。独立复制件和替换模型若可能进入管内而没有已声明插入槽，
会由明确的保守包络检查拒绝，避免把模板名称当作安全插入的证明。

默认装配由376增加至407个网格，新增31个区域，没有删除已有区域。
C2 条板增加固定孔；C3 壳体改由共享 CAD 配方剖分，但其解析环壳形状不变。
完整逐配件审计未发现非闭合网格、材料遗漏或构建错误。

后台证据：

- [本轮之前](../../outputs/condenser-details-20261020/before/audit.json)
  与 [本轮之后](../../outputs/condenser-details-20261020/after/audit.json) 的逐配件审计。
- [实际装配间隙检查](../../outputs/condenser-details-20261020/assembly-clearance.json)：
  检查同柱287个材料区域中与本批部件相关的334对包围盒相交候选，未发现正体积穿插。
- [STEP 回读](../../outputs/condenser-details-20261020/step/step-manifest.json)：
  66个选定材料区域，5个分组文件及1个总装文件，验证实体、体积、边界、名称、材料与颜色。
- [七组装配检查](../../outputs/condenser-details-20261020/variant-audit.json)：
  覆盖三种枪、七种柱配置及两种记录系统，没有网格、材料遗漏或 CAD 构建错误。
- [相关回归结果](../../outputs/condenser-details-20261020/verified-regression.xml)：
  综合回归加修复后的定向复测，按测试身份去重后824项全部通过，0失败、0错误、0跳过。
  这是多批实际测试的汇总，不是一次 pytest 运行；
  [批次与去重记录](../../outputs/condenser-details-20261020/verification.json)
  保留所有原始报告位置。初次电子枪场缓存和真实 STEP 导入受 Windows 沙箱文件权限阻断；
  前者改用工作区生成缓存，后者经获准的后台测试重跑后通过。
  普通运行环境跳过的6项 CAD 测试另在现有独立 CAD 环境中全部执行通过。
- [实际场与传播对照](../../outputs/passive-condenser-transport-20261020T212507Z/report.json)：
  通过保存配置副本并重新解析当前 Nanoprobe Normal 装配，分别改变 Stigmator
  三项支撑厚度、C3 两项端部厚度及其组合。39个场源、6080个有效磁场采样值、
  9条粒子在352个保存截面的轨迹和生产传播计划完全一致，材料形状确实改变。
  X激励由4%改成5%的正对照正常改变场和轨迹，最大Y差约0.963 nm；
  非零场与正对照排除了空场、空路径导致的假通过。

以上均为代码、已保存工程几何和模型一致性验证；未操作桌面界面，也不是实机验证。
