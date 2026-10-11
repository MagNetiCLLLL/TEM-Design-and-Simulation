# 普通侧插式 TEM holder、goniometer 与镜柱入口

核对日期：2026-10-11。范围是 Titan/Spectra 兼容结构族的常温单倾或双倾样品杆，供当前 CAD 装配重建使用。公开资料足以确定侧插拓扑、3 mm 样品规格和细杆直径量级；没有获得普通 Titan/Spectra holder 的完整制造图。因此下列工程分配不能显示为 OEM 实测尺寸，也不构成真空或可制造性认证。

## 来源与证据级别

| 资料 | 可核对的位置 | 本次采用 | 边界 |
|---|---|---|---|
| [Mel Build WO2025134741A1](https://patents.google.com/patent/WO2025134741A1/en) | 原日文 PDF p10，印刷 p8，[0036]；[原页](cad-library/WO2025134741A1/page-10.png) | 文中同时提到 FEI/现 Thermo Fisher 的细轴，以及 **φ7 mm holder shaft** | 专用可收回/加电 tip 专利，仅支持兼容结构族的杆径；不证明普通 Titan/Spectra 全尺寸 |
| [US20100006771A1](https://patents.google.com/patent/US20100006771A1/en) | Fig.1/PDF p2；[0010]/p4；[0097]/p10–11 | 普通杆、外柄、导向套、镜柱及真空密封的关系；杆/前端直径 **6–10 mm** 的设计范围 | 非特定 Titan 型号；没有杆长、柄径、薄 tip 厚度；直径范围不表示圆杆可以穿过更薄极靴间隙 |
| [Gatan Analytical Holders](https://www.gatan.com/products/tem-specimen-holders/analytical-holders) | 643/646 产品表及 tip 说明 | 常温单倾/双倾分析杆；tip 保留束路与 EDS 视线 | Be cradle 是该产品局部材料，不是全杆材料；未公布所需外形尺寸 |
| [LBNL/NCEM 实物目录](https://foundry.lbl.gov/instrumentation/tem-holders/) | ThermoFisher/FEI entries 1–3 | TitanX 等实际使用普通 FEI 单/双倾杆；3 mm 样品、夹子与垫片 | 是设施实际使用证据，不是 OEM 加工图 |
| [Thermo Fisher CrystalPack](https://documents.thermofisher.com/TFS-Assets/MSD/Application-Notes/crystal-pack-module-application-brochure.pdf) | PDF p2，Figs.1–2 | β 轴可非共心，通过其他台轴补偿；需按 holder 校准 | 不支持把两条倾转轴默认当成精确共点，也没有机械尺寸 |
| [Thermo Fisher Spectra Ultra](https://www.thermofisher.com/nl/en/home/electron-microscopy/products/transmission-electron-microscopes/spectra-ultra-tem.html/1000) | S-TWIN/Ultra-X 产品说明 | 宽间隙物镜可与分析双倾杆组合，holder 遮挡会影响 EDS 接收 | 不能由 EDS 立体角反算 holder 或插入口外形 |

另保存两份容易被误用的原始专利：[FEI US20080173813A1](cad-library/US20080173813A1/简介.md) 中被双执行器夹持的是圆样品载体，不是完整侧插长杆；[EP3843120A1](cad-library/EP3843120A1/简介.md) 为另一种集成 goniometer 衍射 holder，给出 200–300 mm 总长等数值，但架构不同，不移植为 Titan 的尺寸默认。它们只提供机构区分和量级对照。

原 PDF、少量未改绘整页图、作者/申请人、公开日期、URL 和 SHA-256 均在 [CAD 资料索引](cad-library/index.json)。Gatan HTML 下载被限流、两份 datasheet 地址返回 HTML；Spectra 页面原文件下载被拒。已记录状态，没有将返回页伪装成 PDF。网页事实已通过原机构页面的检索内容核对。

## 与原有 TOML 的对应

[研究时输入快照](cad-library/stage-holder-config-comparison.json) 保存四个 `C3*.toml` 的 SHA-256 与原始 `sample`、`sample_stage`、`objective_lens` 行，不覆盖后续实施记录。

- `sample_stage.holder_insertion_axis = "x"`；样品位于物镜中心。无 probe corrector 的 local Z 为 825.2 mm，有 probe corrector 时为 1149.2 mm；这不是全仪器绝对 Z。
- `sample_stage` 原有 650 × 500 mm 横向包络和 5.4 mm 轴向跨度，是装配分配，**不是一块填满该空间的实体板**。横向入口必须独立建模，不能让宽板或圆环穿过两块 pole nose。
- 物镜极靴间隙为 5.4 mm，当前外径为 180 mm。7 mm 圆杆不能伸到中心间隙；靠近样品的部分必须变为薄端部，杆与薄端部在极靴外展后衔接。
- `sample.mechanical_outer_diameter_mm = 3` 是样品盘外径；`length_mm ≈ 0.0001` 是 100 nm 样品厚度，**不是 holder tip 或样品夹厚度**。
- `sample_stage.vacuum_inner_diameter_mm = 5` 是原主束路的轴向尺寸，不能拿来当 7 mm 侧插杆的端口孔径。

## 已保存的共享参数及其性质

本轮参数已经保存到 [`configs/subassemblies/sample_stage.toml`](../../configs/subassemblies/sample_stage.toml)，各 column 变体通过 `[[subassemblies]]` 引用并锚定现有 `sample` 中心，避免多份独立尺寸。下表描述已保存的参数，**不代替单独的碰撞检查结果**。所有 X 位置均以样品中心为原点，单位 mm；正 X 为外部手柄方向，插入沿 −X、抽出沿 +X。这是项目坐标选择，不是厂家左右手性规定。

| 部分 | 保存值 | 依据/性质 |
|---|---:|---|
| 样品盘 | 直径 3 | 实际 FEI/Titan 使用资料与既有 TOML 一致 |
| 长杆 | 直径 7，X = 18…235 | 杆径有上述 FEI 兼容结构族证据；起止位置为当前装配工程分配 |
| 薄端部 | 宽 5，厚 1.2 | 适配当前 5.4 mm 间隙的工程重建；没有 OEM tip 厚度证据 |
| 样品定位座 | 直径 3.1，深 0.65 | 从 tip 的 +Z 面开座，座底位于样品中心 Z − 0.05；全部为工程值 |
| tip 束贯孔 | 直径 2.4 | 与定位座独立，留下径向承托台阶；工程值 |
| 外部手柄 | X = 235…315，外径 40 | 只表示普通外部操作柄，不增加加热、冷却或电控设备；无 OEM 尺寸依据 |
| 导向套 | X = 87…180，外径 12、内径 9 | 由杆径及当前镜柱壳位置确定的工程包络 |
| 柱外法兰 | X = 90…100，外径 70 | 工程分配，不宣称真实螺钉数、法兰标准或密封槽尺寸 |
| 柱外台体 | X = 100…170，外径 50 | 仅 goniometer 外部壳体的简化重建，内部轴承/驱动未知 |
| 主壳侧入口 | 直径 12.5 | 容纳外径 12 的导向套；是侧向开孔，与 Z 束孔分开 |
| 机械抽出量 | 默认 0，允许 0…400 | `holder_retraction_mm` / `holder_maximum_retraction_mm`；纯机械展示行程，工程分配 |

当前 tip 为半径 2.5 mm 的圆端与宽 5 mm、厚 1.2 mm 的平臂相并，到 X = 18 mm 转接圆杆。3.1 mm 定位座与 2.4 mm 贯孔之间保留实体台阶；没有把整个定位座贯穿挖空。没有凭空增加压片、真实 TEM grid 厚度或双倾内部驱动。座底比光学样品面低 0.05 mm，并不等于软件已经建立一块 50 μm 厚的 grid；若需要真实夹紧/承托接触或倾转碰撞，需要补入用户确认的 grid/夹持件定义。

`sample_stage` 的 `length_mm=5.4` 继续表示原束路间隙；实际外部 stage 的 Z 范围通过 `actual_z_extent_mm=70` 与明确横向实体给出。原 650 × 500 mm 分配转存为 `legacy_allocation_x_mm/y_mm`，不再作为实心板或新外壳尺寸。`sample_holder` 使用独立 `side_entry_sample_holder` profile，`axial_vacuum_context_only=true`，不能按其 ±20 mm 包络制造新的轴向真空挡板。

抽出量只平移机械 holder，不自动改变 `sample.inserted`、晶体姿态或电子传播状态；因此不能把抽出后的机械画面解读为样品已经从物理计算中移除。

从样品中心到手柄最外端 315 mm 的当前分配不与已获得资料构成直接矛盾，但也不由 EP 专利的 200–300 mm 范围背书：两者长度参考面和架构不同。未知尺寸不能通过示意图像素或照片比例补成“真实值”。

## 装配方向、入口与碰撞边界

未倾转时，电子沿项目 **+Z** 通过 XY 样品平面；holder 沿 **X** 由侧面插入，α 围绕杆轴 X，β 在 tip 内围绕与杆轴垂直的局部轴。β 的真实偏心量未知，当前若采用共心简化必须标明，而不能以软件补偿资料证明机械共心。

需要在实际相交的物镜外壳/真空边界上形成沿 X 的连续插入口，并让导向套与它相接。只有在内部磁轭确实与该路径相交时才处理相应缺口；不能为了视觉方便贯穿切掉真实线圈或整块磁芯。主 Z 束通道必须开放，样品座仍保留可见的支撑。

后续碰撞审查至少要区分：薄 tip 与两块 pole nose、7 mm 杆与锥形间隙、导向套与壳孔、手柄/台体与主壳、holder 对 EDS 视线的影响。9 mm 套孔对 7 mm 居中杆只提供每侧 1 mm 间隙，不等同于所有台平移或倾转状态均可通行；α 旋转圆杆本身不改变轴线，β 则不应误旋转整根圆杆。

本资料只约束 CAD 的来源与机械拓扑。增加这些实体不能自动声称已将 holder 遮挡、EDS 固体角、样品真实微结构或台运动校准接入粒子/波动求解；这些是需要单独说明的计算接口。

## 原生界面预览

[`scripts/preview_stage_holder.py`](../../scripts/preview_stage_holder.py) 实际创建 Qt `PartModelView` 并抓取 widget；只改变相机、剖面和可见配件，不修改网格或后期拼图。输出位于 `outputs/stage_holder_20261011/preview/`：原配置/新装配对照、样品附近剖面与独立配件、整体剖面。`preview_manifest.json` 记录实际模型、材料类别、源文件及网格 SHA、相机和显示省略项。

旧配置从 `outputs/stage_holder_20261011/originals/configs/instruments/column/` 独立载入，没有使用新的共享 stage 文件；旧的未定义 holder 与侧孔不被新模型补齐。对照画面省略没有实体定义的旧 stage 大包络，并明确展示该原始分配数值。预览隐藏了其他光学单元及 EDS 配件，仅用于此处的机械关系观察，不能当作整柱全部配件无碰撞的证据。
