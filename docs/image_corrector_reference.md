# CETCOR 成像校正器：光路、共轭关系与装配证据

核对日期：2026-10-11。范围是经典两六极 CETCOR 成像校正器，不把 CESCOR 探针校正器倒放后直接当作本机设计。此文记录修改前输入；它不是已经校正成功的计算报告。`CETCOR2` 在本次查到的 CEOS 官方资料中没有被确认为独立产品/硬件型号，因此不据此引入另一套部件。

本次核对之后的实际修改、默认参数、GPU 结果与新版 CAD 见 [更新报告](IMAGE_CORRECTOR_UPDATE_20261011.md)。下文“修改前”表格保留为原始对照，不再代表当前配置。

## 原始资料

| 资料 | 直接支持的事项 | 文件/位置 |
|---|---|---|
| Martin Linck，CEOS，How do aberration correctors work?，2025-08-31 | 理想/实际两六极成像光路、附加对准通道；有限长 HP 同时产生二阶和负三阶作用 | [原 PDF](references/cad-library/CEOS-Linck-2025/original.pdf)，p22、24、26、29；[p26 原图](references/cad-library/CEOS-Linck-2025/page-26.png)；[官方来源](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/01_How-do-aberration-correctors-work-pdf.pdf) |
| Peng-Han Lu，Forschungszentrum Jülich，CEOS workshop，2025-08-31 | CETCOR 与 CESCOR 对照；物镜前/后半场、样品/BFP、两族基准光线 | [原 PDF](references/cad-library/CEOS-Lu-2025/original.pdf)，[p2 原图](references/cad-library/CEOS-Lu-2025/page-02.png)；[官方托管来源](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf) |
| P. Hartel、H. Müller、S. Uhlemann、M. Haider，CEOS，Residual aberrations of hexapole-type Cs-correctors | TL1 倍率/HP 强度调 C3；TL21/TL22 中间像位置影响 D4；C5 与离轴彗差的取舍 | [网页存档](references/cad-library/CEOS-CETCOR-residuals/original.html)；[官方来源](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR) |
| CEOS CETCOR 产品页，德文 | 产品尺寸及适用型号 | [网页存档](references/cad-library/CEOS-CETCOR-product/original.html)；[官方来源](https://www.ceos-gmbh.de/de/produkte/temkorrektoren/cetcor) |
| US8841630B2 | 两 HP 的典型负倍率成像；经典中平面到中平面 −I；磁双透镜反极性与旋转；也明确存在非单位倍率变体 | [原 PDF](references/cad-library/US8841630B2/original.pdf)，p14、16、20、22；[专利](https://patents.google.com/patent/US8841630B2/en) |
| H. Müller 等，Physics Procedia 1 (2008), 167–178 | CETCOR 模型、两 HP 对称性、磁芯/线圈/磁轭结构族、raw element 与组合对准工具的区别 | [书目与可用性记录](references/cad-library/Mueller2008/source.json)；[DOI](https://doi.org/10.1016/j.phpro.2008.07.093)。本次原 PDF 下载不可用，仅核对作者论文的公开文本，不能声称已存原图。 |

两份 2025 原始讲义的完整 PDF 和未经改绘的整页图均已保存。Lu 的光路图标明来自 CEOS residuals 页面，因此它与该页是同一图源，不能算两份独立工程测量。Linck 讲义与 CEOS 原始论文提供交叉支持。

## 沿电子前进方向的光学顺序

在样品之后：**同一物镜的 OL post 半场 → HPol/QPol/DP11 区域 → TL11 → DP12 → TL12 → HP1（DPH1 同 Z）→ TL21 附近 DP21 → 中间像 → DP22/TL22 → HP2（DPH2 同 Z）→ ADL → ISh → DSh/DSt → SAD 像面**。

这规定的是有效场的顺序/重合关系，不是把示意符号外缘当作机械端面：

- `OL pre` 和 `OL post` 位于同一个跨样品物镜的上下半部；图中没有额外串联的第二个物镜实体。
- DP12 在第一传输双透镜之间，与 TL12 是不同有效平面。图不能给出二者毫米间距。
- DPH1/DPH2 画在相应 HP 内，支持重合的偶极/六极作用区域。公开图不足以唯一确定 DPH 是哪种独立绕组、极针复用还是具体径向尺寸。
- HPol/QPol/DP11 是入口对准功能；不能仅按相邻图标推导它们共用一个磁芯或每个都有独立壳体。经典 image 图没有 probe 扩展 `QPH1/2` 标签。
- DP21/DP22 位于 TL21/TL22 区域两侧，细节图的重叠不等于已经确认 coil 在哪个极靴槽内。

## 应验证的共轭条件

在同一横向实验室坐标下，机械斜率或归一化正则动量必须分别一致使用。记同一能量的一阶映射

\[
\begin{bmatrix}r_2\\u_2\end{bmatrix}=
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}r_1\\u_1\end{bmatrix}.
\]

**位置像共轭**检查 `B≈0`；`A` 给倍率、旋转、剪切。**角度/傅里叶关系**检查指定起始平面到观察平面的 `A≈0`，不是看到几条彩色轨迹交轴就判定。厚场中的 coma-free aperture plane 接近焦面，但不应未经计算就与几何 BFP 标签完全等同。

经典 CETCOR 图的两族平面是：

1. 样品像族：样品 → TL11/TL12 之间的中间像 → TL21/TL22 对称面附近的中间像 → SAD。**SAD aperture 所在面在这里是样品像面**；名字中的 diffraction 不表示该孔径安装在衍射面。
2. 孔径族：物镜 coma-free aperture plane → HP1 中面 → HP2 中面。TL11/TL12 把物镜与校正器的孔径共轭匹配；倍率参与有效校正强度。
3. HP1→HP2：经典对称、无净 Larmor 旋转、单位负倍率模型的中平面映射为 `−I`。应分别报告 `B`、`C`、`A/D` 与净旋转，而不是只报一个倍率。专利也描述非单位倍率变体，所以不能把 `−I` 视作一切两 HP 设计的定义。

中间像落在 TL21/TL22 对称面，是抑制 D4 的条件之一。有限 HP、透镜像差、场尾及三阶组合项仍要保留；一阶 relay 满足不等于 C3/A2/D4 已达到目标。相对 HP 方位、励磁及传输旋转必须共同匹配，不能无视转动坐标只给两个 HP 填相反符号。C3-free、C5-free、离轴彗差最小也不总是同一设置。

## 修改前 TOML 对照

下表单位 mm，为模块 **local Z**，不是全仪器绝对 Z。合并模块的这些 image 元件相对 image-only 模块全部平移 +324 mm。完整值与 SHA 在 [对照记录](references/cad-library/image-corrector-config-comparison.json)。

| 元件 | image-only 中心 | 机械/通道区间 | 结论 |
|---|---:|---|---|
| 样品 / objective assembly | 825.2 | 615.0–1035.4 | 已有跨样品物镜 |
| 额外 image_ol_post_lens | 1062.9 | 1040.4–1085.4 | 下游另一个实体，不能当成图中的 OL post 半场；若有实际独立 relay/mini lens 依据，应单独命名和追溯 |
| HPol / QPol / DP11 | 1092.9 / 1098.9 / 1104.9 | 各 6 mm 通道区间 | 顺序可作为入口区，宿主未证实 |
| TL11 | 1136.9 | 1113.9–1159.9 | 有独立实体 |
| DP12 / TL12 | 1172.9 / 1172.9 | 1165.4–1180.4 / 1172.4–1173.4 | **明确与官方光学顺序不符**：同 Z 及“DP12 包含 TL12”没有公开依据 |
| HP1 / DPH1 | 1202.4 / 1202.4 | HP 1180.4–1224.4；DPH 1187.4–1217.4 | 中心/轴向包含与图相容；未证明真实绕组/CAD |
| TL21 / DP21 | 1256.4 / 1231.4 | TL 1238.4–1274.4 | DP21 当前在透镜前；官方图偏向靠出口。应检查有效场/对准响应，不从符号比例直接挪机械件 |
| TL22 / DP22 | 1306.4 / 1281.4 | TL 1288.4–1324.4 | DP22 在前，顺序相容 |
| HP2 / DPH2 | 1346.4 / 1346.4 | HP 1324.4–1368.4；DPH 1331.4–1361.4 | 同 HP1 的证据边界 |
| ADL / ISh / DSh / DSt / SAD | 1386.4 / 1420.4 / 1444.4 / 1452.4 / 1476.4 | 见对照记录 | 主要顺序相容；DSh/DSt 的独立机械宿主未确认 |

TL11 末端至 HP1 入口只有 20.5 mm；DP12 原区间已占 15 mm。保留这两个端面和 DP12 时，独立 TL12 仅剩 5.5 mm（含全部空隙）。本文件其他 transfer lens 为 36/46 mm、极隙 20 mm，不能在不重排的情况下补画同规格 TL12 实体。一个 1 mm 的有效平面不等于 1 mm 的完整圆磁透镜。优先修复独立传输自由度，再按共同光路约束调整局部轴向分配；任何新增包络/线圈尺寸必须标工程重建并复算成像，不可从 CEOS 示意比例取毫米。

## 外形与实体细节边界

厂家德文页明确 **H×B×T = 250×396×306 mm**，即高度 250、宽度 396、深度 306。页面未给接口/法兰基准或明确哪些外接耦合透镜属于外形，因此不能把 OL→SAD 全光程强制缩成 250 mm。当前 HP1 入口至 HP2 出口的核心区为 188 mm；TL11 入口到 ADL 出口为 290.5 mm；额外 OL post 入口到 SAD 为 436 mm。这些边界不同，不能直接称相互矛盾或相符。

Müller2008 的 Fig5/6 支持 CETCOR/CESCOR 设计用极针、线圈、磁轭及轴向镜板形成有限长 HP；其示例为 154 mm 磁轭外径、30 mm 长、4.05 mm 孔半径。已存 Müller2006 的六针实体照片支持同一 CEOS 双六极结构族，但其 152 mm 外径/4 mm 半径属于另一具体示例。选择六极针通用结构有公开依据，精确极面、绕组、材料牌号、现役 CETCOR 内部尺寸仍无依据。不能把 2006 的 probe 照片冒充 image 装配原图。

CAD 可以表达已确认的主结构族及 TOML 输入包络；对于 DPH/QPol 等仅确认功能或场位置的部件，应注明通用重建/宿主未知。未知冷却孔、螺钉、支架、独立 QPH、新壳体不应为增加细节而发明。本文不修改计算源、已加载样品、场强或几何文件。
