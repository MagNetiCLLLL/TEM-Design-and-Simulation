# CAD 结构参考库

这里只归档可溯源资料，不将图的精细程度等同于本机的已知精度。原始 TOML 的坐标、边界和材料定义仍是当前装配输入。专利给出的可实现拓扑，不能自动证明本机存在同样的冷却管、支架、线圈或隐藏零件。

每项包含 `source.json`、中文简介、原始 PDF（可获得时）和少量未经改绘的原文整页图。索引中的“PDF”表示实际文件已保存并通过 PDF 解析，非仅有链接。完整文件 SHA-256 在各项元数据中。

[原始 TOML 配件对应表](component-mapping.json)记录本次检查的 11 份模块文件及 SHA-256，并列出资料对应的真实配件键。表中未解析的自然语言类别只表示结构族，不是新增或已存在零件的声明。

| ID | 适用配件/功能 | 原始文件状态 |
|---|---|---|
| [JP2967966B2](JP2967966B2/简介.md) | 普通圆磁透镜 | 原 PDF/照片未获得，见说明 |
| [WO2021170762A1](WO2021170762A1/简介.md) | 物镜 | PDF 已保存 |
| [US4306149A](US4306149A/简介.md) | 辅助 mini condenser 与物镜 | PDF 已保存 |
| [US8841630B2](US8841630B2/简介.md) | 校正器传输透镜 | PDF 已保存 |
| [US20090079531A1](US20090079531A1/简介.md) | CEOS 多极线圈与 DPH 插入件 | PDF 已保存 |
| [US20240371597A1](US20240371597A1/简介.md) | 通用多极磁芯/绕组/磁轭 | PDF 已保存 |
| [US6548816B2](US6548816B2/简介.md) | 消像散器 | PDF 已保存 |
| [US5481164A](US5481164A/简介.md) | 同位置 normal/skew 消像散器 | PDF 已保存 |
| [US9548182B2](US9548182B2/简介.md) | 枪与加速电极 | PDF 已保存 |
| [US6844548B2](US6844548B2/简介.md) | Wien 过滤器 | PDF 已保存 |
| [US9966219B2](US9966219B2/简介.md) | EELS 能滤与狭缝 | PDF 已保存 |
| [US8373137B2](US8373137B2/简介.md) | 能滤复用多极元件 | PDF 已保存 |
| [Mueller2006a](Mueller2006a/简介.md) | CEOS CESCOR 光路与六极实体 | PDF 已保存 |
| [FEI-Titan-training](FEI-Titan-training/简介.md) | Titan mini condenser 与照明系统 | PDF 已保存 |
| [CEOS-Linck-2025](CEOS-Linck-2025/简介.md) | CETCOR 成像校正器光路与对准通道 | PDF 与 p22/24/26/29 整页图已保存 |
| [CEOS-Lu-2025](CEOS-Lu-2025/简介.md) | CETCOR/CESCOR 两种光路、OL pre/post 及基准光线 | PDF 与 p1/2 整页图已保存 |
| [CEOS-CETCOR-residuals](CEOS-CETCOR-residuals/简介.md) | 对称面、中间像与残余像差 | 官方网页原文已保存 |
| [CEOS-CETCOR-product](CEOS-CETCOR-product/简介.md) | CETCOR 高/宽/深及适用范围 | 德文官方网页原文已保存 |
| [Mueller2008](Mueller2008/简介.md) | CETCOR 原始建模论文及 HP 结构族 | 原 PDF 下载不可用，仅书目与核对记录 |
| [legacy-user-photos](legacy-user-photos/简介.md) | 旧用户照片线索 | 原 PDF/照片未获得，见说明 |

## 使用边界

- 本库不是 Titan/Spectra/CESCOR 制造图。对照时使用相同模块/变体的原始 TOML，不能拿另一型号的专利数字强行统一。
- Mueller 2006 的实际六极示例与项目当前 HP 尺寸不同；保留差异，不冒充完全匹配。
- 旧照片仅有线索，不得作为确认孔径材料、真实厚度或隐藏装配的凭据。
- 专利的初始申请日、公开日、申请人是来源元数据，不是当前法律状态或所有权结论。
- 公开访问不等于开放许可；原文保留作者/机构权利，图页不得作为本项目原创或 OEM 交付件销售。

关联说明：[圆透镜证据](../cad-round-lenses.md)、[多极证据](../cad-multipoles.md)、[枪/能滤/孔径证据](../cad-gun-filter-apertures.md)。

成像校正器另见 [CETCOR 光路/装配对照](../../image_corrector_reference.md) 和 [修改前 TOML 数值/SHA](image-corrector-config-comparison.json)。该记录的 local Z 与全仪器绝对 Z 不同；后续修复不覆盖此原始比较证据。

## 侧插式样品杆与样品台

[尺寸/装配边界说明](../cad-stage-holder.md)。本组资料对应普通侧插结构族；特殊专利机构没有自动加入本机。

- [US20100006771A1](US20100006771A1/简介.md)：original_pdf_saved。
- [WO2025134741A1](WO2025134741A1/简介.md)：original_pdf_saved。
- [US20080173813A1](US20080173813A1/简介.md)：original_pdf_saved。
- [EP3843120A1](EP3843120A1/简介.md)：original_pdf_saved。
- [Gatan-analytical-holders](Gatan-analytical-holders/简介.md)：record_only_original_download_unavailable。
- [Thermo-CrystalPack](Thermo-CrystalPack/简介.md)：original_pdf_saved。
- [Thermo-Spectra-Ultra-holder](Thermo-Spectra-Ultra-holder/简介.md)：record_only_original_download_unavailable。
- [LBNL-TEM-holders](LBNL-TEM-holders/简介.md)：original_html_saved。

## EDS 与 EELS camera 材料

[接收面不变、工程机械尺寸与资料边界](../cad-detectors.md)。新增四份原始 PDF、图页、简介及哈希已编入索引：

- [US8410439B2](US8410439B2/简介.md)：SDD、窗口选择、冷屏与绝缘支承结构族。
- [US7642513B2](US7642513B2/简介.md)：间接 EELS 转换层与光学传递结构族。
- [DS0510](Thermo-Iliad-Ultra-DS0510/简介.md)：五条间接探测 strip 的公开规格。
- [KETEK VITUS](KETEK-VITUS/简介.md)：同类 Si 吸收层厚度的数量级参考，不是 Ultra-X 图纸。

Image corrector 的八个偶极线包参见 [CEOS 图与专利构造对照](../cad-image-corrector-channels.md)，原始文件校验记录见 [CETCOR channel 条目](CETCOR-channel-construction/source-entry.json)。
