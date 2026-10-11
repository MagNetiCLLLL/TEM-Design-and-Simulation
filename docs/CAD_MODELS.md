# 可编辑 CAD 与软件 3D 模型

2026-10-11。本轮交付的是按现有装配尺寸建立的材料实体模型。公开专利和厂家示意图用于判断结构类别；没有把未标尺寸的示意图比例当作测量值。

**当前整库交付：** `outputs/detector_corrector_cad_20261011/` 已加入 EDS、EELS camera 和 image corrector 的八组偏转线包。11 种模块有 187 个配件/模块 STEP，另有含／不含 energy filter 的全局整机装配。详见 [探测器与 corrector 更新报告](DETECTOR_CORRECTOR_CAD_20261011.md)，包括未公开尺寸的边界、编辑能力与干涉验证。下文保留首次 CAD 阶段记录，不作为当前版本清单。

**同日后续更新：** 下文 `outputs/cad_models_20261011` 和全库压缩包记录首次 CAD 交付。之后已按 CETCOR 资料修正 image corrector 的光学身份、局部布局、默认圆透镜激励及 HP 内部表示；新的两个 image 模块单独交付在 `outputs/image_corrector_20261011/cad/`。需要 image corrector 时以这个新版为准，参见 [CETCOR 更新报告](IMAGE_CORRECTOR_UPDATE_20261011.md)。首次全库的统计数和“光学未改变”仅适用于首次 CAD 阶段。

## 已提供的模型

交付目录为 `outputs/cad_models_20261011`。`step/` 包含 **154 个 STEP 文件**：11 种配置模块的完整装配，以及这些模块中的 143 个光学单元。共有 745 个几何实体；这是所有备选模块累计的数量，并非一台电镜的部件数。不同柱配置、电子枪和记录系统应分别选用，不能叠放全部模块。

光学单元中有依据的极靴、整体绕组、磁轭、外壳、电极或绝缘件分别保留；未知内部只保留明确标识的配置包络。材料类别以颜色、名称和 STEP 分类层记录。单位为毫米，使用配置模块坐标。整模块 STEP 保留单元层级和装配位置。软件完整装配视图再使用现有模块安装位置；能量过滤器保持柱底部的 90° 支路。

| 部件类别 | 本轮模型内容 |
| --- | --- |
| 圆磁透镜，包括 condenser、relay、objective、mini condenser、projector | 由现有材料截面生成解析旋转实体，保留上下极靴、线圈、磁轭、外壳和已定义的分段结构 |
| 主 HP1/HP2 | CEOS 原文确认的六个磁极针、六组整体线包和一个磁轭；无证据的外壳、绝缘支架不添加 |
| 其他磁多极件和 stigmator | 已明确线圈拓扑的显示整体线包；其他未知结构保留配置包络，不由光学谐波推断实体极数 |
| DPH 与普通 deflector、AC scan coil | 同一套部件格式；保留实际轴向中心及同一 Z 内的径向嵌套关系，线圈材料与驱动通道分开 |
| Tip、枪电极与 accelerator | Tip 实际几何、已定义厚度的导体电极与级间绝缘段；缺电极厚度的模块保留包络；发射宽度不作为金属 Tip 半径 |
| Aperture | 有限厚度的孔径片，使用已指定的工作孔和位置；未知孔径不擅自填入一个数值 |
| 静电 blanker | 两块电极板，保留工作间隙和轴向长度，不添加未确认的绝缘支架 |
| Energy Filter | 弯转磁极、支路多极件、电极、狭缝片与漂移管；未给定厚度的探测器有效面不制造为虚构实体 |

## 在软件中查看和导出

1. 重启运行中的软件以加载本轮代码，进入 **Physical Layout → 3D** 查看整机。
2. 选择部件，点击 **Open in 3D Parts**；也可直接进入 **Physical Layout → 3D Parts** 打开所需模块。
3. 使用 **Section** 查看内部结构，使用材料/部件可见性控制逐项查看。普通已定义尺寸仍通过现有参数编辑界面修改。
4. 点击文件栏中 **Save copy…** 右侧的 **Export STEP…**，选择输出父文件夹。
5. 软件创建新的 `step-日期-时间` 子目录，在后台导出当前模块的每个光学单元及整模块。导出包含当前未保存草稿和已传入的运行时孔径/狭缝状态；输出 `catalog-input.json` 保留该次输入。完成前可取消。

未给定运行时狭缝间隙时显示、导出的是配置中的最大机械行程开口；这不代表当前能量选择宽度。导出完成提示只在 STEP 回读验证和输入身份核对通过后出现。

新加入的重建材料区可以查看，但不会自动变成材料场求解器的参数域。编辑器现已接通 STEP 导入、尺寸缩放、开孔和切槽：这些编辑与原生 3D 使用同一套操作参数，导出保留精确 B-rep。详见 [STEP 编辑操作说明](CAD_STEP_EDITING.md)。

## 如何编辑，以及“同步”的含义

STEP 文件包含可由 CAD 软件编辑的解析边界实体，不是把三角网格封装成 STEP。可在支持 STEP 的 CAD 工具中选择独立材料实体、修改几何或重新装配。

STEP 本身不保留完整的建模操作历史。项目同时提供 `catalog.json` 与生成器，保存截面、布尔构造、放置变换和材料信息，便于重建。软件内 3D 视图与 STEP 使用相同的部件构造规则和原有参数；改变软件中的受支持参数，再次导出即可保持一致。外部修改的 STEP 可通过 **Import STEP…** 显式导入为新机械件或替换已选材料实体。源 B-rep、原文件摘要、名称路径、颜色和本程序新增的编辑记录嵌入 TOML；不会自动反推透镜激励、材料本构或探测器接受区。

STEP 导出使用独立的可选环境，软件的粒子和波动计算环境不安装或加载 OpenCascade：

```powershell
python -m venv environment/cad
environment/cad/Scripts/python.exe -m pip install -r requirements-cad.txt
```

按当前项目配置重新生成全套模型：

```powershell
.venv/Scripts/python.exe -m temsim.cad_catalog --output outputs/cad_models/catalog.json
environment/cad/Scripts/python.exe scripts/export_cad_models.py outputs/cad_models/catalog.json --output outputs/cad_models/step
```

## 证据和模型边界

结构依据按部件归档：[原始文献、专利与图页目录](references/cad-library/README.md)、[统一细节规则](references/CAD_DETAIL_POLICY.md)、[圆透镜](references/cad-round-lenses.md)、[多极件与偏转器](references/cad-multipoles.md)、[枪、孔径与能量过滤器](references/cad-gun-filter-apertures.md)。每个实体在 `catalog.json` 中还包含 `status`、`note` 和 `evidence`：

- `configured_geometry`：依据当前已定义的尺寸生成；不等于厂家实测认证。
- `engineering_reconstruction`：有结构类型依据，但部分内部尺寸为显式重建值。
- `configured_envelope`：只有配置中的外形或包络可靠，内部结构仍未确定。
- `mixed_material_cartridge_unresolved`：保留已知共享载体外形，内部材料分区尚未确认。

最终清单中有 669 个重建实体、29 个配置几何、42 个未知内部包络和 5 个未细分材料的共享载体。后三类的标签尤其不能被理解为制造精度或真实材料体积。

本轮没有找到足以推翻特定现有装配外尺寸的直接证据，因此 CAD 构建保留这些尺寸。没有擅自修改电子源、激励值、光学中心、孔径工作尺寸或传播算法。绕线匝数、冷却通道、螺钉、密封、材料牌号和 B–H 曲线等未确认信息不作为已知输入；这些模型目前不能作为制造图纸或新的材料场校准依据。

例如 `FEG_Mono` 的双叶片狭缝尚无已知工作缝宽，当前保留载体包络；部分探测器封装、样品腔和热发射阴极也仅有已配置外形。它们在清单中明确标识，不能把本轮成果理解为所有内部细节均已确定。

## 验证结果

154 个 STEP 均执行导出后回读：检查实体有效、体积为正、尺寸/体积保持、名称层级、颜色、材料分类层及毫米单位。与生成实体相比，最大边界差约 `3.81e-10 mm`，最大相对体积差约 `2.28e-9`；这是格式往返误差，不是物理尺寸准确度。

最终清单为 `step/step-manifest.json` 和 `step-final-verification.json`。按统一细节与证据规则修订后，已全量重新导出、回读，并核对全部文件 SHA。旧的阶段性验收清单不纳入交付压缩包。

原生预览保存在 `preview/native_material_previews.png`，由实际 `PartModelView` 离屏绘制并截图，不是图片生成器或手绘效果图。预览仅为观察内部结构隐藏了部分外围组件，详情见 `preview/preview_manifest.json`。

与原 TOML 的对照位于 `assembly-comparison/README.md`，其中保存原始模块和共享定义的原文、SHA、逐件数据及 18 张同尺度双栏图。覆盖 11 模块、493 条部件记录和 204 个独立键：362 条原生网格完全不变，66 条显示几何或表示方式改变，51 条为非材料通道，14 条没有可定义表面。原输入、身份、父子次序与光学中心保留；整机坐标转换最大误差约 `4.55e-13 mm`，适用的包络/孔隙检查未发现新增负余量。HP/DPH 径向间隙仍为 `0.1 mm`。

原物镜线圈按父件的上下两段实际材料区间绘制，其范围本来就超出自身的等效轴向段；报告同时保留初始数值差和正确的物理区间，未为让比较通过而修改线圈。这些检查不是整机三维碰撞或制造公差认证。

CAD 专项几何、GUI 和 CI 注册检查，以及相关旧 3D/装配回归已通过。STEP 后端在独立 CadQuery 环境验证，新增独立 GitHub workflow；这不是声称远端 GitHub 已运行。本轮没有重新执行无关的长时间相干传播计算。
