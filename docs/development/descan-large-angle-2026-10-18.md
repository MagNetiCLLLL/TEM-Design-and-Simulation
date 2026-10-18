# STEM Nanoprobe Large Angle：Descan 与 Lower Objective

日期：2026-10-18。验证范围为 **C3 + Probe Corrector、STEM Nanoprobe Large Angle、300 kV、Diffraction 投影、1 µm 方形扫描视场**。全程通过后台代码执行，没有操作桌面。

## 结论与工作点

前一轮采用的共享 Image/Diffraction–Descan 位置适用于本次 Large Angle 工作点，不需要再次移动硬件。上、下线圈中心仍为全局 **1674.4 / 1699.4 mm**，线圈长度 15 mm、中心间距 25 mm；Descan 是同一双级偏转器的动态通道，静态 Image/Diffraction 对准和共享峰值限制保持不变。选区光阑和主荧光屏仍在 **1824.4 / 2714.15 mm**。

使用当前模式重新执行 **Position and angle** 校准后，有限粒子传播在两个观测平面都保持了亚纳米的扫描束心稳定性和小于 0.002 µrad 的机械入射角变化。这是限定工作点的数值结果，不能扩展为所有视场、粒子数、样品或真实仪器的精度指标。

Large Angle 必须通过实际模式预设加载。仅修改 `nanoprobe_range` 标签不等于改变光学系统。本次 C2、C3、ADL、TL12 激发分别为 **25.2577272%、46.0598152%、79.8069434%、49.9770148%**，Objective 为 **68.9801%**。C3 为强激发，实际有限束斑在 C2–C3 和 C3–ADL 两段均存在 crossover。

## 位移与入射角：实际有限电子束

重新生成本次 Large Angle 的 3000 个物理 Tip 样本。比较两个校准约束、Descan 开/关、扫描中心及四角，使用 0.1 / 0.05 mm 积分步长，共 **40 个实际生产场计划 GPU 主比较案例**，每个案例另有束腰局部细化重放；另对两个约束的最后一角进行 CPU 重放。所有比较复用同一批实际执行的上游状态、权重、能量和粒子 ID。

完成输入与粒子来源校验后，另重放当前联合校准的中心及一角，结果与主批次逐项相同；再将 Normal 已接受的 held AC 和 Descan 驱动恢复到 Large Angle，在 0.05 mm 步长重放中心及四角，整个过程没有重新求解 held 驱动。

下表采用 0.05 mm 数据，数值为四角相对扫描中心的**最大二维矢量变化**，不是单轴峰值，也不是两个相反角之间的全幅变化。

| Descan 状态 | SAA 位移 / nm | SAA 机械角变化 / µrad | 主荧光屏位移 / nm | 主荧光屏机械角变化 / µrad |
|---|---:|---:|---:|---:|
| 关闭 | 16130.430 | 79.9407 | 4.47552 | 14.1533 |
| 仅位置校准 | 0.424794 | 95.2761 | 108669.534 | 258.517 |
| 沿用 Normal held 联合校准 | 41.024331 | 0.110059 | 106.375346 | 0.289027 |
| 位置与角度联合校准 | **0.424867** | **0.00174968** | **0.418449** | **0.00136140** |

![实际有限粒子传播的位移和角度比较](assets/descan-large-angle-2026-10-18.png)

仅位置约束消除了 SAA 束心移动，但没有同时控制入射角，因此会使下游屏幕移动。关闭 Descan 时，当前 Diffraction 光路的屏幕位移本来较小，但 SAA 位移和屏幕入射角仍变化。联合校准同时处理这些量。双级成像侧偏转器用于分别控制位置和角度的物理动机可参见 [JEOL de-scan 说明](https://www.jeol.com/words/emterms/20160713.103817.php)；本项目的具体位置和数值来自实际模拟传播。

40 次传播中，均保留完全相同的 **2401 / 3000 个粒子，源电流比例 80.0333%**，Descan 调整未新增截断。与上一轮 Normal 的 2662 个存活粒子不同，这是 Large Angle 上游传播的真实差异；没有补回被挡粒子、重新居中或强制归一化电流。

中心束绝对束心仍保留。例如 SAA 约为 `(−1.963635, +0.424234) µm`，屏幕约为 `(−0.667780, −4.259660) µm`；对应机械平均斜率分别约为 `(−6.03708, +1.30434) µrad` 和 `(−1.86125, −11.87353) µrad`。中心束在 SAA 和屏幕的 RMS 半径分别约 **440.133 / 944.585 µm**。亚纳米扫描残差表示图样对扫描近乎静止，不表示束斑本身只有亚纳米大小。

## 从 Normal 切换后的保持校准

直接保留 Normal 的 held 驱动再切换 Large Angle，会保留真实的失配。TL12 从 36.0045587% 变为 49.9770148%，其有限场支持范围 **1315.2–1417.4 mm** 与 AC 线圈重叠。AC 上级中心 1387.5 mm 处的 TL12 轴向磁场由约 **1.40615 mT** 变为 **1.95184 mT**，因此 AC 响应发生变化。

一阶响应验证中，旧 Normal held 校准在 Large Angle 下造成 SAA / 屏幕约 **40.5525 / 106.0372 nm 的最大单轴扫描位移**，机械角单轴峰值约 **0.108572 / 0.288031 µrad**。样品扫描位置映射相对误差约 **0.05355%**。这些是单轴一阶指标，与上表二维有限束斑指标不同。

实际有限粒子重放也确认了失配：SAA / 屏幕的最大二维位移为 **41.0243 / 106.3753 nm**，机械角变化为 **0.110059 / 0.289027 µrad**；存活 ID、电流和原来的 Normal held 矩阵均未改变。重新求 Large Angle 的完整校准后，分别下降到上表的约 0.425 / 0.418 nm。

重新校准后，AC 命令矩阵改变约 **0.09599%**，上下级耦合改变约 **0.08183%**；下游透镜和期望样品扫描映射相同，因此新的 Descan 物理驱动与 Normal 的联合校准数值几乎相同。这不意味着两个模式的上游电子束或 held 校准可以直接混用。

操作上：切换到 Large Angle 后选择 **Position and angle**，执行 **Calibrate and hold**，同时重新求 AC 与 Descan。若需要保留失配用于研究，可以继续使用 held；程序不会隐式重拟合它。仅执行 **Calibrate Descan (keep AC)** 不会修正旧 AC 的样品扫描映射。

## 共同倾转中心与有限束腰

重新校准 AC 后，不同扫描束心的一阶轨迹在全局 **Z≈1605.193496、1625.735134 mm** 有共同交点，交点处机械斜率非零。它们落在 Lower Objective 区域；Lower Objective 下极靴几何范围为 1601.9–1663.9 mm，中心为 1632.9 mm，这些定义不能混用。

将 Descan 在 SAA 产生的相空间增量沿同一线性光学系统向上游作等效延拓，在上述交点得到与 AC 相反的斜率、近零的位置增量。尺度化传递基条件数约 **162.314**；AC 与等效 Descan 的位置和斜率之和分别小于 **7×10⁻⁹ nm、2×10⁻⁹ µrad**。这是依赖于 SAA 联合抵消关系的线性一致性检查，并非独立的粒子束腰证据。真实 Descan 场在这些上游点的直接响应为零，不能解释为偏转器影响了已经通过的电子。

补偿后，下游一阶扫描轨迹重合，不能把每一个零响应截面都称为新的独立 tilt 中心。有限电子束的 crossover 则通过同一存活粒子群的二维包络及中心化位置–斜率协方差单独判断：

- C2–C3：执行观测点 **Z≈668.19107 mm**，二维 RMS 半径约 0.00923 nm。
- C3–ADL：执行观测点 **Z≈1017.93389 mm**，二维 RMS 半径约 0.05516 nm。
- Lower Objective 内的样品后径向 RMS 束腰：**Z≈1612.47754 mm，括定 ±0.01 mm**。在中心及 ±0.005 / ±0.01 mm 精确观测，0.05 / 0.025 mm 步长分别得到约 **0.216127 / 0.216131 nm** 的中心 RMS 半径；同一 2401 个粒子的中心化位置–斜率协方差在束腰前负、束腰后正。这个有限束腰与 1605.19 / 1625.74 mm 的扫描共同交点不同。

两个上游束腰均在约 ±0.01 mm 的精确观测括区内验证了先收敛、后发散；0.1 / 0.05 mm 积分步长使用相同的 2401 个正权重粒子并给出一致拓扑。这里的极小半径属于几何粒子模型的束腰，不含波动衍射极限；打印出的 Z 小数位也不是位置置信区间。

## 数值与复用边界

0.1→0.05 mm 比较中，联合校准在 SAA / 屏幕的最大绝对束心变化为 **0.00562 / 0.00320 nm**，机械斜率变化约 **0.0000272 / 0.0000101 µrad**。CPU/GPU 对比中联合校准束心差不超过 **3.45×10⁻¹⁰ nm**，电流和存活 ID 相同。

一阶位置研究也复核了原位置和当前安装位置：当前联合校准的最大单轴物理驱动约 **0.242460 mrad**，原位置约 **0.559525 mrad**，下降约 **56.7%**。这些是模型中的积分名义偏转角，不能直接当作线圈电流。尺度化求解条件数由原位置约 6.85 升至当前约 62.98；移动降低了驱动，不能宣称改善了条件数。0.1 / 0.05 mm 和 0.5 / 1 / 2 µrad 导数扰动检查支持当前解。没有再次改动机械位置，因此本轮不重复宣称新的 CAD 间隙认证。

本轮改进了研究脚本的模式选择与证据边界：拒绝跨 Normal/Large Angle 复用研究文件、候选及上游粒子；修复验证脚本给 Descan 动态添加不存在的 wobble 字段、导致导出参数不能回读的问题。原始粒子传播不消费该字段，物理结果不受它影响。

保存的上游粒子现在绑定完整数值摘要和实际上游传播依赖：原始 ID、坐标、斜率、权重、能量、存活状态和 Z，以及实际场计划、电子源、光阑和柱壁。改变上游场轮廓或孔径后拒绝复用；CPU/GPU 选择和纯下游 Descan 变化不冒充上游物理变化。新计算在执行时记录输入与摘要。本轮先完成的原始批次采用明确标记的事后关联：从保存的原始枪出口，按原来的 2200 个观测节点和 0.05 mm 步长独立传播至 Z=1379.999 mm，所有坐标/斜率与保存的前缀逐位相同，ID、权重、能量、2401 个存活粒子也一致，再建立校验记录。没有把事后摘要标成当时已记录的证据。

验证范围是当前场模型的真空电子光学传播，不包括样品散射、探测器像素强度、相干波 phase plate、其他投影模式、粒子数收敛或实机标定。SAA 在该工作点仍是 mixed 分类，联合校准没有将其重命名为像平面。

## 工作点与重现

可加载 `configs/design_candidates/nanoprobe_large_angle_descan_position_angle_20261018.toml`；配套 `.temwp` 是包含配置与 held 驱动的完整输入包。发布包从本次范围提交的干净检出重新生成，两种格式均检查回读及共享硬件和校准恢复；原研究目录中的旧包回读证据仍保留原身份。输入包恢复会校验归档输入完整性，允许当前兼容实现读取输入而不改写原始实现身份。它不含已保存的粒子计算结果，所有物理结果需重新计算；也不替换默认模式预设或 Normal 工作点。

```powershell
.venv/Scripts/python.exe scripts/study_descan_phase_space.py --nanoprobe-range large_angle --output outputs/descan_large_angle_repeat/linear --baseline-pair-center-mm 1786.9 --shifts -100 0 --pivot-audit --backward-equivalent-pivot-audit --validate-all-candidates --compare-normal-held
.venv/Scripts/python.exe scripts/validate_descan_transport.py --nanoprobe-range large_angle --output outputs/descan_large_angle_repeat/particles --compare-constraints --upstream-checkpoints --backend gpu --step-mm 0.1 --step-refinement --compare-backend cpu --no-chief-diagnostics
.venv/Scripts/python.exe scripts/validate_descan_transport.py --nanoprobe-range large_angle --output outputs/descan_large_angle_repeat/installed --replay-prefix-source outputs/descan_large_angle_repeat/particles --backend gpu
.venv/Scripts/python.exe scripts/validate_descan_transport.py --nanoprobe-range large_angle --output outputs/descan_large_angle_repeat/held --replay-prefix-source outputs/descan_large_angle_repeat/particles --normal-held-study outputs/descan_large_angle_repeat/linear/study.json --backend gpu
```

本次证据目录为 `outputs/descan_large_angle_20261018/`：

- `linear/study.json`：一阶求解、位置比较、扰动/步长、共同交点与 Normal held 对比。
- `particles/report.json`、`particles/summary.json`：40 个生产传播主案例及局部细化、原始绝对束心/角度/电流、同 ID 和 CPU/GPU 比较。
- `particles/installed-sealed/report.json`：输入和数值校验后的当前安装重放，以及 Lower Objective 内有限束腰括定。
- `particles/normal-held/report.json`：保留 Normal held 驱动的五点实际传播。
- `particles/prefix-provenance.json`、`particles/prefix-reconstruction-validation.json`：上游输入和粒子完整性校验，以及独立重放建立的历史关联。
- `topology/topology-comparison.json`：重新执行的上游二维束腰和步长比较。
- `topology/mode-and-held-scan-audit.json`：实际模式激发及 TL12–AC 场重叠。
- `profile-roundtrip.json`：TOML 与输入 Working Point 的恢复检查。
- `core-regression.xml`：444 项扫描、held 校准、有限场、共享硬件、模式、拓扑、输入包与缓存回归。
- `final-protocol-regression.xml`：39 项模式协议、参数回读、跨模式拒绝、虚拟延拓及粒子来源防误用回归。
- `final-test-summary.json`：两组测试身份不重叠，合计 **483 项通过，0 失败、0 错误、0 跳过**。

## 上传前的独立范围验证

将本次暂存内容单独检出后，从新的 3000 粒子 Tip 源重新执行全部 40 个 GPU 主案例和 2 个 CPU 角点对照。上游前缀及 40 个案例文件中的 290 个数组逐位相同；与原研究共有的 598076 个数值观测量最大差为零，仍为 2401 个存活粒子，电流比例 0.8003333333333333。比较排除了耗时、源码身份及描述性元数据，没有改写原始研究的身份。新前缀在计算时记录来源。

[发布复核记录](descan-publication-verification-2026-10-18.json)保留两份报告的 SHA256、源代码身份、比较范围和数值结果。该独立版本只包含 Descan 及所需电子枪几何场积分精度、缓存准则和模式缓存修改，不依赖原工作区的 Image 成像、可选源采样或辅助结构重建。Normal 和 Large Angle 的 TOML 与输入专用 Working Point 均从此版本生成，已回读验证原 held 驱动和共享偏转器关系；输入包还在第二份独立检出中成功恢复。

独立检出的相关回归最终为 **683 项不同测试通过，0 失败、0 错误、0 跳过**。首轮发现既有缓存测试的替代函数缺少现有接口的 `preserve_playback` 关键字；纳入该一行修正后，完整重跑缓存模块的 38 项并全部通过。最终统计以此复跑覆盖首轮缓存模块结果，没有将重复测试相加。复核记录同时保留首轮与修正后测试记录的 SHA256，以及自动验收注册检查。
