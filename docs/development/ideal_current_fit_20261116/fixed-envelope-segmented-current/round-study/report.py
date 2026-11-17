"""Prepare the field-only fixed-envelope segmentation report and figures."""
from pathlib import Path
import csv
import json
from hashlib import sha256
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'results.json').read_text())
joint=json.loads((HERE/'joint-results.json').read_text())
library=json.loads((HERE/'candidate-library.json').read_text())
precision=json.loads((HERE/'precision-audit.json').read_text())
profiles=np.load(HERE/'profiles.npz')

rows=[]
for owner in raw['owners']:
    for cutoff,metrics in owner['selected_metrics'].items():
        rows.append(dict(owner=owner['owner'],rcond=cutoff,
            physical_turns=owner['original_turn_count'],segments=metrics['segment_count'],
            full_axis_q24_relative_l2_error=metrics['holdout_q24_relative_l2'],
            maximum_current_a=metrics['maximum_absolute_current_a'],
            q12_to_q24_relative_l2=metrics['q12_to_q24_field_relative_l2'],
            retained_condition=metrics['retained_condition'],
            current_error_gain_upper_bound=metrics['independent_fractional_current_error_gain_upper_bound']))
with (HERE/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

fig,axes=plt.subplots(4,4,figsize=(16,13),layout='constrained')
for ax,owner in zip(axes.flat,raw['owners']):
    key=owner['owner'];z=profiles[key+'_z_m'];target=profiles[key+'_target_t'];fitted=profiles[key+'_candidate_t']
    center=float(z[len(z)//2]);metrics=owner['selected_metrics']['1e-10']
    ax.plot((z-center)*1000,target,color='black',label='Gaussian reference')
    ax.plot((z-center)*1000,fitted,color='tab:blue',label='Fixed-envelope signed segments')
    ax.set(xlim=(-90,90),xlabel='Relative Z (mm)',ylabel='Bz (T)',
        title=f"{key}\nL2 {100*metrics['holdout_q24_relative_l2']:.3f}% · {metrics['segment_count']} cells")
for ax in list(axes.flat)[len(raw['owners']):]:ax.set_visible(False)
axes.flat[0].legend(fontsize=7)
fig.suptitle('Original winding envelopes and physical turns unchanged · internal prescribed current only\nSVD cutoff 1e-10 is an explored numerical conditioning choice, not a field or focus acceptance gate')
fig.savefig(HERE/'per-owner-field-comparison.png',dpi=160);plt.close(fig)

lines=['# Fixed-envelope segmented prescribed-current round-lens study','',
    'Status: field-only feasibility and candidate-current evidence. No emitter, finite-source particle transport, coherent propagation, resistance or constant-power calculation was run. No production source or configuration file was edited by this study.',
    '', '## Geometry and source constraints', '',
    'The reference is the exact archived ideal-coil geometry and controls in mixed-run-v3. The old resized/relocated inverse-design candidates are not used. All 13 enabled round owners retain the original winding inner radius, outer radius, axial interval, pose, physical turn total, and conductor-metal-area records. Objective retains 580 turns in each of its two unchanged axial intervals.',
    '', 'Each original rectangular radial-axial current cross-section is tiled by internal cells. Every cell has a positive integer physical-turn allocation. Radial strip widths are proportional to strip turns; axial lengths within each strip are proportional to cell turns. Thus each cell area is exactly its turn fraction times the parent area, to floating-point reconstruction precision. Equal current gives exactly the original continuous uniform azimuthal current density.',
    '', 'This is electrical segmentation of the declared homogenized ideal-current cross-section. It is not proof that each arbitrary internal cell boundary aligns with the existing individual-wire CAD layout. The original CAD and wire diameter are not changed. Actual turn-addressed wiring would require grouping the existing wire coordinates separately.',
    '', 'A production uniform-current fast path can merge equal-current cells into the original parent field source. This avoids numerical differences caused solely by repartitioning the original q12 quadrature. Those differences are explicitly recorded for the unmerged experimental basis.',
    '', '## Method', '',
    'Axial-only splits use 1, 2, 4, 8, 16 and 32 requested cells per original interval; additional radial-axial nested tilings are explored, with no more cells than allocated physical turns. Signed currents are found by weighted SVD on the full real axis. Finite central panels and rational infinite-tail transformations retain all tails. Fits are repeated for SVD cutoffs 1e-6 through 1e-14. These cutoffs are experiments in conditioning, not changes to project tolerances.',
    '', 'Selected per-owner solutions are independently evaluated on a finer whole-axis quadrature and original q12 versus refined q24 source quadrature. Native AnnularCoilPotential constructors and axis evaluations are checked. MP80 spot checks show the error introduced by strong signed-current cancellation. A smaller least-squares residual alone is not admitted as a focused or numerically qualified working point.',
    '', '## Per-owner results at SVD cutoff 1e-10', '',
    '| Owner | Original physical turns | Cells | Full-axis q24 relative L2 error | Maximum absolute current (A) | q12 to q24 relative field difference |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for owner in raw['owners']:
    m=owner['selected_metrics']['1e-10']
    lines.append(f"| {owner['owner']} | {owner['original_turn_count']} | {m['segment_count']} | {100*m['holdout_q24_relative_l2']:.6f}% | {m['maximum_absolute_current_a']:.8g} | {m['q12_to_q24_field_relative_l2']:.4g} |")
lines+=['', '## Joint fixed-working-point fit', '',
    'The joint study uses 545 internal cells chosen from the per-owner designs and permits all signed cell currents to fit the complete Gaussian Bz sum. It is a fixed-point field fit, not a calibration preserving each logical lens control response. Independent C2 lens/C3 lens changes must be re-evaluated rather than inheriting the old Gaussian gain.',
    '', '| SVD cutoff | Retained rank | Full-axis q24 L2 error | Maximum absolute current (A) | q12 to q24 relative field difference |',
    '| --- | ---: | ---: | ---: | ---: |']
for row in joint['regularizations']:
    lines.append(f"| {row['rcond']:g} | {row['retained_rank']} | {100*row['holdout_q24_relative_l2']:.6f}% | {row['maximum_absolute_current_a']:.8g} | {row['q12_to_q24_field_relative_l2']:.4g} |")
lines+=['', 'Larger retained condition numbers yield smaller fitted Gaussian residuals but much larger currents and cancellation sensitivity. The 1e-14 joint solution needs about 8.12e14 A in its largest ideal segment. Ideal zero resistance does not make floating-point cancellation disappear. No thermal or hardware limit is imposed here; the numerical losses are still reported.',
    '', '## Independent high-precision checks', '',
    'MP80 evaluates the same finite cells using exact serialized binary64 geometry/current inputs and high-precision 12-point Gauss-Legendre nodes and weights. This isolates errors that can be hidden by repeated binary64 calculations.',
    '', '| Owner / cutoff | Maximum native binary64 versus MP80 axis B error (T) |',
    '| --- | ---: |']
for case in precision['cases']:
    lines.append(f"| {case['owner']} / {case['rcond']} | {case['maximum_native_simple_sum_error_t']:.8g} |")
lines+=['', '## Limits and next use', '',
    'The fixed envelopes can provide substantially better axial fields by internal opposite-current regions, but the explored finite-dimensional fits do not reconstruct every old narrow Gaussian equally well. The relay lenses are particularly ill-conditioned. Aggressive truncation choices must not be installed merely because the residual is smaller.',
    '', 'No focus, crossover topology, current transmission, D95, convergence angle, source-count convergence or scanning image has been qualified. Field fitting and actual finite-source focus are separate checks. The candidate-library JSON preserves every signed current and physical cell; the joint library additionally preserves its fixed-working-point control limitation.',
    '', 'Files: results.json and summary.csv (per-owner sweep); candidate-library.json (physical cell and current definitions); joint-results.json and joint-candidate-library.json; precision-audit.json; per-owner-field-comparison.png; joint-field-comparison.png; profiles.npz and joint-profiles.npz. The exact first fit script is retained as executed-fit-v1.py; fit.py only corrects its explicit project import root for portability.', '']
(HERE/'README.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={path.name:sha256(path.read_bytes()).hexdigest() for path in HERE.iterdir() if path.is_file() and path.name!='checksums.json'}
(HERE/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(owners=len(raw['owners']),joint_cells=joint['cell_count'],written='README.md')))
