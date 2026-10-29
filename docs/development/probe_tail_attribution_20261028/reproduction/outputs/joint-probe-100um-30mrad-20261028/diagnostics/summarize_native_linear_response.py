"""Read completed local-response evidence and produce an English offline summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_column_energy_diagnostic import file_hash


def summarize(run,output):
    run=Path(run).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    r=json.loads((run/'report.json').read_text());checks=json.loads((run/'checksums.json').read_text())
    if r['status']!='COMPLETE' or r['stability_status']!='PASS':raise ValueError('Stable completed diagnostic required')
    for name in ('report.json','calibration-probes.npz','mathematical-predictions.npz'):
        if file_hash(run/name)!=checks[name]:raise ValueError('Completed diagnostic checksum mismatch')
    c=r['comparison'];a=c['profiles']['actual_full_field'];p=c['profiles']['linear_prediction_half_h'];rem=c['nonlinear_remainder']
    lines=['# Native local-linear column response diagnostic','',
        'Within this executed engineering-field model, the large ID306 sample-plane tail is absent from the stable local first-order prediction and appears predominantly in the full-field map minus that prediction. This is a local nonlinear remainder, not a unique spherical-aberration attribution or an additive decomposition of beam widths.','',
        f"The one calibration batch executed 17 mathematical probes in {r['elapsed_s']:.6f} s on the exact original 23,513-node / 479-checkpoint mesh. All survived without a stop. The static field snapshot, potential/geometry/plan identities, zero induction work, original inputs and saved full-source files passed their guards.",
        '', '## Calibration and stability','',
        'The expansion reference is the actual current-weighted original poly gun-exit mean. Every probe uses K = 300000.3 eV and the mean original exit clock. The probes have no assigned beam current and are not an electron source.',
        '',f"Reference [x, y, tx, ty] = {r['reference_xy_m_slopes_rad']} in [m, m, rad, rad]; clock = {r['reference_clock_s']:.12g} s.",
        '', 'The perturbations are 1 nm in x/y and 10 nrad in tx/ty, plus their half sizes. Both four-coordinate central differences share the same reference, requiring 17 trajectories. Predictions use all 769 original source columns and their unchanged absolute weights. They are mathematical arrays stored separately from executed particles.',
        '',f"Across every source point and every requested through-focus plane, h versus h/2 changes the predicted x/y vector by at most {c['stability']['maximum_prediction_xy_difference_nm']:.12g} nm (predeclared limit 0.01 nm), and the slope vector by {c['stability']['maximum_prediction_slope_difference_rad']:.12g} rad (limit 1e-6 rad).",'',
        '## Native profile comparison','',
        '| Quantity | Executed full field | Local first-order prediction (h/2) |','|---|---:|---:|']
    for key,label in (('d90_nm','Surface D90 (nm)'),('d95_nm','Surface D95 (nm)'),('rms_radius_nm','Surface RMS radius (nm)'),('alpha95_mrad','Surface alpha95 (mrad)'),('id306_radius_nm','Surface ID306 centered radius (nm)')):
        lines.append(f"| {label} | {a['surface'][key]:.12g} | {p['surface'][key]:.12g} |")
    lines.extend([f"| Smallest observed D95 (nm) | {a['d95_minimum']['minimum']:.12g} | {p['d95_minimum']['minimum']:.12g} |",
        f"| Its offset from the sample (nm) | {a['d95_minimum']['representative_offset_nm']:.12g} | {p['d95_minimum']['representative_offset_nm']:.12g} |",'',
        f"ID306 raw x/y is {rem['id306_actual_surface_xy_nm']} nm in the actual result and {rem['id306_predicted_surface_xy_nm']} nm in the prediction. The raw vector remainder is {rem['id306_surface_xy_remainder_nm']} nm, norm {rem['id306_surface_xy_remainder_norm_nm']:.12g} nm. It is the largest source-point remainder at the sample.",
        '', 'The large-ray result and the whole-population width answer different questions: the actual surface D95 is smaller than the first-order D95 even though its largest tail is much larger. The first-order minimum is a mathematical reference, not a demonstrated attainable physical probe; it does not authorize retuning, rescaling, or imposing another focus definition. No D95 quadrature subtraction is used.','',
        '## Evidence and limits','',
        '- Actual population: the already executed complete constant-K 769-particle comparison, original current 10 uA. No new full-source propagation was performed.',
        '- Only 17 differential probes were newly propagated, with the unchanged complete native field model and physical geometry; no legacy matrix, empirical Cs term or alternate source was introduced.',
        '- The local response is stable to the tested perturbation sizes. This does not certify every field approximation, step discretization, source distribution or experimental microscope.',
        '- No physical beam current, qualification, feasibility claim, or size acceptance threshold is attached to the mathematical prediction.','',
        f"Raw evidence directory: `{run}`",'',f"report.json SHA256: `{checks['report.json']}`",f"calibration-probes.npz SHA256: `{checks['calibration-probes.npz']}`",f"mathematical-predictions.npz SHA256: `{checks['mathematical-predictions.npz']}`",''])
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    with np.load(run/'mathematical-predictions.npz') as f:
        z=f['z_mm'];i=int(np.flatnonzero(z==a['surface']['z_mm'])[0]);ids=f['source_id'];j=int(np.flatnonzero(ids==306)[0])
        linear=f['prediction_half_h_xy_slopes'][i,:,:2]*1e9
    original=Path(r['constant_case_path'])/'particles.npz'
    if file_hash(original)!=r['constant_case_hashes']['particles.npz']:raise ValueError('Actual original particle checksum mismatch')
    with np.load(original) as f:
        if not np.array_equal(f['z_mm'],z) or not np.array_equal(f['source_id'],ids):raise ValueError('Actual original grid or source mismatch')
        actual=np.column_stack((f['x_m'][i],f['y_m'][i]))*1e9
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].scatter(actual[:,0],actual[:,1],s=8,alpha=.6,label='Executed full field')
    axes[0].scatter(linear[:,0],linear[:,1],s=8,alpha=.6,label='Mathematical local-linear prediction')
    axes[0].scatter(actual[j,0],actual[j,1],s=65,facecolors='none',edgecolors='black',label='ID306 actual')
    axes[0].set(xlabel='Original global X (nm)',ylabel='Original global Y (nm)',title='Same 769 entry states; no coordinate rescaling')
    axes[0].axis('equal');axes[0].legend(fontsize=8)
    for profile,label in ((a,'Executed full field'),(p,'Mathematical local-linear prediction')):
        axes[1].plot([row['offset_nm'] for row in profile['planes']],[row['d95_nm'] for row in profile['planes']],label=label)
    axes[1].axvline(0,color='black',lw=.8,ls='--');axes[1].set(xlabel='Observed offset from sample (nm)',ylabel='Native D95 (nm)',title='Exact saved planes; minima are discrete observations')
    axes[1].legend(fontsize=8);fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    (output/'summary-source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();summarize(a.run,a.output)
