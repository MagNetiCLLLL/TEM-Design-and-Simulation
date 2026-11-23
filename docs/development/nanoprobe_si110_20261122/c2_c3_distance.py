"""Test whether a C2–C3 crossover closer to C2 increases occupied specimen angle."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from condenser_comparison import probe_image, read, run_stations
from angle_origin import measure
from material_trial import HERE, source_and_electrodes
from pupil_audit import audit


LABELS = ('c1-zoom-foil', 'c1-zoom-focused', 'c1-zoom-near-c2')
C2_EXIT_MM = 750.
OUT = HERE/'c1-zoom'


def inventory(bias, lens):
    values = {v for k, v in bias.items() if k.startswith(lens+'/')}
    if len(values)!=1:
        raise ValueError('Expected a single common current in the selected lens')
    return next(iter(values))


def analyze():
    source = source_and_electrodes()
    common_bias = reference = baseline_dc = None
    rows, envelopes = [], []
    for name in LABELS:
        folder = HERE/name
        record, data = read(folder)
        bias = record['field']['bias']
        frozen = {k:v for k,v in bias.items() if not k.startswith(('condenser_lens_2/', 'condenser_lens_3/'))}
        with np.load(record['gun_source']) as gun:
            initial = [gun[k] for k in ('launch','weight','ids','emitted_A')]
        if reference is None:
            reference, common_bias, baseline_dc = initial, frozen, record['field']['mesh_id']
        if frozen != common_bias or record['field']['mesh_id']!=baseline_dc:
            raise ValueError('An uncontrolled supply or the material mesh changed')
        for expected, actual in zip(reference, initial,strict=True):
            np.testing.assert_array_equal(expected, actual)
        intermediate = [z for z in record['waists_mm'] if z<1595.]
        if len(intermediate)!=4 or not all(lo<z<hi for z,(lo,hi) in zip(intermediate,
                [(450.,550.),(C2_EXIT_MM,830.),(1179.1,1237.2),(1420.,1512.5)],strict=True)):
            raise ValueError(f'{name}: invalid crossover branch {intermediate}')
        if not (folder/'pupil.json').exists():
            audit(folder)
        pupil = json.loads((folder/'pupil.json').read_text())
        stations = run_stations(folder, source)['stations']
        surface = record['surface']
        central = np.linalg.norm(data['surface_phase'][:,:2],axis=1)<=50e-12
        halves = []
        for ix in np.array_split(np.arange(record['count']), 2):
            subset = ix[data['loss'][ix]==-1]
            halves.append(1000*measure(data['surface_phase'][subset],data['weights'][subset])['alpha95_mrad'])
        half_step = None
        if name in (LABELS[0],LABELS[-1]):
            refined = run_stations(folder,source,half_step=True)
            half_step = dict(step_mm=refined['step_mm'],surface=refined['surface'],
                             max_landing_change_nm=refined['max_displacement_nm_relative_to_saved'])
        rows.append(dict(run=name, source_count=record['count'], effective_count=record['finite_source_effective'],
            mesh_id=baseline_dc, field_sha256=record['field']['solution_sha256'],
            C2_A=inventory(bias,'condenser_lens_2'),C3_A=inventory(bias,'condenser_lens_3'),
            crossover_mm=intermediate[1],distance_from_C2_exit_mm=intermediate[1]-C2_EXIT_MM,
            intermediate_waists_mm=intermediate, terminal_rms_waists_mm=record['waists_mm'][4:],
            specimen=surface, axis_50pm_packet_count=int(np.count_nonzero(central&(data['loss']==-1))),
            disjoint_source_halves_alpha95_urad=halves, refined_integration=half_step,
            rms_angle_urad=1000*stations['specimen']['rms_angle_mrad'],
            upper_pole_radius95_um=stations['upper_pole_entrance']['axis_r95_um'],
            Mini_radius95_um=stations['Mini']['axis_r95_um'], C2_foil_faces=pupil['foil_faces'],
            formal_C2_edge_mrad=pupil['formal_c2_edge_mrad'],formal_column_limit_mrad=pupil['limiting_geometric_semiangle_mrad'],
            formal_limiting_element=pupil['limiting_element'],required_30mrad_radius_mm=pupil['required_peak_radius_30mrad_mini_to_surface_mm']))
        envelopes.append((data['z_mm'],data['stats'][:,0]))
    xa = np.array([r['distance_from_C2_exit_mm'] for r in rows]); ya = np.array([r['specimen']['alpha95_mrad']*1000 for r in rows])
    comparison = dict(C2_exit_mm=C2_EXIT_MM,fixed_winding_currents_A=common_bias,
            source_identity='Identical emitted phase, original IDs, absolute weights and emitted current; gun transport repeated in each total DC field.',
            rows=rows,measured_relative_gain_percent=100*(ya[-1]/ya[0]-1),
            finite_interval_secant_urad_per_mm=(ya[0]-ya[-1])/(xa[0]-xa[-1]))
    (OUT/'c2-c3-distance.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    fig,(env,angle) = plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    colors=('#6d7783','#c57135','#006b74')
    for row,(z,rad),c in zip(rows,envelopes,colors,strict=True):
        use=(z>=744.)&(z<=785.)
        tag=f'{row["distance_from_C2_exit_mm"]:.2f} mm from exit'
        env.plot(z[use],rad[use]*1000,c=c,lw=1.8,label=tag)
        zz=row['crossover_mm']
        env.scatter(zz,1000*np.interp(zz,z,rad),s=28,c=c,zorder=3)
    env.axvspan(744,C2_EXIT_MM,color='#c7d2d7',alpha=.40)
    env.axvline(765.,ls=':',color='#526066',lw=1.)
    env.text(765.5,.97,'C2 aperture',transform=env.get_xaxis_transform(),va='top',fontsize=8)
    env.text(747,.06,'C2 lens',rotation=90,transform=env.get_xaxis_transform(),ha='center',fontsize=8)
    env.set(xlim=(744,785),ylim=(0,None),xlabel='Distance from effective source (mm)',ylabel='RMS beam radius (nm)')
    env.set_title('Crossover between C2 and C3 lenses',loc='left',fontsize=11,weight='bold')
    env.legend(frameon=False,fontsize=8,loc='upper left')
    angle.plot(xa,ya,color='#839197',lw=1.,zorder=1)
    for x,y,c in zip(xa,ya,colors,strict=True):
        angle.scatter(x,y,c=c,s=52,zorder=2)
        angle.annotate(f'{y:.2f}',(x,y),xytext=(4,6),textcoords='offset points',color=c,fontsize=10,weight='bold')
    angle.set(xlim=(0,max(xa)+2),ylim=(min(ya)-.32,max(ya)+.35),xlabel='Crossover distance below C2 lens exit (mm)',ylabel='Semi-angle containing 95% of specimen current (µrad)')
    angle.set_title('Same C1 lens and objective currents',loc='left',fontsize=11,weight='bold')
    angle.yaxis.set_major_locator(MaxNLocator(6))
    for ax in (env,angle):
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.19)
    fig.savefig(OUT/'c2-c3-distance.png',dpi=180)
    plt.close(fig)
    titles = [f'Crossover {r["distance_from_C2_exit_mm"]:.2f} mm below C2 lens exit' for r in rows]
    probe_image([read(HERE/n) for n in LABELS],(0,len(rows)-1),titles,'c2-c3-specimen-probes.png')
    return comparison


if __name__=='__main__':
    analyze()
