"""Compare original electron trajectories, angular current and surface landings for the C1 trial."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from angle_origin import measure
from material_trial import HERE, Column, Field, source_and_electrodes
from condenser_pair_column import specimen
from relay_transfer import advance


FOLDERS = ('relay-edge', 'c1-strong-control', 'c1-zoom-fixed-objective', 'c1-zoom-focused')
LABELS = ('Previous', 'C1 alone', 'C1–C3, previous objective', 'C1–C3 and objective focus')
COLORS = ('#697581', '#ab5f23', '#b56328', '#006d76')
OUT = HERE/'c1-zoom'


def read(folder):
    record = json.loads((folder/'tracking.json').read_text())
    with np.load(folder/'column-and-landings.npz') as f:
        data = {k: f[k] for k in ('surface_phase', 'weights', 'loss', 'source_A', 'source_ids', 'z_mm', 'stats')}
    return record, data


def run_stations(folder, source, half_step=False):
    record, landing = read(folder)
    field = Field(Path(record['solution']), step=record['bore_step_mm'])
    roots = record['waists_mm']
    stations = dict(C1_waist=roots[0], C2_foil_in=764.9, C2_foil_out=765.1,
                    C3_entrance=830., C3_center=915., ADL=1088.5, TL22=1166.35, TL21=1249.95,
                    scan_waist=roots[3], Mini=1522.5, upper_pole_entrance=1534.5, specimen=record['surface_z_mm'])
    step = record['step_mm'] / (2 if half_step else 1)
    model = Column(source, field, Path(record['gun_source']), record['count'], step, observation_mm=list(stations.values()))
    idx = model.indices(np.array(list(stations.values()))*.001)
    sampled = advance(model.t, model.full, model.t.input, model.en, model.phi0, model.loss, 0, idx)
    result = {}
    for (label, position), (state, loss) in zip(stations.items(), sampled, strict=True):
        if np.any(loss < -1):
            raise ValueError('An electron left the numerical bore')
        live = loss == -1
        w = model.t.weights[live]
        result[label] = dict(z_mm=position, n=int(live.sum()), current_uA=1e6*model.t.current*w.sum(),
                             **measure(state[live],w))
    np.testing.assert_array_equal(loss, landing['loss'])
    np.testing.assert_array_equal(model.t.ids, landing['source_ids'])
    np.testing.assert_array_equal(model.t.weights, landing['weights'])
    if not half_step:
        np.testing.assert_allclose(result['specimen']['alpha95_mrad'],record['surface']['alpha95_mrad'],rtol=2e-5)
    peak_delta = np.linalg.norm((state[live,:2]-landing['surface_phase'][live,:2]),axis=1).max()
    return dict(step_mm=step, stations=result, surface=specimen(state[live],w,model.t.current),
                max_displacement_nm_relative_to_saved=1e9*peak_delta)


def comparison(records):
    fig, (envelope, angle) = plt.subplots(2, 1, figsize=(8.6,6.55), layout='constrained')
    for i in (0,3):
        rec, data = records[i]
        inside = (data['z_mm']>=450) & (data['z_mm']<=1088.5)
        envelope.plot(data['z_mm'][inside], data['stats'][inside,0]*1000, c=COLORS[i], lw=1.9, label=LABELS[i])
        for root in rec['waists_mm'][:2]:
            envelope.scatter(root, np.interp(root,data['z_mm'],data['stats'][:,0])*1000,
                             s=27, c=COLORS[i], edgecolor='white', linewidth=.6, zorder=5)
        live = data['loss']==-1
        alpha = 1e6*np.arctan(np.linalg.norm(data['surface_phase'][live,2:4],axis=1))
        w = data['weights'][live]*data['source_A']*1e6
        order = np.argsort(alpha)
        angle.step(alpha[order],w[order].cumsum(),where='post', c=COLORS[i], lw=1.8, label=LABELS[i])
        a95 = rec['surface']['alpha95_mrad']*1000
        angle.plot([a95,a95],[0,.95*w.sum()],ls=':',c=COLORS[i],lw=1.2)
        angle.scatter(a95,.95*w.sum(),s=28,c=COLORS[i],zorder=4)
        angle.annotate(f'{a95:.2f} µrad',(a95,8),xytext=(5,0),textcoords='offset points',c=COLORS[i],fontsize=9)
    for lo, hi, label in [(450,550,'C1 lens'),(550,750,'C2 lens'),(830,1000,'C3 lens')]:
        envelope.axvspan(lo,hi,color='#c9d5da',alpha=.30,zorder=-2)
        envelope.text((lo+hi)/2,.96,label,transform=envelope.get_xaxis_transform(),ha='center',va='top',fontsize=9)
    envelope.axvline(765,c='#9c5c24',ls='--',lw=.85)
    envelope.text(771,.07,'C2 aperture',transform=envelope.get_xaxis_transform(),fontsize=8,c='#875023')
    envelope.set(yscale='log',xlim=(450,1088.5), xlabel='Distance from effective source (mm)',ylabel='RMS beam radius (nm)')
    envelope.margins(y=.21)
    envelope.set_title('Condenser illumination',loc='left',weight='bold')
    angle.set(xlim=(0,14),ylim=(0,15.6),xlabel='Specimen semi-angle to the column axis (µrad)',ylabel='Enclosed current (µA)')
    angle.set_title('Current inside the specimen cone',loc='left',weight='bold')
    angle.legend(frameon=False,loc='lower right')
    for ax in (envelope,angle):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.17)
    fig.savefig(OUT/'condenser-and-angle.png',dpi=165)
    plt.close(fig)


def probe_image(records, indices=(0,3), titles=LABELS, filename='specimen-probes.png'):
    fig, axes = plt.subplots(1, 2, figsize=(9.3,4.4),layout='constrained',sharex=True,sharey=True)
    bins = np.linspace(-10,10,61)
    pitch = bins[1]-bins[0]
    maps = []
    for i in indices:
        rec, data = records[i]
        mask = data['loss']==-1
        x,y = data['surface_phase'][mask,:2].T*1e9
        currents = data['weights'][mask]*data['source_A']*1e9
        density = np.histogram2d(x,y,bins=[bins,bins],weights=currents)[0].T/pitch**2
        maps.append((density, density.sum()*pitch**2/currents.sum()))
    upper = max(d.max() for d,_ in maps)
    for ax,i,(d,fraction) in zip(axes,indices,maps,strict=True):
        s = records[i][0]['surface']
        img = ax.pcolormesh(bins,bins,d,cmap='magma',vmin=0,vmax=upper,shading='flat')
        ax.add_patch(Circle(np.array(s['centroid_um'])*1e3,500*s['D95_um'],fill=False,color='white',ls='--',lw=1.2))
        ax.set_title(f'{titles[i]}\nD95 {1000*s["D95_um"]:.2f} nm · {100*fraction:.2f}% current in view',fontsize=10)
        ax.set(xlabel='x (nm)',aspect='equal',xticks=[-10,-5,0,5,10],yticks=[-10,-5,0,5,10])
    axes[0].set_ylabel('y (nm)')
    fig.colorbar(img,ax=axes,shrink=.78,label='Landing current density (nA/nm²)')
    fig.suptitle('Geometrical probe at the specimen upper surface',weight='bold',fontsize=14)
    fig.savefig(OUT/filename,dpi=180)
    plt.close(fig)


def main(plots_only=False):
    records = [read(HERE/name) for name in FOLDERS]
    for rec, data in records:
        np.testing.assert_array_equal(data['source_ids'], records[0][1]['source_ids'][:len(data['source_ids'])])
        np.testing.assert_allclose(data['source_A'],records[0][1]['source_A'],rtol=1e-12,atol=0.)
        if len(data['weights'])==len(records[0][1]['weights']):
            np.testing.assert_array_equal(data['weights'],records[0][1]['weights'])
    comparison(records)
    probe_image(records)
    if plots_only:
        return
    source = source_and_electrodes()
    final = HERE/FOLDERS[-1]
    result = {FOLDERS[i]:run_stations(HERE/FOLDERS[i],source) for i in (0,3)}
    result['final_half_step'] = run_stations(final,source,half_step=True)
    prefixes = {}
    r, d = records[-1]
    for n in (8192,16384):
        live = d['loss'][:n]==-1
        prefixes[str(n)] = specimen(d['surface_phase'][:n][live], d['weights'][:n][live],float(d['source_A']))
    result['same_emission_prefix_absolute_weights'] = prefixes
    result['working_points'] = {}
    for name, (record, data) in zip(FOLDERS,records,strict=True):
        bias = record['field']['bias']
        selected = {k:v for k,v in bias.items() if k.startswith(('condenser_lens_', 'objective_lens/'))}
        live = data['loss']==-1
        central = np.linalg.norm(data['surface_phase'][live,:2],axis=1)<=50e-12
        result['working_points'][name] = dict(currents_A=selected, roots_mm=record['waists_mm'],
            field_sha=record['field']['solution_sha256'], positive_current_source_count=record['count'],
            surface=record['surface'], axis_50pm_packets=int(central.sum()))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true')
    main(parser.parse_args().plots_only)
