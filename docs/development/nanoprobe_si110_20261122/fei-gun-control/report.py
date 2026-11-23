"""Figures and current ledgers for the physical-tip cold-gun voltage sweep."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from gun_field import HERE, WORK, key, study_field_reader
from transport import field_at


BIASES = (50, 500, 1000, 1200, 1500, 1800, 2000)
COLORS = {'extractor': '#bb5219', 'electrostatic_lens': '#8950a6', 'gun_aperture': '#008d86',
          'exit_anode': '#986019', 'downstream': '#778598', 'exit': '#17578c'}
NAMES = {'extractor': 'Extractor', 'electrostatic_lens': 'Gun lens', 'gun_aperture': '24 mm plate',
         'exit_anode': 'Independent anode', 'downstream': 'Other losses', 'exit': 'Transmitted'}


def write_csv(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_record(voltage):
    name = key('anode', True, voltage) + '-step0.5'
    return json.loads((HERE / (name+'-currents.json')).read_text())


def category(key_):
    return key_ if key_ in COLORS else 'downstream'


def summarize(record):
    ledger = record['ledger']
    current = {name: row['uA'] for name, row in ledger.items()}
    upstream = sum(current.get(k, 0.) for k in
                   ('extractor', 'electrostatic_lens', 'tip_return', 'feg_emitter_enclosure', 'feg_cathode_holder'))
    after_plate = 15. - upstream - current['gun_aperture']
    return dict(lens_kV=record['voltage_V']/1000., emitted_uA=15.,
                extractor_uA=current['extractor'], gun_lens_uA=current['electrostatic_lens'],
                gun_aperture_uA=current['gun_aperture'], after_gun_aperture_net_uA=after_plate,
                independent_anode_uA=current.get('exit_anode', 0.),
                downstream_losses_uA=after_plate-current.get('exit_anode', 0.)-current['exit'],
                accelerator_rings_uA=sum(v for k,v in current.items() if k.startswith('accelerator:')),
                accelerator_bore_uA=current['accelerator_bore'],
                missed_outlet_entrance_uA=current['outside_outlet_entrance'],
                outlet_tube_uA=sum(v for k,v in current.items() if k.startswith('grounded_outlet')),
                C1_aperture_uA=current['C1_aperture'], exit_nA=1000*current['exit'],
                original_particles=record['n_original'], effective_original=record['weight_effective'],
                exit_particles=ledger['exit']['count'], effective_exit=ledger['exit']['effective_count'])


def configure():
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#8a97a5',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
                         'xtick.color':'#3e4a59','ytick.color':'#3e4a59','text.color':'#172e43',
                         'axes.labelcolor':'#263c50','font.size':10.5, 'axes.titlesize':11.5,
                         'axes.titleweight':'bold','legend.frameon':False,'savefig.dpi':165})


def selected_paths(voltage):
    p=f'v{voltage}_'
    with np.load(HERE / 'representative-trajectories.npz') as data:
        labels = data[p+'labels']
        categories = np.array([category(s) for s in labels])
        path = np.linalg.norm(data[p+'xy_m'].astype(float), axis=2)*1000.
        z = data['z_m']*1000.
        end = data[p+'end_xyz_m']*1000.
        angles = data[p+'theta']
    traces = []
    for group in ('downstream', 'exit', 'gun_aperture', 'electrostatic_lens', 'extractor', 'exit_anode'):
        available = np.flatnonzero(categories == group)
        if not available.size:
            continue
        indices = available[np.argsort(angles[available])]
        for i in np.unique(indices[np.linspace(0, len(indices)-1, min(len(indices),12)).astype(int)]):
            good = np.isfinite(path[i])
            zz, rr = z[good], path[i,good]
            hit_z, hit_r = end[i,2], np.hypot(*end[i,:2])
            if not len(zz) or hit_z < zz[-1]-1e-5:
                continue  # Backtracking is retained in the current ledger.
            traces.append((np.r_[zz,hit_z],np.r_[rr,hit_r],group))
    return traces


def draw_hardware(ax, rows, full, ymax):
    for q in rows:
        comp = q['key']
        a,b,inner,outer = [1000*q[k] for k in ('start_m','stop_m','inner_m','outer_m')]
        if b < 0 or a > (451 if full else 38) or inner >= ymax:
            continue
        if comp == 'outside_outlet_entrance':
            ax.vlines(a, inner, ymax, colors='#556674', linestyle=':', linewidth=1.,zorder=2)
            continue
        if comp.startswith('grounded_outlet_joint:') or comp=='gun_aperture' or comp=='C1_aperture':
            ax.vlines((a+b)/2,inner,min(ymax,outer),colors='#5b6771',lw=1.3,zorder=3)
        ax.add_patch(Rectangle((a,inner),b-a, max(0.,min(ymax,outer)-inner),
                               facecolor='#d8dfe5',edgecolor='#7c8b98',linewidth=.6,zorder=1))
    for at, label in ([(100,'Accelerator bore'),(370,'Tube inlet'),(444,'C1 ap.')] if full else
                      [(1.8,'Extractor'),(18.,'Gun lens'),(24.,'Plate'),(34.,'Anode')]):
        ax.text(at,.955*ymax,label,va='top',ha='right' if label=='C1 ap.' else 'center',
                fontsize=8.,zorder=6,color='#31475c',clip_on=True,
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.))


def ray_plot():
    fig, axes = plt.subplots(3,2,figsize=(12.1,8.6),sharex='col',gridspec_kw={'width_ratios':[1,1.05]})
    fig.subplots_adjust(left=.072,right=.986,top=.87,bottom=.11,hspace=.31,wspace=.17)
    for row,v in enumerate((50,500,1200)):
        curves = selected_paths(v)
        parts = json.loads((HERE / (key('anode',True,v)+'.json')).read_text())['stops']
        for col,ax in enumerate(axes[row]):
            full = col==1
            ymax=5.85
            draw_hardware(ax,parts,full,ymax)
            for zz,rr,group in curves:
                color=COLORS[group]
                ax.plot(zz,rr,color=color,lw=.85 if group!='downstream' else .7,
                        alpha=.70 if group!='downstream' else .62,zorder=4)
                ax.scatter(zz[-1],rr[-1],s=7,color=color,zorder=5,linewidths=0.)
            ax.set(xlim=(-.30,38) if not full else (-1.,451.),ylim=(0,ymax))
            ax.grid(axis='y',alpha=.18)
            ax.set_title(f'+{v/1000:.2f} kV  |  '+('complete gun' if full else 'tip to independent anode'),loc='left')
            if not full:
                ax.set_ylabel('Ray radius (mm)')
            if row==2:
                ax.set_xlabel('Distance from tip, z (mm)')
    fig.suptitle('Where electrons strike inside the cold field-emission gun',x=.072,ha='left',
                 y=.985,fontsize=19,fontweight='bold')
    fig.text(.072,.927,'Each curve is one primary electron. Color identifies its first intercept or transmission to 450 mm.',
             fontsize=10.8)
    handles=[Line2D([],[],color=COLORS[k],lw=2,label=NAMES[k]) for k in
             ('extractor','electrostatic_lens','gun_aperture','exit_anode','downstream','exit')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.003),ncol=6,fontsize=9.2)
    fig.savefig(HERE/'gun-trajectories.png')
    plt.close(fig)


def extractor_plot():
    fig,(a,b)=plt.subplots(1,2,figsize=(10.6,4.2),gridspec_kw={'width_ratios':[1,1.15]})
    fig.subplots_adjust(left=.075,right=.985,top=.78,bottom=.24,wspace=.24)
    fig.suptitle('Primary interception at +1.20 kV gun-lens bias',x=.075,ha='left',
                 y=.975,fontsize=16.5,fontweight='bold')
    parts=json.loads((HERE / (key('anode',True,1200)+'.json')).read_text())['stops']
    for q in parts:
        if q['key']=='extractor':
            lo,hi,ri=[1000*q[k] for k in ('start_m','stop_m','inner_m')]
            if ri<1:
                a.add_patch(Rectangle((lo,ri),hi-lo,1-ri,color='#d8dfe5',ec='#7c8b98',lw=.7))
    a.text(.78,.69,'Extractor throat',fontsize=9.,ha='left')
    a.set(xlim=(-.035,1.48),ylim=(0,.77),title='Tip and extractor',ylabel='Ray radius (mm)')
    draw_hardware(b,parts,False,5.1)
    b.set(xlim=(10,37.5),ylim=(0,5.1),title='Gun lens, plate and independent anode')
    curves=selected_paths(1200)
    for ax in (a,b):
        for zz,rr,group in curves:
            ax.plot(zz,rr,color=COLORS[group],alpha=.75,lw=.8)
            ax.scatter(zz[-1],rr[-1],s=7,color=COLORS[group],zorder=4,linewidths=0.)
        ax.set_xlabel('Distance from tip, z (mm)')
        ax.grid(axis='y',alpha=.18)
    handles=[Line2D([],[],color=COLORS[k],lw=2,label=NAMES[k]) for k in
             ('extractor','electrostatic_lens','gun_aperture','downstream','exit')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),ncol=5,fontsize=9.5)
    fig.savefig(HERE/'gun-front-end.png')
    plt.close(fig)


def response_plot(rows):
    v = np.array([d['lens_kV'] for d in rows])
    fig,(a,b)=plt.subplots(1,2,figsize=(11.4,4.8),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.075,right=.978,top=.82,bottom=.15,wspace=.25)
    fig.suptitle('How gun-lens bias redistributes 15 µA of emission',x=.075,ha='left',
                 y=.97,fontsize=17,fontweight='bold')
    for col,group,label in [('extractor_uA','extractor','Extractor'),('gun_lens_uA','electrostatic_lens','Gun lens'),
                            ('gun_aperture_uA','gun_aperture','24 mm plate'),
                            ('downstream_losses_uA','downstream','Downstream losses')]:
        a.plot(v,[d[col] for d in rows],'-o',color=COLORS[group],label=label,lw=1.8,ms=4)
    a.set(title='First interception',xlabel='Gun-lens potential relative to tip (kV)',ylabel='Collected current (µA)',
          ylim=(-.28,11.3))
    a.legend(loc='upper right',ncol=2,fontsize=8.5)
    b.semilogy(v,[1000*d['after_gun_aperture_net_uA'] for d in rows],'-o',
               color=COLORS['gun_aperture'],lw=1.8,ms=4,label='Just past the 24 mm plate')
    b.semilogy(v,[d['exit_nA'] for d in rows],'-o',color=COLORS['exit'],lw=1.8,ms=4,label='Gun exit at 450 mm')
    b.set(title='Net transmitted current',xlabel='Gun-lens potential relative to tip (kV)',ylabel='Beam current (nA)')
    b.legend(loc='center right',fontsize=8.5)
    for ax in (a,b):
        ax.grid(alpha=.22)
        ax.set_xlim(-.01,2.06)
    fig.savefig(HERE/'interception-scan.png')
    plt.close(fig)


def axial_plot():
    load_field=study_field_reader()
    fig,(a,b)=plt.subplots(1,2,figsize=(10.9,4.55))
    fig.subplots_adjust(left=.077,right=.982,top=.81,bottom=.15,wspace=.24)
    fig.suptitle('Axial energy change through the electrostatic gun lens',
                 x=.077,ha='left',y=.975,fontsize=17,fontweight='bold')
    z=np.linspace(1.4,36.6,2201)
    samples=[]
    for v,color in [(50,'#bb5219'),(500,'#008981'),(1200,'#17578c'),(2000,'#8950a6')]:
        ef,_=load_field('anode',True,v)
        curve=np.array([field_at(0.,0.,pos/1000.,ef) for pos in z])
        a.plot(z,curve[:,0]/1000.,color=color,lw=1.65,label=f'+{v/1000:.2f} kV')
        b.plot(z,-curve[:,3]/1e6,color=color,lw=1.55)
        samples.extend(dict(lens_kV=v/1000.,z_mm=pos,axis_phi_V=val[0],axis_Ez_V_per_m=val[3])
                       for pos,val in zip(z,curve))
    for ax in (a,b):
        for x1,x2,tag in [(14,22,'Lens'),(23.9,24.1,'Plate'),(31.5,36.5,'Anode')]:
            ax.axvspan(x1,x2,color='#b3c2cd',alpha=.23)
            ax.text((x1+x2)/2,1.013,tag,transform=ax.get_xaxis_transform(),ha='center',fontsize=8.)
        ax.set(xlim=(1.4,36.6),xlabel='Distance from tip, z (mm)')
        ax.grid(alpha=.2)
    a.set_ylabel('On-axis potential relative to tip (kV)')
    a.legend(loc='upper left',fontsize=9)
    b.set_ylabel('Axial force per unit charge magnitude, −E$_z$ (MV/m)')
    b.axhline(0,color='#202a38',lw=.7)
    fig.savefig(HERE/'axial-electric-field.png')
    plt.close(fig)
    write_csv('axial-field.csv',samples)


def report():
    configure()
    records={v:main_record(v) for v in BIASES}
    archive={}
    loss_rows=[]
    rows=[]
    for voltage,record in records.items():
        row=summarize(record)
        with np.load(WORK / (record['name']+'-rays.npz')) as data:
            if not archive:
                archive.update(source_xyz_m_u_over_c=data['phase'],source_id=data['ids'],weight_uA=data['weight_uA'],
                               surface_normal_polar_rad=data['polar_angle_rad'],labels=data['labels'])
            np.testing.assert_array_equal(data['phase'],archive['source_xyz_m_u_over_c'])
            np.testing.assert_array_equal(data['weight_uA'],archive['weight_uA'])
            np.testing.assert_array_equal(data['labels'],archive['labels'])
            labels=data['labels'][data['label']]
            end=data['end']
            ex=labels=='exit'
            assert np.allclose(end[ex,2],.45,atol=1e-15,rtol=0.)
            code=str(voltage)+'V_'
            archive[code+'first_hit']=data['label'].astype(np.uint8)
            archive[code+'hit_z_m']=end[:,2].astype(np.float32)
            archive[code+'hit_r_m']=np.hypot(end[:,0],end[:,1]).astype(np.float32)
            archive[code+'exit_id']=data['ids'][ex]
            archive[code+'exit_state_xyz_u_time']=end[ex,:7]
            ang=np.arctan2(np.hypot(end[ex,3],end[ex,4]),end[ex,5])
            order=np.argsort(ang)
            w=data['weight_uA'][ex][order]
            row['exit_axis_semiangle95_mrad']=1000.*ang[order][np.searchsorted(np.cumsum(w),.95*w.sum())]
        rows.append(row)
        for component,amount in record['ledger'].items():
            loss_rows.append(dict(lens_kV=voltage/1000.,component=component,current_uA=amount['uA'],
                                  primary_count=amount['count'],effective_count=amount['effective_count']))
    np.savez_compressed(HERE/'source-and-first-hits.npz',**archive)
    write_csv('sweep.csv',rows)
    write_csv('all-collectors.csv',loss_rows)
    controls=[]
    for file in sorted(HERE.glob('*-currents.json')):
        record=json.loads(file.read_text())
        controls.append(dict(case=record['name'],mesh=record['mesh'],step=record['step_scale'],
                             source_field_V_per_nm=record['source']['apex_V_per_nm'],**summarize(record)))
    write_csv('controls.csv',controls)
    ray_plot()
    extractor_plot()
    response_plot(rows)
    axial_plot()


if __name__=='__main__':
    report()
