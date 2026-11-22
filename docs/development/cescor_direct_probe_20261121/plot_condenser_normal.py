"""Compare the same source IDs in the two physical gaps of the Normal branch."""

from hashlib import sha256
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_normal import ACTIVE, FIELD, HERE, OWNER, WORK, spans_mm
from relay_transfer import HP1, HP2, PARTS, START, SURFACE, TL21, TL22


def read(name, step, *, include_sample=False):
    prefix = f"condenser-{name}-32768-{step:g}"
    row = json.loads((WORK / (prefix+".json")).read_text())
    with np.load(WORK / (prefix+".npz")) as raw:
        tracks = {k:raw[k] for k in ("z_m", "moments", "paths", "path_ids", "path_band", "path_C2_fill")}
        if include_sample:
            for key in ('source_id', 'weights', 'phase', 'loss'):
                tracks[key] = raw[key]
            tracks['energy_offset_eV'] = raw['energy_offset_eV'] if 'energy_offset_eV' in raw else raw['energy_eV']
    return row, tracks


def plot_column(tracks, selected, colors, focus):
    fig = plt.figure(figsize=(13, 7.6), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.12, 1))
    full, relay, sample = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    z_mm = tracks['z_m'] * 1000
    surface = SURFACE * 1000
    for ax, lo, hi in ((full, START * 1000, surface), (relay, 1070., 1425.), (sample, surface-.16, surface)):
        keep = (z_mm >= lo) & (z_mm <= hi)
        abscissa = (z_mm[keep]-surface)*1000 if ax is sample else z_mm[keep]
        for row, color in zip(selected, colors):
            ax.plot(abscissa, tracks['paths'][keep, row, 0]*1e6, color=color, alpha=.80, lw=.83)
        ax.axhline(0, color='#56636a', lw=.5, zorder=0)
        ax.set_xlim(abscissa[0], abscissa[-1])
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(xlabel='Distance to specimen upper surface (µm)' if ax is sample else 'Column z (mm)', ylabel='x (µm)')
    full.set_title('Gun exit to specimen upper surface', fontsize=12, loc='left')
    relay.set_title('Corrector and scan coils', fontsize=12, loc='left')
    sample.set_title('Last 160 µm of illumination', fontsize=12, loc='left')
    for ki, (left, right) in zip(('C2 lens','C3 lens','ADL'),spans_mm()):
        full.axvspan(left, right, color='#dce3e9', alpha=.75, lw=0, zorder=-2)
        full.text((left+right)/2, .97, ki, ha='center', va='top', fontsize=9,
                  transform=full.get_xaxis_transform())
    full.axvline(765., color='#636a62', ls=':', lw=.9)
    full.text(770, .85, 'C2 aperture', fontsize=8, transform=full.get_xaxis_transform())
    for position, label in ((1522.5, 'Mini Condenser'),
                             (START*1000 + PARTS['objective_upper_pole']['local_center_z_mm'], 'Upper objective')):
        full.axvline(position, color='#8a93a1', lw=.7, zorder=-1)
        full.text(position-3, .96 if label == 'Mini Condenser' else .74, label, va='top', ha='right',
                  fontsize=8, transform=full.get_xaxis_transform())
    for position, name in ((HP2*1000,'HP2'), (TL22*1000,'TL22'), (TL21*1000,'TL21'),
                           (HP1*1000,'HP1'), (1400.,'Scan pair')):
        relay.axvline(position, color='#8a93a1', lw=.7, zorder=-1)
        relay.text(position, .96, name, ha='center', va='top', fontsize=8,
                   transform=relay.get_xaxis_transform())
    minimum = focus['D95_minimum_bracket_nm'][1] / 1000
    sample.axvline(minimum, color='#732855', ls='--', lw=1.)
    sample.text(minimum-2, .96, f'D95 minimum\n{minimum:g} µm', color='#732855', ha='right', va='top',
                fontsize=9, transform=sample.get_xaxis_transform())
    sample.axvline(0, color='#202a30', lw=1., clip_on=False)
    sample.text(-2, .96, 'Upper\nsurface', ha='right', va='top', fontsize=9, transform=sample.get_xaxis_transform())
    fig.suptitle('Nanoprobe Normal  |  Original electron paths at the adjusted condenser currents', fontsize=16)
    fig.savefig(HERE / 'condenser-normal-full-column.png', dpi=160)
    plt.close(fig)


def main():
    before, a = read("before", .0375, include_sample=True)
    after, b = read("uniform", .0375, include_sample=True)
    for key in ('path_ids', 'source_id', 'weights', 'energy_offset_eV', 'z_m'):
        np.testing.assert_array_equal(a[key], b[key])
    if before["source_digest"] != after["source_digest"]:
        raise ValueError("The two settings require identical source particles and weights")
    faces = spans_mm()
    records = dict(previous=before, corrected=after)
    for case in ("c2-only", "g2-1548", "g3-024", "uniform", "wider-footprint", "larger-angle"):
        records[case] = read(case,.075)[0]
    focus = json.loads((WORK/"condenser-uniform-axial.json").read_text())
    waist = json.loads((WORK/"condenser-uniform-waist.json").read_text())
    with np.load(FIELD) as field:
        corrector = json.loads(str(field['report']))['currents_A']
    with np.load(HERE / 'evidence/last-point-trajectories.npz') as old:
        np.testing.assert_array_equal(old['source_id'], a['path_ids'])
        path_weight = old['source_weight']
        path_energy = old['energy_offset_eV']
    with np.load(WORK / 'normal-condenser-screen.npz') as sweep:
        gain, result = sweep['gains'], sweep['result']
        weak = (gain[:, 0] > 0) & (gain[:, 1] <= .55)
        valid = weak & (result[:, 0] == 1) & (result[:, 1] == 0) & (result[:, 7] == 0)
        screen = dict(model='Paraxial source correlation, candidate selection only',
                      gain_min=gain[weak].min(axis=0).tolist(), gain_max=gain[weak].max(axis=0).tolist(),
                      cases=int(weak.sum()), correct_gap_counts_and_clear_C3_center_to_ADL_center=int(valid.sum()),
                      maximum_estimated_filled_radius_fraction=float(result[valid, 5].max()))
    report = dict(geometry_mm=dict(zip(("C2_lens","C3_lens","ADL"),faces.tolist())),
                  gain_reference="Previous 2.296 micrometre D95 working point", condenser_gains=[1.542,.26],
                  source_z_mm=450., specimen_z_mm=1599.1999975, C2_aperture_diameter_um=100,
                  field_file=FIELD.name, field_sha256=sha256(FIELD.read_bytes()).hexdigest(),
                  currents_A=after['currents_A'], objective_At=after['objective_At'], source_digest=after['source_digest'],
                  corrector_currents_A=corrector, round_lens_winding_owner=OWNER[ACTIVE].tolist(),
                  corrected_gap_waist=waist, common_gain_screen=screen,
                  D95_waist_bracket_nm=focus["D95_minimum_bracket_nm"], RMS_waist_bracket_nm=focus["RMS_minimum_bracket_nm"],
                  native_runs=records)
    (HERE/"evidence/condenser-normal.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(HERE/"evidence/condenser-normal-tracks.npz",z_m=a["z_m"],source_id=a["path_ids"],
                        source_weight=path_weight, energy_offset_eV=path_energy,
                        all_source_ids=a['source_id'], all_source_weights=a['weights'], all_energy_offset_eV=a['energy_offset_eV'],
                        before_sample_phase=a['phase'], corrected_sample_phase=b['phase'],
                        before_loss=a['loss'], corrected_loss=b['loss'],
                        before_phase=a["paths"],corrected_phase=b["paths"],before_moments=a["moments"],
                        corrected_moments=b["moments"],before_foil_r_over_R=a["path_C2_fill"],
                        corrected_foil_r_over_R=b["path_C2_fill"])
    z=a["z_m"]*1000
    fig,axes=plt.subplots(3,2,figsize=(13,8.5),layout="constrained",gridspec_kw={"height_ratios":[1,1,.8]})
    windows=[(718,860),(967,1090)]
    sections=[((faces[0],"C2 lens"),(faces[1],"C3 lens")),((faces[1],"C3 lens"),(faces[2],"ADL"))]
    fig.suptitle("Nanoprobe Normal  |  Source trajectories between the lens bodies",fontsize=17)
    # The two colors mark the original source x half-planes in every panel.
    # The RMS curves include the entire specimen-surviving cohort per setting.
    selected = np.r_[0:36:3,36:42]
    selected = selected[selected < len(a["path_ids"])]
    az = np.arctan2(a["paths"][0,selected,1],a["paths"][0,selected,0])
    colors=np.where(np.cos(az)>0,"#16736c","#a65327")
    for j,((lo,hi),parts) in enumerate(zip(windows,sections)):
        keep=(z>=lo)&(z<=hi)
        for k,((root,tracks),label) in enumerate(zip(((before,a),(after,b)),("Before","After"))):
            ax=axes[k,j]
            for h,c in zip(selected,colors):
                ax.plot(z[keep],1e6*tracks["paths"][keep,h,0],color=c,alpha=.75,lw=.87)
            ax.axhline(0,color="#45535c",lw=.65)
            ax.set_ylabel(f"{label}  •  x (µm)")
            bracket=root["x_y_waists_mm"][("C2_C3_gap","C3_ADL_gap")[j]][0]
            if bracket:
                mid=np.mean(bracket[0])
                ax.axvline(mid,color="#732855",ls="--",lw=1.1)
                ax.text(mid+1.5,.035,f"{mid:.1f} mm",transform=ax.get_xaxis_transform(),
                        fontsize=9,color="#732855",bbox=dict(facecolor="white",edgecolor="none",pad=.5))
        ax=axes[2,j]
        for tracks,color,label in ((a,"#79727d","Before"),(b,"#125e8b","After")):
            for col,ls in ((1,"-"),(2,"--")):
                ax.plot(z[keep],1e6*tracks["moments"][keep,col],color=color,ls=ls,
                        lw=1.6 if col==1 else 1.1,label=label + (" σx" if col==1 else " σy"))
        ax.set(yscale="log",ylabel="RMS width (µm)",xlabel="Column z (mm)")
        for k,ax in enumerate(axes[:,j]):
            for (left,right),name in parts:
                ax.axvspan(max(left,lo),min(right,hi),color="#dce3e9",alpha=.72,zorder=-2,lw=0)
                if k==0:
                    ax.text((max(left,lo)+min(right,hi))*.5,.96,name,ha="center",va="top",fontsize=9,
                            color="#374858",transform=ax.get_xaxis_transform())
            if j==0:
                ax.axvline(765.,color="#6f706a",ls=":",lw=1)
                if k==0:
                    ax.text(766.2,.96,"C2 aperture",fontsize=8,va="top",transform=ax.get_xaxis_transform())
            ax.set_xlim(lo,hi)
            ax.grid(axis="y",lw=.4,alpha=.25)
            ax.spines[["right","top"]].set_visible(False)
            if k<2:
                ax.tick_params(labelbottom=False)
        axes[0,j].set_title(("C2 lens exit to C3 lens entrance  ·  750–830 mm",
                             "C3 lens exit to ADL entrance  ·  1000–1070 mm")[j],fontsize=12,pad=7)
    axes[2,0].legend(loc="lower left",ncol=2,fontsize=8,framealpha=.92)
    fig.savefig(HERE/"condenser-normal-trajectories.png",dpi=160)
    plt.close(fig)
    plot_column(b, selected, colors, focus)


if __name__=="__main__":
    main()
