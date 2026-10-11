"""Bounded, reproducible image-corrector matrix and CUDA optical audit.

This is an optical diagnostic population, not a physical Tip emission run.
No lens, source or assembly defaults are fitted or written by this script.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("C3 + Image Corrector", "C3 + Probe Corrector + Image Corrector")


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2), encoding="utf-8")


def inventory():
    paths = sorted((ROOT / "src/temsim").rglob("*.py"))
    paths += sorted((ROOT / "configs/instruments").rglob("*.toml"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def matrix_metrics(matrix):
    matrix = np.asarray(matrix, float)
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    u, singular, vt = np.linalg.svd(a)
    rotation = u @ vt
    # Equal position/angle scales at both ends; removes arbitrary SI conditioning.
    scales = np.diag([1e-6, 1e-6, 1e-3, 1e-3])
    m = np.linalg.solve(scales, matrix @ scales)
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    return dict(matrix=matrix, A=a, B_m_per_rad=b, C_rad_per_m=c, D=d,
                A_singular_values=singular, A_rotation_deg=np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])),
                B_operator_norm_m_per_rad=np.linalg.norm(b, 2),
                blur_at_1_mrad_m=np.linalg.norm(b, 2)*1e-3,
                A_minus_negative_identity_norm=np.linalg.norm(a+np.eye(2), 2),
                scaled_symplectic_residual=np.linalg.norm(m.T@j@m-j)/np.linalg.norm(j))


def paired_matrices(state, source, targets):
    from temsim.optics.direct_alignment import canonical_transfers, canonical_source_basis
    transfers = canonical_transfers(state, source, targets)
    source_basis = canonical_source_basis(state, source)
    result = {}
    for z, transfer in transfers.items():
        target_basis = canonical_source_basis(state, z)
        cm = transfer.matrix
        result[str(z)] = dict(source_z_mm=source, target_z_mm=z,
            canonical_to_mechanical=matrix_metrics(cm),
            canonical_to_canonical=matrix_metrics(np.linalg.solve(target_basis, cm)),
            mechanical_to_mechanical=matrix_metrics(cm @ np.linalg.inv(source_basis)),
            source_canonical_to_mechanical_basis=source_basis,
            target_canonical_to_mechanical_basis=target_basis,
            affine_position_m=transfer.position_offset_m, affine_angle_rad=transfer.angle_offset_rad)
    return result


def optical_rings():
    theta = np.arange(64)*2*np.pi/64
    radii = np.repeat([1e-3, 5e-3, 1e-2], len(theta))
    angles = np.tile(theta, 3)
    return np.vstack((np.zeros(len(radii)+1), np.zeros(len(radii)+1),
                      np.r_[0., radii*np.cos(angles)], np.r_[0., radii*np.sin(angles)]))


def trace_rings(state, source, targets, initial, *, spherical, hexapoles, step, history_sink=None):
    from temsim.physics.core import propagate
    from temsim.physics.instrument_magnetic import active_column_events
    previous_step=state.step_mm
    state.step_mm=step
    try:
        result = propagate(state, source, max(targets), initial[0].copy(), initial[2].copy(),
            initial[1].copy(), initial[3].copy(), events=active_column_events(state),
            include_spherical_aberration=spherical, include_hexapole=hexapoles,
            maximum_step_mm=step, save_z_mm=targets, checkpoint_z_mm=targets, return_checkpoints=True)
    finally:
        state.step_mm=previous_step
    if history_sink is not None:
        history_sink['z_mm']=result[0]
        history_sink['phase']=np.asarray((result[1],result[3],result[2],result[4]))
    points = result[-1]
    return {str(float(z)): np.array((points.x_m[i], points.y_m[i], points.tx_rad[i], points.ty_rad[i]))
            for i, z in enumerate(points.z_mm)}


def ring_metrics(value, linear):
    answer = []
    for index, angle in enumerate((1e-3, 5e-3, 1e-2)):
        selection = slice(1+64*index, 1+64*(index+1))
        xy = value[:2, selection]-value[:2, :1]
        ref = linear[:2, selection]-linear[:2, :1]
        cov = (xy-xy.mean(axis=1, keepdims=True)) @ (xy-xy.mean(axis=1, keepdims=True)).T/64
        answer.append(dict(sample_launch_semiangle_rad=angle,
            centroid_m=xy.mean(axis=1), radial_rms_m=np.sqrt(np.mean(np.sum(xy*xy, axis=0))),
            principal_rms_m=np.sqrt(np.maximum(0, np.linalg.eigvalsh(cov))),
            nonlinear_displacement_rms_m=np.sqrt(np.mean(np.sum((xy-ref)**2, axis=0)))))
    return answer


def render_paths(output,report):
    """Plot already executed paths; never propagate for a screenshot."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    data=np.load(output/'image-rays.npz')
    variant=report['variants'][VARIANTS[0]]
    planes=variant['planes_mm'];start=planes['sample']
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    for ax,kind,title in zip(axes,('linear','cs_and_hp_fine'),('Linear field observer','Finite Cs + HP rays')):
        z=data['history_'+kind+'_z_mm']
        phase=data['history_'+kind+'_phase']
        for index,name,colour in ((1,'launch +X','#de4c3d'),(17,'launch +Y','#31945d'),(33,'launch -X','#b45b8b'),(49,'launch -Y','#3b72bd')):
            ax.plot(z-start,(phase[0,:,index]-phase[0,:,0])*1e6,color=colour,lw=1.1,label=name+' / X')
            ax.plot(z-start,(phase[1,:,index]-phase[1,:,0])*1e6,color=colour,lw=.8,ls=':',alpha=.7)
        for component in variant['image_components']:
            key=component['key']
            if key in ('image_tl11_lens','image_tl12_lens','image_tl21_lens','image_tl22_lens','image_adapter_lens','image_hp1_hexapole','image_hp2_hexapole'):
                centre=component['z_mm']-start;length=component['mechanical_length_mm'] or 0
                ax.axvspan(centre-length/2,centre+length/2,color='#9d91aa' if 'hexapole' in key else '#8a9dae',alpha=.12)
                label=key.removeprefix('image_').replace('_lens','').replace('_hexapole','').upper()
                ax.text(centre,1.015,label,rotation=90,ha='center',va='bottom',fontsize=8,transform=ax.get_xaxis_transform())
        for key,label in (('first_middle','I1'),('middle','I2'),('sad','SAD')):
            ax.axvline(planes[key]-start,color='#83479b',ls='--',lw=.9)
            ax.text(planes[key]-start,.985,label,va='top',ha='right',fontsize=8,transform=ax.get_xaxis_transform())
        ax.set(ylabel='Lab X / Y minus axis ray (um)');ax.grid(alpha=.2)
        ax.text(.015,.95,title,transform=ax.transAxes,va='top',ha='left',fontsize=11,
                bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    axes[0].legend(ncol=4,fontsize=8,loc='lower center')
    axes[-1].set_xlabel('Distance after specimen (mm); dotted curves are lab Y, solid curves lab X')
    fig.suptitle('Executed CUDA image-corrector paths: 1 mrad sample-point diagnostic\nComputed I1 / I2 / SAD positions from full 4x4 maps; lab projection crossing alone is not conjugacy',fontsize=12)
    fig.subplots_adjust(left=.08,right=.995,bottom=.085,top=.80,hspace=.33)
    fig.savefig(output/'internal-ray-paths.png',dpi=170,bbox_inches='tight');plt.close(fig)


def execute(args):
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.compute_backend import choose_ray_backend
    from temsim.physics.backend_execution import backend_receipts
    args.output.mkdir(parents=True, exist_ok=True)
    before = inventory(); save(args.output/"implementation-before.json", before)
    report = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Current saved assembly defaults, fixed 300 keV synthetic optical diagnostic; not physical Tip emission, full wave, material clipping or aberration qualification",
        coordinates="(x,y,theta_x,theta_y), SI metres/radians; electron +Z; full lab-frame 4x4. Canonical momentum is divided by the common reference momentum.",
        protocol=dict(matrix_step_mm=.025, ring_step_mm=.05, refined_ring_step_mm=.025,
            finite_population="193 rays: axis plus 64 equal-angle rays at each 1,5,10 mrad from the specimen centre; no source fitting",
            expected_relations="Classical symmetric two-hexapole relay: HP1/HP2 midplanes conjugate, unit magnification and negative identity in one lab frame. Specimen image near TL21/TL22 symmetry plane. These are diagnostics, not an automatic correction certificate.",
            refinement_gate_relative=1e-3, refinement_gate_absolute_m=1e-10), variants={})
    report['first_order_acceptance']=dict(maximum_B_m_per_rad=1e-7,maximum_unit_magnification_error=1e-5,
        scope='Position conjugacy only at 300 kV; no full negative-identity, finite-HP cancellation or objective-Cs qualification')
    save(args.output/"report.json", report)
    start = perf_counter()
    candidate=None
    if args.candidate_json:
        candidate=json.loads(args.candidate_json.read_text(encoding='utf-8'))['candidates'][0]['percent']
        report['candidate_override']=dict(percent=candidate,path=str(args.candidate_json.resolve()),
            sha256=hashlib.sha256(args.candidate_json.read_bytes()).hexdigest(),scope='Detached validation state only; no defaults or source changes')
    selected, reason = choose_ray_backend("Require GPU", acceleration_enabled=True, ray_count=193)
    if selected != "CUDA GPU" or reason: raise RuntimeError(f"CUDA required: {selected}: {reason}")
    catalog = AssemblyCatalog()
    for variant in VARIANTS:
        print(variant, flush=True)
        state = default_state(); catalog.apply(state, AssemblySelection("FEG", variant, "No Energy Filter"))
        if candidate:
            lenses={lens.key:lens for lens in state.lenses}
            for key,value in candidate.items():lenses[key].percent=float(value)
        state.acceleration_enabled = True; state.acceleration_backend = "Require GPU"; state.step_mm = .025
        system = state.image_corrector_system
        parts = {p.key:p for p in system.components}
        planes = dict(sample=float(state.sample.z_mm), objective_image_reference=float(state.objective_image_plane_z_mm),
            hp1=float(system.hp1_hexapole.z_mm), hp2=float(system.hp2_hexapole.z_mm),
            first_middle=.5*(parts["image_tl11_lens"].z_mm+parts["image_tl12_lens"].z_mm),
            middle=.5*(parts["image_tl21_lens"].z_mm+parts["image_tl22_lens"].z_mm),
            sad=float(system.sad_plane.z_mm), camera=float(next(p.z_mm for p in state.recording_planes if p.key=="camera")))
        slug = "double" if "Probe" in variant else "image"
        save(args.output/f"{slug}-inputs.json", capture_instrument_snapshot(state).to_dict())
        current = dict(planes_mm=planes, simulation_mode=state.simulation_mode, beam_voltage_kv=state.beam_voltage_kv,
            image_components=[dict(key=p.key, z_mm=p.z_mm, enabled=getattr(p,"enabled",None),
                mechanical_length_mm=getattr(p,"mechanical_length_mm",None),
                percent=getattr(p,"percent",None), polarity=getattr(p,"polarity",None),
                strength_m3=getattr(p,"strength_m3",None), orientation_rad=getattr(p,"orientation_rad",None)) for p in system.components],
            matrices={}, backend_receipts=[], finite_rings={})
        report["variants"][variant] = current
        with backend_receipts(current["backend_receipts"]):
            for source_name, target_names in (("sample",("first_middle","hp1","middle","hp2","sad","camera")),
                    ("hp1",("hp2","sad")),("hp2",("sad",)),
                    ("objective_image_reference",("hp1","middle","sad"))):
                matrices = paired_matrices(state, planes[source_name], [planes[name] for name in target_names])
                for target in target_names:
                    current["matrices"][source_name+"_to_"+target] = matrices[str(planes[target])]
            save(args.output/"report.json", report)
            if args.rays:
                rays = optical_rings(); targets = [planes[x] for x in ("hp1","middle","hp2","sad")]
                fields = {}
                histories={}
                for name, cs, hp, step in (("linear",False,False,.05),("cs_only",True,False,.05),
                        ("cs_and_hp",True,True,.05),("cs_and_hp_fine",True,True,.025)):
                    history={}
                    fields[name] = trace_rings(state,planes["sample"],targets,rays,spherical=cs,hexapoles=hp,step=step,history_sink=history)
                    histories.update({'history_'+name+'_'+key:value for key,value in history.items()})
                np.savez_compressed(args.output/f"{slug}-rays.npz", initial=rays,
                    **{name+"__"+z.replace('.','_'):value for name, rows in fields.items() for z,value in rows.items()},**histories)
                for z in targets:
                    current["finite_rings"][str(z)] = {name:ring_metrics(rows[str(z)],fields["linear"][str(z)]) for name, rows in fields.items()}
                coarse=fields["cs_and_hp"][str(planes["sad"])][:2]
                fine=fields["cs_and_hp_fine"][str(planes["sad"])][:2]
                error=float(np.sqrt(np.mean(np.sum((coarse-fine)**2,axis=0))))
                scale=float(np.sqrt(np.mean(np.sum(fine*fine,axis=0))))
                current["finite_step_check"] = dict(position_rms_difference_m=error, fine_position_rms_m=scale,
                    passed=error<=max(1e-10,1e-3*scale), scope="All 193 unclipped optical rays at SAD; step .05 versus .025 mm")
        receipts=[r for r in current["backend_receipts"] if r.get("stage")=="column" and r.get("outcome")=="returned"]
        current["cuda_verified"] = bool(receipts) and all(r.get("actual")=="CUDA GPU" for r in receipts)
        current['first_order_checks']={name:current['matrices'][name]['canonical_to_canonical']['B_operator_norm_m_per_rad']<=1e-7
            for name in ('sample_to_first_middle','sample_to_middle','sample_to_sad','hp1_to_hp2')}
        current['first_order_checks']['hp_unit_position_magnification']=bool(np.max(np.abs(np.asarray(
            current['matrices']['hp1_to_hp2']['canonical_to_canonical']['A_singular_values'])-1.))<=1e-5)
        save(args.output/"report.json", report)
    after=inventory(); save(args.output/"implementation-after.json",after)
    report["implementation_and_saved_inputs_unchanged"] = before==after
    report["elapsed_s"] = perf_counter()-start
    report["status"] = "AUDIT_COMPLETE" if before==after else "FAILED_IMPLEMENTATION_CHANGED_DURING_AUDIT"
    report['first_order_passed']=all(all(v['first_order_checks'].values()) and v['cuda_verified'] for v in report['variants'].values())
    report['nonlinear_qualification']='NOT_ESTABLISHED: HP relay is not full -I; native objective thin spatial Cs kick cannot qualify outgoing specimen-angle aberration. Finite HP pair residual remains separate evidence.'
    save(args.output/"report.json", report)
    if args.rays:render_paths(args.output,report)
    print(report["status"],report["elapsed_s"],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/image_corrector_20261011/baseline")
    parser.add_argument("--rays",action="store_true",help="Also execute bounded CUDA finite-ring diagnostics")
    parser.add_argument("--candidate-json",type=Path,help="Explicit optional candidate-report percent override on detached validation states")
    execute(parser.parse_args())
