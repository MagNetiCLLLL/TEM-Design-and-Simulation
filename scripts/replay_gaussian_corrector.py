"""Replay a prepared physical checkpoint with experimental coherent Gaussian packets.

This CPU diagnostic approximates cubic/quartic actions locally, avoiding a full
Cartesian grid through a strong corrector. It is NOT the application wave solver
or a newly executed source-to-specimen calculation. Inputs are a prepared case
directory with state.json, metadata.json and outcome-all.json. The preparation
must have verified unchanged upstream optics for any reused historical waves.
No fitted source, automatic refocus, probability normalization or mode averaging
is performed. Compare packet width AND axial step independently before using a
phase plot. The reported triangle error bound may be too loose to certify it.

Example:
  python scripts/replay_gaussian_corrector.py --case-directory CASE --output OUT
The NPZ output contains complex Gaussian coefficients, not a standard checkpoint.
"""
import os, sys
if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['NUMBA_DISABLE_CUDA']='1'
    sys.modules['cupy']=None
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,hashlib,psutil
from dataclasses import replace
import numpy as np
from scripts.experimental_gaussian_beamlets import GaussianBeamlets,propagate_affine,apply_taylor_screen,append_physical_column_step
from scripts.experimental_gaussian_beamlets import _inverse2
from temsim.instrument_snapshot import decode_instrument
from temsim.immutable_json import json_digest
from temsim.physics.wave_checkpoint_store import _StoredModes
from temsim.physics.wave_grid import refine_plane_wave,WaveGridNumerics
from temsim.physics.column_wave import _prepare_column,_column_transports,_electric_coefficients
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.optics.electron_gun.tip_coherence import wavelength_m


def decompose(wave,wavelength,*,sigma_cells=.8):
    ny,nx=wave.amplitude.shape
    basis=wave.basis_m
    sigma=sigma_cells*min(np.linalg.svd(basis,compute_uv=False))
    k=2*np.pi/wavelength
    iy,ix=np.meshgrid(np.arange(ny)-ny//2,np.arange(nx)-nx//2,indexing='ij')
    delta=np.stack((ix.ravel(),iy.ravel()),axis=1)@basis.T
    # Periodic Gaussian collocation solves the full sampled envelope, retaining
    # complex coefficients. Actual non-periodic reconstruction is measured below.
    kernel=np.exp(-np.sum(delta**2,axis=1).reshape(ny,nx)/(2*sigma*sigma))
    symbol=np.fft.fft2(np.fft.ifftshift(kernel))
    density=wave.amplitude/np.sqrt(abs(np.linalg.det(basis)))
    b=np.fft.ifft2(np.fft.fft2(density)/symbol).ravel()
    curvature=np.zeros((2,2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt=np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase=k*(.5*np.einsum('bi,ij,bj->b',delta,curvature,delta)+delta@tilt)
    precision=np.broadcast_to(curvature+1j*np.eye(2)/(k*sigma*sigma),(nx*ny,2,2))
    normalizer=1/(np.sqrt(np.pi)*sigma)
    packets=GaussianBeamlets(wave.origin_m+delta,delta@curvature.T+tilt,precision,
        b/normalizer*np.exp(1j*phase),wavelength)
    return packets,dict(method='Gaussian envelope collocation with complex deconvolution; full quadratic carrier preserved analytically',
        sigma_amplitude_m=sigma,sigma_intensity_m=sigma/np.sqrt(2),shape=list(wave.amplitude.shape),
        spectral_symbol_min=float(abs(symbol).min()),coherent_coefficient_l1=float(abs(packets.coefficients).sum()))


def validate_prepared_input(manifest, prepared, state_graph, metadata, mode_index, pixels, budget):
    """Bind the actual stored field to its case before allocating mode arrays."""
    identity = json_digest({key:value for key,value in manifest.items() if key!='manifest_digest'})
    storage = prepared['storage']
    if (identity != manifest.get('manifest_digest') or identity != storage.get('manifest_digest')
            or manifest.get('key') != storage.get('key')
            or manifest.get('dependency') != storage.get('dependency')):
        raise ValueError('Executed checkpoint identity does not match the prepared outcome')
    record = manifest['record']
    if (record.get('case') != prepared['case'] or record.get('case') != metadata['case']
            or json_digest(record.get('state')) != json_digest(state_graph)
            or json_digest(state_graph) != metadata['state_sha256']):
        raise ValueError('Executed checkpoint belongs to a different instrument or comparison case')
    if not 0 <= mode_index < len(manifest['modes']):
        raise ValueError('Requested energy/spatial mode is absent from the executed checkpoint')
    declared = manifest['modes'][mode_index]['arrays']['amplitude']['shape']
    if len(declared)!=2 or any(type(size) is not int or size<=0 for size in declared):
        raise ValueError('Invalid declared checkpoint grid')
    shape = tuple(max(pixels,size) for size in declared)
    if max(shape)>1024 or 2048*int(np.prod(shape))>budget:
        raise MemoryError('Packet propagation exceeds the diagnostic working-memory budget')
    return shape


def largest_symmetric_eigenvalue2(values):
    """Closed-form 2x2 Hermitian eigenvalue, with eigvalsh's lower triangle."""
    half_trace=(values[...,0,0]+values[...,1,1])/2
    return half_trace+np.hypot((values[...,0,0]-values[...,1,1])/2,values[...,1,0])


def preflight_stored_amplitude(reader, mode_record):
    """Check the verified NPY header before PlaneWave owns an immutable copy.

    The manifest admission bounds its declared grid. A separately generated but
    inconsistent manifest must not allow a larger actual payload to be copied
    before that mismatch is detected. The store's checksum/path validation is
    retained, and its read-only mmap does not materialize the amplitude.
    """
    item = mode_record['arrays']['amplitude']
    amplitude = reader._array(item)
    try:
        if amplitude.shape != tuple(item['shape']):
            raise ValueError('Stored amplitude shape differs from the admitted checkpoint metadata')
    finally:
        del amplitude


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode-index',type=int,default=0)
    parser.add_argument('--pixels',type=int,default=360)
    parser.add_argument('--maximum-step-mm',type=float,default=.5)
    parser.add_argument('--axial-order',type=int,choices=(2,4),default=4)
    parser.add_argument('--quadratic-step-mm',type=float,default=.0625,
                        help='Independent fine small-matrix integration step; 0 keeps midpoint-frozen quadratic stages')
    parser.add_argument('--stop-z-mm',type=float)
    parser.add_argument('--maximum-working-mib',type=int,default=1024)
    args=parser.parse_args()
    case_root=args.case_directory
    prepared=json.loads((case_root/'outcome-all.json').read_text(encoding='utf-8'))
    metadata=json.loads((case_root/'metadata.json').read_text(encoding='utf-8'))
    state_graph=json.loads((case_root/'state.json').read_text(encoding='utf-8'))
    if prepared.get('status')!='computed' or json_digest(state_graph)!=metadata['state_sha256']:
        raise ValueError('Prepared input failed or instrument snapshot does not match its receipt')
    if metadata.get('upstream450_to_c2_changes') or metadata.get('auto_refocus'):
        raise ValueError('Prepared source replay must verify unchanged upstream optics and no refocus')
    case=prepared['case']
    if metadata['case']!=case:
        raise ValueError('Prepared case identity differs from instrument metadata')
    root=Path(prepared['storage']['directory'])
    out=args.output
    mode_index=args.mode_index
    n=args.pixels
    if (not 32<=n<=1024 or not 0<args.maximum_step_mm<=1 or args.maximum_working_mib<=0
            or not np.isfinite(args.quadratic_step_mm) or args.quadratic_step_mm<0):
        raise ValueError('Use 32..1024 packet grid points, a positive step <=1 mm and a positive memory budget')
    budget=args.maximum_working_mib*1024**2
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    declared_shape=validate_prepared_input(manifest,prepared,state_graph,metadata,mode_index,n,budget)
    reader=_StoredModes(root,(manifest['modes'][mode_index],),manifest['reference_plane'])
    preflight_stored_amplitude(reader,manifest['modes'][mode_index])
    state=decode_instrument(state_graph)
    mode=reader[0]
    stop=state.sample.upper_surface_z_mm if args.stop_z_mm is None else args.stop_z_mm
    if not manifest['plane_z_mm']<stop<=state.sample.upper_surface_z_mm:
        raise ValueError('This diagnostic propagates forward only, no farther than the specimen entrance')
    shape=tuple(max(n,s) for s in mode.plane.amplitude.shape)
    if shape!=declared_shape:
        raise ValueError('Loaded field shape differs from the admitted checkpoint metadata')
    # Includes copied immutable packet state, moment recurrence and 2x readout.
    # This is a conservative allocation admission, not an OS RSS guarantee.
    if max(shape)>1024 or 2048*int(np.prod(shape))>budget:
        raise MemoryError('Packet propagation exceeds the diagnostic working-memory budget')
    out.mkdir(parents=True,exist_ok=True)
    if any((out/f'mode{mode_index}-n{n}-{suffix}').exists() for suffix in ('packets.npz','receipt.json')):
        raise FileExistsError('Output already exists; use a new directory to preserve previous evidence')
    wave=refine_plane_wave(mode.plane,shape,numerics=WaveGridNumerics(maximum_pixels=2048,maximum_working_bytes=budget))
    wavelength=float(wavelength_m(mode.energy_kev*1000));started=time.perf_counter()
    def emit(kind,**data): print(json.dumps(dict(kind=kind,**{'elapsed_s':time.perf_counter()-started,**data})),flush=True)
    packets,receipt=decompose(wave,wavelength)
    receipt.update(case=case,state_file=str(case_root/'state.json'),
        state_file_sha256=hashlib.sha256((case_root/'state.json').read_bytes()).hexdigest(),
        lens_percent={l.key:l.percent for l in state.lenses if l.enabled},objective_control='common upper and lower objective excitation',
        scope='EXPERIMENTAL semiclassical Gaussian replay; actual historical executed input, not a newly executed tip-to-specimen calculation',
        input_manifest=str(root/'manifest.json'),input_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        mode_id=mode.mode_id,energy_kev=mode.energy_kev,weight=mode.weight_per_reference_electron,
        start_z_mm=manifest['plane_z_mm'],stop_z_mm=stop,packet_count=packets.count,
        implementation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('experimental_gaussian_beamlets.py'))})
    source_root=Path(__file__).resolve().parents[1]/'src'/'temsim'
    source_hashes={str(p.relative_to(source_root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(source_root.rglob('*.py'))}
    receipt['current_production_source_sha256']=json_digest(source_hashes)
    receipt['implementation_sha256'][str(Path(__file__).with_name('gaussian_column_stages.py'))]=hashlib.sha256(
        Path(__file__).with_name('gaussian_column_stages.py').read_bytes()).hexdigest()
    receipt['implementation_sha256'][str(Path(__file__).with_name('gaussian_quadratic_paths.py'))]=hashlib.sha256(
        Path(__file__).with_name('gaussian_quadratic_paths.py').read_bytes()).hexdigest()
    # Twice the node density measures off-node and finite-window reconstruction,
    # rather than checking only the collocation points used to obtain weights.
    fine=refine_plane_wave(wave,tuple(2*s for s in shape),numerics=WaveGridNumerics(maximum_pixels=4096,maximum_working_bytes=budget))
    reconstructed,tail=packets.sample_plane(fine.amplitude.shape,fine.basis_m,fine.origin_m,return_record=True,method="recurrence",maximum_working_bytes=budget)
    delta=fine.coordinates_m()-fine.origin_m[:,None,None]
    curvature=np.zeros((2,2)) if fine.curvature_m1 is None else fine.curvature_m1
    tilt=np.zeros(2) if fine.tilt_rad is None else fine.tilt_rad
    carrier=(2*np.pi/wavelength)*(.5*np.einsum('iyx,ij,jyx->yx',delta,curvature,delta)+np.einsum('i,iyx->yx',tilt,delta))
    reconstructed*=np.exp(-1j*carrier)
    expected=fine.amplitude
    receipt['initial_reconstruction']=dict(relative_complex_l2=float(np.linalg.norm(reconstructed-expected)/np.linalg.norm(expected)),
        probability_ratio=float(np.vdot(reconstructed,reconstructed).real/fine.probability),evaluation=tail)
    del fine,reconstructed,expected,delta,carrier
    emit('decomposed',**receipt)
    step_mm=args.maximum_step_mm
    receipt['maximum_column_step_mm']=step_mm
    receipt['working_memory_admission_bytes']=budget
    receipt['source_preparation_metadata_sha256']=hashlib.sha256((case_root/'metadata.json').read_bytes()).hexdigest()
    plan,radii,stops,owners=_prepare_column(state,manifest['plane_z_mm'],stop,step_mm)
    receipt['physical_plan_signature']=plan.signature
    receipt['bore_bound_scope']='Gaussian-tail omission at original physical axial nodes; not a continuous between-node interception bound'
    if stops or getattr(plan,'mapped_fields',()) or getattr(plan,'posed_spherical_kicks',()) or getattr(plan,'intrinsic_aberration_kicks',()):
        raise ValueError('Experimental beamlet path does not implement physical hard apertures/posed fields')
    if any(o.get('dynamic',False) for o in owners): raise ValueError('Dynamic hardware unsupported')
    physical_radii=radii
    stage_count=1
    if args.axial_order==4:
        from scripts.gaussian_column_stages import fourth_order_column_stages
        plan,electric=fourth_order_column_stages(state,plan)
        stage_count=3
    else:
        electric=_electric_coefficients(plan)
    receipt['axial_order']=args.axial_order
    receipt['axial_method']='Yoshida signed midpoint Strang with fresh physical fields' if stage_count==3 else 'midpoint Strang'
    if args.quadratic_step_mm:
        from scripts.gaussian_quadratic_paths import integrated_quadratic_stage_paths
        paths,quadrature_record=integrated_quadratic_stage_paths(state,plan,mode.energy_kev,
            maximum_step_mm=args.quadratic_step_mm,maximum_working_bytes=budget,retained_bytes=packets.bytes)
        receipt['quadratic_path_integration']=quadrature_record
    else:
        paths=tuple(_column_transports(plan,mode.energy_kev,electric=electric))
    p0,_=_momentum_velocity(mode.energy_kev*1000)
    scale=float(getattr(plan,'reference_momentum_kg_m_s',p0))/p0
    bound=0.;events=0;max_sigma=0.;min_wall_clearance=float('inf');bore_l2_bound=0.
    cached_stats=[None,None]
    combined=CanonicalPath(1e-3)
    def flush(value):
        nonlocal combined
        value=propagate_affine(value,combined.matrix,combined.offset,**combined.phase_kwargs())
        combined=CanonicalPath(1e-3)
        return value
    def containment(value,matrix,offset,radius):
        nonlocal min_wall_clearance,bore_l2_bound,max_sigma
        if cached_stats[0] is not value:
            covq=_inverse2(value.precision_m1.imag)/(2*(2*np.pi/wavelength))
            covp=value.precision_m1.real@covq@value.precision_m1.real+value.precision_m1.imag/(2*(2*np.pi/wavelength))
            cached_stats[:]=[value,(np.sqrt(largest_symmetric_eigenvalue2(covq).max()),np.sqrt(largest_symmetric_eigenvalue2(covp).max()))]
        sq,sp=cached_stats[1]
        a,b=matrix[:2,:2],matrix[:2,2:]
        mapped=value.centres_m@a.T+value.momenta@b.T+offset[:2]
        sigma_bound=np.linalg.norm(a,2)*sq+np.linalg.norm(b,2)*sp
        max_sigma=max(max_sigma,float(sigma_bound))
        clearance=float(radius*1e-3-np.linalg.norm(mapped,axis=1).max()-12*sigma_bound)
        min_wall_clearance=min(min_wall_clearance,clearance)
        if clearance<0: raise ValueError('Experimental propagation cannot establish 12-sigma bore containment')
        bore_l2_bound+=float(abs(value.coefficients).sum()*np.exp(-36))
    containment(packets,np.eye(4),np.zeros(4),radii[0])
    for i,path in enumerate(paths):
        hn=plan.midpoint_hex_normal_m3[i]*plan.step_m[i]*scale/2
        hs=plan.midpoint_hex_skew_m3[i]*plan.step_m[i]*scale/2
        if hn or hs:
            packets=flush(packets)
            packets,evidence=apply_taylor_screen(packets,normal_m2=hn,skew_m2=hs)
            bound+=evidence['coherent_absolute_l2_error_upper_bound']; events+=1
        # Follow the actual generator through all intermediate caustics.
        append_physical_column_step(combined,path)
        j=i+1
        if hn or hs or plan.cs_kick_m3[j] or plan.kick_x_rad[j] or plan.kick_y_rad[j] or j==len(paths):
            packets=flush(packets)
            if hn or hs:
                packets,evidence=apply_taylor_screen(packets,normal_m2=hn,skew_m2=hs)
                bound+=evidence['coherent_absolute_l2_error_upper_bound'];events+=1
            if plan.kick_x_rad[j] or plan.kick_y_rad[j]:
                p_local,_=_momentum_velocity(mode.energy_kev*1000+electric[0][j]-electric[0][0])
                packets=propagate_affine(packets,np.eye(4),np.r_[0.,0.,plan.kick_x_rad[j]*p_local/p0,plan.kick_y_rad[j]*p_local/p0])
            if plan.cs_kick_m3[j]:
                p_local,_=_momentum_velocity(mode.energy_kev*1000+electric[0][j]-electric[0][0])
                packets,evidence=apply_taylor_screen(packets,spherical_m3=plan.cs_kick_m3[j]*p_local/p0)
                bound+=evidence['coherent_absolute_l2_error_upper_bound'];events+=1
        if j%stage_count==0:
            containment(packets,combined.matrix,combined.offset,physical_radii[j//stage_count])
        if i%64==0 or j==len(paths):
            emit('propagating',z_mm=float(plan.z_mm[j]),events=events,triangle_l2_bound=bound,maximum_packet_sigma_m=max_sigma,
                minimum_twelve_sigma_wall_clearance_m=min_wall_clearance)
    receipt.update(peak_process_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),triangle_l2_bound=bound,bore_omission_l2_bound=bore_l2_bound,screen_events=events,maximum_packet_sigma_m=max_sigma,
        minimum_twelve_sigma_wall_clearance_m=min_wall_clearance,elapsed_s=time.perf_counter()-started)
    np.savez(out/f'mode{mode_index}-n{n}-packets.npz',centres_m=packets.centres_m,momenta=packets.momenta,
        precision_m1=packets.precision_m1,coefficients=packets.coefficients,wavelength_m=wavelength)
    (out/f'mode{mode_index}-n{n}-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    emit('finished',**receipt)
if __name__=='__main__':main()
