"""Fixed-envelope ideal-current coil fields versus the default Gaussian fields.

No magnetic material/FEM solve, source emission, particle trace or wave run.
All trial currents are detached field calculations, never saved working points.
"""
from copy import copy,deepcopy
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq,minimize_scalar

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider,_analytic_gaussian_terms
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode


ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/ideal-coil-fields-20261115'


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def feature(function,grid):
    grid=np.unique(grid);values=np.abs(function(grid))
    indices=np.flatnonzero((values[1:-1]>=values[:-2])&(values[1:-1]>=values[2:]))+1
    maxima=[]
    for i in indices:
        if values[i]<values.max()*1e-6:continue
        origin=grid[i]
        fit=minimize_scalar(lambda x:-abs(float(function(x+origin))),
            bounds=(grid[i-1]-origin,grid[i+1]-origin),method='bounded',options={'xatol':1e-9})
        maxima.append(dict(z_mm=float(origin+fit.x),magnitude_t=-float(fit.fun)))
    peak=max(maxima,key=lambda row:row['magnitude_t']);half=peak['magnitude_t']/2
    mask=values>=half;edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
    regions=[]
    for a,b in zip(edges[::2],edges[1::2]):
        if a==0 or b==len(grid):raise ValueError('FWHM extends outside feature grid')
        left=brentq(lambda z:abs(float(function(z)))-half,grid[a-1],grid[a],xtol=1e-9)
        right=brentq(lambda z:abs(float(function(z)))-half,grid[b-1],grid[b],xtol=1e-9)
        regions.append(dict(start_mm=left,end_mm=right,width_mm=right-left))
    containing=next(row for row in regions if row['start_mm']<=peak['z_mm']<=row['end_mm'])
    return dict(peak_magnitude_t=peak['magnitude_t'],peak_z_mm=peak['z_mm'],
        signed_peak_t=float(function(peak['z_mm'])),peak_search_candidates=maxima,
        half_max_regions=regions,half_max_region_count=len(regions),
        main_lobe_fwhm_mm=containing['width_mm'],
        half_max_bounding_span_mm=regions[-1]['end_mm']-regions[0]['start_mm'])


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/protected-files.json').read_text())
    assert all(sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in protected.items())
    inventory=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/inventory.json').read_text())
    catalog=AssemblyCatalog();selection=catalog.default_selection();state=default_state()
    catalog.apply(state,selection);switch_mode(state,'unified')
    apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=selection.column,recording_name=selection.recording)
    from temsim.column.state_layout import apply_physical_layout_to_state
    apply_physical_layout_to_state(state);state.step_mm=.05
    baseline=state.to_dict()
    assert json.loads(json.dumps(baseline))==json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/standard-startup-state.json').read_text())
    dump(OUT/'default-state.json',baseline)
    ideal=copy(state);ideal.lenses=deepcopy(state.lenses);ideal.round_lens_field_model='copper_coil'
    ideal._runtime_lens_field_provider_cache={};ideal._runtime_copper_coil_provider_cache={}
    records=[];arrays={}
    for row in inventory:
        key=row['owner'];lens=next(lens for lens in state.lenses if lens.key==key)
        native=state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens
        old=resolve_runtime_lens_field_provider(state,key,native)
        terms=_analytic_gaussian_terms(native,old.gaussian_width_scale)
        potential=capture_round_lens_potential(ideal,key).potential
        assert all(c.quadrature_order==12 for c in potential.coils)
        assert all(c.registration.origin_global_m==(0.,0.,0.) and
                   c.registration.rotation_local_to_global==((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)) for c in potential.coils)
        gaussian=lambda z:old.magnetic_field_t(z)
        coil=lambda z:sum(c.axis_bz_t(np.asarray(z)*.001) for c in potential.coils)
        lo=min((c.center_z_m-c.length_m/2)*1000 for c in potential.coils)
        hi=max((c.center_z_m+c.length_m/2)*1000 for c in potential.coils)
        scale=max(hi-lo,max(c.outer_radius_m*1000 for c in potential.coils))
        grid=np.unique(np.r_[np.linspace(lo-10*scale,hi+10*scale,20001),
            np.concatenate([np.linspace((c-10*s)*1000,(c+10*s)*1000,2001) for _,c,s in terms])])
        gf=feature(gaussian,grid);cf=feature(coil,grid)
        old_values=gaussian(grid);coil_values=coil(grid)
        # Verify the axial shortcut against the unchanged public vector sampler.
        check=grid[::max(1,len(grid)//129)]
        xyz=np.zeros((len(check),3));xyz[:,2]=check*.001
        np.testing.assert_allclose(coil(check),potential.sample(xyz)['B_t'][:,2],rtol=4e-14,atol=1e-20)
        total_ni=math.fsum(c.ampere_turns for c in potential.coils)
        peak_scale=gf['peak_magnitude_t']/cf['peak_magnitude_t']
        required_ni=abs(total_ni)*peak_scale
        # Integral comparison is a separate metric, not a focus qualification.
        old_integral=math.fsum(a*b*math.sqrt(2*math.pi)*s*t/math.sqrt(s*s+t*t)*
            math.exp(-((c-d)**2)/(2*(s*s+t*t))) for a,c,s in terms for b,d,t in terms)
        middle=(lo+hi)/2
        integral,error=quad(lambda x:float(coil(middle+scale*x))**2*scale*.001,
            -np.inf,np.inf,epsabs=1e-18,epsrel=2e-10,limit=300)
        integral_scale=math.sqrt(old_integral/integral)
        current_limit=3. if key=='objective_lens' else 1.
        n=math.ceil(required_ni/current_limit);actual=required_ni/n
        percent=row['excitation_percent']*.01
        full_n=math.ceil(required_ni/(current_limit*percent));full_i100=required_ni/(full_n*percent)
        existing_area=(row['outer_diameter_mm']-row['inner_diameter_mm'])/2*row['occupied_axial_length_mm']
        record=dict(key=key,lens=row['label'],excitation_percent=row['excitation_percent'],polarity=row['polarity'],
            gaussian=gf,ideal_at_original_reference_ni=cf,original_signed_ni_a_turn=total_ni,
            peak_matched_required_ni_magnitude_a_turn=required_ni,
            peak_matched_scale=peak_scale,peak_matched_field_at_old_peak_t=float(coil(gf['peak_z_mm']))*peak_scale,
            gaussian_integral_b2_t2_m=old_integral,ideal_reference_integral_b2_t2_m=integral,
            integral_quadrature_error_estimate=error,
            peak_matched_integral_to_gaussian_ratio=integral*peak_scale**2/old_integral,
            integral_matched_required_ni_magnitude_a_turn=abs(total_ni)*integral_scale,
            gaussian_effective_b2_length_mm=old_integral/gf['peak_magnitude_t']**2*1000,
            ideal_effective_b2_length_mm=integral/cf['peak_magnitude_t']**2*1000,
            existing_envelope_cross_section_mm2=existing_area,
            workpoint_actual_current_design=dict(current_limit_a=current_limit,total_turns=n,actual_current_a=actual,
                i100_setting_a=actual/percent,bare_wire_area_mm2=actual/3,
                bare_wire_diameter_mm=math.sqrt(4*actual/(3*math.pi)),pack_area_mm2=required_ni/(3*.6),
                sensitivity_pack_area_mm2=[required_ni/(5*.7),required_ni/(2*.5)]),
            full_excitation_current_design=dict(i100_limit_a=current_limit,total_turns=full_n,
                actual_i100_a=full_i100,workpoint_current_a=full_i100*percent,
                wire_area_at_i100_mm2=full_i100/3,pack_area_at_i100_mm2=required_ni/(percent*3*.6)))
        records.append(record)
        arrays[key+'__z_mm']=grid;arrays[key+'__gaussian_bz_t']=old_values
        arrays[key+'__ideal_reference_bz_t']=coil_values
        arrays[key+'__ideal_peak_matched_bz_t']=coil_values*peak_scale
        arrays[key+'__ideal_integral_matched_bz_t']=coil_values*integral_scale
        print(row['label'],'FWHM Gaussian/coil',round(gf['main_lobe_fwhm_mm'],4),round(cf['main_lobe_fwhm_mm'],4),
              'regions',cf['half_max_region_count'],'NI peak',round(required_ni,4),flush=True)
    assert baseline==state.to_dict()
    assert all(sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in protected.items())
    np.savez_compressed(OUT/'curves.npz',**arrays)
    dump(OUT/'results.json',dict(scope='Ideal prescribed-current coils only; no magnetic material, yoke or pole response',
        fixed_defaults=True,fixed_solver_tolerances=True,protected_files_unchanged=len(protected),
        source_particles_emitted=0,coherent_runs=0,material_fem_runs=0,quadrature_order=12,
        design_assumptions=dict(j_a_per_mm2=3.,metal_fill=.6,current_density_sensitivity=[2,5],fill_sensitivity=[.5,.7]),
        note='Conditional electrical and packing budgets at the original envelope. No physical geometry or working point is updated. Altered envelope requires a new field solution.',lenses=records))
    flat=[]
    for r in records:
        w=r['workpoint_actual_current_design'];f=r['full_excitation_current_design']
        flat.append(dict(lens=r['lens'],gaussian_peak_t=r['gaussian']['peak_magnitude_t'],
            gaussian_fwhm_mm=r['gaussian']['main_lobe_fwhm_mm'],
            ideal_reference_peak_t=r['ideal_at_original_reference_ni']['peak_magnitude_t'],
            ideal_main_lobe_fwhm_mm=r['ideal_at_original_reference_ni']['main_lobe_fwhm_mm'],
            ideal_half_max_regions=r['ideal_at_original_reference_ni']['half_max_region_count'],
            required_peak_ni_a_turn=r['peak_matched_required_ni_magnitude_a_turn'],
            actual_current_limit_a=w['current_limit_a'],turns_at_workpoint_current=w['total_turns'],
            pack_area_at_workpoint_mm2=w['pack_area_mm2'],i100_from_workpoint_design_a=w['i100_setting_a'],
            turns_at_i100_limit=f['total_turns'],pack_area_at_i100_limit_mm2=f['pack_area_at_i100_mm2']))
    with (OUT/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


def forbidden(*_args,**_kwargs):
    raise AssertionError('Ideal field study must not emit or transport particles')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden),patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        run()
