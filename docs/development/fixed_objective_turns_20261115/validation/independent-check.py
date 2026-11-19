"""Fixed photo-derived upper winding count; ideal fields only, no input edits."""
from dataclasses import replace
from pathlib import Path
import json
import math
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize_scalar

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.simulation_modes import switch_mode
from temsim.physics.potential_capture import capture_round_lens_potential

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
inventory=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/inventory.json').read_text())
protected=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/protected-files.json').read_text())
assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
state=default_state();cat=AssemblyCatalog();selection=cat.default_selection();cat.apply(state,selection)
switch_mode(state,'unified');apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=selection.column,recording_name=selection.recording)
from temsim.column.state_layout import apply_physical_layout_to_state
apply_physical_layout_to_state(state);state.step_mm=.05
baseline=state.to_dict()
obj=next(l for l in state.lenses if l.key=='objective_lens')
upper_centre=obj.upper_field_center_z_mm;sample_top=state.sample.upper_surface_z_mm
scale=obj.percent/100*obj.polarity
upper_gaussian=lambda z:scale*obj.upper_b0_t*obj._profile(np.asarray(z),obj.upper_field_center_z_mm,obj.upper_a_mm,obj.upper_gaussian)
total_gaussian=lambda z:obj.magnetic_field_t(np.asarray(z))
state.round_lens_field_model='copper_coil'
potential=capture_round_lens_potential(state,'objective_lens').potential
assert len(potential.coils)==2
original_upper,original_lower=sorted(potential.coils,key=lambda c:c.center_z_m)
n=586
upper=replace(original_upper,ampere_turns=float(n))
upper_b=lambda z:upper.axis_bz_t(np.asarray(z)*.001)
lower_b=lambda z:original_lower.axis_bz_t(np.asarray(z)*.001)
def peak(fun,lo,hi):
 middle=(lo+hi)/2
 fit=minimize_scalar(lambda z:-abs(float(fun(z+middle))),bounds=(lo-middle,hi-middle),method='bounded',options={'xatol':1e-9})
 return abs(float(fit.fun)),float(fit.x+middle)
up_g_peak,up_g_z=peak(upper_gaussian,upper_centre-30,upper_centre+30)
up_c_peak,up_c_z=peak(upper_b,original_upper.center_z_m*1000-50,original_upper.center_z_m*1000+50)
total_peak,total_z=peak(total_gaussian,upper_centre-30,upper_centre+30)
cases=[]
def add(name,target,per_a,z,definition):
 current=target/per_a
 cases.append(dict(name=name,target_field_t=target,upper_field_per_actual_ampere_t=per_a,
                   z_mm=z,upper_turns=n,current_a=current,i100_a=current/(obj.percent*.01),
                   current_range_for_580_to_590_turns_a=[current*n/590,current*n/580],definition=definition))
add('upper_component_global_peak',up_g_peak,up_c_peak,None,'Match upper component peak magnitudes at their separate peak positions; does not match location or sample field')
add('upper_component_at_original_upper_field_centre',float(upper_gaussian(upper_centre)),float(upper_b(upper_centre)),upper_centre,'Match only upper Gaussian contribution at the same original upper-field centre plane')
add('upper_component_at_specimen_upper_surface',float(upper_gaussian(sample_top)),float(upper_b(sample_top)),sample_top,'Match only upper Gaussian contribution at the same specimen entrance; lower winding remains separately unspecified')
target_total=float(total_gaussian(sample_top));retained_lower=float(lower_b(sample_top))
add('whole_objective_at_specimen_surface_lower_legacy_NI_held',target_total-retained_lower,float(upper_b(sample_top)),sample_top,'Match whole Objective field at specimen entrance by upper current alone, holding original lower ideal NI, not lower turn count')
result=dict(scope='Independent ideal-coil component arithmetic; no particles, waves, material solve or persisted input edit',
            photo_turn_count=dict(region='upper only',value=586,estimated_range=[580,590],one_circle_is_one_turn=True),
            original_optical_percent=obj.percent,original_polarity=obj.polarity,
            upper_gaussian_peak_t=up_g_peak,upper_gaussian_peak_z_mm=up_g_z,
            upper_ideal_at_1a_peak_t=up_c_peak,upper_ideal_peak_z_mm=up_c_z,
            whole_gaussian_peak_t=total_peak,whole_gaussian_peak_z_mm=total_z,
            specimen_upper_surface_z_mm=sample_top,lower_original_ni_a_turn=original_lower.ampere_turns,
            upper_original_ni_a_turn=original_upper.ampere_turns,retained_lower_ideal_bz_at_specimen_t=retained_lower,
            lower_turn_count=None,wire_diameter_status='Not inferred from unscaled photo',cases=cases,
            recommendation='Keep photo evidence as a per-upper-region fixed count. Current legacy record spans both regions; do not store586 as its combined total or assume lower586 without authorization.')
lower_symmetric=replace(original_lower,ampere_turns=float(n))
series_b=lambda z:upper_b(z)+lower_symmetric.axis_bz_t(np.asarray(z)*.001)
series_peak_candidates=[peak(series_b,c.center_z_m*1000-50,c.center_z_m*1000+50)
                        for c in (original_upper,original_lower)]
series_peak,series_peak_z=max(series_peak_candidates)
global_current=total_peak/series_peak
surface_current=float(total_gaussian(sample_top))/float(series_b(sample_top))
old_peak_plane_current=total_peak/float(series_b(total_z))
result['later_user_authorized_symmetric_scenario']=dict(
    upper_turns=586,lower_turns=586,total_series_turns=1172,
    provenance='Upper count from user photograph estimate; lower count explicitly authorized by user as a symmetry assumption',
    current_definition='Same actual conductor current through both series winding regions; each region has586I ampere-turns',
    global_peak_matching=dict(target_t=total_peak,current_a=global_current,
        i100_a=global_current/(obj.percent*.01),ideal_peak_z_mm=series_peak_z,
        field_at_old_gaussian_peak_plane_t=float(series_b(total_z))*global_current,
        field_at_specimen_upper_surface_t=float(series_b(sample_top))*global_current,
        current_interval_if_both_counts_580_to590_a=[global_current*n/590,global_current*n/580]),
    same_old_gaussian_peak_plane_matching=dict(z_mm=total_z,target_t=total_peak,
        current_a=old_peak_plane_current,i100_a=old_peak_plane_current/(obj.percent*.01),
        resulting_global_peak_t=series_peak*old_peak_plane_current),
    same_specimen_upper_surface_matching=dict(z_mm=sample_top,target_t=float(total_gaussian(sample_top)),
        current_a=surface_current,i100_a=surface_current/(obj.percent*.01),
        resulting_global_peak_t=series_peak*surface_current),
    source_shape_unchanged=True,wire_diameter_defined=False,
    note='Scalar current matching does not match both peak amplitude and its axial position or restore the original focused working point')
result['recommendation']='User now explicitly authorizes upper586 plus lower586. Store combined total1172 if using the existing two-region winding record, with per-region586 provenance. Wire diameter remains unknown; do not fabricate it.'
state.round_lens_field_model=baseline['simulation_mode_profiles'].get('unified',{}).get('round_lens_field_model','gaussian')
assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
