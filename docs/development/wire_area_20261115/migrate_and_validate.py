"""Install a nominal 9 mm2 turn allocation and verify unchanged ideal fields.

The user-facing CAD size is conductor metal area, a separate optional input.
This authoring migration changes only estimated N and its compensating I100.
Original photo reference counts are retained; Objective counts may be explicitly
adjusted within the subsequent user-authorized 20% bound. Geometry and the full
optical state remain fixed. No emission, transport, focus or coherent run occurs.
"""
from argparse import ArgumentParser
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from unittest.mock import patch

import numpy as np

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.cad_round_winding import round_winding_envelope
from temsim.coil_turn_estimation import estimate_winding_turns_from_allocated_area
from temsim.coil_winding import OWNER_CURRENT_FIELD, INTERVAL_TURNS_FIELD, WIRE_AREA_FIELD
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.round_lens_models import round_lens_cache_identity
from temsim.simulation_modes import switch_mode

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/wire-area-20261115'
CURATED=Path(__file__).resolve().parent/'validation'
PREVIOUS=ROOT/'docs/development/all_lens_current_20261115/validation'
ELECTRICAL={'winding_turn_count',INTERVAL_TURNS_FIELD,OWNER_CURRENT_FIELD}
ROUND={'magnetic_lens_assembly','integrated_magnetic_lens_channel'}


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def preserved(name):
    for base in (OUT,CURATED):
        path=base/name
        if path.is_file():return path
    raise FileNotFoundError(name)


def startup(catalog,column,recording):
    state=default_state();catalog.apply(state,AssemblySelection('FEG',column,recording))
    switch_mode(state,'unified');selected=catalog.default_selection()
    if (column,recording)==(selected.column,selected.recording):
        apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=column,recording_name=recording)
    apply_physical_layout_to_state(state);state.step_mm=.05
    return state


def source_records():
    catalog=AssemblyCatalog();states=[];fields=[]
    for column in catalog.columns:
        for recording in catalog.recording_systems:
            state=startup(catalog,column.name,recording.name);control=deepcopy(state.to_dict())
            states.append(dict(column=column.name,recording=recording.name,state=json.loads(json.dumps(control))))
            selected=copy(state);selected.lenses=deepcopy(state.lenses);selected.round_lens_field_model='copper_coil'
            selected._runtime_copper_coil_provider_cache={};selected._runtime_lens_field_provider_cache={}
            identities=round_lens_cache_identity(selected)
            parts={p.key:p for p in state._resolved_assembly.parts}
            for lens in selected.lenses:
                owner=parts.get(lens.key)
                if owner is None or owner.data.get('mechanical_profile') not in ROUND:continue
                percent,enabled=lens.percent,lens.enabled;lens.percent=100.;lens.enabled=True
                potential=capture_round_lens_potential(selected,lens.key).potential
                z=lens.z_mm*.001
                points=np.array([[0.,0.,z],[0.,0.,z-.08],[0.,0.,z+.08],
                                 [1e-4,-2e-4,z+.02],[2e-4,1e-4,z-.04]])
                coefficients=[{k:getattr(c,k) for k in ('inner_radius_m','outer_radius_m','length_m',
                                                      'center_z_m','ampere_turns','quadrature_order')} for c in potential.coils]
                fields.append(dict(column=column.name,recording=recording.name,owner=lens.key,
                    original_percent=percent,original_enabled=enabled,polarity=lens.polarity,
                    coefficients=coefficients,points_m=points.tolist(),reference_b_t=potential.sample(points)['B_t'].tolist()))
                lens.percent=percent;lens.enabled=enabled
            assert state.to_dict()==control
            states[-1]['ideal_cache_identity']=identities
            print('Captured',column.name,'/',recording.name,flush=True)
    return states,fields


def apply():
    if (OUT/'before/baseline.json').exists():raise ValueError('Do not overwrite the pre-migration baseline')
    policy=tomllib.loads((ROOT/'configs/physics/coil_winding_estimates.toml').read_text())
    nominal=policy['nominal_allocated_cross_section_mm2_per_turn'];assert nominal==9.
    files=json.loads((PREVIOUS/'manifest.json').read_text())['installed_inputs']
    baseline={}
    for name in files:
        data=(ROOT/name).read_bytes();baseline[name]=sha256(data).hexdigest()
        dest=OUT/'before'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    states,fields=source_records()
    save('before/baseline.json',dict(files=baseline,states=states,fields=fields))
    updates_by_file={};entries=[]
    for name in files:
        document=module_manifest.read_document(ROOT/name);parts={p['key']:p for p in document['parts']};updates={}
        for coil in document['parts']:
            owner=parts.get(coil.get('field_source_key'),{})
            if coil.get('mechanical_profile')!='magnetic_excitation_coil' or owner.get('mechanical_profile') not in ROUND:continue
            old_n=coil['winding_turn_count'];old_i=owner[OWNER_CURRENT_FIELD]
            estimate=estimate_winding_turns_from_allocated_area(round_winding_envelope(coil,parts),nominal)
            fixed=owner['key']=='objective_lens'
            new_n=old_n if fixed else estimate['winding_turn_count']
            new_i=old_i if new_n==old_n else (old_n*old_i)/new_n
            if fixed:assert old_n==1172 and list(coil[INTERVAL_TURNS_FIELD])==[586,586]
            else:
                updates['parts',coil['key'],'winding_turn_count']=new_n
                updates['parts',coil['key'],INTERVAL_TURNS_FIELD]=estimate[INTERVAL_TURNS_FIELD]
                meta=deepcopy(coil.get('parameter_metadata',{}))
                note=(f"User-authorized nominal integer allocation on 2026-11-15: "
                      f"N=round_half_up({estimate['cross_section_mm2']:.17g} mm2 / 9 mm2)={new_n}. "
                      "The 9 mm2 value includes insulation and residual pack space, not conductor metal. "
                      "Metal area remains an independent optional user input. This is an engineering "
                      "estimate, not measured/OEM data; fixed counts do not follow later geometry edits automatically.")
                for field in ('winding_turn_count',INTERVAL_TURNS_FIELD):meta[field]=dict(source_kind='estimated',source_note=note)
                updates['parts',coil['key'],'parameter_metadata']=meta
                if new_i!=old_i:
                    updates['parts',owner['key'],OWNER_CURRENT_FIELD]=new_i
                    meta=deepcopy(owner.get('parameter_metadata',{}));entry=meta.setdefault(OWNER_CURRENT_FIELD,{})
                    entry['source_kind']='user_defined'
                    entry['source_note']=(f"2026-11-15 nominal-allocation migration: prescribed reference current "
                        f"changed from {old_i:.17g} A at N={old_n} to {new_i:.17g} A at N={new_n}, "
                        "preserving the original NI and full ideal field within unchanged binary64 checks. "
                        "The prior single-plane field condition remains unchanged; no focus or current rating is asserted.\n"
                        +entry.get('source_note',''))
                    updates['parts',owner['key'],'parameter_metadata']=meta
            entries.append(dict(file=name,owner=owner['key'],coil=coil['key'],old_turns=old_n,new_turns=new_n,
                old_reference_current_a=old_i,new_reference_current_a=new_i,fixed_photo_count_preserved=fixed,
                old_reference_ni=old_n*old_i,new_reference_ni=new_n*new_i,estimate=estimate))
        updates_by_file[name]=updates
    staged={}
    for name,updates in updates_by_file.items():
        path=ROOT/name
        assert sha256(path.read_bytes()).hexdigest()==baseline[name]
        staged[name]=module_manifest.stage_manifest_text(path.read_bytes().decode('utf-8'),updates,
            source_path=path,validate=not name.endswith('projector_stack.toml'))
        # These part-only electrical edits also validate in every assembled host below.
    for name,text in staged.items():(ROOT/name).write_bytes(text.encode('utf-8'))
    save('migration.json',dict(nominal_allocated_area_per_turn_mm2=nominal,entries=entries))


def stripped(document):
    value=deepcopy(document)
    for part in value.get('parts',()):
        for key in ELECTRICAL:part.pop(key,None)
        metadata=part.get('parameter_metadata',{})
        for key in ELECTRICAL:metadata.pop(key,None)
        if not metadata:part.pop('parameter_metadata',None)
    return value


def adjust_objective_within_authorized_bound():
    """Apply the subsequent explicit 20% permission without rewriting history."""
    path=OUT/'migration.json';migration=json.loads(preserved('migration.json').read_text())
    if (OUT/'before-objective-adjustment.json').exists():
        raise ValueError('The bounded Objective adjustment is already recorded')
    policy=tomllib.loads((ROOT/'configs/physics/coil_winding_estimates.toml').read_text())
    fraction=float(policy['maximum_turn_adjustment_fraction']);assert fraction==.20
    (OUT/'nominal-only-migration.json').write_bytes(path.read_bytes())
    if (OUT/'summary.json').exists():(OUT/'nominal-only-summary.json').write_bytes((OUT/'summary.json').read_bytes())
    snapshots={};staged={}
    for row in migration['entries']:
        if row['owner']!='objective_lens':continue
        name=row['file'];source=ROOT/name
        doc=module_manifest.read_document(source);parts={p['key']:p for p in doc['parts']}
        coil=parts[row['coil']];owner=parts[row['owner']]
        assert coil['winding_turn_count']==1172 and list(coil[INTERVAL_TURNS_FIELD])==[586,586]
        estimate=estimate_winding_turns_from_allocated_area(round_winding_envelope(coil,parts),9.)
        new_n=estimate['winding_turn_count'];new_counts=list(estimate[INTERVAL_TURNS_FIELD])
        assert new_n==1160 and new_counts==[580,580]
        assert abs(new_n/row['old_turns']-1)<=fraction
        old_i=owner[OWNER_CURRENT_FIELD];new_i=(1172*old_i)/new_n
        snapshots[name]=dict(sha256=sha256(source.read_bytes()).hexdigest(),objective=owner,coil=coil)
        dest=OUT/'before-objective-adjustment'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        meta=deepcopy(coil.get('parameter_metadata',{}))
        note=('Explicit user-authorized bounded modelling adjustment on 2026-11-15: '
              'upper and lower each change from the original adopted 586 turns to 580 (-1.02389%), '
              'within the allowed +/-20%. The original upper photograph was estimated at 580-590 '
              'cross-sections with 586 adopted; the lower 586 was a symmetry assumption. '
              '580 is the current engineering model count, not a new measurement. '
              'At fixed ID60/OD132 and occupied145mm per region, allocated area/N is now exactly9mm2. '
              'This allocation includes insulation; conductor metal area remains an independent user input. '
              'Counts remain fixed after this explicit edit; geometry/current changes do not re-estimate them.')
        for field in ('winding_turn_count',INTERVAL_TURNS_FIELD):meta[field]=dict(source_kind='estimated',source_note=note)
        current_meta=deepcopy(owner.get('parameter_metadata',{}));entry=current_meta.setdefault(OWNER_CURRENT_FIELD,{})
        entry['source_kind']='user_defined'
        entry['source_note']=(f'2026-11-15 explicit +/-20% turn permission: I100 changes from {old_i:.17g} A '
            f'at 586/586 turns to {new_i:.17g} A at 580/580 turns, preserving per-region NI and full ideal field. '
            'The original specimen-upper-surface matching condition, excitation, geometry and tolerances remain unchanged. '
            'No focus or current-rating qualification is asserted.\n'+entry.get('source_note',''))
        updates={('parts',row['coil'],'winding_turn_count'):new_n,
                 ('parts',row['coil'],INTERVAL_TURNS_FIELD):new_counts,
                 ('parts',row['coil'],'parameter_metadata'):meta,
                 ('parts',row['owner'],OWNER_CURRENT_FIELD):new_i,
                 ('parts',row['owner'],'parameter_metadata'):current_meta}
        staged[name]=module_manifest.stage_manifest_text(source.read_bytes().decode('utf-8'),updates,source_path=source)
        row.update(new_turns=new_n,new_reference_current_a=new_i,new_reference_ni=new_n*new_i,
                   fixed_photo_count_preserved=False,original_photo_reference_preserved=True,
                   original_interval_turn_counts=[586,586],new_interval_turn_counts=new_counts,
                   authorized_adjustment_fraction=fraction,relative_turn_change=new_n/1172-1)
    for name,text in staged.items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==snapshots[name]['sha256']
        (ROOT/name).write_bytes(text.encode('utf-8'))
    migration['authorized_adjustment_fraction']=fraction
    save('before-objective-adjustment.json',snapshots);save('migration.json',migration)


def validate():
    baseline=json.loads(preserved('before/baseline.json').read_text());migration=json.loads(preserved('migration.json').read_text())
    for name in baseline['files']:
        before=tomllib.loads(preserved('before/'+name).read_text(encoding='utf-8'))
        after=tomllib.loads((ROOT/name).read_text(encoding='utf-8'))
        assert stripped(before)==stripped(after),'Winding envelope or non-electrical inputs changed'
    states,fields=source_records();checks=[]
    assert len(states)==14 and len(fields)==196
    for before,after in zip(baseline['states'],states,strict=True):
        assert before['state']==after['state'],'Optical/source/numerical state changed'
        assert before['ideal_cache_identity']!=after['ideal_cache_identity']
    for before,after in zip(baseline['fields'],fields,strict=True):
        assert (before['column'],before['recording'],before['owner'])==(after['column'],after['recording'],after['owner'])
        for old,new in zip(before['coefficients'],after['coefficients'],strict=True):
            for key in old:
                if key!='ampere_turns':assert old[key]==new[key]
            np.testing.assert_allclose(new['ampere_turns'],old['ampere_turns'],rtol=2e-14,atol=0)
        np.testing.assert_allclose(after['reference_b_t'],before['reference_b_t'],rtol=2e-14,atol=1e-20)
        scale=max(abs(x) for row in before['reference_b_t'] for x in row)
        error=float(np.max(np.abs(np.array(after['reference_b_t'])-before['reference_b_t']))/scale)
        checks.append(dict(column=after['column'],recording=after['recording'],owner=after['owner'],peak_normalized_change=error))
    authorized=migration.get('authorized_adjustment_fraction',0.)
    assert all(abs(row['new_turns']/row['old_turns']-1)<=max(authorized,.20) for row in migration['entries'])
    objective_adjusted=authorized>0
    assert all(row['old_turns']==1172 and row['new_turns']==(1160 if objective_adjusted else 1172)
               for row in migration['entries'] if row['owner']=='objective_lens')
    save('field-checks.json',checks)
    save('summary.json',dict(status='passed',nominal_allocated_area_per_turn_mm2=9.,source_definitions=9,
        source_owner_instances=78,assemblies=14,field_checks=196,pointwise_rtol=2e-14,pointwise_atol_t=1e-20,
        maximum_peak_normalized_field_change=max(r['peak_normalized_change'] for r in checks),
        changed_estimated_counts=sum(r['old_turns']!=r['new_turns'] for r in migration['entries']),
        original_objective_photo_reference_preserved=True,
        current_objective_interval_turns=[580,580] if objective_adjusted else [586,586],
        authorized_turn_adjustment_fraction=authorized,all_turn_adjustments_within_20_percent=True,
        envelopes_and_optical_source_numerics_unchanged=True,
        all_ideal_cache_identities_changed=True,default_metal_area_inferred=False,
        source_transport_focus_or_coherent_executed=False))
    print('Validated',len(checks),'unchanged ideal fields',flush=True)


def forbidden(*args,**kwargs):raise AssertionError('Field-only authoring must not emit or transport particles')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--apply-objective-adjustment',action='store_true');args=parser.parse_args()
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        if args.apply:apply()
        if args.apply_objective_adjustment:adjust_objective_within_authorized_bound()
        validate()
