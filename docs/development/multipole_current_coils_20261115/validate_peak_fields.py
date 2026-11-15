"""Check fixed-current fits against native A-polynomials and finer quadrature.

Native unit controls are detached test references and restored immediately.
No source particle, trajectory, coherent grid or working-point edit is used.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from migrate_inputs import forbidden, originals, state_for
from temsim.assembly_catalog import AssemblyCatalog
from temsim.immutable_json import thaw_json
from temsim.physics.multipole_coil_field import _bindings, multipole_field_coefficients
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.physics.transverse_coil_field import TransverseCoilPotential, _unit_nodes


V=Path(__file__).resolve().parent/'validation'
LIMIT=1e-8


def native_polynomial_target(state,component,order,count):
    """Read the old executor's Az polynomial, independently of current fitting."""
    strength=f'strength_m{order}'
    skew=f'skew_strength_m{order}'
    previous=(component.enabled,getattr(component,strength),getattr(component,skew,None),hasattr(component,skew))
    result=[]
    try:
        component.enabled=True
        for channel in range(count):
            if order==2 and channel==1 and not callable(getattr(component,'skew_quadrupole_strength_m2',None)):
                # Legacy Image QPol exposes only a normal native input. Its
                # new second electrical winding is explicitly defined as the
                # independent 45-degree quadrupole, not a fabricated old knob.
                normal,skew_value=result[0]
                result.append([-skew_value,normal])
                continue
            setattr(component,strength,1. if channel==0 else 0.)
            if previous[3] or count==2 and (order==3 or callable(getattr(component,'skew_quadrupole_strength_m2',None))):
                setattr(component,skew,1. if channel==1 else 0.)
            rows=capture_posed_column_fields(state,include_aligned=True)
            field=next(row for row in rows if row.component_key==component.key)
            terms={(x,y):value for x,y,value in field.polynomial_terms}
            if order==2:
                # Bx=d_y Az, By=-d_x Az, evaluated on the native envelope peak.
                normal=-terms.get((2,0),0.)+terms.get((0,2),0.)
                skew_value=-terms.get((1,1),0.)
            else:
                normal=.75*(-terms.get((3,0),0.)+terms.get((1,2),0.))
                skew_value=.75*(-terms.get((2,1),0.)+terms.get((0,3),0.))
            result.append([normal,skew_value])
    finally:
        component.enabled=previous[0]
        setattr(component,strength,previous[1])
        if previous[3]:
            setattr(component,skew,previous[2])
        elif hasattr(component,skew):
            delattr(component,skew)
    return np.asarray(result).T


def main():
    saved=json.loads((V/'current-calibration-final.json').read_text(encoding='utf-8'))
    fits={(row['path'],row['host_key']):row for row in saved['records']}
    catalog=AssemblyCatalog()
    columns={Path(option.file).name:option.name for option in catalog.columns}
    cache={};records=[]
    for original in originals():
        selection,state=state_for(original,catalog,columns)
        before=json.dumps(state.to_dict(),sort_keys=True)
        for part,component,definition,coils,electrical in _bindings(state,require_current=True):
            if (original['path'],part['key']) not in fits:
                continue
            fit=fits[original['path'],part['key']]
            order=definition['order'];count=definition['electrical_channel_count']
            target=native_polynomial_target(state,component,order,count)
            np.testing.assert_allclose(target,fit['native_target_per_unit'],rtol=2e-14,atol=2e-17)
            # Translation invariance is exact. Full binary64 descriptors are
            # retained in the key; no rounded geometry is substituted.
            center=coils[0]['center_z_mm']
            normalized=[]
            for item in coils:
                row=thaw_json(item)
                for name in ('center_z_mm','start_z_mm','end_z_mm'):
                    if name in row:
                        row[name]-=center
                normalized.append(row)
            offset=component.z_mm-center
            identity=json.dumps(dict(coils=normalized,N=definition['turn_count_per_coil'],
                order=order,channels=count,reference_offset_mm=offset),sort_keys=True,separators=(',',':'))
            if identity not in cache:
                response=np.zeros((2,count))
                point=np.array([[0.,0.,offset*.001]])
                for channel in range(count):
                    current=[0.,0.];current[channel]=1.
                    source=TransverseCoilPotential('independent_quadrature',normalized,
                        definition['turn_count_per_coil'],tuple(current),pack_order=24,path_order=48)
                    response[:,channel]=multipole_field_coefficients(source,point,order)[0]
                cache[identity]=response
                _unit_nodes.cache_clear()
            observed=cache[identity]@np.asarray(electrical['matrix_a_per_unit'])
            error=float(np.max(abs(observed-target))/np.max(abs(target)))
            assert error<LIMIT, (part['key'],error)
            records.append(dict(path=original['path'],selection=asdict(selection),key=part['key'],
                multipole_order=order,turn_count_per_coil=definition['turn_count_per_coil'],
                reference_orders=[24,48],installed_orders=[12,24],
                native_A_polynomial_target_per_unit=target.tolist(),observed_field_per_unit=observed.tolist(),
                independent_unit_current_response=cache[identity].tolist(),
                maximum_absolute_error=float(np.max(abs(observed-target))),peak_normalized_error=error,
                native_reference_basis=('Legacy normal native polynomial plus explicit 45-degree electrical extension'
                    if order==2 and not callable(getattr(component,'skew_quadrupole_strength_m2',None))
                    else 'Existing native normal/skew polynomial basis'),
                native_controls_restored=True))
            print('Independent native/quad check:',part['key'],f'{error:.3g}',flush=True)
        assert json.dumps(state.to_dict(),sort_keys=True)==before
    assert len(records)==40
    receipt=dict(status='passed',source_files=5,hosts=len(records),
        peak_normalized_error_limit=LIMIT,maximum_peak_normalized_error=max(r['peak_normalized_error'] for r in records),
        unique_exact_centered_circuit_responses=len(cache),reference_orders=[24,48],installed_orders=[12,24],
        native_target_method='Analytic derivatives of unchanged legacy executor Az unit polynomials',
        source_particle_or_wave_execution=False,records=records)
    (V/'independent-peak-field-checks.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2))


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
