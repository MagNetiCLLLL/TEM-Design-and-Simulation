"""Independent doubled-quadrature checks of every installed peak calibration.

This is a field-array calculation with forbidden source emission/transport.
The unchanged acceptance threshold is declared before evaluating the results.
It does not qualify finite-source focus or coherent transport.
"""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog,AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.transverse_coil_field import calibrate_transverse_current_matrix
from temsim.physics.transverse_coil_field import _bindings
from temsim.immutable_json import json_digest
from temsim.simulation_modes import switch_mode
from temsim.transverse_current import MATRIX_FIELD


ROOT=Path(__file__).resolve().parents[3]
VALIDATION=Path(__file__).resolve().parent/'validation'
PEAK_NORMALIZED_ERROR_LIMIT=1e-8


def forbidden(*args,**kwargs):
    raise AssertionError('Peak-field checks must not emit or transport particles')


def main():
    inputs=json.loads((VALIDATION/'current-calibration-final.json').read_text(encoding='utf-8'))
    catalog=AssemblyCatalog();default=catalog.default_selection();states={};bindings={};rows=[];responses={}
    for record in inputs['records']:
        selection=AssemblySelection(**record['selection'])
        if selection not in states:
            state=default_state();catalog.apply(state,selection);switch_mode(state,'unified')
            if selection.column==default.column:
                apply_operating_mode_pair(state,'nano_probe','diffraction',
                    column_name=selection.column,recording_name=selection.recording)
            apply_physical_layout_to_state(state);states[selection]=state
            bindings[selection]={r[0]['key']:r for r in _bindings(state,require_current=False)}
        state=states[selection]
        part=next(p for p in state._resolved_assembly.parts if p.key==record['host_key'])
        matrix=np.asarray(part.data[MATRIX_FIELD])
        np.testing.assert_array_equal(matrix,record['matrix_a_per_unit'])
        # Translation invariance permits exact reuse of a completed unit-current
        # response for an identical centered circuit. No rounding or geometry
        # quantization is used; unequal floating-point descriptors do not match.
        bound=bindings[selection][record['host_key']]
        origin=record['reference_planes_z_mm'][0]
        centered=[]
        for source in bound[3]:
            coil=dict(source)
            for name in ('center_z_mm','start_z_mm','end_z_mm'):
                if name in coil:coil[name]-=origin
            centered.append(coil)
        geometry_key=json_digest(dict(coils=centered,turns=record['turn_count_per_coil'],
            reference_planes_mm=[p-origin for p in record['reference_planes_z_mm']],orders=[24,48]))
        if geometry_key not in responses:
            reference=calibrate_transverse_current_matrix(state,record['host_key'],pack_order=24,path_order=48)
            responses[geometry_key]=(reference,dict(path=record['path'],host_key=record['host_key']))
        reference,reused_from=responses[geometry_key]
        target=np.asarray(record['native_target_per_unit'])
        actual=np.asarray(reference['unit_current_response'])@matrix
        scale=np.max(np.abs(target))
        error=float(np.max(np.abs(actual-target))/scale)
        assert error<=PEAK_NORMALIZED_ERROR_LIMIT, (record['path'],record['host_key'],error)
        rows.append(dict(path=record['path'],host_key=record['host_key'],
            reference_planes_z_mm=record['reference_planes_z_mm'],command_unit=record['command_unit'],
            installed_pack_order=record['pack_order'],installed_path_order=record['path_order'],
            reference_pack_order=24,reference_path_order=48,
            native_target_per_unit=target.tolist(),independent_observed_per_unit=actual.tolist(),
            maximum_absolute_error=float(np.max(np.abs(actual-target))),peak_normalized_error=error,
            response_condition=reference['response_condition_number'],
            centered_circuit_digest=geometry_key,unit_current_response_origin=reused_from))
        print('Doubled-quadrature peak check:',record['host_key'],f'{error:.3g}',flush=True)
    receipt=dict(status='passed',physical_hosts=len(rows),source_files=len({r['path'] for r in rows}),
        peak_normalized_error_limit=PEAK_NORMALIZED_ERROR_LIMIT,
        maximum_peak_normalized_error=max(r['peak_normalized_error'] for r in rows),
        independently_evaluated_centered_circuits=len(responses),
        reference_orders=[24,48],records=rows,source_or_particle_transport_or_wave_executed=False)
    (VALIDATION/'independent-peak-field-checks.json').write_text(
        json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2))


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
