"""Record current readbacks and plot detached signed multipole field profiles."""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from migrate_inputs import originals, state_for, forbidden
from temsim.assembly_catalog import AssemblyCatalog
from temsim.physics.multipole_coil_field import (
    _bindings, capture_multipole_coil_fields, multipole_field_coefficients,
)
from temsim.physics.transverse_coil_field import TransverseCoilPotential


V=Path(__file__).resolve().parent/'validation'


def main():
    fits=json.loads((V/'current-calibration-final.json').read_text(encoding='utf-8'))['records']
    references={(row['path'],row['host_key']):row for row in fits}
    catalog=AssemblyCatalog()
    columns={Path(option.file).name:option.name for option in catalog.columns}
    input_rows={Path(row['path']).name:row for row in originals()}
    states={}
    readbacks=[]
    for name in ('C3_ProbeCorrector.toml','C3_ImageCorrector.toml'):
        original=input_rows[name]
        selection,state=state_for(original,catalog,columns)
        state.round_lens_field_model='copper_coil'
        before=json.dumps(state.to_dict(),sort_keys=True)
        sources=capture_multipole_coil_fields(state)
        rows=[]
        for source in sources:
            data=source.potential.provenance
            rows.append(dict(component_key=source.component_key,multipole_order=data['multipole_order'],
                turns_per_coil=source.potential.turn_count_per_coil,
                currents_a=list(source.potential.currents_a),
                physical_coil_count=len(source.potential.coils),
                native_orientation_rad=data.get('native_orientation_rad'),
                coil_azimuths_deg=[c['azimuth_deg'] for c in source.potential.coils]))
        assert json.dumps(state.to_dict(),sort_keys=True)==before
        readbacks.append(dict(selection=asdict(selection),sources=rows))
        states[name]=(original,selection,state)
    (V/'default-current-readbacks.json').write_text(json.dumps(dict(status='passed',
        scope='Detached existing controls to physical amperes; no particle or field propagation',
        source_particle_or_coherent_execution=False,states=readbacks),indent=2)+'\n',encoding='utf-8')
    figure,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    cases=(('C3_ProbeCorrector.toml','probe_hp2_hexapole','Probe HP2: six coils, N=2'),
           ('C3_ProbeCorrector.toml','probe_qpol_quadrupole','Probe QPol: eight coils, N=384'),
           ('C3_ImageCorrector.toml','image_hpol_hexapole','Image HPol: twelve coils, N=63'))
    data=[]
    for ax,(name,key,title) in zip(axes,cases):
        original,selection,state=states[name]
        row=next(r for r in _bindings(state,require_current=True) if r[0]['key']==key)
        part,component,definition,coils,electrical=row
        fit=references[original['path'],key]
        current=np.asarray(electrical['matrix_a_per_unit'])[:,0]
        currents=tuple(current) if len(current)==2 else (float(current[0]),0.)
        source=TransverseCoilPotential(key,coils,definition['turn_count_per_coil'],currents)
        offset=np.linspace(-60.,60.,81)
        points=np.column_stack((np.zeros((len(offset),2)),(component.z_mm+offset)*.001))
        coefficients=multipole_field_coefficients(source,points,definition['order'])
        target=np.asarray(fit['native_target_per_unit'])[:,0]
        projected=(coefficients@target)/(target@target)
        assert abs(projected[len(offset)//2]-1)<1e-8
        width=component.effective_length_mm
        dense=np.linspace(-60.,60.,1001)
        old=np.exp(-.5*(dense/(width/2.355))**2)
        ax.plot(dense,old,color='#64748b',ls='--',label='Previous native Gaussian')
        ax.plot(offset,projected,color='#c46022',lw=2,label='Ideal closed-coil coefficient')
        ax.axvline(0,color='#94a3b8',lw=.7)
        ax.axhline(0,color='#94a3b8',lw=.5)
        ax.set(title=title,xlabel='Offset from original field-reference plane (mm)',
            ylabel='Signed coefficient / original reference coefficient')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        data.append(dict(key=key,selection=asdict(selection),reference_plane_z_mm=component.z_mm,
            multipole_order=definition['order'],turns_per_coil=definition['turn_count_per_coil'],
            native_unit_command='normal coefficient +1 in native m^-order units; detached reference only',
            channel_currents_a=list(currents),native_reference_coefficient=target.tolist(),
            source_reference_error=abs(float(projected[len(offset)//2])-1),
            offset_mm=offset.tolist(),source_coefficient_normal_skew=coefficients.tolist(),
            source_signed_normalized_projection=projected.tolist(),native_fwhm_mm=width))
        print('Rendered signed field profile:',key,flush=True)
    figure.suptitle('Reference coefficient matching preserves one condition; axial field profiles change',fontsize=13)
    figure.savefig(V/'peak-matched-field-profiles.png',dpi=160)
    plt.close(figure)
    (V/'field-profile-data.json').write_text(json.dumps(dict(scope='Static field arrays only; no beam/waist/A1/Cs qualification',
        curve_spacing_mm=1.5,source_quadrature_orders=[12,24],records=data),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
