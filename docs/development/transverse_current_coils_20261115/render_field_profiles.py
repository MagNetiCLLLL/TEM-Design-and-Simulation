"""Show the actual peak-matched coil field and retained native reference law.

The two nonzero commands are detached illustrative reference inputs. No optical
working point, file, beam source, particle trajectory or coherent wave is changed.
"""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.column import default_state
from temsim.physics.transverse_coil_field import (
    TransverseCoilPotential,_bindings,quadrupole_gradient_t_per_m,
)
from temsim.lens_pose import lens_pose_registration
from temsim.simulation_modes import switch_mode


DESTINATION=Path(__file__).resolve().parent/'validation'


def forbidden(*args,**kwargs):
    raise AssertionError('Field profiles must not emit or transport particles')


def central_fwhm(x,y):
    values=np.abs(y);middle=len(x)//2;half=values.max()/2
    left=right=middle
    while left>0 and values[left]>=half:left-=1
    while right<len(x)-1 and values[right]>=half:right+=1
    if left==0 or right==len(x)-1:return None
    lo=np.interp(half,values[left:left+2],x[left:left+2])
    hi=np.interp(half,values[right-1:right+1][::-1],x[right-1:right+1][::-1])
    return float(hi-lo)


def main():
    catalog=AssemblyCatalog();selection=catalog.default_selection()
    state=default_state();catalog.apply(state,selection);switch_mode(state,'unified')
    apply_physical_layout_to_state(state)
    original=json.dumps(state.to_dict(),sort_keys=True)
    bindings={row[0]['key']:row for row in _bindings(state,require_current=True)}
    figure,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    records=[]
    cases=(('probe_dph2_coils',[1.,0.],'DPH2: +1 mrad X reference command','By (mT)'),
           ('condenser_stigmator',[100.,0.],'Condenser stigmator: +100% X reference command','Normal gradient (T/m)'))
    for ax,(key,command,title,label) in zip(axes,cases):
        part,component,definition,coils,electrical,gun=bindings[key]
        assert len({c['station_index'] for c in coils})==1 and not gun
        current=np.asarray(electrical['matrix_a_per_unit'])@command+electrical['current_bias_a']
        potential=TransverseCoilPotential(component.key,coils,definition['turn_count_per_coil'],
            tuple(current),lens_pose_registration(state,component.key))
        reference=float(component.z_mm)
        offset=np.linspace(-50.,50.,201)
        z=reference+offset
        points=np.column_stack((np.zeros(len(z)),np.zeros(len(z)),z*.001))
        if key=='probe_dph2_coils':
            from temsim.physics.core import electron
            charge,momentum,_=electron(state)
            peak=-momentum/charge*.001/(component.effective_thickness_mm*.001)
            native=np.where(abs(offset)<=component.effective_thickness_mm/2,peak,0.)
            actual=potential.field_at_global_positions_t(points)[:,1]
            display=1000.
            native_integral=peak*component.effective_thickness_mm*.001
        else:
            from temsim.physics.core import electron
            charge,momentum,_=electron(state)
            peak=momentum/charge*component.max_strength_m2
            native=peak*np.exp(-.5*(offset/(component.length_mm/2.355))**2)
            actual=quadrupole_gradient_t_per_m(potential,points)[:,0]
            display=1.
            native_integral=peak*np.sqrt(2*np.pi)*(component.length_mm/2.355)*.001
        error=abs(actual[len(z)//2]-peak)/abs(peak)
        assert error<1e-8
        ax.plot(offset,native*display,label='Previous effective field',color='#64748b',ls='--')
        ax.plot(offset,actual*display,label='Ideal closed-coil field',color='#c2410c',lw=2)
        ax.axvline(0,color='#94a3b8',lw=.7)
        ax.set(xlabel='Axial offset from original reference plane (mm)',ylabel=label,title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=9)
        records.append(dict(host_key=key,component_key=component.key,
            command=command,channel_currents_a=current.tolist(),turns_per_coil=definition['turn_count_per_coil'],
            reference_z_mm=reference,reference_peak_error=error,
            field_fwhm_mm=central_fwhm(offset,actual),native_fwhm_mm=(
                component.effective_thickness_mm if key=='probe_dph2_coils' else component.length_mm),
            integration_interval_offset_mm=[-50.,50.],
            native_integral=float(native_integral),native_integral_method='analytic native envelope',
            coil_integral=float(np.trapezoid(actual,z*.001)),
            offset_mm=offset.tolist(),native=native.tolist(),coil=actual.tolist()))
        print('Rendered peak-matched field:',key,flush=True)
    figure.suptitle('Peak matching preserves one field condition; axial shape and integrals change',fontsize=12)
    figure.savefig(DESTINATION/'peak-matched-field-profiles.png',dpi=160)
    plt.close(figure)
    assert json.dumps(state.to_dict(),sort_keys=True)==original
    (DESTINATION/'field-profile-data.json').write_text(json.dumps(dict(selection=asdict(selection),
        scope='Detached reference-command field arrays; no particle/focus/coherent calculation',
        profile_spacing_mm=.5,records=records),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
