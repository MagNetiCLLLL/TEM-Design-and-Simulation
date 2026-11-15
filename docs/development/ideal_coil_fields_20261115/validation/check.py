"""Independent axis-only ideal-coil verification, without particles or waves."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0
from scipy.optimize import brentq, minimize_scalar

from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def exact_axis(coil,z):
    """Integrate axial coordinate and radius analytically for one rectangle."""
    x=np.asarray(z)-coil.center_z_m
    def primitive(a):
        nonzero=a!=0
        out=np.zeros_like(a,dtype=float)
        out[nonzero]=a[nonzero]*(np.arcsinh(coil.outer_radius_m/np.abs(a[nonzero]))
                               -np.arcsinh(coil.inner_radius_m/np.abs(a[nonzero])))
        return out
    return mu_0*coil.ampere_turns/(2*coil.length_m*(coil.outer_radius_m-coil.inner_radius_m))*(
        primitive(x+coil.length_m/2)-primitive(x-coil.length_m/2))


def half_max_regions(z,values):
    half=values.max()/2;above=values>=half
    crossings=np.flatnonzero(above[1:]!=above[:-1])
    ends=[float(np.interp(half,np.sort([values[i],values[i+1]]),
         np.array([z[i],z[i+1]])[np.argsort([values[i],values[i+1]])])) for i in crossings]
    return [dict(start_mm=1000*a,end_mm=1000*b,width_mm=1000*(b-a)) for a,b in zip(ends[::2],ends[1::2])]


def run():
    rows=json.loads((ROOT/'docs/development/coil_inventory_20261115/validation/inventory.json').read_text())
    summary=[]
    for row in rows:
        old=CopperCoilAssemblyPotential.from_dict(row['captured_ideal_source'])
        coils=old.coils
        lo=min(c.center_z_m-c.length_m/2 for c in coils)
        hi=max(c.center_z_m+c.length_m/2 for c in coils)
        reach=max(hi-lo,max(c.outer_radius_m for c in coils))
        z=np.linspace(lo-3*reach,hi+3*reach,4001)
        exact=sum(exact_axis(c,z) for c in coils)
        fields={q:sum(replace(c,quadrature_order=q).axis_bz_t(z) for c in coils) for q in (12,24,32)}
        peak=np.max(np.abs(exact))
        record=dict(lens=row['label'],key=row['owner'],source_intervals=len(coils),
            original_quadrature_order=[c.quadrature_order for c in coils],
            sample_points=len(z),peak_reference_t=float(peak),
            absolute_error_t={q:float(np.max(np.abs(v-exact))) for q,v in fields.items()},
            peak_normalized_max_error={q:float(np.max(np.abs(v-exact))/peak) for q,v in fields.items()},
            exact_half_max_regions=half_max_regions(z,np.abs(exact)))
        if row['owner']=='objective_lens':
            axis=lambda zz:sum(exact_axis(c,np.atleast_1d(zz)) for c in coils)
            candidates=[]
            for c in coils:
                mid=c.center_z_m
                opt=minimize_scalar(lambda t:-abs(float(axis(mid+t)[0])),bounds=(-c.length_m/2,c.length_m/2),
                                    method='bounded',options={'xatol':1e-13})
                candidates.append(dict(z_mm=float((mid+opt.x)*1000),magnitude_t=float(-opt.fun)))
            record['exact_lobe_peaks']=candidates
            record['exact_central_gap_field_t']=float(axis((lo+hi)/2)[0])
            record['fwhm_warning']='Report each above-half-height interval separately; outer bounding span is not a single-lobe FWHM.'
        summary.append(record)
        print(row['label'],record['peak_normalized_max_error'],flush=True)
    result=dict(scope='Read-only on-axis operator/analytic checks, no source emission or coherent propagation',
                geometry='Original complete uniform annular envelopes; no material/yoke/pole response',
                production_inputs_or_tolerances_changed=False,reference_method='Analytic axial and radial integration using scipy.constants.mu_0',
                lenses=summary)
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':run()
