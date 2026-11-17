"""Independent high-precision axis checks of fixed-envelope current fits."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import math
import time
import mpmath as mp
import numpy as np

from fit import HERE,REFERENCE,axis_basis,gaussian,write_json,AnnularCoilPotential


def main() -> None:
    mp.mp.dps=80
    started=time.perf_counter()
    library=json.loads((HERE/'candidate-library.json').read_text())
    raw=json.loads(REFERENCE.read_text())
    nodes,weights=mp.gauss_quadrature(12,'legendre')
    cases=[]
    for owner,cutoff in (('condenser_lens_1','1e-10'),('probe_tl22_lens','1e-10'),
                         ('probe_tl22_lens','1e-14'),('objective_lens','1e-10')):
        cells=library[owner][cutoff]['cells']
        terms=np.array(raw['gaussian'][owner]['terms_t_m'])
        center=float(np.average(terms[:,1],weights=np.abs(terms[:,0])))
        zs=center+np.array((-.035,-.01,0.,.013,.041))
        matrix_value=axis_basis(zs,cells,12)@np.array([row['current_a'] for row in cells])
        per_cell=[]
        for row in cells:
            native=AnnularCoilPotential(row['source_id'],row['inner_radius_m'],row['outer_radius_m'],row['length_m'],row['turn_count']*row['current_a'],row['center_z_m'])
            per_cell.append(native.axis_bz_t(zs))
        per_cell=np.array(per_cell)
        simple=per_cell.sum(axis=0)
        fsum=np.array([math.fsum(column) for column in per_cell.T])
        precise=[]
        for z in zs:
            terms_mp=[]
            for row in cells:
                ri=mp.mpf(row['inner_radius_m']);ro=mp.mpf(row['outer_radius_m'])
                length=mp.mpf(row['length_m']);zc=mp.mpf(row['center_z_m'])
                # Current and N are the exact serialized physical input values.
                ampere_turns=mp.mpf(row['current_a'])*row['turn_count']
                prefactor=mp.mpf(float(__import__('scipy').constants.mu_0))*ampere_turns/8
                for i in range(12):
                    radius=(ri+ro)/2+(ro-ri)*nodes[i]/2
                    for j in range(12):
                        dz=mp.mpf(float(z))-zc-length*nodes[j]/2
                        terms_mp.append(prefactor*weights[i]*weights[j]*radius**2/(radius**2+dz**2)**mp.mpf('1.5'))
            precise.append(float(mp.fsum(terms_mp)))
        precise=np.array(precise)
        reference_peak=float(max(abs(gaussian(np.linspace(center-.1,center+.1,4001),terms))))
        case=dict(owner=owner,rcond=cutoff,segment_count=len(cells),z_mm=(zs*1000).tolist(),
            high_precision_axis_bz_t=precise.tolist(),matrix_q12_axis_bz_t=matrix_value.tolist(),
            native_simple_sum_axis_bz_t=simple.tolist(),native_fsum_axis_bz_t=fsum.tolist(),
            maximum_matrix_error_t=float(np.max(np.abs(matrix_value-precise))),
            maximum_native_simple_sum_error_t=float(np.max(np.abs(simple-precise))),
            maximum_native_fsum_error_t=float(np.max(np.abs(fsum-precise))),
            maximum_matrix_error_over_reference_peak=float(np.max(np.abs(matrix_value-precise))/reference_peak),
            maximum_native_simple_error_over_reference_peak=float(np.max(np.abs(simple-precise))/reference_peak),
            reference_peak_t=reference_peak)
        cases.append(case)
        print(json.dumps(dict(owner=owner,rcond=cutoff,max_native_error_t=case['maximum_native_simple_sum_error_t'],elapsed_s=time.perf_counter()-started)),flush=True)
    write_json(HERE/'precision-audit.json',dict(schema='fixed-envelope-current-cancellation-precision-audit-v1',
        scope='Pure fields only. MP80 sum uses exact serialized binary64 geometry/current inputs and high-precision 12-point Gauss-Legendre nodes and weights. No source, particle or wave.',
        source_library_sha256=sha256((HERE/'candidate-library.json').read_bytes()).hexdigest(),
        cases=cases,elapsed_s=time.perf_counter()-started,
        qualification='No unchanged project tolerance or focus criterion is declared passed by this diagnostic.'))


if __name__=='__main__':main()
