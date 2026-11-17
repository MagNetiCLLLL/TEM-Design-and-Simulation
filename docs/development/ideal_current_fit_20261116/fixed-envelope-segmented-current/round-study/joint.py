"""Fixed-working-point joint current fit on original physical coil cells."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import math
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fit import HERE,REFERENCE,whole_axis,axis_basis,gaussian,solve_current,write_json,RCONDS


def main() -> None:
    start=time.perf_counter()
    raw=json.loads(REFERENCE.read_text())
    library=json.loads((HERE/'candidate-library.json').read_text())
    cells=[];owner_slices={};terms=[]
    for owner,variants in library.items():
        first=len(cells)
        cells.extend([dict(cell,owner=owner) for cell in variants['1e-10']['cells']])
        owner_slices[owner]=(first,len(cells))
        terms.extend(raw['gaussian'][owner]['terms_t_m'])
    terms=np.array(terms)
    z,weights=whole_axis(24);target=gaussian(z,terms)
    print(json.dumps(dict(stage='Building original-envelope joint basis',segments=len(cells))),flush=True)
    basis=axis_basis(z,cells,12)
    summaries,currents=solve_current(basis,target,weights)
    hold,hold_weights=whole_axis(48);hold_target=gaussian(hold,terms)
    print(json.dumps(dict(stage='Independent full-tail holdout and q24',elapsed_s=time.perf_counter()-start)),flush=True)
    h12=axis_basis(hold,cells,12)
    h24=axis_basis(hold,cells,24)
    target_norm=np.linalg.norm(hold_target*np.sqrt(hold_weights))
    output=[];candidate={}
    for metrics,current in zip(summaries,currents):
        f12=h12@current; f24=h24@current
        record=dict(metrics,holdout_q12_relative_l2=float(np.linalg.norm((f12-hold_target)*np.sqrt(hold_weights))/target_norm),
            holdout_q24_relative_l2=float(np.linalg.norm((f24-hold_target)*np.sqrt(hold_weights))/target_norm),
            q12_to_q24_field_relative_l2=float(np.linalg.norm((f24-f12)*np.sqrt(hold_weights))/target_norm),
            q24_total_integral_b2_t2_m=float(np.dot(hold_weights,f24*f24)),
            reference_total_integral_b2_t2_m=float(np.dot(hold_weights,hold_target*hold_target)))
        output.append(record)
        candidate[str(metrics['rcond'])]=dict(metrics=record,cells=[dict(cell,current_a=float(amp),current_ratio=float(amp/library[cell['owner']]['1e-10']['original_shared_current_a'])) for cell,amp in zip(cells,current)])
    result=dict(schema='fixed-envelope-joint-current-field-fit-v1',
        status='FIXED_WORKING_POINT_FIELD_ONLY_NOT_FOCUS_OR_RUNTIME_QUALIFIED',
        source_library_sha256=sha256((HERE/'candidate-library.json').read_bytes()).hexdigest(),
        constraints='All original physical winding rectangles and turn counts remain unchanged; only internal segment currents differ. No resistance, heat or constant power constraint.',
        control_scope='Joint fixed operating-point field fit. It does not preserve independent Gaussian response of every logical lens control; a lens-current edit must be re-evaluated.',
        numerical_scope='SVD cutoffs explore conditioning; no original project tolerance is relaxed or declared passed.',
        cell_count=len(cells),owners=owner_slices,regularizations=output,elapsed_s=time.perf_counter()-start)
    write_json(HERE/'joint-results.json',result);write_json(HERE/'joint-candidate-library.json',candidate)
    plot_z=np.linspace(.35,2.85,6001); plot_target=gaussian(plot_z,terms)
    plot_basis=axis_basis(plot_z,cells,12)
    arrays=dict(z_m=plot_z,target_t=plot_target)
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
    axes[0].plot(plot_z*1000,plot_target,color='black',label='Gaussian working-field reference')
    for cutoff in (1e-8,1e-10,1e-12):
        index=RCONDS.index(cutoff);field=plot_basis@currents[index]
        arrays[f'current_cutoff_{cutoff:g}']=field
        axes[0].plot(plot_z*1000,field,'--',label=f'Segmented original envelope, SVD {cutoff:g}')
        axes[1].plot(plot_z*1000,field-plot_target,label=f'SVD {cutoff:g}')
    axes[0].set(xlabel='Global Z (mm)',ylabel='Bz (T)',title='Fixed geometry: joint current-only field fitting')
    axes[1].set(xlabel='Global Z (mm)',ylabel='Candidate minus reference Bz (T)')
    axes[0].legend();axes[1].legend()
    fig.savefig(HERE/'joint-field-comparison.png',dpi=160);plt.close(fig)
    np.savez_compressed(HERE/'joint-profiles.npz',**arrays)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
