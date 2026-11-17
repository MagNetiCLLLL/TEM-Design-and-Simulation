"""Signed-front and symmetric-bucking branch search, source-free operator only."""
from __future__ import annotations
import csv, importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transfer_v1', HERE/'design_transfer.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


def main():
    started=time.perf_counter()
    raw=json.loads((old.BASE/'round-lens-field-match/captured-field-coefficients.json').read_text())
    power={row['key']:row for row in csv.DictReader((old.BASE/'round-lens-field-match/candidate-currents.csv').open())}
    cells=[];scales=[];initial=[];owners={};terms=[]
    for owner,record in raw['ideal_current_coils'].items():
        md=record['metadata']; master=md['provenance']['winding_current_definition']['current_a']
        parents=[]
        for row in md['coils']:
            parent=dict(row['metadata']);parent['turns']=int(round(parent['ampere_turns']/master));parents.append(parent)
        start=len(cells)
        cc=old.tools.cells_for(owner,parents,1,4 if owner=='condenser_lens_1' else 2)
        for cell in cc:cell['owner']=owner;cell['original_actual_master_a']=master
        cells+=cc
        scale=abs(float(power[owner]['isolated_b2_current_a']))
        scales.extend([scale]*len(cc));initial.extend([float(np.sign(master))]*len(cc))
        owners[owner]=(start,len(cells));terms.extend(raw['gaussian'][owner]['terms_t_m'])
    scales=np.array(scales);initial=np.array(initial);terms=np.array(terms)
    coarse=old.TransferModel(cells,scales,terms,.2)
    def residual(x, anchor, endpoint_weight=.5, anchor_weight=1e-6):
        r=coarse.residual(x,anchor,anchor_weight)
        maps=coarse.evaluate(x)
        # Explicit signed-field endpoint targets, in tesla. This is an objective
        # weight, not an acceptance tolerance or thermal/power constraint.
        endpoint=np.array([maps[0]['initial_bz_t']-coarse.targets[0]['initial_bz_t']]+
                          [a['final_bz_t']-b['final_bz_t'] for a,b in zip(maps,coarse.targets)])
        return np.r_[r, endpoint_weight*endpoint]
    seeds=[]
    for pattern in ([1,1,1,1],[-2,-1,2,3],[-3,3,3,-3],[-8,8,8,-8],[-15,15,15,-15],[-3,-1,1,4],[3,-3,-3,3],[-6,-2,2,8]):
        x=initial.copy();x[:4]=pattern;seeds.append(x)
    attempts=[];best=None
    for i,seed in enumerate(seeds):
        fit=least_squares(lambda x:residual(x,seed),seed,bounds=(-100.,100.),x_scale='jac',
                          max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        score=float(np.linalg.norm(residual(fit.x,seed,anchor_weight=0.)))
        row=dict(index=i,score=score,nfev=fit.nfev,success=bool(fit.success),values=fit.x.tolist(),current_a=(fit.x*scales).tolist())
        attempts.append(row)
        old.write_json(HERE/'attempts-v2.json',dict(attempts=attempts,elapsed_s=time.perf_counter()-started))
        print(json.dumps(dict(stage='Signed internal branch',attempt=i,score=score,nfev=fit.nfev,elapsed_s=time.perf_counter()-started)),flush=True)
        if best is None or score<best[0]:best=(score,fit.x.copy())
    old.write_json(HERE/'candidate-v2.json',dict(status='SOURCE_FREE_UNQUALIFIED_BRANCH_SEARCH',cells=cells,scales=scales.tolist(),
        owners=owners,best_values=best[1].tolist(),score=best[0],attempts=attempts,
        maps=old.serializable(coarse.evaluate(best[1])),targets=old.serializable(coarse.targets),elapsed_s=time.perf_counter()-started))
    print(json.dumps(dict(stage='DONE',score=best[0],elapsed_s=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
