"""Independently measure the frozen AC pivot and the Mini/upper-objective front plane."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from material_trial import HERE, PREVIOUS, WORK, Column, Field, Circuit, Global, bore, digest, launch, load_bias, source_and_electrodes, tangent
from joint_pivot_tuning import scan_response
from material_relay_column import MaterialScan


MIX_FILE = PREVIOUS/'evidence/material-relay/post-scan-registered-calibration/scan-calibration.json'
CHANNELS = [f'ac_{station}_{axis}' for station in (0, 1) for axis in 'xy']


def prepare_fields(solution, spacing):
    for key in CHANNELS:
        try:
            Field(solution, key, step=spacing)
        except FileNotFoundError:
            tangent(solution, CHANNELS, spacing)
            break
    original = load_bias(solution)
    pair_bias = {k: (v if k.startswith(('mini_condenser/', 'objective_lens/')) else 0.)
                 for k, v in original.items()}
    isolated = WORK / ('global-' + digest(pair_bias)[:16] + '.npz')
    if not isolated.exists():
        Circuit(Global(9)).solve(pair_bias, WORK/'global-0611340872b558b1.npz', isolated)
    if load_bias(isolated) != pair_bias:
        raise ValueError('Isolated pair excitation does not match the working point')
    bore.process(isolated, ['dc'], spacing)
    return isolated


def verify(folder, count, *, pivot_only=False):
    record = json.loads((folder/'tracking.json').read_text())
    solution, spacing = Path(record['solution']), record['bore_step_mm']
    source, dc = source_and_electrodes(), Field(solution, step=spacing)
    pair = None if pivot_only else prepare_fields(solution, spacing)
    seed = launch(source, dc, count)
    post = record['waists_mm'][3]
    model = Column(source, dc, seed, count, record['step_mm'], observation_mm=(post,))
    mix = np.asarray(json.loads(MIX_FILE.read_text())['mix'])
    saved = folder/'ac-pivot.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        if (result['total_field_sha256']!=dc.receipt['solution_sha256'] or result['particles']!=count
                or result['postscan_static_waist_mm']!=post or not np.array_equal(result['pivot']['mix'],mix)):
            raise ValueError('Saved pivot differs from the requested field, cohort or mixing')
    else:
        trace = MaterialScan(model, [Field(solution, k, step=spacing) for k in CHANNELS], from_source=True)
        with patch('joint_pivot_tuning.DrivenTrace', lambda m: trace):
            pivot = scan_response(model, None, mix, post)
        response_at_waist = pivot.pop('front_focus_response')
        downstream = [x for x in pivot['candidates'] if 1420. < x['z_mm'] < 1512.5]
        closest = min(downstream, key=lambda p: abs(p['z_mm']-post)) if downstream else None
        result = dict(postscan_static_waist_mm=post, registration='Prior calibration kept fixed for both signed commands',
                      calibration_source=str(MIX_FILE.relative_to(HERE.parent)), pivot=pivot, postscan_pivot=closest,
                      response_at_static_waist=response_at_waist,signed_commands_from_effective_source=True,
                      static_scan_current_A=[0.]*4,particles=count,total_field_solution=solution.name,
                      total_field_sha256=dc.receipt['solution_sha256'])
        saved.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if pivot_only:
        return
    isolated = Field(pair, step=spacing)
    if isolated.receipt['mesh_id'] != dc.receipt['mesh_id']:
        raise ValueError('Different magnetic meshes')
    model.pair = model.axis_field(isolated)
    result.update(isolated_pair_fronts=model.fronts(),isolated_pair_solution=pair.name)
    (folder/'ac-and-front-planes.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Pivot near waist',result['postscan_pivot'],'isolated fronts',result['isolated_pair_fronts'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--count', type=int, default=2048)
    parser.add_argument('--pivot-only', action='store_true')
    args = parser.parse_args()
    verify(HERE/args.label, args.count, pivot_only=args.pivot_only)
