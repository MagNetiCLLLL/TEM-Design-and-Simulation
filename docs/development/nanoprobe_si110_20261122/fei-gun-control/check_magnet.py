"""Test the archived column field against its volume solution over the gun acceptance."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'cescor_direct_probe_20261121'))
from material_relay_column import Magnet
from material_relay_global import impressed
from temsim.physics.corrector_vacuum_field import _induced_cuda, _sources


def archived():
    folder = HERE.parent / 'c1-zoom' / 'transport-inputs'
    entry = json.loads((folder / 'index.json').read_text())['trials']['c1-zoom-near-c2-focused']
    path = folder / entry['bore_field']['file']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['bore_field']['sha256']
    with np.load(path) as data:
        return Magnet(data['z'], data['coef']), json.loads(str(data['receipt']))


def check():
    narrow, receipt = archived()
    volume = HERE.parents[3] / 'outputs/material-relay-20261122' / receipt['source']
    with volume.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['solution_sha256']
    with np.load(volume) as data:
        centres, moments = _sources(data['corners_m'], data['moments_Am2'], 4)
        recipe = json.loads(str(data['recipe']))['sources']
    samples = []
    for z in np.unique(np.r_[np.arange(0., 370., 14), np.arange(370., 451., 4), 450.]):
        maximum = 8.0 if z < 370 else 3.0
        angle = .173 * z + np.arange(7) * .73
        radius = np.array([0., .18, .42, .67, .8, 1., 1.]) * maximum
        samples.extend(np.stack((radius*np.cos(angle), radius*np.sin(angle), np.full_like(radius,z)), axis=1))
    check_points = np.asarray(samples) * .001
    z = np.arange(-2., 452.1, 2.) * .001
    scale = .008
    grid = np.linspace(-1., 1., 5)
    xy = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    terms = [(i,d-i) for d in range(5) for i in range(d+1)]
    def basis(points):
        return np.column_stack([points[:,0]**i * points[:,1]**j for i,j in terms])
    sites = np.column_stack((np.tile(xy*scale,(len(z),1)),np.repeat(z,len(xy))))
    local = np.vstack((sites,check_points))
    local[:, 2] -= .45
    direct = np.zeros_like(local)
    for phi in np.arange(3) * 2*np.pi/3:
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        p = local.copy()
        p[:,:2] = local[:,:2] @ rot.T
        b = _induced_cuda(p, centres, moments)
        for key, amp in receipt['bias'].items():
            if amp:
                b += amp * impressed(recipe[key], p)
        b[:,:2] = b[:,:2] @ rot
        direct += b/3
    training, direct = np.split(direct, [len(sites)])
    coeff = np.einsum('ij,zjc->zic',np.linalg.pinv(basis(xy)),training.reshape(len(z),len(xy),3))
    fitted = np.einsum('ni,nic->nc',basis(check_points[:,:2]/scale),CubicSpline(z,coeff,axis=0)(check_points[:,2]))
    xyz = check_points
    diff = np.linalg.norm(fitted-direct,axis=1)
    output = dict(volume_sha256=receipt['solution_sha256'], coordinates='xyz in mm from tip',
                  reference='four quadrature points per material cell plus impressed currents, previous C3 symmetry',
                  samples=len(xyz), peak_error_T=float(diff.max()), positions_mm=(xyz*1000).tolist(),
                  direct_B_T=direct.tolist(), approximation_B_T=fitted.tolist())
    centre = xyz[np.hypot(xyz[:,0],xyz[:,1])==0.]
    previous = narrow.field_at_global_positions_t(centre)
    output['old_axis_max_difference_T'] = float(np.max(np.linalg.norm(previous-CubicSpline(z,coeff[:,0])(centre[:,2]),axis=1)))
    np.savez_compressed(HERE/'gun-volume-field.npz',z_m=z,coeff=coeff,terms=terms,scale_m=scale,
                        volume_sha256=receipt['solution_sha256'])
    (HERE / 'magnetic-check.json').write_text(json.dumps(output,separators=(',', ':'))+'\n')
    for lo,hi in ((0,.035),(.035,.37),(.37,.451)):
        sel=(xyz[:,2] >= lo) & (xyz[:,2] < hi)
        print(lo,hi,'max error T',diff[sel].max(),'peak B T',np.max(np.linalg.norm(direct[sel],axis=1)),flush=True)


if __name__ == '__main__':
    check()
