"""Electrostatic geometry for the opt-in FEI cold-gun comparison (SI)."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / 'outputs' / 'fei-gun-acceptance-20261123'
sys.path.insert(0, str(HERE.parent / 'gun-lens-24mm'))
from gun_focus import prepare
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.continuous_curvature_conductor import ContinuousCurvatureConductor
from temsim.physics.continuous_gun_field import continuous_field_request, mesh_axes, _boundary_arrays
from temsim.physics.planar_gun_field import request_digest


def study_field_reader():
    # Archived column studies also provide modules called "scan".
    return runpy.run_path(str(HERE / 'scan.py'))['load_field']


def ring(key, centre_mm, length_mm, inner_mm, outer_mm, potential):
    return dict(key=key, start_m=(centre_mm-length_mm/2)*.001, stop_m=(centre_mm+length_mm/2)*.001,
                inner_m=inner_mm*.001, outer_m=outer_mm*.001, potential_rise_v=float(potential))


def request_for(gun, voltage, wiring, fine):
    # The same divider is fed from the fixed exit-anode bias (+4 kV, numerically
    # equal to the independently held extractor supply) in all comparisons.
    gun.accelerator.voltage_start_reference = 'extractor'
    gun.electrostatic_lens.voltage_kv = voltage / 1000
    gun.emitter.curvature_nm_inv = .01
    request = continuous_field_request(gun, cells_per_bore=15 if fine else 12,
        exit_extension_mm=50., tip_nodes=195 if fine else 160, apex_cells_per_radius=72 if fine else 48)
    geometry = request['cathode_geometry']
    # This optical trial samples a physical area set by the emission law.
    geometry['emission_support_radius_nm'] = geometry['apex_radius_nm'] * np.sin(np.deg2rad(80.))
    request['boundary_conditions']['cathode']['geometry'] = geometry.copy()
    aperture = gun.dpa_aperture
    assert aperture.enabled and (aperture.z_mm, aperture.radius_mm, aperture.offset_x_mm, aperture.offset_y_mm) == (24., 2., 0., 0.)
    if wiring not in ('anode', 'lens', 'previous'):
        raise ValueError('Unknown aperture connection')
    if wiring != 'previous':
        aperture_v = 4000. if wiring == 'anode' else voltage
        request['rings'].append(ring('gun_aperture', aperture.z_mm, aperture.plate_thickness_mm,
            aperture.radius_mm, aperture.mechanical_outer_diameter_mm/2., aperture_v))
        # Fig. 2: lens-to-anode midplane separation about 16 mm. Dimensions
        # below are rectangular axisymmetric approximations read from the drawing.
        request['rings'].append(ring('exit_anode', 34., 5., 4.5, 50., 4000.))
    request['reference_case'] = dict(patent='US20180323036A1 Fig. 2', aperture_supply=wiring,
        accelerator_start_V=4000., aperture_annular_equivalent=True, lens_bias_V=voltage)
    return request


def conductor_stops(gun, request):
    rows = list(request['rings'])
    # Even in the former field topology the 24 mm mechanical plate absorbs primaries.
    if not any(r['key'] == 'gun_aperture' for r in rows):
        a = gun.dpa_aperture
        rows.append(ring('gun_aperture',a.z_mm,a.plate_thickness_mm,a.radius_mm,a.mechanical_outer_diameter_mm/2,0.))
    a = gun.c1_aperture
    rows.append(ring('C1_aperture',a.z_mm,a.plate_thickness_mm,a.radius_mm,a.mechanical_outer_diameter_mm/2,0.))
    for component, key in ((gun.accelerator, 'accelerator_bore'),(gun.deflector,'gun_deflector_bore'),(gun.stigmator,'gun_stigmator_bore')):
        rows.append(ring(key,component.mechanical_center_from_tip_mm,component.mechanical_length_mm,
            component.mechanical_clear_bore_diameter_mm/2,component.mechanical_outer_diameter_mm/2,0.))
    # Rays that miss the outside rim of the flight tube cannot be counted as
    # transported beam. The housing that finally collects them is unspecified.
    entrance = min((r for r in request['rings'] if r['key'].startswith('grounded_outlet:')),
                   key=lambda r: r['start_m'])
    rows.append(dict(key='outside_outlet_entrance', start_m=entrance['start_m'],
                     stop_m=entrance['start_m'], inner_m=entrance['outer_m'],
                     outer_m=max(r['outer_m'] for r in rows)))
    return rows


def key(wiring, fine, voltage):
    return f'{wiring}-mesh{15 if fine else 12}-{voltage:g}V'


def solve(gun, voltage, wiring, fine):
    name = key(wiring, fine, voltage)
    request = request_for(gun, voltage, wiring, fine)
    digest = request_digest(request)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / (name+'.npz')
    if path.exists():
        with np.load(path) as d:
            if str(d['sha']) == digest:
                receipt = HERE / (name+'.json')
                if receipt.exists():
                    record = json.loads(receipt.read_text())
                    stops = conductor_stops(gun,request)
                    if stops != record['stops']:
                        record['stops'] = stops
                        receipt.write_text(json.dumps(record,indent=2)+'\n')
                    return path
    g = ContinuousCurvatureConductor(**request['cathode_geometry']).validate()
    r,z = mesh_axes(request)
    print('FEM',name,len(r),len(z),flush=True)
    fixed, values = _boundary_arrays(request,r,z,g)
    fem = AxisymmetricCutField(r,z,fixed,values,geometry=g,tolerance=1e-9)
    for row in request['rings']:
        if row['key'] in ('gun_aperture','exit_anode','electrostatic_lens'):
            radial = (r >= row['inner_m']) & (r <= row['outer_m'])
            axial = (z >= row['start_m']) & (z <= row['stop_m'])
            assert np.all(fem.nodal_voltage[np.ix_(radial,axial)] == row['potential_rise_v'])
            assert not np.any(fixed[0,axial])
    # Only the cells cut by the emitter require triangle queries. Compacting
    # these indices leaves the uncut bilinear FEM representation identical.
    parents = np.repeat(fem.cut_cells.ravel(),2)
    triangle_ids = np.unique(fem.lookup[parents]); triangle_ids = triangle_ids[triangle_ids>=0]
    selected = fem.lookup[parents]
    compact = np.where(selected>=0,np.searchsorted(triangle_ids,selected),-1)
    indices = np.full(fem.cut_cells.shape,-1,np.int32)
    indices[fem.cut_cells] = np.arange(fem.cut_cells.sum())
    np.savez_compressed(path,r=r,z=z,voltage=fem.nodal_voltage,cut_index=indices,lookup=compact,
        origin=fem.origin[triangle_ids],inverse=fem.inverse[triangle_ids],grad=fem.gradient[triangle_ids],
        phi0=fem.phi0[triangle_ids],surface_s=fem.boundary_s,surface_z=fem.boundary_z,sha=digest)
    record=dict(request=request,sha=digest,vertices=fem.vertex_count,residual=float(fem.residual),
                stops=conductor_stops(gun,request))
    (HERE/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print('Solved',name,fem.residual,flush=True)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fine',action='store_true')
    p.add_argument('--voltages',type=float,nargs='+',help='Select physical lens biases in volts.')
    p.add_argument('--wiring',choices=['anode','lens','previous'])
    args=p.parse_args()
    gun,_=prepare()
    for wiring,voltages in [('anode',[1200,50,500,1000,1500,1800,2000]),('lens',[1200,50,2000]),('previous',[1200])]:
        if args.wiring and wiring != args.wiring:
            continue
        if args.fine and not args.wiring and wiring != 'anode':
            continue
        for voltage in args.voltages or voltages:
            solve(gun,voltage,wiring,args.fine)
