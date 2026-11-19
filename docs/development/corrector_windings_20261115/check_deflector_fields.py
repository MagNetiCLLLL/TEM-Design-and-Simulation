"""Bounded source-free deflector field-scale and centroid-path checks.

This research calculation does not install a current source in production.
The representative centreline is explicitly an engineering approximation.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.core import electron
from temsim.physics.instrument_magnetic import column_dipole_fields
from temsim.simulation_modes import switch_mode

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/corrector-windings-20261115/deflector-field-check.json'
MU0 = 4e-7 * math.pi


def forbidden(*args, **kwargs):
    raise AssertionError('This check must not emit or propagate particles.')


def field_pair(points, radius, length, phi, order):
    """Closed opposed filament saddles, N=1 each and I=1 A in series.

    The second saddle has reversed winding sense to reinforce the on-axis
    dipole. The first has its +Z straight conductor at azimuth +phi.
    """
    xg, wg = np.polynomial.legendre.leggauss(order)
    result = np.zeros_like(points)
    h = length / 2
    for center, sense in ((0., 1.), (math.pi, -1.)):
        for angle, lo, hi in ((center+phi, -h, h), (center-phi, h, -h)):
            z = .5*(hi+lo) + .5*(hi-lo)*xg
            paths = np.column_stack((np.full(order, radius*math.cos(angle)),
                                     np.full(order, radius*math.sin(angle)), z))
            dl = np.zeros((order, 3)); dl[:, 2] = .5*(hi-lo)*wg*sense
            displacement = points[:, None, :] - paths[None, :, :]
            result += (MU0/(4*math.pi)*np.sum(np.cross(dl, displacement)
                /np.linalg.norm(displacement, axis=2)[..., None]**3, axis=1))
        for z, lo, hi in ((h, center+phi, center-phi), (-h, center-phi, center+phi)):
            angles = .5*(hi+lo) + .5*(hi-lo)*xg
            paths = np.column_stack((radius*np.cos(angles), radius*np.sin(angles), np.full(order, z)))
            tangents = np.column_stack((-radius*np.sin(angles), radius*np.cos(angles), np.zeros(order)))
            dl = tangents*(.5*(hi-lo)*wg*sense)[:, None]
            displacement = points[:, None, :] - paths[None, :, :]
            result += (MU0/(4*math.pi)*np.sum(np.cross(dl, displacement)
                /np.linalg.norm(displacement, axis=2)[..., None]**3, axis=1))
    return result


def run():
    catalog = AssemblyCatalog(); selection = catalog.default_selection()
    state = default_state(); catalog.apply(state, selection); switch_mode(state, 'unified')
    apply_operating_mode_pair(state, 'nano_probe', 'diffraction',
        column_name=selection.column, recording_name=selection.recording)
    apply_physical_layout_to_state(state)
    before = deepcopy(state.to_dict())
    charge, momentum, _ = electron(state)
    native = type(state).from_dict(before, resolved_assembly=state._resolved_assembly)
    target = next(c for c in native.corrector_elements if c.key == 'probe_dph2_deflector')
    target.enabled = True; target.kick_x_mrad = 1.; target.kick_y_mrad = 0.
    fields = [f for f in column_dipole_fields(native) if f.key.startswith(target.key+':')]
    assert len(fields) == 1
    f = fields[0]
    length = f.upper_m-f.lower_m
    expected_by = -momentum/charge*.001/length
    np.testing.assert_allclose(f.by_t, expected_by, rtol=1e-13)
    assert f.bx_t == 0. and f.by_t > 0.
    assert state.to_dict() == before
    native_receipt = dict(selection=asdict(selection), beam_voltage_kv=state.beam_voltage_kv,
        reference_momentum_kg_m_per_s=momentum, signed_charge_c=charge,
        positive_rigidity_Tm=momentum/abs(charge),
        detached_command_example=dict(component=target.key, x_mrad=1., y_mrad=0.,
            effective_length_mm=target.effective_thickness_mm, bx_t=f.bx_t, by_t=f.by_t,
            integrated_bx_Tm=f.bx_t*length, integrated_by_Tm=f.by_t*length),
        original_state_exactly_preserved=True)

    config = ROOT/'configs/instruments/column/C3_ProbeCorrector.toml'
    parts = tomllib.loads(config.read_text(encoding='utf-8'))['parts']
    rows = []
    for p in parts:
        if 'winding_layer_thickness_mm' not in p:
            continue
        copper_length = p.get('winding_axial_length_mm', p['local_end_z_mm']-p['local_start_z_mm'])
        band = p['winding_band_width_mm']; axis_length = (copper_length-band)*1e-3
        pair_rows = []
        for radius_mm in (p['mechanical_inner_diameter_mm']/2+p['winding_layer_thickness_mm']/2,
                          p['mechanical_outer_diameter_mm']/2-p['winding_layer_thickness_mm']/2):
            radius = radius_mm*1e-3
            phi = math.radians(p['winding_angular_span_deg'])/2-band/(2*radius_mm)
            integral = 2*MU0*axis_length*math.sin(phi)/(math.pi*radius)
            pair_rows.append(dict(radius_mm=radius_mm, straight_length_mm=axis_length*1e3,
                half_angle_rad=phi, integral_Tm_per_Aturn=integral,
                Aturn_per_reference_1mrad=(momentum/abs(charge)*.001/integral)))
        rows.append(dict(host=p['key'], copper_length_mm=copper_length, pairs=pair_rows))

    ref = rows[0]['pairs'][0]
    a = ref['radius_mm']*1e-3; ell = ref['straight_length_mm']*1e-3
    phi = ref['half_angle_rad']; h = ell/2
    def on_axis(z):
        return MU0*math.sin(phi)/math.pi * (
            ((h-z)/math.sqrt(a*a+(h-z)**2)+(h+z)/math.sqrt(a*a+(h+z)**2))/a
            +a*((z+h)/(a*a+(z+h)**2)**1.5-(z-h)/(a*a+(z-h)**2)**1.5))
    value, error = quad(on_axis, -math.inf, math.inf, epsabs=1e-15, epsrel=1e-11)
    np.testing.assert_allclose(value, ref['integral_Tm_per_Aturn'], rtol=2e-10)
    points = np.array([[0., 0., 0.], [0., 0., .5*ell], [.00025, .0004, -.002], [-.001, .0008, .005]])
    records = []
    reference = field_pair(points, a, ell, phi, 64)
    assert np.max(np.abs(reference[:, 0])) > 0
    for order in (4, 8, 16, 32):
        actual = field_pair(points, a, ell, phi, order)
        records.append(dict(gauss_order_per_segment=order,
            b_T_per_Aturn=actual.tolist(), max_abs_error_T_per_Aturn=float(np.max(np.abs(actual-reference))),
            peak_normalized_error=float(np.max(np.abs(actual-reference))/np.max(np.abs(reference)))))
    assert records[1]['peak_normalized_error'] < records[0]['peak_normalized_error']
    assert records[2]['peak_normalized_error'] < records[1]['peak_normalized_error']
    assert records[-1]['peak_normalized_error'] < 1e-10
    np.testing.assert_allclose(reference[0], [on_axis(0.), 0., 0.], atol=1e-18, rtol=1e-12)
    np.testing.assert_allclose(field_pair(points, a, ell, phi, 32), reference, atol=1e-14, rtol=1e-9)
    receipt = dict(scope='field-only research; no installed current-source change or experimental validation',
        native=native_receipt, centroid_assumption='one representative closed filament per saddle; N=1 each, opposed pair current senses reinforce; no leads or finite conductor averaging',
        geometry_config=str(config.relative_to(ROOT)), geometry_config_sha256=sha256(config.read_bytes()).hexdigest(),
        centroid_scales=rows, scalar_full_axis_check=dict(numerical_Tm_per_Aturn=value,
            analytic_Tm_per_Aturn=ref['integral_Tm_per_Aturn'], quad_error_estimate=error,
            relative_difference=(value-ref['integral_Tm_per_Aturn'])/ref['integral_Tm_per_Aturn']),
        centroid_vector_convergence=dict(points_m=points.tolist(), reference_gauss_order=64,
            reference_b_T_per_Aturn=reference.tolist(), doubled_quadrature=records),
        source_emission_forbidden=True, particle_propagation_forbidden=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(OUT), native=native_receipt,
        convergence=[(r['gauss_order_per_segment'], r['peak_normalized_error']) for r in records]), indent=2))


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        run()
