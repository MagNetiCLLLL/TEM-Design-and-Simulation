"""Bounded field-only stigmator response study; never installs a new field.

The wire path is an explicitly assumed representative rectangular loop halfway
between each saved bulk frame's inner and outer edges. It is not an OEM turn
path or an envelope-homogenized source. Values are normalized to one turn per
coil and one ampere per four-coil channel. All sources are closed loops.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.constants import mu_0

from temsim.assembly_catalog import AssemblyCatalog
from temsim.cad_multipoles import _stigmator_windings
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.core import electron
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.simulation_modes import switch_mode

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent / 'validation/stigmator-current-study.json'


def segment_field(points_m, start_m, end_m):
    """Exact finite straight-wire Biot-Savart field for 1 A, SI units."""
    direction = np.asarray(end_m)-start_m
    length = np.linalg.norm(direction)
    unit = direction/length
    delta = np.asarray(points_m)-start_m
    s = delta @ unit
    transverse = delta-s[..., None]*unit
    distance2 = np.sum(transverse*transverse, axis=-1)
    if np.any(distance2 <= 0):
        raise ValueError('Study observations must not be collinear with a wire')
    t = s-length
    scale = mu_0/(4*np.pi*distance2)*(s/np.sqrt(distance2+s*s)-t/np.sqrt(distance2+t*t))
    return np.cross(unit, transverse)*scale[..., None]


def loop_field(points_m, vertices_m):
    return sum(segment_field(points_m, a, b)
               for a, b in zip(vertices_m, np.roll(vertices_m, -1, axis=0)))


def source_loops(part):
    result = []
    for index, winding in enumerate(_stigmator_windings(part)):
        frame = winding.shape['shape']
        centre = np.asarray(frame['base']['center_mm'], float)
        _, outer_width, outer_length = frame['base']['size_mm']
        _, inner_width, inner_length = frame['tools'][0]['size_mm']
        # Midway between the frame edges, not the unknown conductor diameter.
        width = (outer_width+inner_width)/2
        length = (outer_length+inner_length)/2
        vertices = centre+np.array([[0, -width/2, -length/2],
            [0, width/2, -length/2], [0, width/2, length/2],
            [0, -width/2, length/2]])
        rotation = np.asarray(winding.shape['rotation'], float)
        vertices = vertices @ rotation.T
        # Work around the material centre. The local axial origin is arbitrary.
        vertices[:, 2] -= centre[2]
        sign = (-1)**(index//2)
        result.append(dict(group=winding.parameters['coil_group'],
            azimuth_deg=winding.parameters['coil_azimuth_deg'],
            sign=sign, vertices_m=vertices*1e-3))
    return result


def channel_field(points_m, loops, group):
    return sum(source['sign']*loop_field(points_m, source['vertices_m'])
               for source in loops if source['group'] == group)


def gradient_projection(field, z_m, h_m):
    z = np.asarray(z_m, float)
    points = np.zeros((len(z), 4, 3))
    points[..., 2] = z[:, None]
    points[:, 0, 0] = h_m
    points[:, 1, 0] = -h_m
    points[:, 2, 1] = h_m
    points[:, 3, 1] = -h_m
    b = field(points.reshape(-1, 3)).reshape(len(z), 4, 3)
    dx, dy = (b[:, 0]-b[:, 1])/(2*h_m), (b[:, 2]-b[:, 3])/(2*h_m)
    return np.stack(((dx[:, 1]+dy[:, 0])/2,
                     (dy[:, 1]-dx[:, 0])/2), axis=-1)


def forbidden(*args, **kwargs):
    raise AssertionError('This study must not emit or transport particles')


def run():
    # An independent closed-square analytic normalization and direction check.
    half_side = .012
    square = np.array([[-half_side, -half_side, 0], [half_side, -half_side, 0],
        [half_side, half_side, 0], [-half_side, half_side, 0]])
    actual = loop_field(np.array([[0., 0., 0.]]), square)[0]
    expected = np.array([0., 0., np.sqrt(2)*mu_0/(np.pi*half_side)])
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-20)

    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, 'unified')
    apply_operating_mode_pair(state, 'nano_probe', 'diffraction',
        column_name=selection.column, recording_name=selection.recording)
    apply_physical_layout_to_state(state)
    before = json.dumps(state.to_dict(), sort_keys=True)
    original_snapshot = capture_instrument_snapshot(state)
    charge, momentum, _ = electron(state)
    by_key = {p.key: dict(p.data) for p in state._resolved_assembly.parts}
    components = tuple(state.stigmators)
    frozen = {f.component_key: f for f in capture_posed_column_fields(state,
        include_aligned=True, component_keys=tuple(c.key for c in components))}
    records = []
    derivative_steps = (1e-5, 5e-6, 2.5e-6, 1.25e-6, 6.25e-7)
    for component in components:
        part = by_key[component.key]
        if part.get('coil_topology') != 'two_interleaved_four_pole_windings':
            continue
        loops = source_loops(part)
        c_steps = []
        for h in derivative_steps:
            c_steps.append(np.column_stack([gradient_projection(
                lambda p, group=group: channel_field(p, loops, group), [0.], h)[0]
                for group in ('x', 'y')]))
        response = c_steps[-1]
        relative_change = np.max(np.abs(c_steps[-1]-c_steps[-2]))/np.max(np.abs(response))
        assert relative_change < 1e-7, relative_change
        assert np.linalg.cond(response) < 1.000001
        axis_b = [channel_field(np.array([[0., 0., 0.]]), loops, g)[0]
                  for g in ('x', 'y')]
        assert np.max(np.abs(axis_b)) < 1e-16
        np.testing.assert_allclose(response[0, 0], 0., atol=1e-14)
        np.testing.assert_allclose(response[1, 1], 0., atol=1e-14)
        np.testing.assert_allclose(response[0, 1], -response[1, 0], rtol=1e-8)

        n, _, s = map(float, component.quadrupole_tensor_m2(component.z_mm))
        target = np.array([n, s])*momentum/charge
        basis_angles = np.deg2rad(2*np.array([
            component.channel_x_angle_deg, component.channel_y_angle_deg]))
        reference_k = component.max_strength_m2*np.stack((
            np.cos(basis_angles), np.sin(basis_angles)))
        reference_gradient = reference_k*momentum/charge
        original_commands = (component.strength_x_percent, component.strength_y_percent)
        nonzero_native_checks = []
        for channel_index, channel in enumerate(('x', 'y')):
            detached = original_snapshot.restore()
            detached_component = next(c for c in detached.stigmators if c.key == component.key)
            assert detached_component is not component
            detached_component.strength_x_percent = 100. if channel == 'x' else 0.
            detached_component.strength_y_percent = 100. if channel == 'y' else 0.
            detached_fields = capture_posed_column_fields(detached, include_aligned=True,
                component_keys=(component.key,))
            assert len(detached_fields) == 1
            nonzero_projected = gradient_projection(
                detached_fields[0].field_at_global_positions_t,
                [detached_component.z_mm*1e-3], 1e-6)[0]
            expected_nonzero = reference_gradient[:, channel_index]
            assert np.max(np.abs(nonzero_projected)) > 0
            np.testing.assert_allclose(nonzero_projected, expected_nonzero,
                rtol=1e-13, atol=1e-15)
            nonzero_native_checks.append(dict(channel=channel,
                detached_commands_percent=[detached_component.strength_x_percent,
                    detached_component.strength_y_percent],
                projected_signed_gradient_t_per_m=nonzero_projected.tolist(),
                expected_signed_gradient_t_per_m=expected_nonzero.tolist(),
                nonzero=True, original_component_untouched=(
                    (component.strength_x_percent, component.strength_y_percent) == original_commands)))
        native = frozen[component.key]
        projected = gradient_projection(native.field_at_global_positions_t,
            [component.z_mm*1e-3], 1e-6)[0]
        np.testing.assert_allclose(projected, target, rtol=1e-13, atol=1e-15)
        illustrative_current = np.linalg.solve(response, target)
        z = np.linspace(-.100, .100, 4001)
        profile = gradient_projection(lambda p: channel_field(p, loops, 'y'), z, 6.25e-7)[:, 0]
        integral = float(np.trapezoid(profile, z))
        fwhm_indices = np.flatnonzero(np.abs(profile) >= abs(profile[len(z)//2])/2)
        assert np.all(np.diff(fwhm_indices) == 1), 'FWHM must describe one connected central lobe'
        fwhm_mm = float((z[fwhm_indices[-1]]-z[fwhm_indices[0]])*1000)
        sigma_m = component.length_mm/2.355*1e-3
        native_integrated_per_peak = np.sqrt(2*np.pi)*sigma_m
        records.append(dict(key=component.key, enabled=component.enabled,
            native_reference_z_mm=component.z_mm,
            native_effective_fwhm_mm=component.length_mm,
            native_maximum_strength_m2=component.max_strength_m2,
            native_commands_percent=[component.strength_x_percent, component.strength_y_percent],
            native_peak_normal_skew_m2=[n, s], native_signed_gradient_t_per_m=target.tolist(),
            native_100percent_separate_channel_tensor_m2=reference_k.tolist(),
            native_100percent_separate_channel_signed_gradient_t_per_m=reference_gradient.tolist(),
            detached_nonzero_native_gradient_checks=nonzero_native_checks,
            engineering_ampere_turns_to_match_separate_native_100percent_channels=np.linalg.solve(
                response, reference_gradient).tolist(),
            native_field_projection_t_per_m=projected.tolist(),
            engineering_one_turn_coils=[{**v, 'vertices_m':v['vertices_m'].tolist()} for v in loops],
            derivative_steps_m=list(derivative_steps),
            response_matrices_t_per_m_per_ampere=[m.tolist() for m in c_steps],
            final_relative_derivative_change=float(relative_change),
            response_condition=float(np.linalg.cond(response)),
            centre_dipole_t_per_ampere=np.asarray(axis_b).tolist(),
            current_for_current_native_peak_if_one_turn_per_coil_a=illustrative_current.tolist(),
            physical_profile_central_gradient_t_per_m_per_ampere=float(profile[len(z)//2]),
            physical_profile_fwhm_mm=fwhm_mm,
            physical_profile_integral_minus100_plus100mm_t_per_ampere=integral,
            physical_integral_over_peak_mm=integral/float(profile[len(z)//2])*1000,
            native_gaussian_integral_over_peak_mm=float(native_integrated_per_peak*1000),
            profile_sample_step_mm=.05, profile_extent_mm=[-100., 100.]))
    assert json.dumps(state.to_dict(), sort_keys=True) == before
    result = dict(status='passed', scope='field_only_engineering_loop_study_not_installed_calibration',
        assumptions=['one representative closed rectangular centreline per physical coil',
            'each channel drives four coils with alternating +1 -1 +1 -1 ampere signs',
            'first coil normal is outward radial; phase and handedness are explicit assumptions',
            'no core, pole, material permeability, resistance, heat, insulation or leads',
            'not an envelope-homogenized pack and not an OEM winding measurement'],
        beam_voltage_kv=state.beam_voltage_kv, charge_c=charge, momentum_kg_m_per_s=momentum,
        signed_p_over_q_t_m=momentum/charge, magnetic_rigidity_t_m=momentum/abs(charge),
        input_selection=asdict(selection), source_emission_and_transport_forbidden=True,
        optical_state_unchanged=True, mu_0_h_per_m=mu_0,
        square_analytic_check_t=dict(actual=actual.tolist(), expected=expected.tolist()), records=records,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], beam_voltage_kv=state.beam_voltage_kv,
        magnetic_rigidity_t_m=result['magnetic_rigidity_t_m'],
        records=[{k:r[k] for k in ('key', 'native_peak_normal_skew_m2',
            'native_signed_gradient_t_per_m', 'final_relative_derivative_change',
            'physical_profile_fwhm_mm', 'physical_integral_over_peak_mm')}
            for r in records]), indent=2))


if __name__ == '__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        run()
