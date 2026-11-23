"""Check the compact field, three-dimensional forces and first-impact logic."""
import json

import numpy as np

from gun_field import HERE, study_field_reader
from temsim.physics.axis_field_interpolation import compiled_evaluate
from transport import field_at, first_hit


def verify():
    ef, receipt = study_field_reader()('anode', True, 1200)
    r, z, phi, cut_index, compact, origins, inverse, gradients, phi0, core, slope = ef
    cut = cut_index >= 0
    lookup = np.full((2 * cut.size, 2), -1, dtype=np.int32)
    lookup[np.repeat(cut.ravel(), 2)] = compact
    rng = np.random.default_rng(161124)
    # Include the conforming tip and its metal side, the near-axis core, bore
    # boundaries, field fringe, and positions far from the emitter in the accelerator.
    near = rng.uniform([-1e-7, -1e-7, -1.2e-7], [1e-7, 1e-7, 2e-7], (512, 3))
    bore = rng.uniform([-.004, -.004, .0003], [.004, .004, .449], (512, 3))
    axis = np.zeros((256, 3))
    axis[:, 2] = np.geomspace(1e-11, .449, len(axis))
    sites = np.vstack((near, bore, axis))
    v_ref, e_ref = compiled_evaluate(sites, r, z, phi, cut, lookup, origins, inverse,
                                     gradients, phi0, core, slope)
    values = np.array([field_at(*point, ef) for point in sites])
    np.testing.assert_allclose(values[:, 0], v_ref, atol=2e-8, rtol=1e-13)
    np.testing.assert_allclose(values[:, 1:], e_ref, atol=.03, rtol=5e-10)
    # The metal potential is fixed. The hole contains vacuum with a field.
    for part, bias in [('gun_aperture', 4000.), ('exit_anode', 4000.), ('electrostatic_lens', 1200.)]:
        row = next(q for q in receipt['request']['rings'] if q['key'] == part)
        centre = (row['start_m'] + row['stop_m']) / 2
        metal = field_at(row['inner_m'] + .001, 0., centre, ef)
        np.testing.assert_allclose(metal, [bias, 0., 0., 0.], atol=2e-7)
        assert abs(field_at(0., 0., centre, ef)[0] - bias) > 5.
    assert field_at(0., 0., .008, ef)[3] > 0. and field_at(0., 0., .022, ef)[3] < 0.
    assert field_at(.0001, 0., .018, ef)[1] > 0.
    # Annulus dimensions are arbitrary and these arrays carry no physical source current.
    parts = np.array([[1., 1.002, 1., 3.], [1.5, 1.55, .5, 3.]])
    cases = [
        ((2., 0., 0., 0., 0., 2.), .5, 0),
        ((.7, 0., 0., 0., 0., 2.), .75, 1),
        ((2., 0., 2., 0., 0., -2.), .225, 1),
        ((0., 0., 1.001, 0., 2., 0.), .5, 0),
        ((4., 0., 1.001, -2., 0., 0.), .5, 0),
        ((.25, 0., 0., 0., 0., 2.), 2., -1),
    ]
    for segment, fraction, surface in cases:
        t, owner = first_hit(*segment, parts)
        assert owner == surface
        np.testing.assert_allclose(t, fraction, atol=1e-13)
    report = dict(field_points=len(sites), first_hit_cases=len(cases),
                  maximum_phi_difference_V=float(np.max(abs(values[:, 0] - v_ref))),
                  maximum_E_component_difference_V_per_m=float(np.max(abs(values[:, 1:] - e_ref))),
                  tested_surfaces=['gun_aperture', 'exit_anode', 'electrostatic_lens'])
    (HERE / 'solver-checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    verify()
