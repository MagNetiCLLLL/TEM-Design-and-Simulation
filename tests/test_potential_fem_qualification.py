"""Bounded P1 potential qualification with independent analytic references.

All quantities are SI. These synthetic, linear, static axisymmetric problems
test the existing solver and the preserved triangular potential together.
They are not an instrument, nonlinear material, wave, or OEM qualification.

The acceptance budgets deliberately remain coarse: 1% relative RMS potential,
5% field/interface response, and decreasing errors under refinement. They
detect a wrong basis, constitutive coefficient, or interface convention; they
are not an electron-optical phase-error budget. Relative norms always use the
fixed nonzero analytic reference, never a changing numerical result.
"""
from __future__ import annotations

import json

import numpy as np

from temsim.physics.axisymmetric_magnetostatics import MU0, solve_axisymmetric
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential


def _relative_norm(value, reference):
    return float(np.linalg.norm(value) / np.linalg.norm(reference))


def _capture(r, z, permeability, current, *, source):
    """Use the real FEM solve, retaining its A_phi rather than fitting nodal B."""
    solution = solve_axisymmetric(r, z, permeability, current)
    assert solution.relative_residual < 1e-7
    potential = AxisymmetricMagneticPotential.from_solution(
        solution,
        source_id=source,
        provenance={
            "solver": "temsim.physics.axisymmetric_magnetostatics.solve_axisymmetric",
            "basis": "triangular P1 A_phi",
            "boundary": "A_phi=0 on the axis and remote radial/axial faces",
            "source": source,
            "scope": "synthetic linear qualification fixture; not an instrument",
        },
    )
    np.testing.assert_array_equal(potential.a_phi_tm, solution.a_phi_tm)
    assert not np.shares_memory(potential.a_phi_tm, solution.a_phi_tm)
    return potential


class _MaterialInterfaceReference:
    """Manufactured solution with a cylindrical mu_r=1/5 interface.

    A_phi=f(r)*g(z), g=1-z^2/L^2. Inside r_i, f=c*r. Outside,
    f=a*r+b/r+d*r^3. The coefficients impose f(R)=0, continuous f(r_i),
    and continuous (f'+f/r)/mu at r_i. Thus B_r=-f*g' is continuous,
    and H_z=(f'+f/r)*g/mu is continuous without a free surface current.

    In each constant-permeability region, the independent strong equation is
    J_phi=-(f''+f'/r-f/r^2)*g/mu - f*g''/mu. Its radial term is zero
    inside and 8*d*r outside. Material-interface nodes are mesh-aligned, so
    the convergence experiment does not also move the material boundary.
    """

    outer_radius_m = .03
    interface_radius_m = .015
    half_length_m = .04
    inner_scale_t = .01
    outer_relative_permeability = 5.

    def __init__(self):
        radius, interface = self.outer_radius_m, self.interface_radius_m
        c, ratio = self.inner_scale_t, self.outer_relative_permeability
        self.d = -c*(ratio*radius**2+(1-ratio)*interface**2)/(radius**2-interface**2)**2
        self.a = c*ratio-2*self.d*interface**2
        self.b = interface**2*c*(1-ratio)+self.d*interface**4

    def radial_potential(self, r):
        inverse_r = np.divide(1., r, out=np.zeros_like(r), where=r != 0.)
        return np.where(r < self.interface_radius_m, self.inner_scale_t*r,
                        self.a*r+self.b*inverse_r+self.d*r**3)

    def permeability(self, r, z):
        return np.where(r < self.interface_radius_m, 1., self.outer_relative_permeability)

    def current(self, r, z):
        g = 1-(z/self.half_length_m)**2
        radial_operator = np.where(r < self.interface_radius_m, 0., 8*self.d*r)
        return (2*self.radial_potential(r)/self.half_length_m**2-radial_operator*g) / (
            MU0*self.permeability(r, z))

    def analytic(self, r, z):
        f = self.radial_potential(r)
        g = 1-(z/self.half_length_m)**2
        br = 2*z*f/self.half_length_m**2
        bz = np.where(r < self.interface_radius_m, 2*self.inner_scale_t,
                      2*self.a+4*self.d*r**2)*g
        return f*g, np.column_stack((br, np.zeros_like(br), bz))


def test_material_interface_potential_and_flux_converge_at_fixed_probes(record_testsuite_property):
    reference = _MaterialInterfaceReference()
    random = np.random.default_rng(14919)
    radial = np.r_[random.uniform(.08, .45, 32), random.uniform(.55, .92, 32)] * reference.outer_radius_m
    axial = random.uniform(-.8, .8, 64)*reference.half_length_m
    probes = np.column_stack((radial, np.zeros_like(radial), axial))
    exact_a, exact_b = reference.analytic(radial, axial)
    interface_z = np.array((-.73, -.47, -.19, .23, .51, .77))*reference.half_length_m
    _, interface_b = reference.analytic(np.full_like(interface_z, reference.interface_radius_m), interface_z)
    exact_h_t = 2*reference.inner_scale_t*(1-(interface_z/reference.half_length_m)**2)/MU0
    errors, flux_errors = [], []

    for count in (17, 33, 65):
        r = np.linspace(0., reference.outer_radius_m, count)
        z = np.linspace(-reference.half_length_m, reference.half_length_m, 2*count-1)
        assert np.count_nonzero(r == reference.interface_radius_m) == 1
        potential = _capture(r, z, reference.permeability, reference.current,
                             source="manufactured-cylindrical-material-interface")
        sample = potential.sample(probes)
        errors.append((_relative_norm(sample["A_t_m"][:, 1]-exact_a, exact_a),
                       _relative_norm(sample["B_t"]-exact_b, exact_b)))

        # Probe the two one-sided element limits. epsilon is far below one
        # element width and never changes the fixed physical interface.
        epsilon = (r[1]-r[0])*1e-7
        sides = [potential.sample(np.column_stack((
            np.full_like(interface_z, reference.interface_radius_m+sign*epsilon),
            np.zeros_like(interface_z), interface_z)))["B_t"] for sign in (-1, 1)]
        h_inside = sides[0][:, 2]/MU0
        h_outside = sides[1][:, 2]/(MU0*reference.outer_relative_permeability)
        normal_jump = _relative_norm(sides[1][:, 0]-sides[0][:, 0], interface_b[:, 0])
        normal_error = max(_relative_norm(side[:, 0]-interface_b[:, 0], interface_b[:, 0]) for side in sides)
        tangential_jump = _relative_norm(h_outside-h_inside, exact_h_t)
        tangential_error = max(_relative_norm(h_inside-exact_h_t, exact_h_t),
                               _relative_norm(h_outside-exact_h_t, exact_h_t))
        flux_errors.append((normal_jump, normal_error, tangential_jump, tangential_error))

    errors, flux_errors = np.asarray(errors), np.asarray(flux_errors)
    evidence = {"radial_nodes": [17, 33, 65], "potential_field_relative_errors": errors.tolist(),
                "interface_errors": flux_errors.tolist(),
                "interface_error_columns": ["B_normal_jump", "B_normal_error", "H_tangent_jump", "H_tangent_error"]}
    record_testsuite_property("material_interface_evidence", json.dumps(evidence, sort_keys=True))
    # Smooth-cell P1 potential is expected to converge faster than its first
    # derivative. Gates permit pre-asymptotic constants without accepting a
    # fixed wrong field. The normal trace is shared by adjacent P1 elements.
    assert np.all(errors[1:, 0] < .5*errors[:-1, 0]), evidence
    assert np.all(errors[1:, 1] < .75*errors[:-1, 1]), evidence
    assert errors[-1, 0] < .01 and errors[-1, 1] < .05, evidence
    assert np.all(flux_errors[:, 0] < 1e-6), evidence
    assert np.all(flux_errors[1:, 2:] < .65*flux_errors[:-1, 2:]), evidence
    assert np.all(flux_errors[-1, 1:] < .05), evidence


class _CompactCurrentReference:
    """Smooth compact current with an exact infinite-vacuum dipole exterior.

    Put u=(r^2+z^2)/a^2 and A_phi=B_scale*r*F(u). For u<=1,
    F=35/8-21*u/4+15*u^2/8; outside F=u^(-3/2). F, F' and F''
    are continuous at u=1. The strong equation gives the compact source
    J_phi=(105/2)*B_scale*r*(1-u)/(mu0*a^2) for u<1, zero outside.

    Every expanded domain contains the identical complete current source.
    Only its zero-potential remote boundary moves. This avoids confusing
    enlargement of a noncompact source with boundary truncation.
    """

    radius_m = .01
    field_scale_t = .004

    @staticmethod
    def permeability(r, z):
        return np.ones_like(r)

    def current(self, r, z):
        u = (r*r+z*z)/self.radius_m**2
        return np.where(u < 1., 52.5*self.field_scale_t*r*(1-u)/(MU0*self.radius_m**2), 0.)

    def analytic(self, probes):
        r, z = probes[:, 0], probes[:, 2]
        u = (r*r+z*z)/self.radius_m**2
        f = np.where(u < 1., 35/8-21/4*u+15/8*u*u, u**-1.5)
        derivative = np.where(u < 1., -21/4+15/4*u, -1.5*u**-2.5)
        br = -2*self.field_scale_t*r*z/self.radius_m**2*derivative
        bz = 2*self.field_scale_t*(f+r*r/self.radius_m**2*derivative)
        return self.field_scale_t*r*f, np.column_stack((br, np.zeros_like(r), bz))

    def probes(self):
        random = np.random.default_rng(1138)
        return np.column_stack((random.uniform(.12, 1.2, 32)*self.radius_m,
                                np.zeros(32), random.uniform(-1.2, 1.2, 32)*self.radius_m))

    def solve(self, extent, cells_per_source_radius):
        step = self.radius_m/cells_per_source_radius
        # Integer lattice coordinates keep the whole source grid identical
        # while adding exterior cells. linspace would permit tiny roundoff
        # changes to internal nodes when the remote boundary changes.
        edge = extent*cells_per_source_radius
        r = np.arange(edge+1)*step
        z = np.arange(-edge, edge+1)*step
        return _capture(r, z, self.permeability, self.current,
                        source="compact-spherical-current-in-linear-vacuum")


def _probe_values(potential, probes):
    sample = potential.sample(probes)
    return sample["A_t_m"][:, 1], sample["B_t"]


def test_remote_boundary_expansion_keeps_source_mesh_and_internal_probes_fixed(record_testsuite_property):
    reference = _CompactCurrentReference()
    probes = reference.probes()
    exact = reference.analytic(probes)
    values, internal_grids = [], []
    for extent in (2, 3, 4, 6, 8):
        potential = reference.solve(extent, cells_per_source_radius=12)
        values.append(_probe_values(potential, probes))
        internal_grids.append((potential.r_m[potential.r_m <= reference.radius_m],
                               potential.z_m[np.abs(potential.z_m) <= reference.radius_m]))
    for grid in internal_grids[1:]:
        for actual, original in zip(grid, internal_grids[0]):
            np.testing.assert_array_equal(actual, original)
    increments = np.array([[_relative_norm(later[i]-earlier[i], exact[i]) for i in range(2)]
                           for earlier, later in zip(values, values[1:])])
    total_errors = [[_relative_norm(row[i]-exact[i], exact[i]) for i in range(2)] for row in values]
    evidence = {"boundary_extent_in_source_radii": [2, 3, 4, 6, 8], "cells_per_source_radius": 12,
                "successive_boundary_increments": increments.tolist(),
                "total_errors_against_infinite_domain": total_errors,
                "scope": "boundary sensitivity at a fixed mesh; increments are not a rigorous infinite-domain error bound"}
    record_testsuite_property("remote_boundary_evidence", json.dumps(evidence, sort_keys=True))
    assert np.all(increments[1:] < .7*increments[:-1]), evidence
    # The last boundary move must perturb A and B by less than 0.25% of
    # their fixed analytic scales, well below the separate 5% mesh budget.
    assert np.all(increments[-1] < .0025), evidence
    # Do not require the total B error to fall monotonically here: at this
    # fixed coarse mesh it approaches a discretisation plateau. The following
    # independent experiment refines that mesh without moving the boundary.


def test_grid_refinement_keeps_remote_boundary_and_probe_coordinates_fixed(record_testsuite_property):
    reference = _CompactCurrentReference()
    probes = reference.probes()
    exact = reference.analytic(probes)
    values = [_probe_values(reference.solve(8, cells), probes) for cells in (6, 12, 24)]
    increments = np.array([[_relative_norm(later[i]-earlier[i], exact[i]) for i in range(2)]
                           for earlier, later in zip(values, values[1:])])
    total_errors = np.array([[_relative_norm(row[i]-exact[i], exact[i]) for i in range(2)] for row in values])
    evidence = {"boundary_extent_in_source_radii": 8, "cells_per_source_radius": [6, 12, 24],
                "successive_grid_increments": increments.tolist(),
                "total_errors_against_infinite_domain": total_errors.tolist(),
                "scope": "fixed-boundary mesh refinement; analytic comparison includes the remaining boundary error"}
    record_testsuite_property("fixed_boundary_grid_evidence", json.dumps(evidence, sort_keys=True))
    assert increments[1, 0] < .5*increments[0, 0], evidence
    assert increments[1, 1] < .75*increments[0, 1], evidence
    assert np.all(total_errors[1:, 0] < .5*total_errors[:-1, 0]), evidence
    assert np.all(total_errors[1:, 1] < .75*total_errors[:-1, 1]), evidence
    assert total_errors[-1, 0] < .01 and total_errors[-1, 1] < .05, evidence
