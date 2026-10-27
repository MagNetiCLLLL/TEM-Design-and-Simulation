# Exact quadratic reduction and bounded design example

`temsim.physics.qualified_field_reduction` provides an optional executable
reduction of a **static, constant-energy, second-order forward magnetic model**.
It does not select a new production mode or certify a nonlinear field surrogate.

The coefficient proof admits an exactly uniform aligned axial field represented
by retained `A_phi = c r` coefficients, and aligned uniform-envelope native
`A_z` polynomials of degree at most two. It rejects unknown maps, nonuniform
FEM coefficients, Gaussian axial envelopes, hexapoles, accelerating electric
fields, dynamic drives, tilted interfaces and z-dependent gauge coefficients.
This is a proof about the represented coefficients, not OEM field validation.

The normalized Hamiltonian is

`h = (u - q A_perp / p)^2 / 2 - q A_z / p`, with `q = -e`.

Positions are metres, `u` is canonical transverse momentum divided by the
actual relativistic momentum, and energy is eV. All sources are composed before
squaring the vector potential. Existing `CanonicalPath` machinery carries the
affine scalar action and continuous metaplectic phase. A result is not recovered
from an endpoint ray matrix alone, and no phase or centroid is fitted away.
The fourth-order kinetic contribution is omitted under a separately reported
conservative phase budget. Spin, back reflection and electric acceleration are
outside this reduction.

Qualification records the exact potential snapshot digest, energy range,
position domain, canonical-momentum bound and axial interval. Execution rejects
changed source coefficients, an out-of-range energy or a conservative swept
support bound leaving the domain. Wave execution retains the analytical carrier,
uses the existing metaplectic grid checks and rejects occupied boundary cells.
Wave-grid convergence remains an independent requirement.

No aperture or other boundary action may occur inside the reduced interval or
at either endpoint. Callers must supply the complete boundary-plane list when
qualifying and executing the interval. `replace_interactions` requires every
exact native source record once, replaces its ownership, and retains unrelated
records. Reintroducing a replaced field is rejected by the interaction registry.
This is a standalone opt-in registry contract, not automatic insertion into the
existing production plan or an archived full-column restart.

`temsim.physics.bounded_optical_design` evaluates bounded parameter proposals on
separately restored instrument captures. A trusted diagnostic callable returns
a residual vector with explicitly chosen units and scaling. Physical input
mutation by the diagnostic is rejected. All actual evaluations, parameter
bounds, absolute finite-difference steps and input identities are reported.
Finite-difference steps must be converged independently of propagation and
sampling; machine-epsilon perturbations may be hidden by an existing model's
calibration resolution. Optimizer termination is not experimental acceptance or
a guarantee that the requested residual is attainable.
An increment too small to change the represented parameter is rejected before
diagnostic execution and at every Jacobian evaluation. Derivatives use the
actual bounded floating-point increment, not an unrepresentable requested one.

No candidate is applied to the live instrument. No fitted Cs, other inferred
aberration coefficient or optimization result is fed back into the same
propagation as an additional optical action.

## Runnable example

From the repository in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -X utf8 scripts/example_quadratic_field_reduction.py
```

The script prints two separate results and writes no files:

1. A coefficient-qualified uniform-field ray map.
2. A bounded projector-excitation study using the existing thin particle
   executor as its declared diagnostic. A synthetic target generated at 68%
   is recovered from a 50% starting point; the live control stays at 50%.

The second example demonstrates an executable design workflow. It does not
claim that thin particle optics is a qualified replacement for finite-field
dynamics, that an instrument was calibrated, or that nonlinear reduction is
implemented. Independent tests compare uniform-field motion, affine force
action, complex-wave phase and refinement against the unreduced Wilson
operator, and check domain, boundary, ownership and input-mutation rejection.

## Native-potential phase design

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -X utf8 scripts/example_potential_phase_design.py
```

This second script executes the retained native condenser-stigmator potential
with the fourth-order static magnetic Wilson wave operator. It matches both
quadratures of a controlled complex field on a fixed laboratory lattice over
a 100 micrometre interval. The objective keeps absolute flux and scalar phase,
tilt and defocus; it does not align or normalize the propagated fields.
The two physical stigmator drives are bounded independently. Every trial uses
a private instrument capture, and its potential identity changes with the
drives. A separate, finer spatial and axial discretization checks the recovered
parameters. This is parameter consistency, not an estimate of absolute wave
error: identical recovered drives share the same discretization error.
The CLI prints a compact report; the callable retains every complex
residual sample in the optimization history.

A separate refinement test compares the complete propagated fields on shared
physical points at 33, 65 and 129 pixels, with the pixel-area amplitude units
converted explicitly. It independently changes axial step and finite-difference
drive increments. The recorded spatial complex L2 differences decrease from
about 1.22e-4 to 3.13e-5, while the relative drive-derivative difference decreases
from 4.38e-4 to 1.14e-4. Axial differences are below 3e-14 in this short interval.
The much smaller optimization residual must not be quoted as absolute physical
phase accuracy. No phase alignment or flux renormalization enters these checks.

The synthetic target uses X/Y strengths of 0.18/-0.11 percent, starting from
0.05/0.03 percent. The live controls retain those starting values. This is an
isolated native-component phase-design example, not a whole-column design,
geometrical FEM re-solve, physical-source qualification or hardware calibration.
