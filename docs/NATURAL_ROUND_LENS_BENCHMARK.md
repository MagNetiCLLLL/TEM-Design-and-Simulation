# Field-derived round-lens benchmark

`HighOrderRoundLensPotential` is an opt-in near-axis vacuum expansion of the
existing captured Gaussian axial field. It retains the original B0 and does
not change lens excitation, dimensions or any production preset. No empirical
Cs/Cc kick is evaluated inside the benchmark.

## Field and domain

With n starting at zero, the retained vector potential is

```
A_phi(r,z) = sum [(-1)^n r^(2n+1) B0^(2n)(z)
                 / (2^(2n+1) n! (n+1)!)]
```

Supported radial polynomial orders are 1, 3, 5, 7 and 9. B is obtained from
the curl of the same finite potential. This preserves zero divergence while
leaving an explicitly reported truncation residual in curl(B). The first
corrections are `Bz=B0-r^2 B0''/4` and
`Br=-r B0'/2+r^3 B0'''/16`. The construction is consistent with the regular
axisymmetric expansion in [Khan and Jagannathan](https://arxiv.org/abs/2009.13943).

A validity radius is mandatory model information; a finite expansion cannot
establish its own physical aperture merely by producing finite numbers.
`axial_domain="native_support"` rejects exterior queries.
`axial_domain="gaussian_all_z"` explicitly continues the declared Gaussian
formula. It never substitutes zero for an unknown FEM exterior and does not
cut a nonzero transverse vector potential at a nominal support face.

The field identity includes the coefficients, placement, order and domain.
It contains no mutable state reference. Geometry and material reconstruction
remain distinct from this expansion of an already specified axial profile.

## Rays and aberration conventions

`round_lens_aberration_benchmark` integrates the complete magnetic Lorentz slope
equations with fixed relativistic momentum magnitude. Only forward, centred,
axis-aligned round-lens cases are admitted by this measurement helper. A
separate paraxial Larmor equation defines the nominal image plane and pupil.

The benchmark pupil points outward, opposite the converging paraxial image
slopes. At the fixed nominal image plane its displacement convention is

```
delta_r = -Cs * theta * |theta|^2 + Cc * theta * (delta_E/E0).
```

The report includes the corresponding wave-gradient coefficients and their
sign convention. In particular, its Cc is the negative energy derivative of
the fitted wave-gradient C1. These are diagnostic results, not extra actions
to add to a propagation that already used the field.

Energy scans keep the field snapshot, incoming positions and directions, and
observation plane fixed. They do not refocus, change current, align centroids
or normalize measured displacements separately at each energy. Pupil holdout
rays test the fit away from its fitted samples.

## Independent evidence

Tests compare the radial field expansion with an independently integrated
Fourier-Bessel vacuum solution. They refine field order independently of the
ray integration step. The forward slope solver is also compared with the
existing time-domain relativistic Lorentz reference. A separately derived
first/third-order perturbation ODE checks the natural Cs/Cc fit.

For the synthetic 300 keV, 0.3 T, 6 mm Gaussian-width case, the converged fit
gives approximately Cs=78.620661 mm and Cc=23.019878 mm. These numbers describe
that synthetic benchmark and its stated reference planes; they are not a
calibration for a TEMForge preset or a commercial microscope.

The analytical evidence does not replace a resolved FEM lens's own field-mesh,
boundary and aberration convergence checks. The full C3/Probe Corrector pilot,
coherent production admission and retirement of empirical screens are separate
acceptance gates. Existing production propagation remains unchanged.

## Resolved FEM qualification

The original triangular P1 potential passes field consistency tests, but its
piecewise derivatives did **not** converge to a useful natural third-order
coefficient. That negative result is retained. An explicit
`RegularAxisymmetricMagneticPotential` reconstructs A_phi/r as an even radial
polynomial with a longitudinal spline inside a declared source-free cylinder.
It keeps the original A coefficients, reconstruction choices and error report
in its identity. It is a new approximation, not the original FEM basis.

An independent Biot-Savart air-core coil reference qualifies the implemented
scope: 3000 ampere-turns, 4–6 mm coil radius, 10 mm coil length, 300 keV,
4–30 mrad pupils and an 80 mm outer boundary. Refining the local FEM mesh from
250 to 125 to 62.5 micrometres reduces the fitted Cs error from 5.93% to 2.53%
to 0.928%. The finest Cs is approximately 243.945 mm versus 241.702 mm from the
independent field; Cc is 32.6201 mm versus 32.6399 mm. Separate integration-step,
outer-boundary, reconstruction-window and polynomial-order checks are retained.
The normalized vacuum curl-B residual decreases from 1.192% to 0.1444%.

These tolerances qualify this synthetic air-core benchmark, not a nonlinear
ferromagnetic TEM magnetic circuit or the precision of an experimental corrector.
See `test_potential_fem_aberrations.py` and
`test_regular_axisymmetric_magnetic_potential.py` for the actual acceptance gates.

## Same-entrance action and executed wave phase

`test_potential_aberrations.py` also compares actual nonlinear carrier-wave
propagation with a complete Lorentz pupil and an independent canonical H4
integration. This bounded fixture uses 300 keV, a 0.3 T Gaussian with 1 mm axial
width, the central -0.1 to +0.1 mm interval, and a fixed 20 micrometre pupil with
zero incoming canonical transverse momentum. A Gaussian coherent input has
12.5 micrometre intensity width, absolute flux 0.37 and phase 0.37 radians.
Both computations retain the same regular azimuthal gauge and physical planes.

The declared raw-phase budget is 1e-3 radians and the normalized canonical
momentum budget is 1e-10. No piston, tilt, defocus or intensity correction is
applied. An action fit, with separate radial holdout points and eight azimuths,
retains a resolved quartic phase coefficient of about -0.06498 radians at the
pupil edge; its derivative is checked against the executed canonical momentum.
The actual wave's 16, 32 and 64 axial-step raw-phase differences from the
Lorentz action are 1.69e-6, 4.85e-7 and 2.01e-7 radians. The finest canonical
gradient difference is 2.40e-15; spatial 128-to-256 refinement changes phase by
8.94e-10 radians, and radial field order 5-to-7 changes it by 1.34e-10 radians.
The independently integrated H4/full-Lorentz fixed-point action difference is
3.35e-10 radians. Absolute flux remains 0.37 without normalization.

The finite Gaussian's on-axis diffraction phase is retained: halving its width
increases the departure from the classical action from 2.01e-7 to 8.02e-7
radians. This is separate from numerical refinement and the kinetic-order
error. The local radial quartic action is not a focal-plane Cs convention.
This evidence connects a fitted action and its cubic gradient to an executed
wave from the same entrance; it does not compare the complete C3 canonical
pupil coefficients with an arbitrary physical-source wave or qualify an entire
Gaussian lens, a FEM column, or a production imaging route.
