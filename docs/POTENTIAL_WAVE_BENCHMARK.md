# Static magnetic potential wave benchmark

`physics.potential_wave.propagate_static_magnetic_wave` is an opt-in numerical
benchmark, not a replacement for the production coherent column pipeline.
Its result explicitly reports `production_admission=False`. It consumes an
immutable P1 potential snapshot, a complex input wave, physical Cartesian
coordinates, an axial node sequence and kinetic energy. Particle population
size is not a wave input.

## Equation and conventions

For an electron, `q=-e`, and `Pi=-i hbar grad-q A` is mechanical momentum.
For a static magnetic field, momentum magnitude `p` is constant. Expanding the
forward classical beam Hamiltonian `-sqrt(p^2-Pi_perp^2)-q A_z` and removing the
constant `-p` carrier gives

```
H = Pi_perp^2/(2 p) + Pi_perp^4/(8 p^3) - q A_z.
i hbar d(psi)/dz = H psi.
```

This is a declared spinless, quasiparaxial classical-symbol approximation.
It omits back reflection, spin and explicit quantum field corrections; a
relativistic value of `p` does not make it a complete Dirac solver. Electric
acceleration and frozen dynamic drives are rejected. The two- and four-order
kinetic options allow an independent kinetic-truncation comparison.

The scope follows the separation of scalar paraxial, aberrating and explicit
quantum terms discussed by [Khan and Jagannathan](https://arxiv.org/abs/2009.13943).
The present implementation is a numerical discretization of its stated
classical-symbol model, not an implementation of every term in that paper.

## Discrete operator

The transverse covariant Laplacian uses nearest-neighbor Wilson links:

```
U_ij = exp(-i q integral_i^j A.dl / hbar)
L psi_i = sum_neighbors (psi_i - U_ij psi_j)/spacing^2
H_transverse/hbar = L/(2 k) + L^2/(8 k^3), k=p/hbar.
```

Missing neighbors have zero Dirichlet values. Four-point Gaussian quadrature
integrates each link; field quadrature still needs grid refinement. A separate
longitudinal link `exp(i q integral A_z dz/hbar)` transports the wave to the
midpoint and then to the next plane. Under `A'=A+grad(chi)` both the links and
the wave transform consistently, including gauges with x-z and y-z terms.
See [Halvorsen and Kvaal](https://arxiv.org/abs/0905.4177) for the link-variable
construction and the need to preserve gauge covariance during evolution.

The snapshot sums A before forming L and L squared. Overlap terms therefore
come from the complete potential instead of independently adding lens powers.
The finite Hermitian operator is exponentiated with a bounded Taylor series;
the exponentiation tolerance is separate from spatial and axial errors.

## Budgets and flux

The square-root remainder is bounded over the complete represented transverse
lattice spectrum, using `||L|| <= 4/dx^2+4/dy^2`. The operator rejects a spectrum
outside its declared small-angle domain or a remainder above the caller's
phase budget. This conservative bound is not a field-model error estimate.

Amplitude samples include the square root of pixel area. Their squared norm
therefore retains the caller's incident flux units. Boolean aperture masks at
explicit axial nodes remove flux; a conservation check accounts for that loss
without renormalizing the wave. A two-cell boundary guard rejects occupied
Dirichlet edges instead of absorbing or wrapping their intensity.

The complete initial, final and intermediate observation planes are checked
against every selected source domain. A successful step does not establish
that an unknown region is vacuum. Both CPU and optional CUDA execution use
the same finite operator; actual parity and independent convergence tests are
needed before any production promotion.

## Remaining production integration

The `PlaneWave` bridge expands an admitted carrier while preserving tilt and
curvature in the full complex amplitude. The `WaveMode` bridge transfers the
remaining norm into absolute mode weight and advances `AxialWaveReference`
separately. The constant longitudinal carrier is not multiplied into amplitude
again. Existing executed-wave storage can retain and resume these bounded
benchmark modes with a potential-dependent key; synthetic persistence tests
do not establish physical source admission.

This fixed-grid entry point does not provide adaptive canonical propagation.
The separate carrier executor described below does; neither entry point is an
unrestricted full-column option. A real column can produce
carrier curvature far beyond a fixed grid's phase sampling. Natural Cs/Cc
qualification and device ownership replacement remain separate gates. Existing
coherent source admission, image routes and configured aberrations are intact.

## Canonical carrier and observation bridge

`potential_wave_carrier.py` is a separate bounded CPU executor. It evolves the
existing canonical quadratic carrier and a Hermitian residual envelope, including
the fourth-order transverse kinetic term. The complete sum of A is formed before
squaring, preserving overlapping-source cross terms. Exact quadratic evolution
retains affine motion, scalar action and the continuous metaplectic phase branch.
Tilt and defocus are never removed from the physical result.

The residual checks spatial support and the envelope's spectral tail during
every bounded exponential substep. A conservative swept phase-space enclosure
checks the quadratic reference path, including excursions that return to the
same endpoint. A narrow packet cannot silently wrap through a periodic grid or
leave a field's domain and return. Resolved affine lattices use the complete
inverse-transpose Fourier metric, including shear. Ill-conditioned lattices and
unresolved tilted support faces fail explicitly. Electric acceleration and
time-dependent drives remain outside this static magnetic operator.

Optional exact carrier reselection transfers a quadratic phase between carrier
and complex envelope. It preserves their product, including tilt and defocus;
it is not an aberration correction or a fitted phase subtraction. Independent
Fourier, complex-field, gauge and flux checks cover the affine representation.
These representation tests do not grant full-column source admission.

The default quadratic reference uses the midpoint Hamiltonian. Explicit
`reference_quadrature="cf4"` selects two commutator-free Magnus factors at the
two Gaussian quadrature nodes. Each factor carries the complete quadratic,
affine and scalar Hamiltonian through the continuous canonical path; it is not
an endpoint-matrix fit. The fourth-order statement applies to the quadratic
reference only. The general nonquadratic residual and its optional real-scalar
Strang split remain second order. The latter exponentiates only an explicitly
real multiplication operator; the differential part retains the complete
fourth-order covariant momentum operator and all overlapping-source terms.
See the [commutator-free propagation construction](https://arxiv.org/abs/1102.5071).
Independent varying-Hamiltonian complex-field tests retain absolute phase,
including non-Gaussian input and static gauges with longitudinal mixed terms.

Aligned compact Az sources admit a hard face only as an exact integration
boundary. A representable one- or two-ULP interval beside that face uses the
known one-sided native value: zero on the declared exterior, or the original
analytic jet on the interior. Distinct physical observation planes are never
merged. An interior face crossing, tilted face intersecting the lattice, or
unknown FEM exterior is rejected. This rule does not make the two-sided
derivative at a hard face defined and does not add a fringe-field model.

`potential_wave_observation.py` inserts exact requested observation planes,
aligned support faces and explicitly declared circular aperture masks. It returns
the original complex carrier/envelope representation, remaining absolute mode
weight, aperture loss ledger, source identity and gauge. The existing checkpoint
store and specimen/detector interfaces consume these records without adding a
second legacy aberration pupil. The bridge admits controlled entrances; its class
name does not turn a synthetic Gaussian into a qualified physical gun source.
Only explicit thin aperture events are covered here; unlisted thick material or
vacuum walls are never silently inferred as a completed geometry calculation.

## Explicit numerical support projection

`potential_wave_support.py` provides an optional, separately accounted support
projection. It sums both tails directly, applies a declared smooth window and
crops symmetrically about the original index origin. The retained basis,
origin, tilt, curvature and untapered core cells are unchanged. It does not
renormalize amplitude or reinterpret numerical truncation as physical loss.
Sub-roundoff tails cannot be swallowed by subtracting two nearly equal totals
or by adding an absolute epsilon to a user's budget.

An immutable cumulative ledger records absolute projection L2 error and
probability loss in the incident reference units. Checkpoints persist and
restore this ledger; continuation cannot reset its budget. Material branching
can inherit a validated common, nonspendable prefix rather than multiply the
same available budget across outgoing modes. Actual aperture loss is reported
separately. The projection receipts bound the initial window operation, not
the total error after propagation on a different FFT domain. Independent
window, sampling and projection-budget comparisons are therefore mandatory.

`potential_phase_budget.py` separately tracks the accepted kinetic-expansion
phase bounds. The optional immutable ledger binds a source snapshot, energy,
operator order and exact executed planes. It adds only successful residual
bounds; rejected sampling candidates spend nothing. Continuation restores the
saved allowance rather than granting a new budget at each observation plane.
This permits a localized larger bound near a strong objective while keeping
the same strict whole-interval total. The older uniform per-step allocation
remains the default when no ledger is selected. A unit comparison confirms
that changing only this allocation leaves an already admitted complex result
bitwise unchanged.

Metre endpoints remain exact in the low-level ledger. A checkpoint may restore
its original metre endpoint only after verifying its authoritative stored
millimetre value. General adjacent floating-point planes are not merged. The
material bridge retains an explicit native-millimetre/executed-metre receipt
when that conversion is not reversible. A present receipt always controls
the identity check, including malformed values; ordinary converted-plane
equality cannot bypass it.
The phase ledger is independent of the support-loss ledger and mode weight; its
bound is not a bound on pointwise phase in a normalized conditional material
state. Material energy histories need an explicit common-prefix lineage and
their own admitted contraction assumptions.

## Actual controlled C3 + Probe interval

`tests/test_potential_probe_whole_wave.py` is the long acceptance entry point.
It starts at 600 mm and reaches the specimen entrance at 1599.1999975 mm in
Nanoprobe Normal and Large Angle, using 29 captured native magnetic sources,
high-order round-lens potentials and the configured thin C2/C3 apertures.
The same controlled low-NA complex entrance is held fixed for each mode's
baseline and every parameter perturbation. Its Gaussian width and curvature
were selected in advance from the native linear transfer; they are not a
reconstruction of physical FEG phase. The input also retains its non-Gaussian
complex factor, initial constant phase and nonzero momentum centroid.

The predeclared gates are relative complete complex-field L2 below `1e-5`
and raw transverse phase RMS below `1e-3 rad` where both intensities exceed
`1e-6` of their respective peaks. The comparison retains absolute weight and
reports excluded probability; it fits no piston, displacement, tilt, focus or
intensity. Longitudinal action and flight time are retained and checked
separately. The numerical sequence includes 0.5/0.25/0.125 mm axial steps,
fixed-window transverse refinement, window enlargement, radial field orders
3/5 and independently looser and tighter cumulative support budgets. The
whole-interval kinetic remainder allowance is `0.01 rad`, accumulated from
actual admitted residual bounds without resetting at the 10 mm observations.
The former uniform allocation rejected a tighter-window Normal run at
1585.97561 mm: its local bound was `2.22913e-6 rad`, above its artificial
uniform share `1.22049e-6 rad`. The cumulative scheme does not increase the
original whole-interval allowance or change the fourth-order kinetic operator.

Development runs of the midpoint reference completed both modes but failed
these gates: 0.25/0.125 mm relative complex errors were `5.71e-4` and
`8.39e-4`, with raw phase RMS approximately `3.11e-3` and `4.54e-3 rad`.
The CF4 reference at 0.5/0.25 mm reduced these to `2.95e-7` and `1.72e-7`
in complete complex L2, and `1.12e-6` and `5.97e-7 rad` in raw phase RMS.
These two-level development results alone are not the complete acceptance
receipt; the registered long tests also verify the remaining sequences and
effective magnetic length, A1 and A2 responses. A drive response below its
independently measured numerical uncertainty is reported as unresolved. The
response uncertainty includes axial, transverse-grid, window, field-order and
support-budget comparisons, rather than an axial test alone.
These numerical differences remain separate from the declared `0.01 rad`
kinetic-expansion bound and from unmodelled fringes, spin, quantum corrections
and hardware uncertainty. A resolved functional control response in this
scalar model does not certify an OEM fine-step sensitivity or the smallest
experimentally resolvable adjustment.

The completed 2026-10-24 acceptance contains 28 current repository tests,
14 per mode, with no failures or skips. They use 58 complete entrance-to-output
artifacts and 16 distinct full upstream prefixes. The largest observed raw-field
refinement differences and separately accumulated kinetic bounds are:

| Controlled mode | Complete complex L2 difference | Raw phase RMS difference | Kinetic bound, rad |
| --- | ---: | ---: | ---: |
| Nanoprobe Normal | 3.011e-7 | 1.348e-6 rad | 1.23008e-3 |
| Nanoprobe Large Angle | 1.869e-7 | 1.045e-6 rad | 4.45593e-5 |

The tested functional changes are HP2 effective magnetic length times 1.2,
DPH2 steering plus 0.001 mrad, Condenser Stigmator drive plus 0.01 percent,
HP2 strength plus one percent, and HPC strength plus `1e6 m^-3`. The last value
was selected prospectively after the `1e5 m^-3` candidate did not meet the
original response margin. It is an engineering-model functional stress drive:
its coefficient exceeds the nominal HP2 coefficient, and the software's
`1e9 m^-3` cap is not an OEM operating range or a calibrated fine step.

For this HPC stress drive, both the baseline and perturbed fields independently
pass all axial, grid, window, field-order and support-budget comparisons.
`tests/potential_probe_response_helpers.py` additionally compares the unfitted
complex difference of those fields and their raw relative phase. It uses the
fixed incident weight of 0.25, explicit exterior-tail accounting, and independent
twofold/fourfold interpolation for every response and error comparison. The
maximum measured refinement difference is an empirical numerical floor; it is
not a statistical sigma or a rigorous combined bound on the entire model.
The complete-complex response exceeds that floor by 13.9 times in Normal and
111 times in Large Angle. The raw phase response is `1.0203e-6` and
`1.7730e-5 rad`, respectively. The retained `1e5` to `1e6` response magnitude
ratio is approximately ten, with matching raw-phase direction and no fitted
phase, focus, position or intensity.

The smaller HP2 change of 0.1 percent and HPC changes of `100` and `1e5 m^-3`
remain explicitly unresolved under their original whole-field refinement
criterion. Their results and the original two failed `1e5` resolution tests are
retained; passing the separately named stress-drive tests does not relabel them
as resolved. An optional tighter sensitivity family was not completed: its fine
grid was stopped at 730 mm after expensive retained support, while a separate
`1e-7` support-budget run requested 2520 by 2880 cells and exceeded its declared
2048-cell/4-GiB resource budget. Neither result is a completed convergence test
or evidence of a physical failure.

Local final execution took approximately 64 minutes with parallel workers.
The separately recorded unique-prefix and suffix elapsed intervals sum to
about 4.18 hours; that sum is neither campaign wall time nor a timing prediction
for another host. The final repository assertions were replayed against
checksummed completed artifacts after proving all 677 source files and the
propagation-helper syntax trees unchanged. Only assertion logic and the explicit
test-case registry changed. Clean CI executes the portable repository tests
without private research files or that local cache.

Paired parameter tests may reuse the identical 600--1120 mm prefix only after
proving equal gauge, mechanical geometry, apertures and unaffected providers,
and that every changed source's complete old and new native support is
downstream of that interval. This is a test-only exact model equivalence,
excluding FEM or joint material coupling. It preserves the original prefix
complex field, weight, action and both numerical ledgers. A test-only ledger
identity replacement retains every original cost and the original checkpoint
receipt after that complete-support equivalence proof. Production continuation
still rejects a changed snapshot. Changing effective magnetic length in this
test is not a mechanical redesign or a new material-field solve.

This acceptance remains static, magnetic, low-NA and controlled-input. It does
not qualify an electrostatic physical gun-to-sample phase, dynamic scan, all
source modes, unknown field-map exteriors or complete microscope hardware.
