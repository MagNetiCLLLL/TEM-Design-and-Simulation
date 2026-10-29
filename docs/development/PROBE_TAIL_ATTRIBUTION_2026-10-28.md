# Probe-tail attribution with fixed optical controls

The [compact evidence archive](probe_tail_attribution_20261028/README.md)
contains byte-verified scripts, input snapshots and completed reports.
Large particle arrays remain local; the archive documents this reproduction
limit explicitly.

## Current target and scope

The user retains the physical 100 um C2 aperture diameter, specimen-surface
alpha95 target of 30 mrad, and a consistent native D95 waist at that surface.
The former D95 < 0.7 nm hard limit has been withdrawn. D90 and radial RMS are
reported separately. Joint tuning is held during the attribution experiments.

These are classical vacuum-probe calculations with the production source,
finite electromagnetic fields, physical apertures and original particle
weights. No extra empirical Cs/Cc, thin-lens correction, coordinate rescaling,
particle removal or coherent-wave calculation is introduced. The field model
remains an engineering model, not a measured or OEM-qualified microscope.

The full-source experiments use the same 769 emission positions, directions,
IDs and weights. Original source current is 10 uA. Every completed experiment
below transmits all 769 particles. Optical excitations are held fixed for
causal comparisons; the actual alpha95 is measured rather than retuned.
Consequently the full monochromatic source is not itself a 30 mrad working
point. This change of convergence angle must not be hidden by rescaling or
interpreted as a matched-angle probe-size comparison.

## Energy controls

The baseline has column step 0.05 mm, gun maximum trace step 0.2 mm and native
electrode-field resolution 16 cells per bore. Its independent 0.025 mm column
repeat changed surface D95 by only 0.000727 percent. The initial energy law
has mean kinetic energy 0.3 eV and a configured RMS-equivalent FWHM of 0.3 eV;
it is a bounded transformed distribution, not an exact Gaussian spectrum.
Acceleration produces approximately 300 keV electrons.

Three controls distinguish energy-dependent transport acquired before and
after the gun exit:

1. Original polychromatic source and complete transport.
2. The exact original gun-exit positions, slopes, clocks, IDs, weights and
   stops, with only column-entry kinetic energies set to 300000.3 eV. This is
   explicitly a controlled-injection counterfactual, not an installed
   monochromator or an actual gun output. It isolates downstream energy
   dependence while preserving the upstream phase space.
3. Ideal monochromatic emission at the unchanged mean launch energy, followed
   by a fresh complete gun and column calculation. This includes changes to
   the gun-exit phase space and convergence-angle distribution.

| Control | Surface D90 (nm) | Surface D95 (nm) | Smallest observed D95 (nm) | Its offset (nm) | Surface alpha95 (mrad) | ID306 centered radius at surface (nm) |
|---|---:|---:|---:|---:|---:|---:|
| Original source | 1.496154 | 2.562874 | 2.171371 | +15.5 | 29.769616 | 28.694650 |
| Original gun-exit phase; uniform column energy | 1.451434 | 2.442065 | 2.211782 | +13.5 | 29.769633 | 29.174219 |
| Monochromatic emission; complete gun and column | 1.392249 | 2.012389 | 1.576127 | +14.7 | 28.496811 | 3.567496 |

Positive offsets are downstream of the specimen upper surface. These are
minima over actual executed planes, not interpolated continuous minima.
Each report retains its actual neighboring-plane bracket. D95 values are
quantiles: their squared differences are not additive variance components.

Uniform downstream energy does not remove the extreme tail. It changes the
largest paired specimen-plane position by about 0.482 nm; monochromatic
emission changes it by about 25.174 nm. Thus direct energy dependence of the
high-energy column is not the dominant cause of this particular extreme tail
in the tested model. The large full-monochromatic response involves changes
already present in the gun-exit phase space and their downstream mapping.
It must not be identified with a unique measured chromatic coefficient.

The replay guard ran 128 original-energy particles through the production
column and reproduced all saved phase, clocks, energies, activity, identities,
weights, mesh and stopping records exactly before the 769-particle controlled
injection. The complete raw inputs, outputs and hashes are retained in:

- `outputs/probe-tail-attribution-20261028/mono769-h005/`
- `outputs/probe-tail-attribution-20261028/poly-original-replay128-h005/`
- `outputs/probe-tail-attribution-20261028/poly-phase-constant-k769-h005/`
- `outputs/probe-tail-attribution-20261028/energy-comparison-poly-constant-mono/`
- `outputs/probe-tail-attribution-20261028/energy-controls.png` and `.svg`

## Initial energy and transverse momentum

The current source's angular sigma is a local slope scale at the low-energy
emission plane, not a 300 keV gun-exit angle. Its virtual-source FWHM parameter
sets an emission footprint in this executed flat-source branch; it is not an
independent measurement of a virtual image. At fixed direction, relativistic
kinetic momentum is proportional to sqrt(K*(K+2*m*c^2)), and in this sub-eV
regime its leading energy dependence is sqrt(K). Consequently a Gaussian
slope marginal with independent energy spread need not give a single
Gaussian transverse-momentum distribution.

For held-out source ID306, the initial momentum ratio between the original
and monochromatic source is 1.357847. The observed gun-exit radius ratio is
1.361111 and slope-magnitude ratio is 1.375707. An offline, five-parameter
linear gun-exit predictor using initial transverse kinetic momentum reduces
held-out position errors from approximately 56 nm to 0.32 nm per coordinate
compared with an equally sized angle-based predictor. IDs remain grouped
across the two source variants; ID306 itself is held out. Its position-vector
error falls from 221.424 nm to 1.872 nm.

This supports the importance of the initial momentum scale; it is not unique
causal attribution to gun focal chromatic aberration. The finite-sample
kinetic-momentum RMS phase-space areas change by only -0.054 and -0.228
percent in the two transverse directions, while the momentum-tail shape
changes. Do not claim a large RMS emittance increase, or equate a mechanical
momentum area with a canonical emittance without its field/gauge conditions.
No fitted source surrogate was propagated or installed.

Details, held-out residuals, source hashes and mathematical checks are in
`outputs/probe-tail-attribution-20261028/launch-momentum-offline-v2/`.

## Independent numerical and field checks

The gun-step experiment changes only the maximum gun integration step from
0.2 to 0.1 mm. It preserves the entire emitted energy law and all fields.
The exact-plane observer at the original setting reproduces the archived
gun exit and stop ledger exactly. The finer gun source has a maximum exit
position difference of 0.1213 nm. Its subsequent complete column calculation
changes the smallest observed D95 from 2.171371 to 2.171481 nm and retains the
+15.5 nm observed waist. This small tested step effect does not account for
the full-monochromatic tail reduction.

Electrode-field grid checks retain the same geometry, boundary values and
source. The actual starting resolution is 16 cells per bore; 20 and 24 are
additional refinements. Resolution 32 is rejected by the existing 1500000
vertex budget, which was not changed. The accepted meshes contain 721152,
1056215 and 1431062 vertices. These grids discretize a shared electrode
potential over the instrument, so a full-column grid comparison includes
both the resulting gun-exit phase and the shared downstream electric field.
It must not be described as a fixed-field gun-exit injection experiment.

At fixed gun step 0.1 mm, the grid16 and grid24 complete-column results have
minimum D95 values 2.171481 and 2.188638 nm. The observed minimum moves from
+15.5 to +15.2 nm. The 0.3 nm shift exceeds the prior 0.25 nm axial numerical
comparison budget; these two observations do not establish grid convergence.
The grid24 ID306 radius remains about 28.97 nm. The complete grid comparison
is now available:

| Cells per bore, gun step 0.1 mm | Surface D95 (nm) | Minimum D95 (nm) | Observed minimum offset (nm) | Neighbor bracket (nm) | Surface alpha95 (mrad) |
|---|---:|---:|---:|---|---:|
| 16 | 2.562758 | 2.171481 | +15.5 | [15.4, 15.6] | 29.772168 |
| 20 | 2.573641 | 2.201931 | +14.9 | [14.8, 15.0] | 29.873277 |
| 24 | 2.569247 | 2.188638 | +15.2 | [15.1, 15.3] | 29.828871 |

The last refinement again shifts the observed minimum by 0.3 nm, with a
nonmonotonic trend. It does not pass the existing 0.25 nm axial comparison
budget. The corresponding minimum-D95 change is about -0.604 percent. All
479 common actual checkpoints were compared using original phase/current
data. These bounded changes do not explain the approximately 25 nm
full-monochromatic change of ID306, but they leave waist-position numerical
qualification unresolved. No tolerance was relaxed to report convergence.

Evidence is in `gun-precision-full-column-comparison.json` and
`gun-precision-full-column-readout.txt` under the investigation output root.

The round-lens field experiment preserves the original gun-exit phase,
energies, electric fields, multipoles, geometry and exact integration mesh.
It replaces only the radial order of the native round-lens vector potential
using the existing explicit immutable-runtime interface. A full three-order
replay of all 769 particles reproduces the archived source/phase/stop arrays
exactly before changing the order. Increasing radial order from 3 to 5 changes
surface D95 from 2.562874 to 2.558746 nm (-0.161 percent); the largest profile
D95 change is 0.434 percent. The minimum changes from 2.171371 at +15.5 nm to
2.169952 at +15.4 nm. ID306 remains in the tail at radius 28.734554 nm.
This tests a specific engineering-field truncation, not the entire field
model or its agreement with a real microscope.

An independent seventh-order run reused verified immutable three- and
five-order evidence without rerunning them. From order 5 to 7, surface D95
changes by -0.000001702 nm and the maximum relative change across the actual
through-focus profile is 2.6065 ppm. The minimum D95 becomes 2.169949 nm at the
same +15.4 nm observed plane and the same neighboring bracket. ID306 remains
at radius 28.734594 nm. This successive stability does not support attributing
the extreme tail to third-order radial truncation. Full receipts and original
arrays are in `round-radial-3-5-h005/` and `round-radial-7-h005/`.

## Native local-linear response control

A further diagnostic holds the original gun-exit phase and the uniform
column energy fixed. It executes the actual field model for 17 mathematical
response probes: a reference at the current-weighted gun-exit mean and plus/
minus perturbations of each transverse position/slope at h and h/2. Position
h is 1 nm and slope h is 1e-8 rad. The exact original integration mesh and
all 479 checkpoints are retained. Fields are verified static, the reference
clock is declared, and every response probe remains active without a stop.
These calibration probes are not assigned a physical emission current.

Applying the two measured local Jacobians to the existing 769 actual input
coordinates gives a maximum h-versus-h/2 prediction difference of
0.00002028 nm across the actual through-focus window, below the predeclared
0.01 nm comparison budget. Mathematical predictions are saved separately
from all actual full-field particle data.

For ID306 at the specimen, the stable local-linear prediction has centered
radius 2.220619 nm, compared with the already executed full-field value
29.174219 nm at the same fixed energy and input phase space. Its raw-coordinate
nonlinear remainder has magnitude 26.987974 nm. Thus this particular extreme
tail is predominantly absent from the local first-order map and appears in
the full column response beyond that map.

The bulk distribution must be interpreted separately: at the specimen plane,
linear-prediction D95 is 2.857621 nm, greater than actual full-field D95 of
2.442065 nm. The mathematical linear prediction reaches minimum D95 of
0.202749 nm near -48 nm, whereas the actual full-field minimum is 2.211782 nm
near +13.5 nm. That mathematical minimum is neither an achieved physical
probe nor a prediction that a real adjustment will reach it. The remainder
does not uniquely identify Cs, another aberration coefficient, or a single
responsible lens.

Receipts, the real 17-probe trajectories, Jacobians and separate mathematical
predictions are in `native-local-linear17-h005/`. An English summary and a
visually checked comparison are in `native-local-linear17-analysis-v2/`.

## Supported attribution and remaining work

Within this fixed working point and the tested full source, the evidence
supports an energy-dependent launch-momentum and gun-exit phase-space pathway
followed by a strongly nonlinear column response for the extreme tail.
Direct downstream energy dependence at approximately 300 keV does not remove
that tail. Tested column/gun step changes and round-field radial truncation
are much smaller than the full-monochromatic tail change.

This is a controlled model attribution, not unique measurement of chromatic
or spherical aberration coefficients and not experimental source validation.
The actual emission law and its count-dependent energy normalization remain
explicit model assumptions. Complete source-count convergence is not tested
by this investigation. Electrode-grid convergence of the D95 waist position
also remains unresolved at the declared budget.

Joint tuning has therefore not resumed. The next numerical work must resolve
or rigorously bound the remaining electrode-grid waist uncertainty while
preserving physical boundaries and the recorded budget. Subsequent joint
tuning must retain the real source spectrum, restore alpha95 to 30 mrad and
place the native D95 waist at the specimen, without reinstating the removed
0.7 nm hard limit. The ideal single-energy control is not installed as a preset.

## Execution and interpretation boundaries

The new native-D95 observer and solver have no implicit 0.7 nm acceptance cap
or 0.65 nm fitting target. Their updated focused suites passed 118 tests.
The separate source, injection, gun and field diagnostic tests validate their
declared guards; they do not replace source-count, field-grid or experimental
qualification. Historical records retain their original definitions and
acceptance limits and are not reclassified using the new target.

All physical observations here are fixed-control diagnostics. No joint fit,
working preset installation or claim of a continuously exact waist at the
sample follows merely from these results.

The usual chromatic mechanism is energy-dependent trajectory curvature and
focusing in a fixed field; the magnetic field itself does not change for
each electron. Background definitions are available in
[JEOL's chromatic-aberration explanation](https://www.jeol.com/words/emterms/20121023.003500.php)
and [the focal-spread description](https://www.jeol.com/words/emterms/20231204.php).
Those general relations do not determine this instrument model's effective
Cc or convert its native D95 into a Gaussian width.
