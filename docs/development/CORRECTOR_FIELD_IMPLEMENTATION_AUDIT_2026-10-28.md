# Probe-corrector field implementation audit

Date: 2026-10-28. This audit is read-only with respect to production source,
physical inputs and presets. It inspects the current implementation and samples
fields at previously executed particle coordinates. It performs no new particle
transport, gun calculation or coherent calculation.

## Finding

No definite sign, unit conversion, canonical/mechanical coordinate or duplicate
field-ownership defect was found that explains the 854.266 nm specimen D95 in
the strict old-Normal-controls unified replay. This is a bounded audit, not proof
that the complete field model is physically calibrated.

The two compared solvers are materially different nonlinear models. The old
solver uses paraxial round-lens transport plus configured empirical spherical
aberration actions. The unified solver uses full forward Lorentz dynamics and
the third-degree radial vector-potential expansion of the same axial round-lens
profile. These natural aberrations need not equal the empirical coefficients.
The old optimum therefore does not transfer automatically even if first-order
convergence angles remain close.

The strict replay already exhibits a large error before the Objective:
at the common 1400 mm AC plane, RMS radius is 2.458 nm in the current legacy
implementation and 686.102 nm in unified. This evidence comes from the same
174 eventual specimen arrivals, not a changed source population. See the
[executed comparison](CORRECTOR_ANGLE_TRANSFER_COMPARISON_2026-10-28.md).

## Hexapole strength and orientation conventions

The native component defines `strength_m3` as the peak Gaussian envelope
coefficient, with normal/skew components proportional to `cos(3 orientation)`
and `sin(3 orientation)`:
[hexapole.py](../../src/temsim/optics/hexapole.py#L123).
It is not an integrated kick or a separate GUI unit.

Let `h_n`, `h_s` denote the actual normal/skew coefficients at a plane, `p_ref`
the reference momentum and `q` the electron charge. The captured potential is

```text
A_z = -(p_ref/q) [h_n (x^3/3 - x y^2) + h_s (x^2 y - y^3/3)].
```

The implementation retains this `p_ref/q` factor, SI coordinates and the same
Gaussian width `effective_length_mm / 2.355`:
[posed_column_fields.py](../../src/temsim/physics/posed_column_fields.py#L214).
With `B = (partial_y A_z, -partial_x A_z, 0)`, its small-slope Lorentz force at
reference momentum is

```text
x'' = -h_n (x^2 - y^2) - 2 h_s x y
y'' =  2 h_n x y - h_s (x^2 - y^2).
```

These are the native legacy kernel's terms, including skew signs:
[ray_integrator.py](../../src/temsim/physics/ray_integrator.py#L54).
Off-energy particles scale the magnetic force with their actual momentum. The
roughly 10.33-fold installed HP change is consequently a changed physical
setting, not a conversion between these two code paths. It cannot explain the
strict pair that holds the old HP settings equal in both models.

## Round lenses and Lorentz equations

For the production `radial_order=3`, the retained potential and its curl are

```text
A_phi = r B0/2 - r^3 B0''/16
B_z   = B0 - r^2 B0''/4
B_r   = -r B0'/2 + r^3 B0'''/16.
```

The coefficients, Hermite derivatives and curl match these expressions:
[high_order_round_lens.py](../../src/temsim/physics/high_order_round_lens.py#L197).
Independent Fourier-Bessel vacuum-field and numerical curl/jet tests cover the
expansion; all selected operator tests passed below. This supports the signs
and factors, not the accuracy of a particular empirical axial profile for a
real CEOS instrument.

Production selects order three explicitly and disables empirical aberration
and equivalent-image actions:
[unified_field_runtime.py](../../src/temsim/physics/unified_field_runtime.py#L181).
Adding the legacy Cs kick on top would double count an aberration model rather
than repair a demonstrated implementation error.

The unified transverse equations use electron charge `-e`, mechanical slopes,
`p_z = p / sqrt(1 + tx^2 + ty^2)` and the full forward Lorentz slope terms:
[unified_field_particle.py](../../src/temsim/physics/unified_field_particle.py#L362).
The legacy canonical conversion `px = tx - g y`, `py = ty + g x` is local to
each step and converts back to mechanical slopes before returning:
[ray_integrator.py](../../src/temsim/physics/ray_integrator.py#L60).
No evidence was found of feeding canonical momenta into the unified mechanical
slope interface in the strict replay.

## Field ownership and internal configuration

The inspected strict replay has enabled C1, C2, C3, ADL, TL22, TL21, TL12,
Mini Condenser and Objective. TL22 and Mini Condenser have negative polarity;
TL21 has positive polarity. These settings agree between the strict pair.
Image-corrector transfer lenses and image HP elements are disabled. There is no
indication that an enabled overlapping image corrector causes the probe error.

Source selection excludes shared virtual deflector owners and the capture
records each selected physical contribution:
[unified_field_runtime.py](../../src/temsim/physics/unified_field_runtime.py#L84).
The compiled sampler partitions its contributions into selected and remaining
sets, and only adds the latter after compiled sampling:
[unified_potential_sampler.py](../../src/temsim/physics/unified_potential_sampler.py#L80).
The operator tests include a 29-source probe-column fixture with unique owner
checks and original-versus-compiled sampling comparisons.

In the inspected input, HPC, HPOL, QPC, QPH1, QPH2 and QPOL are enabled with
zero strength. They are real available matching/aberration controls if needed
under the newly authorized tuning scope. Their presence is not evidence that
they must be energized or that they can meet the target. A successful fit must
preserve the required crossover chain, actual weights and physical losses.

## A physical model limitation: native multipole fringes

The native HP model is deliberately `A_z = P3(x,y) f(z)` with no transverse
potential components. Its magnetic field is divergence-free by construction,
but it is not a complete source-free vacuum fringe solution. For the harmonic
transverse cubic polynomial, away from support discontinuities,

```text
B      = (P_y f, -P_x f, 0)
curl B = (P_x f', P_y f', 0).
```

This curl is generally nonzero in the fringe even inside the nominal vacuum
bore. The source module explicitly states that it retains the existing
transverse field without introducing a new Maxwell multipole model:
[posed_column_fields.py](../../src/temsim/physics/posed_column_fields.py#L1).

This limitation deserves separate physical qualification when claiming a
quantitative real-corrector result at sub-nanometre scale. It does **not** by
itself explain the old/unified discrepancy: both paths retain the same native
transverse HP coefficients, while their dynamics and round-lens aberrations
differ. This audit does not change the model, invent a fringe correction or
claim that such a correction alone would solve the probe.

## Offline expansion-size check on executed rays

The audit sampled orders 1, 3, 5 and 7 at saved original-source alive-ray
coordinates near each round lens's Gaussian centres and plus/minus one/two
Gaussian widths. It did not integrate the altered fields. The table reports
maxima over those sampled coordinates, not continuous all-trajectory bounds.
The radius/sigma statistic uses the smallest Gaussian width of that lens.

| Lens | Maximum sampled radius, mm | Maximum radius / smallest sigma | Max norm(B3-B1), T | Max norm(B5-B3) / peak norm(B3) |
| --- | ---: | ---: | ---: | ---: |
| ADL | 0.211998 | 0.055064 | 6.797e-6 | 3.213e-7 |
| TL22 | 0.234753 | 0.071137 | 3.120e-4 | 9.654e-7 |
| TL21 | 0.313165 | 0.094898 | 5.390e-4 | 3.263e-6 |
| TL12 | 0.144954 | 0.037650 | 3.149e-5 | 2.536e-8 |
| Mini Condenser | 0.398912 | 0.090662 | 3.292e-4 | 2.780e-6 |
| Objective, incident part | 0.385764 | 0.029225 | 2.904e-5 | 1.083e-9 |

The sampled ray radii are well below the narrowest Gaussian widths. This does
not suggest gross radial-series breakdown in these samples. The order-three
terms are nevertheless appreciable relative to the old first-order field,
especially at TL22/TL21. Order-five increments are much smaller in local field
norm, and order-seven increments are smaller again.

A small local field difference is not a sub-nanometre ray-error bound. Real
order-three versus order-five propagation with identical complete source,
controls and observations remains necessary before accepting a sub-nanometre
working point. Neither changing field order to force a fit nor interpreting
these samples as a converged probe is justified.

Raw offline evidence:
`outputs/finite-field-joint-preset-20261028/saved-field-sample-audit-v2.json`.
Executed helper: `audit_saved_field_samples_v2.py` in the same directory.
Version one omitted the Objective because its nominal centre is 2.5 nm beyond
the sample upper surface; version two includes its incident field. Earlier
files remain unchanged for provenance.

## Verification and next discriminating checks

The targeted field-operator suite passed **92 tests**, with zero failures,
errors or skips, in 25.437 seconds:

```text
tests/test_high_order_round_lens.py
tests/test_potential_plane_sampler.py
tests/test_potential_native_sampler.py
```

The XML is
`outputs/finite-field-joint-preset-20261028/field-operator-audit-tests.xml`.
These are CPU field/value/derivative tests; they do not perform a coherent run
or qualify a new full-column probe. Production source files were not edited.

The useful next controls are segment-local matching and an independently
executed field-order check, followed by exact full-source tracing. Retain
C2/C3 as the primary convergence controls, TL22/TL21 for the internal relay,
ADL/TL12 for entrance/output matching, and Mini Condenser/Objective for focus.
Additional internal multipoles may be evaluated when residual symmetry and
the required optical topology justify them. A local fit must be retraced
before it is used as evidence, and source-count, step, topology and driven-scan
qualification remain separate requirements.

## Independent audit of the subsequent local-transport diagnostic

The parent investigation subsequently executed 17 zero-current phase-space
probes through the production unified response to measure a local mechanical
4-by-4 transfer map. This follow-up review did not execute transport. It
independently checked script/output hashes, finite-difference indexing,
input phase-space ordering, half-amplitude reconstruction, original cohort
IDs, weights and saved-plane matching. The checks reproduced the recorded
quantities.

The linear prediction is the propagated weighted chief plus the measured
local derivative applied to the original input deviations. It is an algebraic
diagnostic at a single weighted mean energy, not a propagated full physical
beam. The figures below all use the same actual 174 specimen survivors.

| Plane and quantity | Centered RMS radius/norm, nm | Centered D95, nm |
| --- | ---: | ---: |
| AC at 1400 mm, actual | 686.101566 | 3116.365350 |
| AC, local half-amplitude linear prediction | 3.781739 | 12.678976 |
| AC, actual minus prediction | 682.713037 | 3103.442551 |
| Specimen, actual | 188.773703 | 854.265910 |
| Specimen, local full-amplitude linear prediction | 0.303018 | 1.063643 |
| Specimen, local half-amplitude linear prediction | 0.204173 | 0.734315 |
| Specimen, actual minus half-amplitude prediction | 188.601164 | 853.443648 |

For residual rows, the norm and diameter describe a distribution of per-ray
position differences; they are not an attainable corrected probe. The parent
report's specimen `actual_minus_local_linear_prediction_position_rms_nm`
value of 188.743151 nm is the **uncentered** norm of the per-ray differences.
It must not be compared as though it were exactly the same statistic as the
188.773703 nm centered physical beam radius.

The full-versus-half derivative amplitude changes the predicted specimen
positions by 0.102967 nm centered RMS. This uncertainty is tiny compared with
the actual 188.774 nm radius, but material for a 0.7 nm D95 acceptance target.
Neither linear prediction is a physically qualified sub-0.7 nm probe.

The diagnostic strongly supports that the hundreds-of-nanometres broadening
lies predominantly outside this local first-order map. It does not uniquely
isolate a pure geometric nonlinear term: actual initial energies range from
300000.01489474834 to 300000.65099111944 eV, whereas all 17 local probes use
the weighted mean energy. The remainder includes chromatic and mixed effects,
finite-amplitude departures and the declared reference approximation. It is
therefore accurate to call it a finite-amplitude/chromatic remainder, but not
to identify a specific aberration coefficient or a single component without
further controlled comparisons.

Raw review:
`outputs/finite-field-corrector-rematch-20261028/local-transport-independent-audit.json`.
Its offline helper is `audit_local_transport_v1.py`; the reviewed executed
helper is `diagnose_local_transport_v1.py`, SHA256
`4a63a24e9b77660e805845bb44cea0738404eca9ce4d88271d25487bb57be5f2`.

## Central matching scale and a direct HP relay preparation

Offline inspection of the saved 59.559740 nm-D95 candidate places its radial
covariance-drift waist at 1201.724210 mm with RMS radius 21.645820 micrometres.
Its executed 1208.15 mm geometric midpoint has RMS radius 32.407761
micrometres. The old-control nominal-energy local linear map instead gives a
covariance-drift waist at 1202.054104 mm. Although these are different
diagnostics, both references are upstream of the geometric midpoint: the
approximately 6 mm displacement cannot be attributed solely to the broad
strong-HP finite-beam statistics.

Existing small full-source differences around `high-angle-restart` give the
following local covariance-waist position sensitivities, in millimetres per
percentage point of excitation magnitude:

| Changed lens | Central covariance-waist Z derivative | AC covariance-waist Z derivative |
| --- | ---: | ---: |
| TL22 | -1.166048 | -4.927588 |
| TL21 | 0 at recorded precision | -8.404130 |
| ADL | -0.005633 | -0.039587 |
| TL12 | 0 at recorded precision | -0.881574 |
| C2 | +0.016804 | +0.028299 |
| C3 | -0.019934 | -0.191376 |
| Mini Condenser | 0 at recorded precision | 0 at recorded precision |

These derivatives come from previously executed 0.001 percentage-point
perturbations and the same 174 original IDs; they are not derivatives at a new
large excitation change. Their scale indicates that the former TL22 trust
interval of only 0.5 percentage point is insufficient to explore a roughly
6 mm central matching displacement. A linear estimate near 5.5 percentage
points is a scale diagnostic, not a validated candidate. TL21 and other
matching controls must then be retraced to preserve the HP relay and
downstream images. Central covariance minima are not automatically common
two-dimensional executed waists or zeros of one Gaussian fundamental ray.

The current Normal entrance condition remains unchanged. C3 and neighbouring
fields act together; no separate C3-ADL crossover is introduced merely to
match the labels on a manufacturer schematic.

`measure_hp_relay_local_v1.py` was prepared in
`outputs/finite-field-corrector-rematch-20261028/` for a direct 17-probe
HP2-centre to HP1-centre map of the best candidate. Its default action is
preparation only; the independent audit did not execute it. It retains all
captured overlapping fields, reports mechanical and normalized canonical
maps, half-amplitude changes and magnification singular values, and has no
minus-identity constraint. Direct measurement avoids composing poorly
conditioned gun-to-HP inverse maps at approximate HP planes.

Raw offline waist evidence is `saved-corrector-waist-audit.json` in that
directory. Neither this preparation nor the offline gradients qualifies a
new preset.

## Review of the downstream branch proposal

The parent's separately executed `propose_downstream_branches_v1.py` uses the
centred paraxial Larmor equation with signed round-field superposition,

```text
u'' + [B0_total / (2 p/e)]^2 u = 0.
```

The audit confirms that the code sums the signed fields before squaring,
preserving Mini Condenser/Objective overlap cross terms. Its Gaussian widths,
SI lengths, relativistic rigidity and RK4 stages are consistent. The selected
input has zero quadrupole and DC steering settings and centred round fields,
so this is a useful small-amplitude branch proposal. It omits finite-amplitude
multipoles, source chromaticity, geometry stops and nonlinear Lorentz effects.

At Mini Condenser 43.790282 percent, the sampled sign-change search found two
specimen-focus roots: Objective 25.448359 percent with no detected internal
zero, and Objective 69.016299 percent with a zero near 1585.931243 mm. The
current 69.016744 percent setting is close to the second root. These are
found roots of the approximate initial-value problem, not an exhaustive root
proof or a full-source crossover count.

Exploring the lower branch while rematching downstream angular transfer is a
meaningful controlled experiment. A smaller Objective excitation alone does
not prove smaller natural Cs or C5: magnification, beam heights, conjugates and
overlap also change. The actual full-source alpha95, D95 and complete crossover
chain must decide whether the branch improves the requested probe. The term
"field-free-axis approximation" in the original helper's introductory text
should be read as a centred paraxial/Larmor approximation; its retained axial
magnetic field is not zero.

## Executed direct HP relay and independent energy check

The parent subsequently executed the prepared direct HP relay diagnostic at
300000.3 eV in 11.30 seconds. This audit independently verified its saved
canonical arrays, finite-difference reconstruction and script/data hashes.
At the exact HP1 centre, the half-amplitude map gives:

| Quantity | Measured value |
| --- | ---: |
| Canonical B-block Frobenius norm | 1.382305965e-7 m |
| Canonical B-block spectral norm | 9.774379219e-8 m |
| Position-magnification singular values | 1.000001602649, 1.000001602649 |
| Full-versus-half-amplitude B-block difference | 3.553397216e-12 m |

The canonical angular coordinate is transverse momentum divided by the
reference momentum; it is approximately an angle in radians near the axis.
Thus the B-block units are metres per unit normalized transverse momentum.
For example, a 1 mrad differential input gives approximately 0.0977 nm of
position coupling from this B block. The 1.38e-7 m matrix norm is **not** a
138 nm physical probe radius. The C block remains nonzero, with diagonal
entries approximately 9.668369 m^-1, so this is not a full minus-identity map.

A separate, inexpensive centred linear Gaussian Larmor matrix calculation
used the same signed superposed axial round fields and both 0.05/0.025 mm
steps. This is an approximate operator audit, not a new production particle
calculation or an emitted beam. At the finer step it gives:

| Initial energy | B-block Frobenius norm |
| --- | ---: |
| 300000.0 eV | 1.184862151e-10 m |
| 300000.3 eV | 1.382629395e-7 m |

At 300000.3 eV its B block differs from the direct production result by only
3.234609e-11 m in Frobenius norm. The energy increment therefore explains the
observed B-block scale. Comparing this 300000.3 eV diagnostic directly with a
nominal 300000.0 eV historical conjugacy would misleadingly suggest loss of
matching. The present evidence does not support a gross first-order HP-core
conjugacy failure as the explanation for the large probe.

This result does not settle finite-amplitude relay aberrations, the central
source-image location, or downstream matching. A near-unit local position
relay can coexist with a broad finite beam and cannot qualify its D95.
Likewise, changing the downstream focus branch alone is not a demonstrated
solution: the parent's old-HP first-branch full-source trial still had D95
approximately 721.819 nm.

Raw evidence is
`outputs/finite-field-corrector-rematch-20261028/hp-relay-best-candidate/` and
the independent `hp-relay-energy-independent-audit.json` in its parent.
The energy audit helper is `audit_hp_relay_energy_v1.py`.

## Direct-focus condition: full-source downstream waist audit

The user subsequently required Mini Condenser and Objective to focus directly
at the specimen, with no additional pre-specimen crossover. The independent
offline audit therefore inspected every saved plane from the Mini Condenser
physical body entrance at 1512.5 mm to the upper specimen surface at
1599.1999975 mm. It also recorded the centre-to-surface and AC-to-surface
intervals separately. All compared cases retain the same actual 174 original
IDs throughout this downstream interval, with unchanged original weights,
90.1554404 percent source transmission and no downstream losses.

Detection requires a negative-to-positive radial position/slope covariance
bracket, a radial minimum, and positive-definite growth of the transverse
position covariance on both sides. It therefore rejects a line focus or
individual ray crossing as a two-dimensional current-carrying waist. Sampled
brackets and interpolated radial candidates remain explicitly distinct.

| Executed case | Downstream evidence before specimen | Specimen D95, nm | Specimen covariance-drift focus offset, nm |
| --- | --- | ---: | ---: |
| Previous strong-HP candidate, Objective approximately 69 percent | Additional executed 2D bracket around 1585.7 mm, plus an unresolved near-surface radial candidate | 59.559740 | -16.390 |
| First downstream focus branch, old HP | No distant 1585 mm waist; a misplaced near-surface waist bracket remains upstream | 721.818932 | -7928.856 |
| First downstream focus branch, strong HP | No pre-specimen 2D waist or radial crossing candidate detected at the declared sampling | 151.869695 | +1560.199 |

The previous strong-HP candidate's extra executed bracket spans
1585.203651293-1586.203390308 mm. Its narrowest stored plane is
1585.703520800 mm; a separate Hermite interpolation suggests 1585.935419596
mm. The interpolated value is not an exact executed minimum. This extra
waist directly conflicts with the newly required direct-focus topology.

The first-branch old-HP case has a broad executed bracket from
1598.700127993 to 1599.199897500 mm, with the narrowest stored node at
1599.199797500 mm. The interpolated radial candidate is instead
1599.192068611 mm, approximately 7.929 micrometres upstream of the specimen.
Reporting the stored node as its exact focus would be misleading. Positive
covariance at each of the final three planes independently shows that the
beam is already diverging when it reaches the specimen.

The first-branch strong-HP case has negative covariance at each of the final
three planes: it is still converging at the specimen. Thus absence of an
earlier waist does not establish focus at the specimen. Its principal-axis
covariance-drift offsets are approximately +1445.945 and +1671.673 nm; these
are local diagnostics, not exact finite-field waist locations. It remains
unqualified for both size and focus.

The maximum downstream observation gap is 0.499869507 mm, with 195 retained
planes spanning the body-entrance interval, and the underlying particle
integration limit is 0.05 mm. This establishes the stated fixed-sampling
evidence, not an exhaustive continuous absence proof. Finer exact-plane and
integration checks remain required, especially near the specimen. The sample
is an endpoint observation and has no downstream two-sided focus bracket in
these incident-only captures.

The corrected-denominator audit is
`outputs/finite-field-corrector-rematch-20261028/direct-focus-waist-audit-v2.json`,
produced by `audit_direct_focus_waists_v2.py`. Each waist records both its local
cohort fraction and its original complete-source current fraction. Version
one retained the lower-level detector's local-cohort denominator in a field
named `source_current_fraction`; version two makes the denominators explicit.
No physical input, source, propagation or saved trajectory was changed.

## Bounded acceleration for proposal diagnostics

The investigation added an ignored-output helper for faster optimization
diagnostics on this workstation. It does not replace a production solver.
`fast_static_diagnostic_v1.py` calls the existing compiled plane-coefficient
and plane-field evaluators inside a Numba loop using the same forward static
Lorentz RK4 equations and production integration nodes. All 193 original
coordinates, energies, clocks, IDs and weights are retained as **zero-current
diagnostic probes**. Material interceptions are intentionally omitted, so its
output cannot qualify physical current, transmission, crossover topology or
specimen size. Every selected candidate still needs full production tracing.

The retained electric field could not be omitted: at the 450 mm gun exit its
sampled axial component is approximately 4.66e-7 V/m, not exactly zero. The
helper therefore retains the same bounded native `r^2-z` bilinear potential
and its exact derivatives, including actual `energy + phi(z) - phi(start)`.
The two gun magnetic alignment potentials are absent only because their
entire declared compact supports are strictly upstream of the diagnostic
interval. No field is deleted using a magnitude threshold.

The diagnostic enforces static aligned native providers, finite coefficient
admission, one-sided compact-field RK endpoints, and native domains. The
guarded API in `fast_static_diagnostic_v3.py` imports the executed v1 numerical
kernel unchanged and adds source/beam-voltage consistency, a conservative
complete-sweep domain certificate including final RK updates, and checks that
the source implementation and input state remained unchanged. Its default
action is preparation only. Version two stopped during preparation because a
derived beam-voltage property was incorrectly looked up as a top-level JSON
key; it executed no diagnostic transport and is retained for provenance.

The one completed baseline diagnostic used 26,728 production integration
nodes and 3,697 observed planes. Thirty-one sampled-plane operator checks gave
exactly zero difference in scalar potential, electric field and magnetic
field. The complete 193-column diagnostic took 15.003 seconds, including
initial integrator JIT compilation. Its maximum stage radius was 0.628974 mm,
within the 9 mm conservative native-cylinder bound.

Comparison against the original production run's 174 actual surviving IDs
gave **exactly zero numerical position and slope difference at every retained
plane**. This validates that particular diagnostic regression; it does not
authorize treating the other 19 physically intercepted rays as transmitted
electrons or assuming every future candidate has the same interception mask.
The full 193-column diagnostic remains saved, including those extra
counterfactual continuations. Only the comparison statistic selects the
independently recorded physical survivor IDs.

Raw baseline evidence:
`outputs/finite-field-corrector-rematch-20261028/fast-static-baseline-v1/report.json`.
The final wrapper's preparation is `fast-static-guarded-preparation-v3/`.
No production source file, default, field formula or cache behavior was
modified for this acceleration.

## Independent review of the first bounded nonlinear proposal search

The read-only review of `optimize_fast_first_stage_v1.py` found no source
filtering, weight replacement, C2/C3 adjustment or bypass of the guarded field
domains. Its reference is the completed `central-angle-matched-v5` physical
run, in which all 193 original particles arrive. The initial guarded
diagnostic reproduced every saved position and slope of all 193 columns
exactly. Its initial metrics are alpha95 29.919607456 mrad, D95
140.003926451 nm and RMS radius 27.126047861 nm. This second exact regression
supports using the accelerator at the new starting point; it does not turn
later diagnostic trials into physical results.

The parameter constructor at helper line 106 changes only common HP
strength, HP1 relative strength, HP1 orientation and Objective excitation.
The declared bounds correspond to respectively +/-30 percent, +/-2 percent,
+/-0.002 rad and +/-0.02 percentage points. C2/C3, Mini Condenser, geometry,
source, energies and all other controls are restored from the same exact
input for each evaluation. `checked_trial_controls` adds the existing
control-domain checks, and every evaluation uses the guarded v3 preparation
and execution functions. Finite-difference probes use the same full cohort.

The objective uses original-weighted centered sample positions, directional
position/slope covariance, angle preservation, regional image penalties and
a downstream radial-focus penalty. Normalizing weights for expectations
does not change the original saved weights or reinterpret transmitted
current. The angular statistic reconstructs unit direction vectors and uses
their current-weighted chief direction, consistent with the production
`transverse_beam_statistics` definition. No free recentering action is applied
to trajectories: centering is only the declared D95/RMS statistic, and the
actual centroid is recorded separately.

The regional radial-root count and 500 nm final-focus screening zone at
helper lines 119-170 and 201-204 are proposal filters. They neither establish
exactly one two-dimensional physical waist in each interval nor prove exact
surface focus. A radial root within that 500 nm zone is not acceptance of a
separate pre-surface waist. All outputs explicitly remain unqualified and
require production interception, exact-plane/finer-grid waist checks and
source/step convergence. The balanced residual may retain a point with a
larger D95 than the minimum encountered if its other diagnostic constraints
are better; the full history must remain available when choosing a physical
validation candidate.

One provenance label at helper line 196 is imprecise:
`input_snapshot_sha256` hashes the reference input file, not that evaluation's
candidate. The immutable reference plus recorded scaled parameters fully
determine each candidate, and the selected candidate has a separate actual
file hash. Treat the per-evaluation field as `reference_input_sha256`; do not
present it as a candidate hash. This was reported to the fitting agent without
editing a helper after execution started.

## Prepared field-order convergence review: no additional execution

The production capture explicitly selects `radial_order=3` in
`src/temsim/physics/unified_field_runtime.py:181`. As documented in
`src/temsim/physics/high_order_round_lens.py:44`, this means the highest radial
power of A_phi, not the order of the integrated ray map. Full Lorentz
transport through this field can still produce fifth- and higher-order ray
responses. Consequently, a fitted fifth-order spot pattern does not by
itself prove that the implementation needs an empirical fifth-order kick.

For a promising size candidate, a controlled comparison should retain its
exact settings and complete physical source while changing only the round
field expansion from order 3 to order 5. The existing immutable
`HighOrderRoundLensPotential` can be replaced in a separate diagnostic
snapshot while preserving its native Gaussian field, pose, validity radius,
axial policy and remote-domain policy. Preserve every non-round electric and
magnetic provider, the gauge and the mechanical snapshot. The resulting
runtime must carry distinct field/runtime digests and explicit metadata for
the order override; it must not replace the default runtime or enter the
shared state cache. `build_unified_plan` already accepts an explicit captured
runtime at `src/temsim/physics/unified_field_particle.py:44`, and the production
executor validates plan/runtime ownership at line 187 onward.

The minimum useful comparison is a fixed-input order-3/order-5 pair with
identical nodes, followed by half-step checks so integration error is not
mistaken for field truncation. Compare original-ID sample position and slope
differences, spatial and angular current quantiles, centroid and directional
focus covariance, as well as actual interception identities and waist
brackets. Each physical run uses its own physically surviving population for
acceptance statistics; additionally report their common IDs with original
weights to distinguish field sensitivity from changes in clipping. If the
order-3/order-5 discrepancy is material relative to the remaining size or
focus margin, an order-5/order-7 comparison is needed before claiming radial
convergence. Do not separately retune each order before this sensitivity
comparison, because that would conceal the fixed-working-point discrepancy.

A zero-current accelerated comparison could help prioritize this work, but
would retain its existing proposal-only scope. The earlier sampled field
differences and the successful native operator tests do not replace a
propagated field-order comparison. No order-5 or order-7 full-source
propagation, model change or new qualification has been performed by this
audit.

## Executed bounded order-3/order-5 sensitivity of the first optimized candidate

After the preceding preparation, the parent authorized one bounded comparison
of the completed `first-focus-hp-optimized-v1` working point. This is an
alternative-field, **zero-current diagnostic**, not a change to the production
field model. The helper `compare_fast_field_orders_v1.py` retains the existing
v3 source and swept-domain guards, full finite electric field, all 193 source
columns and the production integration mesh. It creates detached immutable
order-5 round providers and a new snapshot/runtime, then recompiles native
coefficients; it does not edit packed coefficients, production source, shared
runtime caches or defaults. All 13 selected round providers are changed in
order only, including retained overlapping Gaussian tails. Their native
field fingerprints, poses and domains remain identical.

Both runs used exactly the same 26,727 integration nodes and candidate input.
Each order passed the 31-plane retained phi/E/B operator equivalence check,
with zero numerical differences. The kernel elapsed times were 6.685 seconds
for order 3 and 6.548 seconds for order 5. Maximum swept stage radii were
0.479908069 and 0.479907686 mm respectively, within the identical conservative
9 mm native cylinder. The complete finite electrostatic snapshot is the same
object in both prepared models; initial energies and potential reference are
unchanged. No source particle is deleted or reweighted.

| Fixed-input diagnostic statistic | A_phi order 3 | A_phi order 5 |
| --- | ---: | ---: |
| D95, nm | 7.103511535 | 7.122427598 |
| RMS radius, nm | 1.643926338 | 1.647772395 |
| alpha95, mrad | 29.988918357 | 29.988899004 |
| Local covariance-drift focus offset, nm | +0.068577367 | -0.738192887 |

The order-5 minus order-3 sample position difference has original-weighted
uncentered RMS 0.019976097 nm and centered RMS 0.019954449 nm. D95 changes by
+0.018916063 nm and alpha95 by -0.000019353 mrad. The largest individual
position difference across all retained planes is 0.946966491 nm; this is a
maximum along the column, not the sample probe-size change. The comparison
stores all 193 per-ID sample differences and along-column maxima, together
with full paired trajectories and original IDs/weights.

An independent saved-data audit verified all hashes and recomputed the sample
statistics through production `transverse_beam_statistics`. It also confirmed
that the order-3 diagnostic phase coordinates are bitwise identical to the
completed actual production run at every saved plane for all 193 particles.
This reference happens to have all 193 physical arrivals. The order-5 run
still omits physical interception and cannot independently certify its own
transmission or topology.

At this fixed step and working point, the measured radial-field truncation
sensitivity is much smaller than the remaining approximately 7 nm D95. It
does not explain that residual width by itself. The approximately 0.02 nm
sample difference and 0.807 nm shift of the local focus diagnostic remain
relevant when approaching a sub-nanometre target. No half-step or order-5 to
order-7 comparison has yet been performed, so no field-order convergence
claim follows from this pair.

Evidence is `fast-field-orders-stage1-v1/report.json`,
`paired-zero-current-observations.npz`, and `independent-saved-data-audit.json`
inside `outputs/finite-field-corrector-rematch-20261028/`. The independent
reader is `audit_fast_field_orders_v1.py`. The earlier preparation-only record
remains separately preserved in `fast-field-order-preparation-stage1-v1/`.

## Review of the second bounded correction stage

The read-only review of `optimize_fast_second_stage_v1.py` confirms that its
ten variables retain the first stage's four controls, add normal/skew
coefficients for HPC and HPol, and add the two Condenser Stigmator channels.
C2/C3, Mini Condenser, transfer lenses, component geometry and source remain
fixed. The original HPC/HPol controls are enabled with zero strength and
orientation, so the zero vector reproduces their existing baseline fields.

The auxiliary hexapole conversion is consistent with
`src/temsim/optics/hexapole.py:135`: a nonnegative coefficient magnitude and
`atan2(skew, normal)/3` reproduce the requested peak normal/skew coefficients.
These are peak m^-3 coefficients, not integrated kicks. Their finite Gaussian
lengths and physical supports remain unchanged. The configured search bounds
are +/-2e6 m^-3 per HPC component and +/-1e5 m^-3 per HPol component.

The unchanged `normal_skew` stigmator model uses twice the mechanical channel
angle (`src/temsim/optics/stigmator_field.py:27`). Channels at 0 and 45 degrees
therefore span independent normal and skew quadrupoles; they are not two
independent diagonal focusing powers. With the captured maximum coefficient
of 300 m^-2, each +/-0.2 percentage-point search interval corresponds to
+/-0.6 m^-2. The captured native polynomial uses these trace-free components
in `src/temsim/physics/posed_column_fields.py:240` onward. No field convention
or orientation change is required to use the two correction directions.

The second stage adds the actual original-weighted diagnostic 95-percent
radius as a scalar residual alongside RMS positions and the existing focus,
angle and image-region penalties. This better exposes the user-selected D95
criterion to the objective, although the finite weighted quantile is not
smooth when its selected particle changes. Solver convergence alone cannot
establish optical qualification. Each evaluation now records both the
reference input hash and a separate canonical hash of the complete candidate
input, resolving the first stage's provenance label ambiguity. The same
explicit zero-current, no-interception and 500 nm screening limitations
continue to apply.

## Prepared complete-source count-convergence protocol: not executed

The next count study is held until a useful sub-nanometre candidate exists.
No 769-particle source or column has been executed by this preparation. The
existing `scripts/validate_particle_count_convergence.py` supplies the main
physical workflow: it changes the native emitter count, calls
`incident_checkpoints`, and reports original-weighted statistics. The unified
implementation of that observation API at
`src/temsim/optics/beam_path_audit.py:229` obtains the gun through
`trace_source_to_exit` and executes every column field and mechanical-contact
step. Its checkpoint-only retention reduces memory without skipping physical
steps or deleting original source columns.

The existing count script should not be run unchanged as full current
qualification. Its default waist windows are historical coordinates,
including the rejected old downstream 1585.9 mm branch; it does not export
the full launch reference or explicitly assert the requested count and
complete IDs; its local radial focus statistic alone cannot prove a common
two-direction focus. It is suitable as a basis for a small output-side
observer using the current candidate's planes and the checks below. The
193-specific `measure_probe_fields_v2.py` and accelerated diagnostic helpers
must not be relabelled as 769-particle tools: their count/source guards
deliberately reject that substitution.

### Source family and cache boundary

The present emitter has no custom quadrature, Sobol sampling, surface or
coherent source override. Its native classical sampler is the deterministic
six-dimensional Halton family, with bases 2, 3, 5, 7, 11 and 13 and indices
starting at 18 (`electron_gun/emitter.py:30` and `:422`). Preserve that family
and all physical source settings for 193, 769 and, if needed, 3073 particles.
Do not switch the family to Sobol merely to obtain powers of two.

Spatial and angular Halton coordinates of the prefix are nested, but physical
energy samples are not count-independent: `src/temsim/physics/chromatic.py:38`
centres and rescales the complete sampled energy distribution, followed by
the native positive-energy mapping. Thus increasing N may change energies,
gun-exit phase coordinates and clocks even for a numerically repeated ray
ID. Each count must emit and trace its entire native source. The original
193 gun output cannot supply the prefix of a 769 run, and there is no
justified shortcut that simply appends 576 trajectories.

The native weights for these independently generated quadratures are 1/N;
this is the unchanged source integration rule, not post-interception current
renormalization. Record weights directly from the new source and leave them
unchanged throughout its gun and column. Total emitted current, source size,
angular distribution and energy-distribution parameters remain fixed.
Numeric ray IDs are local to each source population; use the population
digest together with the ray ID when comparing different counts.

The production gun cache key includes `requested_count`
(`electron_gun/field_emission.py:808`), and `gun_dependency_signature` also
binds the current solver implementation and gun flight-time schema
(`physics/particle_sections.py:148`). An exact previously executed 769 gun
artifact could be reused only after those complete signatures and all source
arrays are verified. The existing 193 cache is necessarily incompatible.
For the first 769 observation, use a fresh process and no injected exit NPZ
or prior 193 section; native field-solve caches may still be reused when
their own physical inputs match.

### Required small observer implementation

1. Restore the final candidate JSON exactly and freeze its current solver and
   external-input identities. Change only the emitter's actual `ray_count`;
   compare dictionaries after replacing that one explicitly identified
   field, rather than stripping every key named `ray_count`. Keep the column
   integration step and all gun integration settings fixed in the count
   pair. A half-step trial is a separately labelled numerical comparison.
2. Use `incident_checkpoints` with the final 193 candidate's exact published
   planes. Retain source exit, component and region boundaries, aperture
   entrance/exit observations, central/AC dense waist windows, the complete
   downstream direct-focus interval, and the three exact near-surface
   planes. Retain both directional covariance information and radial
   statistics; no automatic refocus, rejection of tails or new fitting is
   allowed between counts. Confirm integration-node identity across counts
   at the same step. The current 5 mm internal checkpoint grid remains the
   same for all three proposed counts under its 512 MiB budget.
3. Assert exactly N unique original IDs, N original weights, N source energy
   offsets and matching phase-space dimensions. Export the actual gun launch
   reference, gun-exit phase, original weights, energies, alive/blocked
   metadata and flight times. Validate positive finite launch energies and
   finite nonnegative arrival clocks for every ray that reaches the gun
   exit; stopped rays retain the production stopped/unknown-clock semantics.
   Validate ancestry against the recorded emission reference, not inferred
   positions. Compute a new source-population digest from the full arrays.
4. Retain all N columns and physical masks at every observation plane. Check
   all live phase, energy and clock values for finiteness, and that the
   initial column row agrees with the actual new gun exit. Report physical
   aperture losses using original weights and emitted current. Export source
   and observation hashes, snapshot, requested/actual counts, field/runtime
   identity, timing and memory receipts. A saved NPZ is evidence, not a
   replacement configurable downstream source.
5. Recompute RMS, D95, alpha95, centroid, actual current, directional focus
   covariance, central/AC waist brackets and direct-focus topology at each
   N. Different N are distinct quadratures, so report ensemble convergence
   rather than asserting per-ID equality of their energies or trajectories.
   A 193-to-769 pair is a sensitivity observation. If stability is marginal
   relative to the remaining D95/focus margin, extend to 3073. The existing
   count summary requires two successive passing refinements and does not
   consider one stable quantile sufficient evidence of convergence.

The existing numerical comparison budgets are 0.25 percent relative for
alpha95, 2 percent for RMS/D95/current, and 0.25 nm absolute for its local
radial-focus diagnostic. These are existing convergence-screening budgets,
not new user requirements or substitutes for the less-than-0.7 nm D95 target,
30 mrad angle target and common two-direction focus. Report the numerical
changes and remaining acceptance margin explicitly; internal topology and
directional focus require their own checks.

### Storage and time planning

The completed stage-one candidate retained 3,696 observation planes on 26,727
integration nodes. Its production column took 145.432 seconds for 193 rays;
total saved trial time was 180.557 seconds with the gun already cached. The
following estimates preserve all those observation planes and use the
production sparse-history budget formulas. They exclude the electric field
arrays, interpreter, gun history and serialization overhead.

| Complete source N | Raw published arrays, MiB | Sparse executor estimate, MiB | Observation assembly estimate, MiB | Estimated compressed observation file, MB |
| --- | ---: | ---: | ---: | ---: |
| 193 | 33.36 | 123.84 | 99.43 | 24.93 |
| 769 | 132.85 | 481.73 | 395.85 | 99.32 |
| 3073 | 530.78 | 1913.30 | 1581.53 | 396.91 |

The compressed sizes are linear extrapolations of the actual 193 archive,
not guaranteed compression ratios. Keeping a full incident `.temsection`
would instead extrapolate to approximately 849 MB at 769 and 3.39 GB at 3073;
it is unnecessary for this observation-only convergence study. Prefer the
bounded checkpoint observer and retain the actual source receipt separately.
The larger count may require batching and can approach the 2 GiB executor
budget; use its existing admission calculation rather than assuming the
estimate is a measured peak-memory guarantee.

Linear extrapolation of the measured 193 column time gives approximately
579 seconds for 769 and 2316 seconds for 3073, before a new real gun trace
and output serialization. Fixed per-node overhead can make scaling smaller
than linear, whereas other workstation activity and newly sampled slow gun
electrons can increase it. Plan roughly 15 minutes and a cooperative
20-minute cap for the first 769 trial, then update the estimate from its
actual stage timings before deciding on 3073. A cap expiry is an incomplete
observation, not convergence or acceptance. Run one numerical worker and no
concurrent optical optimization. No timing guarantee or new execution is
implied by this preparation.

## Fourth-stage search: source, aperture and topology review

The independent read-only review of `optimize_fast_fourth_stage_v3.py` found
the proposed scope consistent with the current control roles. Its fifteen
variables include C2 as the dedicated 30 mrad angle control, with C3 fixed;
TL22/TL21 adjust the core transfer, while Objective compensates coupled
CM/TL12 and correction changes. The expanded bounds are +/-1 percentage point
for C2/TL22/TL21, +/-3 percentage points for CM/TL12/Objective, +/-2 percent
HP1 relative strength and +/-5e5 m^-3 per HPol component. Original component
polarity, geometry, source and enabling remain protected by exact input
comparison after neutralizing only declared strength controls.

The helper explicitly separates the completed physical source reference
`first-focus-hp-optimized-v1` from the initial phase archive generated by
`fast-third-stage-v1`. It verifies the latter's input/archive hashes, completed
diagnostic status, source identity, solver identity, all 193 IDs and original
weights, initial energies, initial phase and initial clocks. The first new
forward evaluation is a regression against the saved stage-three diagnostic,
not a claim of production agreement for that changed candidate. The
production source continues to provide the actual initial phase/energy/clock
vectors to the guarded kernel.

The initial stage-four preflight correctly rejected missing exact aperture
face observations: these faces were already in the integration grid but were
not retained in the earlier sparse archive. Version three's
`add_aperture_face_captures` extends only the saved-row map. It does not insert
an integration node or interpolate a ray at a face. Each candidate must keep
the full native integration grid bitwise identical; original observations
are extracted through the recorded mapping for regression and moment
analysis. The separately captured face values are used only for aperture
clearance diagnostics. The unchanged v3 kernel still checks source identity,
finite electric fields and every ray's complete swept magnetic domain.

Independent inspection of `fast-fourth-stage-v3-preflight/prepared.json`
confirmed that all eight C2/C3 plate/housing faces are exact existing native
nodes, with zero nominal-to-recorded axial difference. Four extra saved rows
are needed; the original 3,696 observations remain intact and ordered.
The checked helper SHA-256 is
`b76ad4e987f8bf2ff9386269db71d348c2cc92512abceb36b2203c8dc9c1654d`.
The unchanged native-grid byte hash is
`1654ba9b5bd99e534512e41939b2e88fa19013ec391c467399144f15b0744144`.

All 193 columns remain in every statistic. If C2 causes interception, the
reported alpha95 is therefore a **counterfactual zero-current diagnostic**,
not the physical transmitted alpha95. This limitation is explicit in each
metrics record. Requiring positive clearance at the sampled faces is a
conservative search-ranking rule, not proof of clearance between faces and
not a new user requirement for 100 percent transmission. Physical candidate
validation must retain complete material interception and report its actual
current quantiles and losses; the user has not imposed a current floor.

The Normal screening checks one radial covariance sign-change bracket from
C2 to C3 and none from C3 to ADL. Central/AC and downstream tests likewise
remain preliminary radial/finite-depth diagnostics. They do not certify
two-dimensional topology. Nonzero directional stationary-plane separation
is reported relative to finite waist depth, without forcing it to zero or
counting it automatically as two physical source images. The 500 nm final
focus screening zone retains its previous diagnostic-only interpretation.

Each evaluation saves its weighted residual and exact candidate hash; each
completed Jacobian saves the matrix, singular values and distances to bounds.
Those derivatives describe the declared scaled controls and local residual,
not fitted physical aberration coefficients. Although the sample-position
terms are smooth, the retained alpha95 quantile and one-sided screening
penalties need not be globally smooth. The scope review permits the bounded
proposal experiment under the parent's execution schedule; this audit
performed no new forward transport and grants no optical qualification.

### Implemented preparation-only complete-source observer

The bounded output-side observer is now prepared as
`outputs/finite-field-corrector-rematch-20261028/observe_complete_source_count_v2.py`.
Its default action is preparation only. It accepts the complete candidate
JSON, one count from 193/769/3073, a new output directory, optional existing
observation planes, and at most 256 additional in-range observation planes.
There is no argument for injecting an exit bundle, prior gun section or
particle subset. Only `--execute` starts emission and real transport.

Preparation reuses the geometric Normal, central/AC, aperture and near-sample
plane builders, then constructs the actual production integration plan and
mirrors the production sparse-history admission before any new gun work. A
769 preparation using the stage-one candidate retains 3,696 observations and
26,727 integration nodes, fits in one 769-column storage batch, and estimates
481.735 MiB for sparse executor arrays. Source and column execution counts
remain zero. This is a preparation demonstration, not a newly qualified
769-ray optical result.

Execution is designed to call the native emitter for an independent launch
reference and then the production `incident_checkpoints` workflow, which
generates and traces its real complete gun. It requires a fresh process with
no existing executed gun trace cache. All emission positions/directions,
original IDs/weights, actual gun-exit phase/energy/clocks and initial column
states are validated and saved. Production fields, finite electric work,
mechanical contacts and sparse observer batching remain in force. The tool
records current before/after each aperture and actual gun-plane arrivals,
two-direction sample moments and the complete sampled crossover chain. It
does not label one count as convergence or treat an unresolved focus as
qualified. A cooperative wall-clock guard and frozen solver/input identities
remain active during the real gun and column calculation.

Version one is preserved as its original preparation artifact. Version two
adds the exact production sparse-batch and observation-assembly admission
before the gun is started; no numerical propagation formula was changed.
The latest helper SHA-256 is
`6e25b3e95d5c132c8480ca3cd6971db87c96812bd935474de8b9752882c9c162`.
Syntax and preparation checks passed. The explicit execution branch has
**not** been run, so production observations must still validate its complete
runtime path before its output is used as count-convergence evidence.

A bounded input-only audit restored 193, 769 and 3073 variants and confirmed
that the actual emitter count is their only input difference. Their physical
input signature, with only that one count field replaced, is identical.
The prepared 769 gun signature
`25835a2fb85287c3f2f033bab41d549c9cb195931b159682962b0545ad9d0cad`
differs from the existing 193 gun signature
`0dce9e48f36759d5b609d09197eb8fd8ec35d5d9f1c2d17f8dd61bb513f7162f`.
The receipt is `source-count-preparation-769-v2/input-invariant-audit.json`.
That audit neither emitted a source nor propagated particles.

Preparation command, with a new empty directory for every invocation:

```powershell
.venv/Scripts/python.exe outputs/finite-field-corrector-rematch-20261028/observe_complete_source_count_v2.py --inputs CANDIDATE.inputs.json --rays 769 --reference-planes CANDIDATE.observations.npz --output NEW_OUTPUT_DIRECTORY
```

After a suitable candidate is selected, the separately authorized physical
run uses the same arguments, a different new output directory, and
`--execute --max-seconds 1200`. A 3073 trial requires a separately evaluated
time/memory budget. No such execution has been started by this preparation.

### Measured fractions of the first stage-four bounded response

The first stage-four Jacobian has a bounded linear solution with residual
norm 0.2148767271, compared with 1.4110854433 at its stage-three initial point.
That residual combines several diagnostic terms and is not an executed probe
radius. A separate immutable helper,
`outputs/finite-field-corrector-rematch-20261028/probe_bounded_response_fractions_v1.py`,
tested physical parameter increments at fractions 0.05, 0.15 and 0.35. Its
SHA-256 is
`861fd165b530d11240bcbdd08ae116e4577452253e40048b7e180395fa0563c8`.
The saved increment vector already uses physical units; each component was
multiplied by the fraction once. In particular, Objective/round-lens controls
are percentage-point increments, HP orientation is radians, and auxiliary
normal/skew strengths are in inverse cubic metres. HP1's common and relative
strength factors retain their multiplicative composition.

This experiment used the same original 193 column initial states, absolute
weights, energies and clocks as the authoritative physical source, with zero
current and material contacts deliberately omitted. All 26,727 integration
nodes and native field/domain guards were preserved. The same exact existing
C2/C3 face nodes were retained as extra observations. The newly executed
initial point reproduced every original saved phase and flight-time element
bitwise. No production source or optical default was changed.

For each fraction, radial refocus used two actual Objective forwards at
plus/minus 0.00001 percentage points to differentiate the sample covariance.
The resulting Objective update was then propagated through the full fields.
There were at most two Newton updates, with a maximum additional Objective
correction of 0.02 percentage points. Local drift lengths were only reported
as normalized covariance descriptors; no drifted ray bundle replaced an
actual forward. The complete run used 16 forwards and 180.39 seconds on one
CPU thread. Its immutable evidence is in `bounded-response-fractions-v1`.

| Fraction | Unrefocused D95 (nm) | Actually refocused D95 (nm) | Refocused RMS radius (nm) | Refocused alpha95 (mrad) | Additional Objective correction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial point | 5.133828 | 5.133828 | 1.392492 | 29.988771 | 0 |
| 0.05 | 6.234192 | 4.958347 | 1.332423 | 29.987598 | -0.000216620 |
| 0.15 | 64.987558 | 4.346457 | 1.289374 | 29.974950 | -0.001871832 |
| 0.35 | 350.321688 | 8.874791 | 2.264756 | 29.909018 | -0.009852651 |

The final radial covariance focus residuals are respectively -0.000190,
-0.013761 and approximately zero nm. Thus the fraction-0.35 broadening after
refocus is not explained by its remaining scalar radial defocus. It does
**not**, however, establish failure of the direction's higher-order effect.
The additional empirical projection below separates these observations.

All three refocused diagnostics retain one radial covariance sign change
between C2 and C3 and none between C3 and ADL. Their minimum sampled face
clearances are 13.7663, 14.4456 and 15.7984 micrometres. Their only downstream
radial sign-change bracket is the final saved interval
1599.1998975 to 1599.1999975 mm. These are preliminary moment/face checks,
not complete mechanical-contact or two-dimensional topology qualification.
At fraction 0.35, the local directional sample stationary-plane descriptors
are -8.4397 and +8.6209 nm from the sample plane; this finite separation is
reported rather than counted automatically as two source crossovers.

The independently recomputed empirical polynomial analysis uses centered
sample mechanical slopes divided by their weighted RMS, with every bivariate
monomial of total degree at most the stated degree, including the constant.
All 193 original weights remain present in each weighted least-squares fit.
It matches the earlier own-angle projection convention; it neither changes
the observed particles nor fits uniquely identifiable physical Cs/C5 values.

| Fraction | Remainder after degree 1 (nm RMS) | After degree 2 | After degree 3 | After degree 4 | After degree 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial point | 1.388035 | 1.343689 | 1.250039 | 0.917775 | 0.099066 |
| 0.05, actual refocus | 1.329328 | 1.285256 | 1.196548 | 0.878623 | 0.095856 |
| 0.15, actual refocus | 1.286276 | 1.181976 | 1.098722 | 0.805776 | 0.090357 |
| 0.35, actual refocus | 2.259243 | 1.140193 | 0.933739 | 0.679056 | 0.082244 |

The remainder beyond degree three improves monotonically along these tested
fractions. The fraction-0.35 total RMS increase is dominated by additional
empirical quadratic content: its degree-two incremental projection has RMS
1.950420 nm, compared with 0.348052 nm initially and 0.507384 nm at fraction
0.15. This makes a subsequent low-order rematch scientifically worth testing
before rejecting the broader direction. It does not prove that available
auxiliary controls can remove that content while preserving every optical
constraint. Finite source/energy correlations and higher-order functions can
also project onto these lower-order sampled basis functions.

The saved-data audit verified every candidate/input/observation hash, all
original IDs/weights/energies, exact initial phase/clock regression, reported
statistics, and actual centered Objective derivatives. Its receipts are
`independent-saved-data-audit.json` and
`own-angle-polynomial-projections-v1.json` in the fraction output directory.
No fraction reaches D95 below 0.7 nm. This initial experiment stopped at 0.35;
it did not execute a physical correction inferred from polynomial subtraction.

### Extended bounded response and the separate low-order rematch

The improving empirical higher-order remainder justified a separately
authorized extension at fractions 0.65 and 1.0. The immutable new helper is
`probe_bounded_response_fractions_v2.py`, SHA-256
`4f2a9aa27e87871bb042277658479e63701ebf3500c11d8260fda545d7589afd`.
The original v1 helper and results remain unchanged. Version two allows at
most three actual Objective Newton updates per fraction and up to 0.15
additional percentage points, with the total Objective increment explicitly
kept within the original stage-four +/-3 percentage-point bounds. Fractions
never exceed the original bounded structural parameter combination.

The new run completed 15 forwards in 189.96 seconds on one CPU thread. Both
fractions needed two actual Newton updates. The same original 193 particles,
weights, initial energies/clocks, finite electric field, native domains and
26,727 integration nodes remained in force. All saved phase and clock values
at the initial point again matched the original stage-three diagnostic
bitwise. The output directory is `bounded-response-fractions-v2`.

| Fraction | Actual refocused D95 (nm) | RMS radius (nm) | Alpha95 (mrad) | Additional Objective correction (pp) | Remainder after degree 3 (nm RMS) | After degree 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.65 | 24.993190 | 6.039487 | 29.711127 | -0.032467983 | 0.743798 | 0.074701 |
| 1.0 | 54.337471 | 11.931221 | 29.335854 | -0.073079537 | 0.581607 | 0.069469 |

The large total probe sizes are actual diagnostic forwards, not the much
smaller empirical remainders. At fraction 1.0, the incremental degree-two
and degree-three projected components have RMS 11.477675 and 3.105305 nm.
The remainder beyond degree three nevertheless continues to decrease, from
1.250039 nm initially to 0.581607 nm. Thus the measured structural direction
reduces this higher-order empirical remainder while creating substantial
lower-order mismatch and shifting alpha95 away from 30 mrad. It cannot be
installed directly as a tuned optical configuration.

Both final points have effectively zero radial covariance focus residual.
Their local sample directional stationary-plane descriptors are
-24.8017/+25.2643 nm and -49.3871/+50.1279 nm. The corresponding minimum exact
sampled-face clearances are 17.8123 and 20.1379 micrometres. Normal radial root
counts remain one between C2/C3 and zero between C3/ADL; the only downstream
radial bracket remains the final 100 nm observation interval. Those limited
moment/face observations do not qualify the actual contact history or full
two-dimensional crossover chain. No current floor has been added.

Every stored input and observation hash, original particle column, statistic,
and centered Objective derivative passed an independent saved-data audit.
The degree-one-through-five residuals were also independently recomputed
from the saved final phases and agreed with the in-run diagnostic. Evidence
is in `independent-saved-data-audit.json` and
`own-angle-polynomial-projections-v2.json` within this output directory.

The complete fraction-1.0 input is
`015-fraction-1-refocused2.inputs.json`, SHA-256
`068a933698966658093fe126e0ea29af13fbe624895d2962560bfb785d8c4469`;
its full original observation archive has the same filename stem and the
`.npz` suffix. It is a diagnostic structural seed for a separate five-control
rematch, not a new physical source reference.

Independent read-only review of `optimize_fast_hp_rematch_v2.py` confirmed
that this subsequent experiment restores the authoritative stage-one
physical source, then separately validates the selected measured fraction's
input/archive hashes, source identity, original IDs/weights/energies and
initial phase/clocks. Its rebuilt native-grid hash must equal the selected
fraction's grid hash. The first forward must reproduce the selected
diagnostic phase within the declared tolerance and every saved flight time
bitwise. Every later candidate retains exactly that complete integration
grid and the guarded v3 field/source sampler.

The only five varied controls are main-HP common strength, HP1 relative
strength, HP1 orientation, Objective focus and C2 convergence. Their bounded
physical changes are respectively +/-20 percent, +/-5 percent, +/-0.005
radians, +/-0.05 percentage points and +/-0.5 percentage points relative to
the measured seed. C3 and all other structural/auxiliary settings remain
fixed. This preserves C2's primary angle role, Objective's focus role, and
the main HP pair's aberration-rematching role. The objective retains the
original weighted sample positions, sample covariance, 30 mrad angle target
and regional/direct-focus diagnostic screens; D95 is measured separately.
All 193 zero-current columns remain present, including if a trial fails the
sampled-face clearance screen. No contact, current or optical qualification
is inferred from the optimizer. This review itself executed no additional
forward transport.

### Saved HP beam heights and complex fifth-degree correlations

The read-only helper `audit_hp_heights_and_degree5_v1.py` compares the
stage-three seed, refocused fractions 0.35/0.65/1.0, and the completed
five-control HP/C2 rematch. It validates every input/archive hash and the
same original 193 IDs, weights, initial phase, energies and clocks. It also
reproduces the independent stage-one complex fifth-degree result before
analysing the new cases. Its SHA-256 is
`ce9679c0bd167b9211013ad51ff51311975d445fad906dbbe21f4c771d213ec0`.
The complete comparison is
`outputs/finite-field-corrector-rematch-20261028/hp-heights-degree5-with-hp-rematch-v1.json`.
No source generation, field propagation or ray interpolation occurred.

The saved diagnostic grid does **not** contain the exact HP centers. HP2 is
at 1144.5 mm and is bracketed by actual saved planes 1144.318745771 and
1144.818615278 mm. HP1 is at 1300.85 mm and is bracketed by 1300.777901585 and
1301.277771093 mm. The table below uses the nearer, upstream executed plane
for each HP. The JSON also reports the downstream plane, exact offsets,
centroids, axis-referenced radii and angles. These observations are not
misrepresented as exact-center measurements. Across the two saved planes,
RMS beam-height variation is below 0.14 micrometres in these cases, much
smaller than the observed change between the structural candidates. The
bracketing variation is not a rigorous error bound on an unrecorded center.

R95 below is the centered **radius**, not D95. All values use the same
original particle weights, normalized only for statistical evaluation.

| Case | HP2 RMS / R95 (um) | HP1 RMS / R95 (um) | HP2 RMS angle / alpha95 (mrad) | HP1 RMS angle / alpha95 (mrad) | HP1/HP2 beam RMS ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage-three seed | 86.656 / 159.957 | 75.801 / 138.564 | 0.250582 / 0.525676 | 0.345264 / 0.714164 | 0.874734 |
| Fraction 0.35 | 82.617 / 152.433 | 72.481 / 132.871 | 0.231298 / 0.491235 | 0.283954 / 0.594792 | 0.877305 |
| Fraction 0.65 | 79.188 / 146.052 | 69.616 / 127.905 | 0.215958 / 0.458067 | 0.237246 / 0.502493 | 0.879130 |
| Fraction 1.0 | 75.226 / 138.689 | 66.261 / 122.022 | 0.199366 / 0.417594 | 0.189897 / 0.404655 | 0.880822 |
| Five-control rematch | 76.951 / 141.905 | 67.782 / 124.929 | 0.208958 / 0.438950 | 0.200701 / 0.424950 | 0.880846 |

| Case | HP2 peak strength (million m^-3) | HP1 peak strength (million m^-3) | HP1/HP2 peak ratio | HP1 orientation (rad) |
| --- | ---: | ---: | ---: | ---: |
| Stage-three seed | 5.396416 | 4.816534 | 0.892543 | 0.017700765 |
| Fraction 0.35 | 5.274399 | 4.674675 | 0.886295 | 0.017350765 |
| Fraction 0.65 | 5.169813 | 4.554295 | 0.880940 | 0.017050765 |
| Fraction 1.0 | 5.047795 | 4.415267 | 0.874692 | 0.016700765 |
| Five-control rematch | 5.247958 | 4.628651 | 0.881991 | 0.015577131 |

HP2 orientation remains zero. The measured beam-height ratio is a finite-beam
ratio at the reported nearby planes; it is not a linear transfer-map
magnification. No geometry, mechanical length, nominal effective length or
field support was changed. The structural direction jointly reduces C2,
TL22, TL21, TL12 and Mini Condenser strengths while refocusing with Objective
and adjusting the multipoles. Its higher-order improvement therefore cannot
be uniquely assigned to one field from this comparison.

The complex diagnostic subtracts weighted means from `x + i*y` and
`tx + i*ty`, normalizes the latter by its own weighted RMS, and jointly fits
all complex monomials `t^k * conjugate(t)^(d-k)` for degrees zero through five.
The fifth-degree azimuthal index is `m = 2*k - 5`. Both the `m = 1` and
`m = -5` terms decrease along the measured structural direction. The table
reports complex coefficients in nm for this dimensionless own-RMS basis and
the weighted RMS of each separate fitted term.

| Case | m=1 coefficient, real + i imaginary (nm) | m=1 term RMS (nm) | m=-5 coefficient (nm) | m=-5 term RMS (nm) |
| --- | ---: | ---: | ---: | ---: |
| Stage-three seed | 0.551741 + 0.006609i | 5.444566 | -0.079619 + 0.064014i | 1.008050 |
| Fraction 0.35 | 0.416264 + 0.004473i | 4.094748 | -0.060222 + 0.042843i | 0.726974 |
| Fraction 0.65 | 0.326225 + 0.003388i | 3.199312 | -0.047666 + 0.030566i | 0.555286 |
| Fraction 1.0 | 0.243889 + 0.002631i | 2.382824 | -0.036515 + 0.020463i | 0.408934 |
| Five-control rematch | 0.309392 + 0.002659i | 3.029369 | -0.042380 + 0.024835i | 0.480941 |

These columns are nonorthogonal. Individual term RMS values must not be
summed as a variance budget or interpreted as unique calibrated spherical or
sixfold aberration coefficients. Finite-source and energy correlations are
still present. For a normalization cross-check, evaluating each fitted term
at the same 30 mrad slope radius gives m=1 magnitudes
11.7592, 8.8463, 7.0902, 5.5962 and 6.3256 nm, respectively; corresponding
m=-5 magnitudes are 2.1772, 1.5706, 1.2306, 0.9604 and 1.0042 nm. Thus the
decrease is not simply a consequence of the changing own-RMS angular scale.
Those evaluated fitted terms are also not realized corrected probes.

The data contradict an explanation based on increasing corrector beam
height: HP2 and HP1 heights both **decrease** along the useful structural
direction, while their ratio changes by less than one percent. After the
five-control rematch restores alpha95 to approximately 30 mrad, the heights
and these fifth-degree components rise slightly from the unrematched
fraction-1.0 seed but remain below their stage-three values. The supported
next inference is limited: coordinated structure and lower-order rematching
are useful, and a smaller beam/changed field combination is associated with
lower fifth-degree content here. There is no demonstrated monotonic rule
that further increasing or decreasing a single beam height, field or
magnification must improve the accepted probe.

### Eleven-control auxiliary rematch review

The subsequent `optimize_fast_auxiliary_rematch_v2.py` was independently
reviewed before its authorized execution. It uses the completed five-control
input and full saved phase as a diagnostic seed, while retaining the original
stage-one physical source reference. Input/phase hashes, source identity,
original energies/weights/clocks, unchanged native integration nodes, and
the full initial phase/flight-time regression remain mandatory.

The original five controls retain their roles and bounds. Six added controls
are HPC normal/skew (+/-2 million m^-3 each), HPol normal/skew (+/-0.5 million
m^-3 each), and condenser-stigmator normal/skew (+/-0.2 percentage points
each). The stigmator must retain its native normal/skew model and 0/45-degree
channel axes. At a hexapole's center the existing Gaussian envelope equals
one; conversion of these peak normal/skew coefficients using magnitude and
one-third `atan2` therefore matches the native triple-angle convention.
Zero increments preserve the original serialized multipole values exactly.
Mini Condenser and the transfer-lens strengths stay fixed in this stage.

A first diagnostic candidate with D95 below 0.7 nm, absolute radial covariance
focus descriptor at most 1 nm, and the preliminary angle/face/branch screens
is saved as an immutable full-193 checkpoint. Its metadata explicitly sets
`qualified=false` and requires physical contact, two-dimensional focus,
topology and numerical/source checks. The checkpoint is a handoff for those
checks, not a definition of optical acceptance. The read-only review found
no blocking mapping, source, field-domain or control-role error; it performed
no new forward propagation.

### Explicitly expanded structural trials at fractions 1.5 and 2.0

Following the measured reduction of the higher-order empirical remainder,
a new search authorization expanded the structural fractions to 1.5 and
2.0. These points exceed the original stage-four local search bounds and
were explicitly treated as new hypotheses, not predictions guaranteed by
the old linear solution. The decreases in Mini Condenser, TL22 and C2 were
limited to 6, 2 and 2 percentage points; the total Objective increment stayed
within +/-3 percentage points. Actual Newton refocus could add at most
0.5 percentage points and use at most three updates. All original hardware,
polarity, geometry, static-field, finite-electric-field, source and swept
domain guards remained active.

Independent review of the preparation found a direct-focus screening gap:
counting at most one downstream radial root could admit a distinct upstream
waist while a new terminal focus lay just beyond the sample. Version four
therefore also rejects any final refocused downstream bracket whose upper
end lies before sample minus 500 nm. A bracket straddling that boundary is
explicitly unresolved. The 500 nm region remains a diagnostic screen, not a
user acceptance tolerance. Missing/multiple central or AC radial roots,
incorrect Normal counts, or multiple downstream radial roots stop the trial.
An initially defocused single terminal waist remains allowed while executing
the real Objective refocus. A newly available independent target checkpoint
prevents starting another fraction; incomplete checkpoint JSON also stops
conservatively. Versions three and four have separate preparation receipts.

The independently reviewed executed helper is
`probe_bounded_response_fractions_v4.py`, SHA-256
`f58c18ccbd513c45d60d4abd96ca21b6e9dee2dc9d8c481774dde01ab3da3217`.
It completed 15 forwards in 192.88 seconds on one CPU thread. Both points
needed two actual Objective Newton updates. The immutable output directory
is `extended-response-fractions-v4`.

| Fraction | Actual refocused D95 (nm) | RMS radius (nm) | Alpha95 (mrad) | Additional Objective (pp) | Remainder beyond degree 3 (nm RMS) | Beyond degree 4 | Beyond degree 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 97.626354 | 20.917697 | 28.622137 | -0.153699350 | 0.418574 | 0.256798 | 0.064545 |
| 2.0 | 137.039368 | 28.903030 | 27.629217 | -0.256550323 | 0.304251 | 0.168649 | 0.060501 |

These large actual probes are unsuitable configurations. Their smaller
polynomial remainders are empirical projection diagnostics only. At fraction
2.0, incremental quadratic and cubic projected RMS components are 28.1278
and 6.3667 nm. A separate main-HP/C2/focus rematch is necessary to assess
whether the higher-order improvement survives restoration of the lower-order
conditions and 30 mrad target.

All original 193 columns remain present. Both candidates passed the declared
preliminary Normal, central/AC and final direct-focus screens. Their only
final downstream radial bracket lies in the last 100 nm saved interval;
neither far-upstream nor boundary-straddling brackets remained. Minimum
sampled aperture-face clearances were 23.4129 and 26.6293 micrometres. Their
sample radial covariance focus descriptors were approximately zero, while
the local directional descriptors were -87.7954/+88.6160 nm and
-124.5722/+124.9851 nm. These descriptors do not independently establish a
two-dimensional surface waist or certify material survival.

The independent saved-data audit verified every input/observation hash, all
original IDs/weights/energies, exact initial full phase/clock regression,
sample statistics and actual centered Objective derivatives. A separate
projection audit recomputed all degree-one-through-five remainders and
checked agreement between the real and complex polynomial bases. It also
confirmed that the fifth-degree trend persists at a fixed 30 mrad slope
radius: the empirical m=1 single-term amplitude is 4.1266 and 3.1488 nm,
while m=-5 is 0.71385 and 0.56132 nm. Nearby executed HP2/HP1 RMS beam heights
continue to decrease, to 69.644/61.456 and 64.160/56.659 micrometres.

The fraction-2.0 final input is `015-fraction-2-refocused2.inputs.json`,
SHA-256 `f3c8f3f5b41f2b20fda6019abab3e6a53d8dd33234f119e6ce1f28551abe2cb6`.
Its full-193 archive `015-fraction-2-refocused2.npz` has SHA-256
`934b05bbfbe682184c2ec14bc77d93eb7f4dc8857872759fbbab56502889b133`.
Audit receipts are `independent-saved-data-audit.json` and
`independent-polynomial-and-height-audit-v1.json` in the same output directory.
No optical default was changed, and no fraction above 2.0 was executed.

For the fraction-2.0 seed, an additional joint complex degree-zero-through-five
fit saves every fourth- and fifth-degree term in
`degree4-5-complex-terms-v1.json`. Its largest empirical terms are:

| Total degree | Azimuthal index m | Complex coefficient in the own-RMS basis (nm) | Separate term RMS (nm) | Term amplitude at fixed 30 mrad (nm) |
| --- | ---: | ---: | ---: | ---: |
| 4 | -2 | -0.039464 - 0.145504i | 0.726359 | 2.379639 |
| 4 | 4 | -0.009440 + 0.036822i | 0.183142 | 0.599996 |
| 5 | 1 | 0.100069 + 0.001771i | 0.967278 | 3.148805 |
| 5 | -5 | -0.017094 + 0.005112i | 0.172431 | 0.561320 |

These simultaneously fitted columns are nonorthogonal and can cancel each
other and lower-degree components. Their individual term RMS values cannot
be added to recover the actual probe variance. They are not calibrated
physical aberration coefficients. The complete receipt includes all eleven
degree-four/five terms, original input/archive hashes, and both actual saved
planes bracketing each HP center. Nearby HP2 RMS/R95 is
64.160/118.170 micrometres; HP1 is 56.659/104.830 micrometres. Their peak
strengths are 4.699174 and 4.026447 million m^-3, with HP1 orientation
0.015700765 rad. Exact-center values remain unavailable from the saved grid.

The next five-control rematch helper, `optimize_fast_hp_rematch_v3.py`, was
independently diff-reviewed against the previously reviewed v2. Its only
numerical configuration changes are Objective bounds of +/-0.1 percentage
points and C2 bounds of +/-1 percentage point; the default output directory
also changes. HP bounds, full source/energy/weight/clock checks, native-grid
identity and all field/domain guards remain unchanged. This provides the
dedicated C2 control enough declared range to attempt restoring 30 mrad from
the measured fraction-2.0 seed. The review executed no new propagation and
does not assert that the wider rematch will meet the target.

### Independent review of finite-region qualification consumers

The separate qualification patch stack was reviewed in the isolated
`finite-region-consumer-validation-v1` and `finite-region-consumer-validation-v2`
trees. Production `src` and the frozen field/source implementation were not
changed. This review checked actual source IDs, original weights, physical
loss handling, component requirements, bounded sampling and the direct-focus
gate. All new reproductions use small analytical arrays, with physical
propagation entry points disabled.

Three actionable issues were reproduced and corrected in the isolated v2
proposal before its final review:

1. **A known aperture loss could hide forbidden C3-to-ADL evidence in the
   condenser consumer.** The finite-interval adapter divides a capture into
   actual stable populations. A waist in the gap containing an attributed
   aperture loss can therefore be absent from both segments' internal radial
   candidates. The condenser consumer originally inspected only those
   segmented candidates and could return `PASS`, while the same input's
   whole-chain and legacy observers rejected the forbidden waist. The
   independent reproduction uses 32 unequal-weight source IDs 100..131,
   two opening-supported stopped particles, and a constructed waist at
   21.975 mm inside the 21.95-to-22 mm loss bracket. The fixed condenser
   consumer retains negative evidence from every adjacent pair's actual
   common positive-current population. These raw brackets can reject a
   forbidden/boundary crossing but cannot manufacture a required positive
   finite-waist witness. Original weights and pre-loss particles remain
   intact. Evidence: `review_finite_consumer_repros_v2.py` and
   `finite-consumer-independent-repros-v2.json`.
2. **The direct-focus child report replaced supplied source IDs with column
   indices.** With IDs 100..131, the parent/corrector reports retained those
   IDs, but direct-focus finite-region evidence reported 0..31. The new
   optional `source_ids` argument is passed through both direct-focus entry
   points and both consumers. Calls without IDs explicitly label their
   fallback as checkpoint column indices. Evidence:
   `finite-consumer-source-id-forwarding-v1.json`.
3. **The new direct-focus ID argument initially lacked entry validation.**
   Passing short or duplicate IDs produced an unresolved finite-region child
   but still allowed the direct absence gate to return `PASS`, because an
   unresolved terminal endpoint is intentionally not an upstream waist
   failure. Explicit shape, integer-kind and uniqueness checks now reject
   invalid IDs at the direct entry point. Requiring every finite child to be
   resolved would be the wrong fix because a specimen endpoint is one-sided.
   Evidence: `finite-consumer-direct-invalid-ids-v2.json` and the new direct
   invalid-ID regression cases.

The corrected source still rejects the old additional pre-specimen branch
and retains the explicit 1 nm terminal window; no wider tolerance was added
to obtain a pass. Required C2/C3, central and AC evidence is kept separate
from a potentially unresolved full inventory after interception. Known
opening-supported losses do not introduce a 100 percent transmission
requirement, while the CLI separately rejects unattributed incident losses.
Finite-region bracketing does not impose zero directional focus separation
or claim an exact common point waist.

Independent tests on the final isolated v2 passed **63** finite-waist,
direct-focus and consumer array cases in **3.98 seconds**, including the new
regressions. The receipt is
`consumer-independent-pure-tests-targeted-v2.xml`. A broader initial
collection attempt also included older crossover tests that load instrument
configuration files; it stopped during collection because that isolated tree
does not contain `configs/instruments/column/C3_ProbeCorrector.toml`. No
propagation or failed assertion occurred in that attempt, and it is not
counted as a passing broader run. The patch author separately retains the
full-stack and integration validation receipts.

The final independently checked v2 `condenser_topology.py` SHA-256 is
`84af95c578ee47a9b762ab6d7b6e82c0ea2b8eedf0641cfdeef2fa7d02fb3eec`;
`direct_focus_topology.py` is
`2ac0164c0b4d14c34eb0f4a84abbd2159d72e9718b98b27a6804f4f7b50f8150`;
and `waist_evidence.py` is
`c888eb2f8163480f7884bebb60c726d51304ad8233151aa332fda89d8ef2466d`.
No further blocking consumer error was found in this bounded review. This
conclusion concerns the observer/consumer proposal, not qualification of a
particular optical preset or its physical source convergence.

### Current complete-column validation and step refinement

The completed `first-focus-auxiliary-refined-v4` calculation uses the real
gun source and complete column with material contacts. It retains all 193
original source IDs, original positive weights, initial phase, energy offsets
and flight clocks. The independent saved-array audit checks the physical stop
masks at every captured plane and finds no losses. The declared 10 microampere
source therefore delivers 10 microamperes in this sampled calculation. This
is a result for the sampled source, not a new 100 percent transmission
requirement or a source-count convergence claim.

The subsequent `first-focus-auxiliary-half-step-v1` calculation uses exactly
the same optical input SHA-256
`61fefa2b5eed3fc395b18bd162bf36f15c67f4f4abea733f4f33739bd1eed34f`.
It halves the nominal integration step without refocusing, changing controls
or refitting source weights. Both runs retain the frozen physical solver
identity `f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`
and source-population identity
`4e3bac1af215cfe7b8589e64363541ba2039654fb01be49e8a5abee3017c0815`.
The numerical snapshot digest changes with the step, as expected.

| Quantity at the specimen upper surface | 0.05 mm nominal step | 0.025 mm nominal step |
| --- | ---: | ---: |
| Surviving original particles | 193 / 193 | 193 / 193 |
| D95 diameter (nm) | 2.435050059954 | 2.435016063834 |
| RMS radius (nm) | 0.635423178920 | 0.635422895878 |
| Alpha95 (mrad) | 29.999821399677 | 29.999821410001 |
| Local covariance drift descriptor (nm) | -0.000586556287 | -0.001021115463 |
| Current using the declared source current (microamperes) | 10 | 10 |

The D95 difference is -0.0000339961 nm, about -0.00140 percent. This bounded
step check does not explain away the remaining size failure. The one-sided
variance-derivative descriptors are likewise close to the specimen surface;
they are not a claim of an exact common two-dimensional point waist.

The complete material-contact result also agrees with the earlier fast
diagnostic at 3,696 exactly shared saved planes: maximum absolute position,
slope and clock differences are respectively 1.5081e-16 m, 3.0950e-14 rad and
2.7132e-22 s. Physical survival and current are established by the complete
production calculation, not inferred from the zero-current diagnostic.

The separately reviewed observer stack replays these immutable observations
without new propagation. Its observer solver identity is
`7dbc9aab0bd4ee5e23c6ea3113754ab408f6992731fc661efdb382b9e2a28ec8`, distinct
from the historical physical solver identity above. At the explicitly
declared maximum observation spacing of 0.5 mm, with all five condenser guard
planes present, both step cases pass the condenser, corrector and required
whole-chain checks. They contain one required C2-to-C3 finite waist region,
no C3-to-ADL waist, one central corrector region and one AC region. The direct
Mini Condenser-to-specimen interval contains no distinct extra upstream
waist at this sampling. These are finite-region, bounded-sampling statements;
they do not require coincident directional point roots or imply an exact
continuous-space topology proof.

The new consumer's sole remaining fixed-sampling qualification failure for
both cases is the specimen D95 limit. **D95 is approximately 2.435 nm and
remains above the required strict 0.7 nm limit.** The 0.635 nm RMS radius is
a different statistic and cannot substitute for D95. These results therefore
do not qualify a new default preset or the requested scanning image. A real
769-particle source-count comparison has not been performed in this audit.

Evidence is retained in each case's `independent-statistics-v1.json`, in
`first-focus-auxiliary-refined-v4/fast-actual-phase-comparison-v1.json`, and in
`finite-region-consumer-proposal-v2/saved-consumer-step-comparison-v4.json`.
The two production observation archive SHA-256 values are respectively
`5000e63f194b2119a9bc94bab98bba4d8aa0473472d4659ed394e6e2c33967a9`
and `85e00ef3dc19ef3db2dac4f9586c461e82e4b448df39c016fff3ab9a455dbc81`.

The final read-only candidate selector,
`select_focused_d95_candidate_v2.py`, was also reviewed. It ranks genuinely
measured D95 values only after the recorded angle, radial-focus and original
preliminary screens pass; reconstructs the eleven-control input with the
executed arithmetic order; verifies its canonical input hash; and validates
the initial diagnostic and actual-reference archive hashes, all 193 IDs,
positive finite weights and complete initial phase, energy and clock arrays.
Only a selected point whose full archive was actually saved receives a copy
of that archive. Otherwise it explicitly reports that the full phase is
unavailable and preserves only the genuinely measured sample residual. The
selector neither propagates nor claims transmitted-beam qualification.
