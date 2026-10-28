# Optical acceptance before joint excitation and geometry tuning

> Target update, 2026-10-28: the user explicitly selected specimen-surface
> **alpha95 = 30 mrad** and **spatial D95 < 0.7 nm**, with no hard transmitted-
> current lower bound, for the new preset search. This supersedes the physical
> pupil-edge 30 mrad fitting objective below for new work. The physical edge
> remains a separate diagnostic. The topology, original-weight current
> accounting, field/source provenance, numerical convergence, and scan checks
> remain applicable. Historical results below keep their original definitions.

> Topology clarification, 2026-10-28: retain the Normal entrance, with the
> C2-C3 crossover and no additional crossover from C3 to the next optical
> element. C3 stays active; its finite field and the following optics jointly
> form the next crossover. Require exactly one resolved two-dimensional
> source-image waist between TL22/TL21 and one inside the AC coil pair, and
> account for every other observed waist and unresolved interval. Component
> naming need not match CEOS labels. The 1208.15 mm TL geometric midpoint is
> not a verified optical symmetry plane or an OEM position specification.
> Follow the [current physical control contract](../OPTICAL_CONTROL_ROLES.md).

> Downstream clarification, 2026-10-28: Mini Condenser plus Upper Objective
> must focus directly at the specimen upper surface. No distinct intermediate
> crossover is allowed between Mini Condenser and that surface. The final
> focus bracket and its numerical tolerance must be distinguished from an
> additional upstream waist. A small final D95 does not waive this condition.

Executed follow-up: [historical comparison, particle counts and crossover validation](FINITE_FIELD_PARTICLE_VALIDATION_2026-10-27.md).
Subject to the target update above, the contract below remains the acceptance
definition; additional observations do not automatically promote a working
point to qualified status.

Date: 2026-10-27. Scope: the current C3 + Probe Corrector assembly in
Finite-Field Optics, STEM Nanoprobe Normal. This document defines the next
qualification work. It does not install a working point, change geometry, or
certify the existing preset. All newly proposed numerical budgets below are
engineering starting points, not manufacturer specifications.

## 1. Confirmed requirements and the target observable

- The specimen convergence semi-angle is **30 mrad at the physical pupil / the
  calibrated diffraction-disk edge**. The user explicitly selected this
  definition. Report current-contained alpha95 and alpha99 separately.
- Normal requires a source crossover between C2 and C3 and none between C3 and
  ADL. Large Angle is a separate branch and must not be reached silently.
- Preserve the control roles in [OPTICAL_CONTROL_ROLES](../OPTICAL_CONTROL_ROLES.md).
  C1 microscope spot size is not the specimen spot radius. AC is a virtual
  channel on the existing scan coils; Descan shares Image/Diffraction hardware.
- There is no prescribed transmitted-current floor or ceiling. Preserve
  source weights, real interceptions, centroids, tilt and energy spread.
- A static beam waist is distinct from a driven scan pivot and from a pupil
  conjugate / coma-free reference plane. Each requires its own evidence.
- Do not impose equal semi-angles at different source-image crossovers without
  an independently justified relay design requiring it.

The cold-FEG emission mean energy and energy-spread FWHM remain separate
parameters. The current inputs specify 0.3 eV for each; neither is the column
accelerating voltage. Use the executed local kinetic energy at every plane.

### Legacy acceptance boundary and the separate physical-pupil measurement

`beam_statistics.TransverseBeamStatistics.alpha_edge_rad` explicitly returns
`None`. Its legacy `convergence_edge_rad` is the largest **sampled** particle
angle, not a measured physical edge. The illumination/focus fitters, catalog
acceptance and `qualify_finite_field_nanoprobe.py --target` currently use
**alpha95**. They cannot certify the newly selected 30-mrad edge requirement.
Their historical results and failed candidates remain correctly labelled
alpha95 observations; do not reinterpret them as edge-angle measurements.

The separate `scripts/measure_optical_acceptance.py` workflow now traces
zero-current source-to-specimen diagnostic rays, brackets actual mechanical
contacts and reports the nominal-energy reference-source pupil alongside
the unchanged weighted population. It does not change the legacy fitters or
reinterpret their historical alpha95 targets. The diagnostic is a declared
source-position/direction slice, with separate finite-source support witnesses;
it is not a complete mixed-energy phase-space qualification.

Before any further working-point fitting, use the separately measured physical
boundary and its numerical qualification. Unknown edge evidence must return
`NOT_MEASURED`, never fall back to alpha95, a maximum particle angle, or a
scalar rescaling of an older result.

## 2. What 30 mrad does and does not determine

Knowing only the specimen semi-angle does not uniquely determine upstream
semi-angles, lens positions, effective lengths or magnifications. A family of
relays can produce the same specimen cone. Each proposed design must first
define its intended conjugacies and compute its transport maps.

For a single energy, an approximately round, paraxial image relay between
field-free reference planes may be written in canonical coordinates as

\[
\begin{pmatrix}x_j\\p_{x,j}\end{pmatrix}
=\begin{pmatrix}M_{ji}&0\\C_{ji}&1/M_{ji}\end{pmatrix}
\begin{pmatrix}x_i\\p_{x,i}\end{pmatrix}.
\]

For a point-source cone, or when the position-dependent contribution is
negligible for the particular rays being compared,

\[
\alpha_j\simeq\frac{p_i}{p_j|M_{ji}|}\alpha_i,
\qquad
\alpha_i\simeq |M_{si}|\frac{p_s}{p_i}\,(30\ \mathrm{mrad}),
\]

where `M_si` is the **position magnification from plane i to the specimen**,
and `p(K) = sqrt(K*(K+2*m*c^2))/c` is relativistic momentum. After acceleration,
at nearly equal energies, the momentum ratio is close to one. Do not apply
that simplification inside the accelerating gun. Angles and slopes are only
interchangeable in the stated paraxial approximation.

For a finite source, the `C*x` contribution matters. In an equal-energy
one-direction slope representation,

\[
\theta_j=C_\theta x_i+\theta_i/M,
\quad
\sigma_{\theta,j}^2=C_\theta^2\sigma_{x,i}^2+
2C_\theta\operatorname{Cov}(x_i,\theta_i)/M+
\sigma_{\theta,i}^2/M^2.
\]

Consequently, position conjugacy, unit magnification, equal angular quantiles
and waist-to-waist matching are different conditions. For coupled X/Y motion,
use the full affine 4-by-4 transport map and covariance, separately per energy.
For the linear, fixed-population limit, `Sigma_j = R*Sigma_i*R.T`; apply
symplectic/eigen-emittance checks in appropriate canonical coordinates.
Mechanical slope matrices inside magnetic fields must not be treated as
canonical maps. Nonlinear projected RMS emittance growth and aperture losses
are not automatically integration failures.

The full finite-field ray map, including aberrations and all actual apertures,
is the final authority. Propagate the same identified pupil boundary rays to
each reference plane rather than independently fitting a new cone at each one.
Auxiliary boundary rays do not carry invented source current and do not replace
the actual emitted population or its intensity distribution.

### Conditional example, not a set of new targets

The current working point's local first-order mechanical-angle response has
approximately `|D_internal->AC| = 0.500005` and `|D_AC->sample| = 9.595`.
If these full-field angular responses were deliberately retained and the
position-dependent terms were negligible for the compared boundary rays, they
would suggest about **3.1 mrad at the AC source image** and **6.3 mrad at the
internal source image** for a 30-mrad specimen cone. The specimen is inside a
magnetic field: this example uses the computed D blocks, not a general assumption
that a field-internal position magnification is the inverse angular gain.
These are design estimates, not
executed 30-mrad results or accepted angle bounds. Moving elements or changing
their fields changes these magnifications and therefore changes the estimates.
The measured alpha95 values near 2.04/1.02 mrad from the existing working point
cannot be multiplied by `30/9.74` to establish physical edge targets.

## 3. Reference-plane and angular-relationship register

| Plane / interval | Required relationship | Evidence and design limitation |
| --- | --- | --- |
| Gun-internal source image and accelerating region | Actual gun transport supplies the incident phase space, energy and current for all later maps. | The existing column NPZ starts at gun exit; it cannot locate internal gun waists. No absolute gun semi-angle follows from 30 mrad alone. |
| Source image below C1 | Its specimen conjugacy and magnification link its angular cone to the final cone; C1 spot-size/source-demagnification role remains explicit. | A new unchanged-input, 193-particle dense observation supports a 2D waist at 503.2 mm. This is a fixed-sampling observation, not a nominal-edge target. Re-evaluate after C1 or geometry changes. |
| Required C2-to-C3 crossover | Actual two-dimensional waist inside the open interval; use its complete map to the specimen or downstream source image. | Position is an interval constraint, not a fixed historical Z. Angle depends on C2/C3/ADL and the remaining relay. |
| C3-to-ADL in Normal | No source crossover at sufficiently resolved axial sampling. | There is no crossover angle to match in this forbidden interval. A source image shown above ADL in a generic vendor diagram does not override this selected branch. |
| Internal corrector source image between TL22 and TL21 | Actual source-image waist; map it to the downstream source image and the specimen. Assess its relationship to the optical symmetry plane and D4 residual. | The geometric midpoint alone is not proof of optical symmetry. The user did not specify an absolute internal angle. |
| HP2 and HP1 field midplanes | Corrector relay conjugacy, magnification/rotation, beam height and nonlinear cancellation. | These are not generally source-waist planes. Do not replace these requirements with equality of two source-waist angles. |
| Source image between the AC coils | Actual static waist; specimen source-imaging map determines its angular relationship. | Its current roughly twofold relation to the internal image is a diagnostic of the current design, not a universal constraint. |
| Driven AC virtual pivot | Common driven response with vanishing displacement at the pivot and the intended displacement/angle response at the specimen. | Determine separately from the static source waist. A front-focal-plane statement requires a specified complete field system and output reference. |
| Mini Condenser to specimen | Complete finite-field envelope, any extra waists, permitted bores and intended specimen focus. | New dense observation supports an additional 2D waist near 1585.9 mm. Include it in the design-topology review and complete transport chain; neither prohibit nor require it merely from its observation. The specimen lies within the overlapping objective fields. |
| Specimen entrance/reference plane | Physical pupil edge 30 mrad, separately reported filling/current quantiles, focus and aberration quality. | Specify the entrance surface versus centre. The current 5-nm specimen uses its upper surface for incident probe observations. |
| Post-specimen / intermediate / projector crossovers | Mode-specific image or diffraction conjugacies and collection-angle transport. | Incident 30 mrad alone does not fix post-specimen angles. Scattering, energy loss, detector acceptance, Image/Diffraction mode and Descan must be specified; use a vacuum reference for an unscattered-beam comparison. |

The observed numerical coordinates and their sampling limits are in the local
`outputs/optical-acceptance-contract-20261027/inventory.json`; this table states
the design relationships and does not freeze those coordinates during redesign.
For each source-image entry, distinguish the paraxial conjugate plane from the
finite-current beam's measured waist. Aberrations and source extent can separate
them; matching a weighted radius minimum to a geometric midpoint is not itself
proof of the intended paraxial image or D4 cancellation. Report the sign of the
position-angle covariance on both sides along with the positive half-angle.

## 4. How to identify the physical 30-mrad pupil

1. Declare the reference source position/chief ray and reference energy. Trace
   the complete accepted phase-space boundary through the gun and column.
   Identify which aperture or bore actually limits each azimuth, and record
   separately whether configured source angular support limits illumination
   before that mechanical boundary is reached. Do not assume
   that the component named C2 Aperture is automatically the active stop.
   Mechanical/energy reachability and the source's intensity filling are
   separate questions. A finite emitted sample or a statistical source cutoff
   must not substitute for the nominal mechanical pupil boundary.
2. Resolve aperture-rim rays by deterministic angular/azimuthal refinement and,
   where needed, boundary-value shooting. Retain the true upstream source
   admissibility and every downstream interception. A ray injected at an
   aperture with an arbitrary slope is not sufficient boundary evidence.
3. At the specimen, report the full angular boundary `alpha_edge(phi,E)`, its
   angular centre, minimum/maximum radii, ellipticity and any missing sectors.
   A distorted pupil must not be hidden in one fitted radius. At each upstream
   crossover, report the corresponding boundary mapping as well as the actual
   weighted population's alpha95/alpha99 and current.
4. Distinguish geometric acceptance from illumination of that acceptance. An
   underfilled aperture can admit a nominal cone without the source populating
   its edge. Report the pupil intensity/filling and any uncertainty in locating
   an observable disk edge; do not synthesize missing intensity.
5. If a screen/detector disk is used as evidence, calibrate its angular mapping
   with the actual projector configuration, camera length, pixels and detector
   response. A particle angular projection does not itself validate coherent
   CBED interference or a crystal-scattered diffraction pattern.

Only for a uniformly filled circular angular disk in the paraxial limit is
`alpha95 = sqrt(0.95)*alpha_edge`: 30 mrad then gives about 29.24 mrad. This is
a useful analytic test, not a conversion applicable to the present FEG by
assumption. Gaussian source/energy tails also require distinguishing a nominal
reference-energy pupil from the intensity-broadened edge of the mixed beam.

## 5. Excitation and geometry must be solved together

Define a versioned design vector containing excitations, optical centres,
effective field lengths/profiles, physical bodies/bores, lens spacing, hexapole
orientation and aperture selection. Keep its physical meaning and units explicit.
Changing geometry invalidates old transfer maps, aberration fits and scan
calibrations. Recompute the actual shared field/geometry snapshot; do not reuse
an obsolete matrix as a target or silently truncate overlapping fringe fields.

Current HP2 and HP1 have different effective lengths (6 and 10 mm) and mechanical
lengths (18.2 and 18.1 mm). Their strengths and relative rotation are also
different. Equal integrated hexapole action alone does not preserve all higher
orders when effective length changes. Test A2 cancellation, C3 compensation,
and relevant residuals such as D4/C5/A5 using the same aberration convention.
Do not impose an ideal identical-hexapole `-I` map as a universal acceptance
condition on this finite, unequal-length implementation.
Aberration coefficients extracted from the shared field are diagnostics of that
field's action. Do not add an equivalent Cs/Cc kick or phase term on top of an
effect already owned by the unified field executor.

For a chosen symmetric-relay design, a near-negative-unit HP map may be an
explicit design target, but then its conjugacy, rotation and nonlinear effect
must all be verified. Changing the source-image magnification to obtain a
desired source-waist angle does not certify that the HP relay still works.

Mechanical shell length, pole length, effective magnetic length and drawing
extent are distinct. Changing only a shell does not justify changing a Gaussian
field scale. Geometry-to-field coupling must be physically supported, including
fringe fields and valid-domain limits. Enforce positive physical clearances,
mechanical travel/bore constraints and declared material/field validity. Record
intended aperture losses and separately identify unplanned interception.

An appropriate search has an inner excitation solve for each feasible geometry,
followed by an outer design update only when useful. Joint residuals include
the pupil edge, intended conjugacies, source-waist topology, aberration/probe
quality and driven scan response. Do not minimize an angle ratio alone. Keep
the original source and intensity measure fixed; after choosing a candidate,
freeze all controls and geometry for independent numerical convergence checks.

## 6. Acceptance layers and proposed numerical budgets

| Layer | Requirement | Status / proposed budget |
| --- | --- | --- |
| Definitions | Nominal edge, reference energy, source/chief ray, sample plane, active stops and mode are identified. | Standalone nominal-energy reference-source boundary measurement is available; full finite-source qualification remains separate. |
| Pupil | Continuous intended accepted pupil, with physical sample edge near 30 mrad. | Proposed initial budget: every intended azimuth within 29.7-30.3 mrad; separately flag underfilling, centre tilt and distortions. Not an OEM tolerance. |
| Topology | Required/forbidden intervals and all extra resolved waists are recorded using actual 2D evidence. | Adaptive axial refinement; unresolved coverage cannot pass. Historical Z values are not fixed targets. |
| Source relays | Boundary mapping and actual phase-space response agree with the declared design maps, including energy and cross-coupling. | No universal upstream absolute angle. Compare full mapped boundaries, not only alpha95 ratios. Set allowed image/angle residuals from the final probe budget. |
| HP / pupil conjugacies | Corrector action, intended conjugacies, rotation, field-domain validity and residual aberrations are demonstrated. | Optical symmetry and aberration tolerances must be tied to a declared usable-probe requirement; a unit-magnitude matrix alone cannot pass this layer. |
| Specimen focus | Actual local radial focus and separate directional/astigmatic focus diagnostics are reported at the declared plane. | Retain 1 nm as a provisional radial-focus engineering gate; it is not a full probe-size or aberration criterion. |
| Scan | X/Y command gains, cross-talk, pivot, specimen position and incidence-angle residuals across the required scan field are verified. | Scan field and position/angle budgets must be explicitly set before claiming scanning qualification; no equality to the static waist is presumed. |
| Intensity | Actual source/transmitted currents and all stop identities; common-source comparison alongside actual populations. | No arbitrary current floor or ceiling; no numerical particle pruning or renormalization to conceal loss. |
| Numerical convergence | Frozen candidate; independent boundary sampling, axial step and complete source-count refinements. | Proposed changes <=0.25% for edge/angle/relay observables, <=0.25 nm radial focus, <=2% for RMS/D95/current. Current variation is a convergence criterion, not a required current level. Use stricter limits if these consume the physical budget. |
| Image/probe performance | Resolution/encircled-current or aberration budget at 30 mrad, over the requested field/energy distribution. | Not yet specified by 30 mrad alone. Do not invent a spot-size threshold or claim atomic-resolution qualification from cone and focus alone. |

Conjugacy tolerances should be expressed as their propagated position/angle
error relative to the allocated probe budget, rather than an arbitrary raw
threshold on dimensionful matrix entries. Inspect each transverse direction.
For numerical limits, require the observed discrepancy to leave margin inside
the physical acceptance budget.
Relative errors require an explicit nonzero target or design scale. Zero-target
conjugacy residuals, cross-talk and chief-ray tilt instead require absolute
position/angle budgets propagated to the specified reference plane.
Use more than one successive particle refinement
when establishing convergence; do not retune at each count and call the results
converged. The current source sampler restandardizes energies with population
size, so higher-N propagation must regenerate and retrace the complete source.

Statuses should distinguish `TOPOLOGY_RESOLVED`, `PUPIL_EDGE_QUALIFIED`,
`RELAY_MATCHED`, `PROBE_QUALIFIED`, and `SCAN_QUALIFIED`. Missing physical or
numerical evidence remains `NOT_MEASURED` or `UNRESOLVED`; software tests alone
cannot promote a working point to instrument qualification.

## 7. Evidence and next implementation boundary

Current actual observations support the selected Normal topology and the
internal/AC waist brackets at their stated sampling. A new 193-particle,
0.05-mm-step run retained the identical emitted source and unchanged physical
inputs, adding 5,348 observation planes across the incident column. It supports
five interior 2D waist brackets at 503.2, 721.0, 1202.0, 1400.0 and 1585.9 mm.
Their alpha95 values are approximately 0.1086, 0.2015, 2.0637, 1.0238 and
8.0173 mrad respectively. These are actual population statistics, not nominal
edge measurements or new prescribed upstream targets. The local covariance
estimate at the specimen is recorded separately from the interior brackets.

The additional run took 185.322 seconds and completed successfully. Its gun
equal-time history is retained separately; gun coordinates resampled onto a
common Z grid are diagnostic, not exact-plane evidence of a gun-internal waist.
Physical pupil-edge measurement, source-count convergence and driven scan-pivot
evidence remain open. The additional Mini-Condenser-to-specimen waist is a
design-review item; a generic CEOS schematic alone is insufficient to decide
whether to preserve or remove it in this chosen Normal branch.

The standalone physical pupil-boundary measurement and joint report are now
implemented; see the [frozen-preset observation](FINITE_FIELD_PUPIL_ACCEPTANCE_2026-10-27.md).
Complete adaptive crossover coverage, energy-dependent differential-map
qualification and full finite-source boundary validation remain further work
before optical/geometrical optimization can claim complete acceptance.
Keep existing alpha95 fitters explicitly legacy until migrated to the new
observable. No lens, aperture, source, body length or position has been changed
in preparing this contract.

Primary references:

- [CEOS, Muller et al. (2006), Figure 2 and optical-design discussion](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)
- [CEOS, residual aberrations and transfer-system tuning](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR)
- [JEOL, condenser aperture and angular/dose control](https://www.jeol.com/words/emterms/20121023.034559.php)
- [JEOL, convergent-beam electron diffraction](https://www.jeol.co.jp/words/emterms/20121023.043259.html)

These references support the optical principles, not proprietary dimensions or
numerical acceptance tolerances for the current model. Detailed source notes,
inventory and bounded analytic checks are retained locally under
`outputs/optical-acceptance-contract-20261027/`.
