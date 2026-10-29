# Joint condenser, corrector and objective search

> Definition correction after the user's later clarification: the requested
> waist is the minimum of native D95(z). The latest 2026-10-28 instruction
> removes the former D95 < 0.7 nm size cap and requires tail attribution
> before further joint fitting. RMS is auxiliary only. The searches below
> that used RMS focus constraints were stopped and retained as historical
> proposals. Their small RMS focus offsets are not D95-waist certificates.
> See [the current definition](../OPTICAL_CONTROL_ROLES.md#unified-d95-probe-waist-definition).

The requested operating point has a physical **100 um C2 aperture diameter**,
**30 mrad current-weighted alpha95** at the exact specimen upper surface, and
the **native spatial D95 waist at that surface, with no absolute size cap**.
The original 0.7 nm size gate is historical. The current search is not a qualified
operating preset. No default microscope controls are replaced by these scripts.

## Physical scope

`scripts/solve_joint_probe.py` uses the production tip-source and unified
Lorentz transport. It varies installed lens and multipole excitations on a
detached state while retaining geometry, source emission parameters, absolute
particle weights, identities, energy, flight time, and real interception.
There are no extra empirical aberration kicks or equivalent image maps.
Nonlinear terms in the finite-field model are evaluated. Their presence does
not establish that a particular simulated tail is quantitatively physical;
source, integration and field-model validation remain necessary.

The objective's single excitation control scales its existing upper and lower
fields together. The four optional quadrupoles have signed normal strengths;
they do not expose independent skew or orientation controls. The native
Gaussian multipole fields and finite radial expansion of round-lens fields
remain engineering models, not a hardware or FEM-field validation.

The selected Normal entrance and corrector crossover requirements, direct
Mini-to-specimen focus, numerical convergence, and driven-scan checks are
defined in [the optical control contract](../OPTICAL_CONTROL_ROLES.md).
The current target does not impose the older RMS < 0.2 nm screening threshold
or a minimum transmitted current. All actual losses must still be reported.

## Current native-D95 tools

`scripts/measure_probe_waist.py` executes real field checkpoints on both sides
of the specimen upper surface. Both the width and the waist use native D95;
D90 and radial RMS are diagnostics only. Each plane uses its actual positive
current population and current-weighted centroid. The observer preserves the
original arrays, source identities, weights, energies, clocks and stop ledger.
Its default window is +/-100 nm with 1 nm spacing; extra actual checkpoints
can refine a candidate minimum. It never certifies an interpolated minimum.
The entire neighboring observed minimum bracket must lie within the declared
surface placement tolerance of +/-1 nm. Independent numerical, topology and
driven-scan checks are still required for qualification.

```powershell
.\.venv\Scripts\python.exe scripts/measure_probe_waist.py `
  --candidate path/to/full-snapshot-or-best.json `
  --output outputs/new-native-d95-observation --threads 8
```

`scripts/solve_probe_waist.py` provides the corresponding proposal search.
Its objective uses the full-current D95 surrogate, native alpha95, and six
actual neighboring D95 profile differences. It has no RMS size, RMS focus or
covariance focus residual. Small control regularization is declared separately
as an optimization choice. Seven profile samples are a proposal screen, not a
dense waist certificate. A native-screened candidate and the lowest smoothed
cost candidate are saved separately. Physical clipping remains observable;
zero transmitted current is unsupported, never a zero-size successful probe.
Physical fitting is held while the source and tail diagnostics below are
checked. No completed run has established D95 < 0.7 nm at the sample waist.

## Source-shape audit after the Gaussian-emission question

The saved source actually executes the legacy six-dimensional deterministic
Halton path in `ColdFieldEmitter.emit`. Transverse position and launch slopes
are radially truncated Gaussian distributions. The configured virtual source
FWHM is 5 nm; angular sigma is 1 mrad per axis with a radial 3 mrad cutoff.
There is no manually constructed two-population transverse source. The earlier
two-population explanation was a mathematical illustration, not this source.

Energy is a different distribution: the Young-tail and Gaussian-like quantile
terms are bounded and matched to the configured mean and RMS-equivalent
width. It is not a monoenergetic joint Gaussian. Increasing the sample count
also rematches the finite-sample energy mean and variance, so common source
IDs retain their positions and directions but not exactly their energies.
This is an explicit limitation of nested-source convergence comparisons.

In the original 769-particle archive, ID 306 starts at radius 1.136 nm, with
launch slope magnitude 2.628 mrad and launch kinetic energy about 0.553 eV.
Its position is near the 14th radial percentile, while its angle and energy
are near the 98th percentiles. It reaches a specimen-plane radius of about
28.69 nm about the native current centroid. Reconstructed source identity,
weights and launch position/direction match the archives; no indexing mismatch
was found. These facts do not identify the downstream tail's physical cause.

For a circular Gaussian that remains a scaled copy of itself, native radial
RMS and D95 differ by a constant, so their axial minima coincide. Uniform
source marginals or truncated Gaussian marginals alone do not guarantee a
self-similar downstream distribution. The executed finite round-lens and
multipole fields, energy-dependent momenta and full Lorentz trajectories are
not an ideal achromatic linear map. No empirical Cs/Cc kick is applied, but
this does not make the finite-field transport aberration-free.

Audit data, source reconstruction code and input hashes are retained under
`outputs/joint-probe-100um-30mrad-20261028/diagnostics/source-emission-audit.*`.
The first actual native-D95 scan of the unchanged original 769-particle seed
at 0.2 mm column step measured D95 = 2.568 nm at the surface and a sampled
minimum of 2.189 nm near +16 nm. Radial RMS minimized near -13 nm. These are
classical vacuum-probe observations, not a claim of physical tail validation.
Independent 0.05 and 0.025 mm repeats retain the same source and controls and
add actual 0.1 nm checkpoints around the coarse D95 minimum.

Those two repeats completed successfully. Their emitted arrays, IDs, weights
and gun-exit kinetic energies were byte-for-byte equal to the 0.2 mm source.
All 769 particles and the original 10 uA survived the observed column and
through-focus window. Results are stored in `seed769-native-d95-waist-h005`
and `seed769-native-d95-waist-h0025` under the same output root.

| Column step (mm) | Surface D90 (nm) | Surface D95 (nm) | Observed minimum D95 (nm) | Minimum offset (nm) |
|---|---:|---:|---:|---:|
| 0.2 | 1.493467 | 2.567580 | 2.189195 | +16.0 |
| 0.05 | 1.496154 | 2.562874 | 2.171371 | +15.5 |
| 0.025 | 1.496123 | 2.562856 | 2.171365 | +15.5 |

Positive offsets are downstream of the specimen upper surface. The two fine
runs bracket the observed D95 minimum with actual +15.4 and +15.6 nm planes.
Their surface D95 difference is 0.000727 percent and their observed D95 waist
positions coincide on this refined grid. The radial-RMS minimum remains near
-13 nm on its 1 nm observation grid. This does not establish a continuous
global minimum, exclude other near-degenerate minima, or certify the source
count, gun step, gun field grid, radial field expansion or material response.
The unchanged candidate fails both the requested D95 size and waist placement.

The shape audit also distinguishes a large-radius particle from a heavy tail:
ID 306 contributes only about 0.884 percent of radial second moment at the gun
exit but about 61 percent at the specimen surface. Native D95/D50 changes from
2.024 at emission to 2.249 at the gun exit and 4.418 at the original coarse
surface observation. Thus the source was not manually split into two groups;
the current finite-field model changes the shape during focusing. Attribution
to a particular physical aberration remains unestablished.

Reproduction and full arrays are linked by
`diagnostics/native-d95-waist-step-comparison.json`,
`diagnostics/check_native_waist_steps.py`, and
`diagnostics/seed769-shape-evolution-h02.json`.
The new observer, D95-only proposal solver and field-audit test suites passed
141 combined tests. No new physical fit or preset installation followed this
diagnostic. At this historical checkpoint, the prepared gun-only diagnostic
had not been executed; monochromatic and independent gun integration/field-grid
checks were still pending. Those checks and their numerical limitations are
now recorded in [the probe-tail attribution report](PROBE_TAIL_ATTRIBUTION_2026-10-28.md).
A changed physical energy law must not be presented as a fixed-source numerical
refinement.

## Historical mixed-metric proposal and validation commands

The commands below describe retained historical runs. They used RMS focus
with D95 width and must not be used to qualify the current D95-waist target.

Run from the repository with its `.venv` Python. Every output must be a new
directory. `measure` executes a frozen seed once; `fit` performs a bounded
local search, with its budget including numerical-Jacobian propagations.
Cooperative cancellation uses `--cancel-file`; stopped searches retain their
completed trials and best proposal.

```powershell
.\.venv\Scripts\python.exe scripts/solve_joint_probe.py fit `
  --seed path/to/best.json --output outputs/new-joint-search `
  --rays 193 --step .2 --threads 8 --max-evaluations 120 `
  --jacobian-step 1e-6 --focus-scale-nm .1 `
  --d95-target-nm .65 --d95-scale-nm .2 --d95-smoothing-nm .05

.\.venv\Scripts\python.exe scripts/validate_joint_probe.py `
  --candidate outputs/new-joint-search/best.json `
  --output outputs/new-joint-validation
```

The 0.65 nm value is a declared search margin, not a replacement for the user's
strict 0.7 nm acceptance limit. The optional D95 residual is
`max(D95_nm - target_nm, 0) / scale_nm`. The full weighted spatial residual,
angle residual and two focus residuals remain active. None changes the stored
particle coordinates or weights. A local fit retains its seed's transmitted
cohort to keep the objective dimension fixed; this is not a global current
acceptance threshold. A separately measured seed can have different real
physical interception.

Optional `--d95-smoothing-nm` defines a radius bandwidth for a logistic-smoothed
full-current CDF. The search diameter is twice the radius at which that CDF
reaches 0.95. It includes all actual transmitted positive-current particles,
and is separately labelled in each trial's `objective_diagnostics`. This
surrogate is not necessarily an upper bound on empirical D95. Its zero-radius
spot floor is `2 * bandwidth * log(19)`; finite-sample CDF plateaus can also
have a different limiting root from the native left-sided quantile. The exact
reported D95 remains the sole diameter acceptance statistic. A bandwidth of
zero, the default, retains the unsmoothed objective.

The validator restores the complete candidate snapshot, verifies paired file
hashes, and independently propagates at 0.05 and 0.025 mm without refitting.
It checks surface focus, actual waist topology, current, D95, and angle. Source
count refinement requires freshly generated complete sources with all physical
controls frozen. Two successive count refinements must satisfy the numerical
budget; a step comparison alone cannot qualify source sampling. A driven scan
test is separate from static waist measurement.

## Evidence ledger

Raw runs, full candidate snapshots, original-current particle arrays, and
per-trial observations are retained in
`outputs/joint-probe-100um-30mrad-20261028/`.

- `seed193-h02` records the initial failed evaluation. Production objective
  preparation synchronized four derived observation-plane values after input
  capture. The driver now prepares those values explicitly before freezing
  inputs; any unrelated physical change remains an error.
- `seed193-h02-v2` is the successful current-engine seed measurement at
  193 particles and 0.2 mm step: alpha95 29.999860901 mrad, D95
  2.210123156 nm, RMS radius 0.561802072 nm, and local focus offset
  +0.143158834 nm. All 193 original-current particles reach the specimen.
- `joint18-193-h02` is the stopped RMS-proxy search. Its best old objective
  value decreased, but its D95 increased to about 2.333 nm. It is retained as
  failed search evidence, not selected as the best D95 candidate. The exact
  original driver is saved beside the run with its matching SHA256.
- `seed193-fine-validation` independently rechecks the unchanged seed. At
  0.05 mm, alpha95 is 29.999865623 mrad, D95 is 2.199557729 nm, and local focus
  offset is +0.000435347 nm. The required bounded-sampling topology passes;
  the D95 size gate fails. At 0.025 mm the corresponding values are
  29.999865637 mrad, 2.199521493 nm, and -0.000075900 nm, with the same topology
  and size-gate outcome. The independent step comparison passes all declared
  numerical budgets. The original 193 particles and 10 uA reach the specimen
  in both cases. This establishes bounded step agreement, not source-count
  convergence or satisfaction of the requested probe diameter.
- `joint18-d95-193-h02` uses the explicit D95 hinge, a 0.1 nm focus-residual
  scale and a smaller absolute normalized parameter-difference step of
  1e-6. The old 1e-4 increment displaced the objective focus by about 57 nm;
  that secant was too broad for a local D95 derivative. Integration-step
  convergence and parameter-difference sensitivity remain distinct checks.
  It was stopped after 51 completed trials and one cancelled trial. Best
  trial 45 reached alpha95 29.999847508 mrad, D95 2.156548498 nm and local
  focus offset -0.007702368 nm. This modest D95 improvement did not improve
  D90 or the fraction inside the requested diameter.
- `seed769-h02` regenerates the complete 769-particle source at unchanged
  seed controls and 0.2 mm step. Alpha95 is 29.769611321 mrad, D95 is
  2.567413153 nm, and local focus offset is -12.515662042 nm. All 769 original
  particles and 10 uA reach the specimen. This change from 193 particles
  fails the source-count budget; it is not a converged physical prediction.
  Fitting or validating only the smaller source would be insufficient.
- `joint22-smooth95-769-h02` starts from that complete 769-particle seed and
  adds the four already installed quadrupole strengths to the original 18
  controls. It uses a 0.05 nm radius smoothing bandwidth, 0.65 nm search
  diameter target, 0.2 nm diameter-residual scale, 0.5 nm spatial-residual
  scale, 0.03 mrad angle scale, and 0.1 nm focus scale. The larger spatial
  residual scale makes RMS a lighter regularizer; there is still no physical
  RMS size gate. Native D90 and D95 are reported separately from the smooth
  objective. The first joint update restores focus but worsens native D95;
  it is not an accepted candidate.
- `c2-seed-exploration` independently measures C2 excitations one, two and
  three percentage points above the original value at 769 particles. The
  third seed has three actual stops labelled `condenser_aperture_2`: IDs 306
  and 331 at 764.9 mm, and ID 706 at 765.060330386 mm. The original-current
  loss is 0.390117%, leaving 9.960988296 uA at the specimen. The exact-plane
  checkpoint at 765 mm still includes ID 706 and must not be interpreted as
  the physical plate's final exit population. These measurements deliberately
  establish a separate physical cohort; downstream optics require rematching.

Further search results and their independent qualification remain to be added.

## Historical probe-size interpretation and the D90 comparison

The subsequent user question asks whether D90 is easier to converge and what a
direct-beam diffraction disk measures. The user then explicitly confirmed
**keeping D95 < 0.7 nm and reporting D90 alongside** at that stage. The later
attribution request removed the absolute D95 size cap while retaining D95 as
the waist metric and D90 as an auxiliary report. The historical comparison reports
D50, D80, D90, D95, D99, and the original-current fraction within a 0.7 nm
diameter circle about the full-current centroid. No particles are removed to
calculate these containment fractions.

| Classical observation at 0.2 mm step | D50 (nm) | D90 (nm) | D95 (nm) | Current inside 0.7 nm diameter |
| --- | ---: | ---: | ---: | ---: |
| Original 193-particle seed | 0.567509 | 1.381171 | 2.210123 | 64.7668% |
| 193-particle joint-search trial 45 | 0.572974 | 1.396756 | 2.156548 | 64.2487% |
| Original 769-particle seed | 0.581151 | 1.493397 | 2.567413 | 64.4993% |

The complete numerical and file-identity ledger is
`outputs/joint-probe-100um-30mrad-20261028/diagnostics/current-containment-comparison.json`.
Existing D90 values also exceed 0.7 nm; changing the target quantile alone
would not make these candidates compliant.

D90 permits 10% of the transmitted current outside the specified diameter;
D95 permits 5%. D90 is therefore an easier size requirement at the same bound,
but a discrete weighted quantile can still switch its active particle and
remain nonsmooth during optimization. Neither definition removes the need
for independent complete-source and integration-step convergence.

FWHM describes the central intensity peak, while encircled-current diameters
describe current localization and probe tails. They need not imply the same
imaging resolution. A STEM study explicitly compares FWHM and D59 to show
different tail behavior at similar central widths:
[Evaluation of probe size in STEM imaging at 30 and 60 kV](https://www.sciencedirect.com/science/article/abs/pii/S0968432811001867).
The sparse classical particle ensemble here does not establish a measured
continuous-intensity FWHM or a wave-optical probe profile.

A vacuum direct-beam disk in the diffraction plane gives angular intensity,
not the real-space containment function. For a coherent probe, the pupil
amplitude and its phase jointly determine the inverse-Fourier-transformed
probe; the intensity of a single vacuum diffraction pattern does not provide
that phase. Ronchigrams use specimen-induced structure to constrain focus and
aberrations, and ptychographic measurements can recover probe amplitude and
phase. See [JEOL's Ronchigram description](https://www.jeol.com/words/emterms/20121023.111557.php)
and [Humphry et al., reconstructed probe and free-space diffraction pattern](https://www.nature.com/articles/ncomms1733).
Disk intensity integrated over an angular region estimates detected electron
fractions after detector corrections. It is not a spatial encircled-current
measurement at the specimen or a measurement of deposited specimen energy.
