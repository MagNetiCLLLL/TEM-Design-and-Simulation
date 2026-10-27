# Physical pupil and joint incident-optics acceptance

Follow-up: [particle-count, exact gun and downstream validation](FINITE_FIELD_PARTICLE_VALIDATION_2026-10-27.md)
adds observations without changing this frozen pupil measurement. It also
compares the historical alpha95 target with the current field model. This
report retains its original measurement scope and limitations.

Date: 2026-10-27. Frozen C3 + Probe Corrector working point, Finite-Field
Optics, STEM Nanoprobe Normal. No excitation, source parameter, dimension,
position, particle weight or propagation formula was adjusted.

## Result

The nominal-energy reference-source mechanical pupil maps to
**53.537956-54.115499 mrad** at the specimen entrance. It fails the requested
**30 mrad aperture-edge** target (initial engineering interval 29.7-30.3 mrad).
This is a mapped physical rim, not an assertion that the emitted beam fills it
or that a crystal diffraction disk was simulated.

The actual archived 193-particle beam has **alpha95 = 9.819338 mrad**,
alpha99 = 11.604294 mrad and current **10 microampere**. All 193 particles
survive to the specimen. Current quantiles are never substituted for a
mechanical edge and no current limit is imposed.

All 32 source-azimuth boundary searches identify **C2 Aperture** as the first
limiting mechanical element. Its entrance plate face is at 764.9 mm; the
nominal plate centre is 765 mm and its radius is 50 micrometres. The accepted
launch slope is 0.0139453125, bracketed against a rejected slope of
0.0139501953125. The configured source-direction cutoff is only 0.003.
All mechanical-rim diagnostic rays therefore lie outside the configured
source angular support and carry exactly zero current.

| Aperture | Actual sampled maximum radius / opening radius | Current-95 radius / opening radius | Nominal-energy support witnesses / opening radius |
| --- | ---: | ---: | ---: |
| C2 | 22.521% | 17.815% | 21.762% |
| C3 | 3.296% | 2.607% | 3.185% |

These are radial occupancy metrics at the plate midplanes, not illuminated
area fractions or reconstructed entrance losses. A separate set of 288
zero-current witnesses samples the source centre and eight source-support
rim positions, with 32 angular-rim directions each. All survive; their
specimen angles at the nominal emission energy span **11.640696-11.993093
mrad**. This is distinct from the real mixed-energy population.
Emission mean energy **0.3 eV** and energy-spread FWHM **0.3 eV** remain
separate settings.

## Crossovers and angular transfer

The original 5,348-plane incident capture was reused after checking every
physical input and recomputing its original gun-exit source identity from
coordinates, slopes, clock, energy offsets, weights, survival and IDs.
Five executed two-dimensional waist brackets remain present:

| Reference | Z (mm) | Actual alpha95 (mrad) | Same physical rim mapped here (mrad) |
| --- | ---: | ---: | ---: |
| After C1 | 503.2 | 0.108583 | 0.607587 |
| C2-C3 | 721.0 | 0.201505 | 1.127645 |
| Corrector internal | 1202.0 | 2.063677 | 9.976802-12.927880 |
| Between AC coils | 1400.0 | 1.023846 | 5.621409-5.708396 |
| Mini Condenser to specimen | 1585.9 | 8.017274 | 43.694525-44.137348 |
| Specimen entrance | 1599.1999975 | 9.819338 | 53.537956-54.115499 |

C2-C3 contains the required crossover; C3-ADL contains none at the bounded
observation spacing. The **1585.9 mm** waist is retained as an actual observation
and design-review item. The specimen row is an endpoint, not a two-sided waist
bracket. The same physical rim is evaluated at the weighted-beam waist planes;
these planes need not be waists of the unilluminated rim bundle itself.

The measured internal-corrector-to-AC rim angular ratio varies from
**0.434828 to 0.572167** by source azimuth. The corresponding current-alpha95
ratio is **0.496127**. Neither equality nor one universal scalar magnification
describes every ray. Complete per-azimuth values and successive transfer ratios
are in the local report. Empirical 4-by-4 particle regressions are explicitly
not canonical differential maps or proof of HP2/HP1 conjugacy.

## Implementation and validation

- `pupil_acceptance.py` traces explicit zero-current source probes through
  the production gun and shared-field column, preserving real energy, clock
  and all physical stops. It brackets the connected chief-ray pupil and refuses
  missing, unstable or observed disconnected boundaries.
- `crossover_acceptance.py` inspects all sampled incident intervals, retaining
  pre-loss waists and reporting actual and common-ID populations separately.
- `measure_optical_acceptance.py` produces a joint JSON receipt with physical
  stop identities, source filling, current statistics, all crossover angles,
  numerical checks and content-addressed local diagnostic caches.
- `render_optical_acceptance_report.py` produces its readable English report.
  The older qualifier's `--target` is now explicitly labelled as alpha95 and
  cannot certify the aperture-edge requirement.

The boundary search used **1,472 diagnostic probes** across 32 azimuths,
followed by independent accepted-rim, chief-ray and source-support traces.
The completed new diagnostic calls total about **993.3 s**; regenerating the
verified joint report reused all ten local diagnostic cache records.

Halving the maximum **column** step from 0.05 to 0.025 mm changes the specimen
rim angle by at most **1.451855e-7 mrad** and preserves all rim rays. The maximum
relative launch-boundary bracket width is **0.000350018**. Nested 16/32-azimuth
sampling gives identical sampled minimum and maximum angles. These are bounded
checks: the fine-step check transports the same accepted launches, and does not
independently solve every fine-step mechanical boundary or certify continuous
azimuth coverage.

**23 targeted regression cases passed**: 16 new physical-boundary, source
handoff and crossover cases; six selected existing regression cases; one CI
inventory check. The three new test files are registered in the lens-geometry
acceptance scope. No broad coherent or GUI suite was run. Existing Pydantic
deprecation warnings remain unrelated to this change.

## Qualification limits

The working point is **NOT_QUALIFIED**, rather than adjusted to appear to pass.
Particle-count convergence remains unestablished. The archived radial
covariance predicts a local drift waist offset of 4.286598 nm, compared with
the provisional 1 nm budget; this is not a new exact focus calculation inside
the objective field.

The measured mechanical pupil is a declared source-position/direction slice
at nominal energy. Full finite-source/mixed-energy acceptance, nonlinear
angular-domain folds, exact gun-internal waist brackets, post-specimen optics,
HP conjugacies/aberration performance and the driven AC pivot are not qualified
by this run. Gun-history interpolation candidates are retained in JSON as
diagnostics only. No equal-angle rule is imposed on different source images.

## Local evidence and reproduction

- [Full joint report](../../outputs/pupil-joint-acceptance-20261027/verified/report.md)
- [Machine-readable receipt](../../outputs/pupil-joint-acceptance-20261027/verified/report.json)
- [Measured overview](../../outputs/pupil-joint-acceptance-20261027/verified/measurement_overview.png)
- [Original acceptance contract](FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md)

The `outputs` links are local, Git-ignored evidence, not published repository
assets. The receipt preserves the solver identity actually used for transport;
the subsequent CI-list registration does not change any physical executor.

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -X utf8 scripts/measure_optical_acceptance.py `
  --inputs outputs/optical-acceptance-contract-20261027/full-column-baseline-n193-h005.inputs.json `
  --particles outputs/optical-acceptance-contract-20261027/full-column-baseline-n193-h005.npz `
  --particle-report outputs/optical-acceptance-contract-20261027/full-column-baseline-n193-h005.json `
  --output outputs/pupil-acceptance-new-run --azimuths 32
```

Use a new output directory to preserve completed reports. An optional
`--cache-dir` reuses only matching complete snapshot, solver, launch and
observation identities; physical or implementation changes invalidate reuse.
