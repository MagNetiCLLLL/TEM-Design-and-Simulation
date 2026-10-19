# Camera and fluorescent-screen reception

**Illuminating Image > Camera / screen** displays the selected installed receiver
in a captured calculation. Run **Run high-accuracy once**, then **Calculate
imaging** on this page. The latter reuses valid upstream particle calculations
and calculates sample interactions/downstream transport when required. It does
not request STEM detector acquisition, EDS, or a coherent-wave calculation.

Selecting a receiver changes the display only. It does not insert/retract the
screen or camera, or bypass an upstream detector. Retracted receivers, disabled
readout, incomplete trajectories, and missing calculations have explicit status
messages. A completed result with zero intercepted probability is a valid black
image, distinct from an uncalculated image. Editing the instrument marks the
retained image as a previous calculation; its receiver list and coordinates still
come from its own snapshot.

## Image projection at the recording plane

Applying **Image** now refocuses the downstream projection at the first inserted
TEM receiver: the main fluorescent screen when it is inserted, otherwise the
Camera. If both are retracted, the Camera is only an optical reference; no
exposure is implied. Upstream inserted STEM detectors still block TEM reception
and must be retracted explicitly.

The Image solve holds the Objective and illumination fixed. It adjusts D/I from
the stored Image preset, keeps P1/P2 at their preset values and preserves the
sample magnification of that preset at the Camera. It checks the complete
sample-canonical position transfer at a finer integration step before publishing
the result. An unreachable focus leaves the previous settings intact. This solves
first-order conjugacy; it does not remove the configured Cs, A1 or B2 terms.

After changing receiver insertion or optical settings while already in Image,
use **Optical operating preset > Focus Image at receiver**. This explicit action
keeps illumination, Objective, held scan controls and receiver insertion intact.
The ordinary Apply button still reapplies both full presets when the mode keys
are unchanged; a full Nanoprobe preset retracts the TEM receivers. Image solves
run in the background, and results from cancelled or superseded settings are
discarded. Recalculate the particles and receiver image after a successful solve.
Selecting a receiver in the image page alone does not refocus the optics.

## Views

- **Current position** bins the actual first physical hits on the selected
  receiver, including available geometric specimen-exit scattering branches.
  It applies the configured detector point spread. Image/diffraction projector
  settings affect downstream propagation; the display does not invent a pattern
  from the mode label. Broad illumination follows the simulated source profile
  and aperture transmission rather than being forced uniform.
- **Single scan preview** averages translated copies of that retained distribution
  over one captured, calibrated AC/descan raster. It is calculated once and held.
- **Scan position preview** shows the moving beam at the selected raster pixel.
  Use the slider or **Next position** for manual inspection, or **Play scan**
  for timed playback. The cyan centroid trajectory is an optional geometric
  overlay; the yellow point marks the chosen position.
- **Frame build-up preview** accumulates the raster through the current position.
  Signal uses the original full-frame dwell weights, so a partial frame is not
  renormalized to look like a complete exposure. Contrast stays referenced to
  the complete frame during playback.

## Continuous camera observation

**Play scan / Pause scan**, **Reset scan**, **Loop**, and **Speed** control only
the camera/screen preview. With Loop off, playback stops at the last pixel and
retains the completed frame. With Loop on, the next preview frame starts after
the completed frame is displayed. Switching to Current position or Single scan
preview, dragging the position slider, or pressing Next position pauses playback.

**Follow STEM scan** follows the clock of the matching Scanning Image result.
It never connects one calculation's clock to a different instrument snapshot.
When the STEM frame completes, Loop can continue camera observation from the
retained data. HAADF/DF/BF acquisition remains single-frame and is not restarted.
Turning Follow STEM scan off allows independent local preview playback.

To compare compensation, enable **AC Scan Coils > Enable raster drive** and
disable **Descan control (Image/Diffraction) > Enable raster drive** in Scanning
Parameters. Calculate imaging, select an inserted receiver, and play Scan
position preview. Re-enable Descan or adjust its coupling, recalculate, and
compare the trajectory. Disabling the raster removes its time-varying command;
it does not disable the shared Image/Diffraction deflector or static offsets.

The preview does not change live scan settings. Input edits pause it and mark
the retained data stale; Calculate imaging is required before playing updated
settings. Hidden receiver pages stop image work and local timers; local playback
resumes when shown, while a followed scan retains only its latest position.

All scan views are explicitly **geometric previews**, not acquired physical
exposures. They use first-order displacement relative to the actual captured
simulation time, including descan and cross-axis rotation. They do not recalculate
scan-dependent specimen scattering or upstream clipping, and particles missed
by the static calculation cannot reappear. Moved pixels use a conservative
sensor-centre acceptance approximation; stationary accepted hits are preserved.
No detector current or physical electron counts are reported for a preview.
Curved energy-filter paths without a supported scan response remain unavailable.

## Units and data

Receiver coordinates are laboratory X/Y in millimetres. The static image starts
with a 512 × 512 grid over the active-area bounding square. **Fit receiver**
returns to that extent. **Fit beam** and manual zoom rebin the retained physical
hit coordinates into the displayed region, rather than enlarging the original
coarse pixels. The displayed pixel pitch and exported region describe this new
sampling; changing it does not trace new particles or create additional physical
resolution. The view is retained across new calculations for comparison.

**Pixels** selects 32, 64, 128, 256, 512 or 1024 pixels per axis for the static
view. It rebins the same hits at the same physical extent; it does not change
the integration step or source population. The default remains 512. Scan
previews retain their separate 192-pixel grid.

The status reports positive-weight hits in view, occupied bins and the median
weighted effective hit count per occupied bin. These diagnostics use actual
hit weights, with effective count `(sum w)^2 / sum(w^2)`; they do not derive
sample counts from blurred intensities or from the exposure. Deterministic
source samples and branches sharing an ancestor are not independent Poisson
events. These numbers are therefore neither a statistical error bar nor a
convergence certificate. Raw and cropped counts, squared-weight sums and their
interpretation are retained in exported metadata.

**Detector blur** displays the configured physical point-spread response when
enabled; disabling it shows the ideal arrival histogram. Gaussian blur includes
hits outside the displayed crop and is never renormalized to fill the crop.
The full receiver's incident probability and current remain distinct from the
probability inside the displayed region. Detector blur can still hide small
optical changes even when the arrival sampling resolves them. Contrast and log
display affect presentation only; sparse particle sampling remains visible.
For Gaussian clouds with a maximum expansion radius of 0.001 PSF standard
deviations, an integrated sixth-order translation expansion preserves all mixed
moments through degree six without depositing particles onto an intermediate
grid. It retains the physical centroid, source weights, PSF width and crop.
The recorded numerical metadata includes its expansion origin and a global L1
truncation bound; that bound excludes floating-point evaluation error. This
path supports isotropic and axis-aligned anisotropic Gaussian responses.
Other clouds use direct integration for small calculations or a bounded
physical grid with subpixel deposition and a recorded evaluation pitch for
large calculations. A very small viewport samples that smooth response density;
its smaller display pixels do not imply a narrower PSF. Disabling Detector blur
skips this calculation entirely.
Scan views remain bounded geometric previews of the retained seed, not new
finely sampled physical exposures.

Static intensity is probability per emitted electron per pixel. **Expected
incident electrons** uses the captured source current and the chosen static
exposure; it is not the count of numerical rays, a noisy camera ADU signal, or
an assumed quantum-efficiency model. NPZ export retains the raw ideal/response
probabilities, millimetre axes, exposure, capture provenance, and available count
expectation, with pixel pitch and displayed-region metadata. Preview exports include trajectory/time arrays and their explicit
preview status, with no physical count array.
When Detector blur is disabled, exports contain ideal arrivals and their expected
incident counts, and explicitly omit uncomputed detector-response arrays.

The implementation prepares the physical seed and scan geometry once per result
and receiver, then uses bounded CPU image shifts or accumulation. It renders
only visible pages, limits refresh to 10 Hz, and keeps a few display products
rather than all pixel images. Local frames shorter than 0.4 seconds at the
selected speed are slowed for visibility; this does not change physical raster
timing. Fast playback may skip displayed positions; Next position inspects each
retained pixel (the existing geometry may be decimated above 128 pixels/axis).
Following STEM uses its captured clock and coalesces display updates. No CUDA or
repeated particle, sample, or STEM calculation is needed for playback.

## Coherent imaging

Classical particles can show the direct beam and calculated scattering, but
cannot reproduce coherent Bragg interference by themselves. **Stored wave /
reference** retains the previous coherent-image display and separate specimen
exit-wave diffraction reference. Its existing source-admission gate is unchanged.
The specimen Fourier reference is not presented as the selected physical
camera's received image. A complete scan-dependent coherent sensor exposure
requires additional wave transport, not relabelling this geometric preview.

A Ronchigram is a convergent-beam specimen pattern observed in the diffraction
plane. Ray transport can describe geometric shadow distortion, but the current
particle scattering path does not retain a complex wave or coherent specimen
interference. It therefore cannot produce a physical coherent Ronchigram merely
by selecting Diffraction. A wave-based implementation must propagate the
aberrated incident wave through an explicit specimen and then to the diffraction
receiver; an empty-beam disk is not a Ronchigram. See the
[JEOL Ronchigram definition](https://www.jeol.com/words/emterms/20121023.111557.php).

## Reproducible C2 particle comparison

Run `scripts/validate_c2_screen_aberrations.py` with the project environment to
compare C2 Cs, A1 and B2 in **C3 + Probe Corrector / Nanoprobe Normal**. The default
run uses 3000 physical-source trajectories, a retracted vacuum specimen, three
fixed C2-waist sections and the main screen in both projection modes. One set of
projector controls is retained across all cases. The coefficient increments are
controlled sensitivity probes, not measured instrument calibration values.

The script compares 0.05 and 0.025 mm integration steps using matched source IDs,
reports aperture survival separately, and keeps position-response stability
separate from binned-intensity stability. These checks do not establish particle
population convergence or a diffraction-limited probe resolution. Screen PSF
plots retain the configured provisional response. The output directory contains
numeric results, retained coordinates and comparison figures; `--analyse-only`
reanalyses those saved coordinates without tracing new particles. Source identity
checks reject mixed implementation versions, so keep simulator source files
unchanged during the run.

## Image intensity convergence

An aberration's paired particle displacement and its binned intensity difference
are separate observables. A small integration error can push one particle
across a pixel boundary, changing two cells by its complete weight even when
the paired displacement is stable. More display pixels can make this problem
worse at a fixed particle count. Neither interpolation nor detector blur proves
that the unblurred histogram has converged.

For a reproducible comparison, keep projector currents, optical parameters,
physical receiver, observation extent and source model fixed. Change one of
the following at a time:

- Particle count: emit and propagate each requested population independently.
  The historical joint cold-FEG sampler uses batch-dependent energy moments;
  a prefix of a larger run is not the same physical source as a smaller run.
- Integration step: retain the same emitted population and controls while
  refining column field integration after the fixed gun exit. This axis does
  not change the gun solver's own integration settings. Do not re-focus at
  each step.
- Pixel grid: rebin retained physical hits with fixed edges and extent. Also
  check sensitivity to the edge origin. Summing nested fine bins back into
  coarse bins checks conservation, not intensity-density convergence.

Compare signed changes relative to the corresponding baseline before taking
an L1 norm. Report both the absolute difference and its ratio to the resolved
signal. Two zero differences mean no signal was resolved; they must not pass
a relative convergence criterion. Pixel-density comparisons use the same
physical area and account for pixel area rather than comparing array values
with different units. Detector-response comparisons retain the configured
physical PSF and remain separate from raw-arrival comparisons.

`scripts/validate_image_intensity_convergence.py` collects a separate matrix:
3000 particles at steps 0.05, 0.025, 0.0125 and 0.00625 mm, and populations
3000, 12000 and 48000 at 0.00625 mm. Each cell includes baseline, Cs, A1 and B2
with the same fixed Image projector controls. It restores the previous
comparison's input design for new execution, without relabelling historical
result caches. Each population gets a newly executed physical gun; exact
source coordinates, weights, energies and ancestry are checked within that
population. The default requires GPU column transport; physical gun transport
retains its CPU implementation and backend receipts disclose both stages.

Run the collector with the project Python environment, then run
`scripts/analyse_image_intensity_convergence.py --input <collector-output>
--output <analysis-output> --phase-offsets` to compare the independent axes.
The collector needs the preceding C2 comparison's `baseline.snapshot.json` and
`comparison.json` in one directory: first run
`scripts/validate_c2_screen_aberrations.py`, or supply that directory through
`--reference`. It restores only those recorded inputs and executes new sources
and propagation; a historical result is not accepted as a new population.
The collector's `--resume` accepts only matching protocol, source, script and
input identities with verified saved files. Analysis never creates new
particles or changes optics. It reports the original 5% criterion, absolute
signal and error, single-particle bin-crossing weights, missing comparison
axes and unresolved zero signals. The compact-cloud Gaussian reference has
an explicit domain and Taylor remainder check, and is independently checked
against direct Gaussian pixel integrals. It remains separate from the actual
raw-arrival histogram and does not certify instrument accuracy.

### Measured Image-screen limits (2026-10-17)

The complete 24-run matrix used C3 + Probe Corrector / STEM Nanoprobe Normal,
a vacuum specimen, fixed Image projector settings and the main screen at
2714.15 mm. C2 changes were Cs +20 mm, A1 +0.03 mm and B2 +1 mm. These are
controlled sensitivity probes, not measured instrument aberrations. The raw
view was fixed at approximately +/-223.925 nm; the configured Gaussian response
used sigma = 0.6 mm and a +/-2.4 mm view. Different physical observables have
different views; each view stayed fixed throughout its convergence comparison.

- Column integration: successive baseline screen-position RMS changes were
  0.024026, 0.0015518 and 0.00009813 nm, approximately fourth-order refinement.
  For the finest step pair (0.0125 to 0.00625 mm), the physical-response intensity
  differences passed the unchanged 5% criterion for all three aberrations.
- Pixel sampling: at 48000 particles, refinement from 256 to 512 pixels per axis
  changed the physical-response difference density by about 1.14%, 0.88% and
  0.88% for Cs, A1 and B2. Raw-arrival difference densities did not converge.
- Particle population: refinement from 12000 to 48000 particles changed the
  physical-response differences by about 346%, 220% and 100%, respectively.
  All three failed the 5% criterion. The finest Cs difference was only
  1.17e-12 in L1 probability per emitted electron. A1 changed 29 survivor
  memberships with a net loss of five particles; B2 had a net loss of one.
  Such rare interception events and small residual source moments can dominate
  these differences even when the total beam intensity and paired trajectories
  are much more stable.

The old grid deposition erased compact-cloud width changes below its cell
scale. Its error for the finest Cs response was 8.12%, measured against stable
Gaussian pixel integration. The compact-cloud evaluator addresses this rendering
defect: all 90 production-response comparisons passed, with a maximum error of
0.0088% after the fix. It does not make a finite-particle intensity estimate
population-converged.
No tilt, defocus or centroid shift was removed, and no cropped intensity was
renormalised to force agreement. A single CPU/GPU baseline check used identical
source records and survivor IDs, with a screen-position RMS difference of
2.75e-8 nm.

Local evidence is retained under `outputs/image-intensity-convergence-20261017/`:
`manifest.executed.json`, `transport-verification.json`, the archived executed
source, and separate `analysis-before-renderer-fix/` and `analysis-final/`
reports. Analysis records its renderer identity separately from the original
propagation identity. These results establish numerical behaviour only within
the recorded configuration; they are not full microscope qualification.
The selected screen, Image-conjugacy, CPU/GPU transport and analysis regressions
passed 258 distinct test cases, with no failures or skips. Screen-dependent
suites were rerun after the renderer change; GUI checks ran offscreen.

### Source sampling and rare aperture interception

The population comparison above exposed two different errors. The historical
joint sampler recalibrates its energy moments from the current batch, so an
existing source ray can change energy when the requested count increases.
Separately, a single equal-weight ray crossing an aperture changes transmitted
probability by `1/N`. In the 48000-ray comparison, A1 lost 17 source rays and
gained 12; B2 lost one. The B2 loss contribution was approximately 418 times
larger than the intensity change of its common surviving rays. Increasing
screen pixels cannot resolve that source-integration error.

`EmissionSampling(seed=0)` selects `tip-sobol-antithetic-v1` explicitly on the
cold-FEG emitter. Complete power-of-two populations from 16 through 1048576
have nested six-dimensional source coordinates. Adjacent source pairs invert
local position and direction and share energy; downstream positions are never
recentered. The positive kinetic-energy map is calibrated against a fixed CDF
integral, independently of population and seed. It retains the historical
Young plus logistic energy-shape approximation, rather than asserting a new
Gaussian broadening model. Version 1 requires a finite positive energy half
range. Finite populations need not have exactly the target energy moments.
This replaces the historical finite-batch energy fit with a fixed continuous
CDF approximation of the same distribution family. Individual energies are
not bitwise equivalent to the historical sampler at a given finite count;
the new method guarantees its own count-independent source prefix. Historical
records continue to use their original method.
The method, seed and full physical source parameters enter saved inputs and
cache identity. An absent method retains historical behaviour.
The complete-net requirement follows the
[SciPy Sobol sampling contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).

The source method can be collected using
`validate_image_intensity_convergence.py --source-mode sobol-antithetic`.
Its default population sequence is 4096, 16384 and 65536 at the fixed
0.00625 mm column step. `--emit-only` checks emitted tables only and is not
evidence of screen convergence. Use High accuracy for these complete tables;
Preview/Medium budgets and added tuning probes cannot silently truncate or
repartition them. The new method is an explicit backend/profile option, not
a changed default source or a new GUI control.

For aperture-boundary integration, `EmissionCDFQuadrature` records the complete
six-dimensional CDF table, original nonuniform probabilities, source signs and
ray IDs. A physical quadrature covers the original source measure with total
weight one. A `boundary-probes` table carries zero weights and predicts zero
exposure; the gun's configured current remains a nominal input. Neither role
renormalizes surviving rays. Complete tables are preserved in profiles,
snapshots and gun cache identity. Source-model mixtures and mismatched budgets
are rejected before emission.

`source_radial_quadrature.py` is an offline validation helper. It locates
actual signed-aperture-margin roots in angular CDF coordinates and takes their
union across all compared cases and source signs. Gauss nodes then cover every
resulting interval, including intervals whose particles are blocked. Every
case uses the same complete table. This avoids an equal-weight single-ray
step in the integrated transmitted probability without softening the physical
aperture. Root brackets, source-measure conservation, membership at quadrature
nodes and callback batch sensitivity are separate checks. Finite scans cannot
certify that arbitrarily narrow unobserved pass bands are absent; a small
root-solver tolerance alone does not bound transport or quadrature error.
The batched bracket solver uses
[SciPy's elementwise root interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.find_root.html).

`validate_conditional_source_convergence.py` executes these boundary probes and
new weighted gun populations through the original four C2 cases. Its optional
common-azimuth quadrature rotates both local source azimuths together before
actual propagation. With `M` independent five-dimensional Sobol bases and `K`
source orientations, the existing positive/negative pair completes the orbit;
there are `M*K/2` radial bases. Source radii, relative azimuth, energies and
original probabilities are retained. This changes how the isotropic source
integral is sampled; downstream images and centroids are never symmetrized.
Fixed zero-weight endpoint probes diagnose changes caused by the gun solver's
shared adaptive time step. Their count is reported separately from weighted
quadrature nodes.

`analyse_conditional_source_convergence.py` compares independent-base count,
common-azimuth order and radial Gauss order separately. It reports signed
transmitted probability as well as spatial intensity difference. A resolved
spatial response does not imply that a much smaller net flux is resolved.
Located-boundary probability bounds enter a conservative error budget, while
finite-scan coverage and observed transport drift retain their own limits.

The analysis report separates common-survivor motion, lost probability and
gained probability, including their signed flux and cancellation. Physical
screen centroids, original weights and absolute intensity changes are retained
throughout; no tilt, defocus or physical displacement is removed.

For a new backend validation run, reuse the captured optics reference produced
by `validate_c2_screen_aberrations.py` and choose a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/validate_conditional_source_convergence.py `
  --reference outputs/c2-screen-aberrations-20261017 `
  --base-count 64 --azimuth-orbit-size 8 --orders 2 4 `
  --stage full --step 0.00625 --u-tol 1e-12 --maxiter 14 `
  --scan-intervals 8 --scan-refinements 1 --threads 8 `
  --output outputs/source-check-M64K8
```

The reference supplies captured physical controls. Every new source table
executes a fresh physical gun. Complete-case checkpoints are shared only when
their source and physical identities match. `--resume` with the original
arguments verifies archived arrays and code identities before reusing completed
work; it does not relabel an old source or replace a failed propagation.
The recorded validation environment uses SciPy 1.18.0; the batched boundary
solver requires SciPy's `optimize.elementwise` interface.

The uniform-source refinement exposed an additional rare event at the angular
cutoff: near a launch energy of 0.146 eV, a perturbed case can change from a
located radial stop to a fully transmitted interval. Comparing matched roots
alone would omit that real intensity change. The collector instead integrates
all pass intervals, including intervals ending at CDF one. In the recorded
M32-to-M64, K8, G4 comparison, spatial difference errors were 1.65% for Cs,
35.86% for A1 and 0.69% for B2. A1 therefore failed the 5% source-refinement
criterion even after radial and common-azimuth refinement.

`source_energy_strata.py` supplies an optional exact change of source integration
coordinates. For each original Young coordinate it divides the complete
Boersch CDF into three intervals: below, inside and above a fixed launch-energy
band. Each interval retains its original probability, and its nodes pass
through the unchanged source energy law and actual gun. Clipping atoms whose
energy lies inside or on either boundary of the selected band belong to the
inclusive middle interval; atoms outside retain their low/high interval
probability. Only intervals with exactly zero probability
are omitted. The band is a quadrature choice, not an energy filter or a
replacement aperture. It is recorded together with the energy calibration,
original coordinates, interval probabilities and source implementation.
`--energy-band 0.143 0.149` selects this method for the current captured source.
The collector checks sampled low/high energy endpoint containment against all
four actual C2 cases and checks every weighted node's aperture membership
before screen propagation. These checks cannot certify unobserved pass bands.
Located-root uncertainty is weighted by each original interval probability;
uniform-source uncertainty formulas cannot be reused for these tables.

The geometric-field gun's discrete-gradient step controller now checks
transverse momentum and each active ray's transverse position as well as
total momentum and displacement. Longitudinal acceleration must not hide a
transverse error, and a distant ray must not lend its position-error budget
to another ray. Nonfinite candidates are rejected. A successfully computed
half step is reused as the next full-step candidate after halving; implicit
solver failure invalidates that reuse. The numerical criteria enter gun and
downstream cache identity. The force model is unchanged. Geometry gun steps
and physical plane localization both explicitly use an inner implicit residual
tolerance of 1e-13; this numerical policy is also included in the cache identity.
The general static-field stepper retains its separate default of 1e-11.

Gun-only tests with identical source anchors reduced the observed common-ray
M16/M64 position discrepancy from 0.846 nm to 0.444 nm with the selected
transverse check. Further tightening was expensive and not monotonically
convergent within each population, so this is not an absolute trajectory-error
bound or a field-grid qualification. A separate replay of complete source
prefixes from the old actual screen records found spatial-response changes
below 0.22%, including located-root uncertainty. That bounded replay indicates
that the observed gun batch dependence is much smaller than the unresolved
35.86% A1 source-integration change; it does not replace fresh energy-stratified
screen runs. Reports with different source methods or gun criteria remain
separate physical-protocol groups.

For the energy-stratified protocol, add `--energy-band 0.143 0.149` to the
collector command above, keep `--gun-step-scale 1`, and use a fresh output
directory. The band is specific to this captured source and optics reference;
changed optics require actual containment and convergence checks again.
Refine `--base-count`, `--azimuth-orbit-size` and `--orders` independently.
The number of independent source bases is not the transported particle count:
energy strata, azimuth nodes, radial Gauss nodes and zero-weight diagnostics
are each recorded separately. A former angular or particle-count comparison
does not qualify this changed source quadrature.

The initial energy-stratified protocol, with inner gun tolerance 1e-11, gives spatial
difference changes for its M16/K8 radial G2-to-G4 comparison of
0.00000943%, 0.02947% and 3.40168% for Cs, A1 and B2.
At M16/G4, K8-to-K16 gives 0.00000136%, 0.08012% and 0.00000798%.
Both comparisons meet the recorded 5% criterion at that base count. Independent
M16-to-M32 refinement still gives 2.55676%, 5.06375% and 1.56111%; A1 therefore
remains unresolved. B2's much smaller net-flux difference changes by 17.22358%,
which must not be hidden by its passing spatial-response comparison.

The first M64/K8 energy-stratified run is retained as a failed diagnostic. Its
2,620 located roots met their bracket tolerance, but observed endpoint drift
corresponded to 5.077986e-12 in angular CDF, exceeding the requested 1e-12.
The collector stopped before constructing the final weighted source or screen.
Those partial boundary records cannot supply a passing M32-to-M64 screen result.
Replaying the two complete offending source batches with inner tolerance 1e-13
made the original worst gun anchors identical. Actual C2 propagation of those
fresh gun results reduced the maximum endpoint-drift estimate to 1.486e-13 CDF,
using the original located-root slopes. This is a paired numerical diagnostic,
not a retroactive pass for the failed run or a substitute for a new full source
refinement. Neither the root-width criterion nor the 5% screen criterion changed.
The retained report and subsequent diagnostic evidence are indexed in
`outputs/source-sampling-improvement-20261017/README.md`.

### Fresh energy-stratified comparison with the tighter inner solver

The new M32/K8/G2,G4 and M64/K8/G4 runs completed twelve actual Image-screen
cases with unchanged optics, source probabilities, physical PSF and 5% criterion.
Both complete partitions passed the original 1e-12 angular-CDF drift gate:
the observed maxima were 1.791636e-13 and 1.486000e-13, respectively. Exact
resume verification retained all source, partition and screen records unchanged.

At fixed K8/G4, M32-to-M64 spatial-difference changes were 0.863190%,
1.227609% and 0.531540% for Cs, A1 and B2. Including located-root and Gaussian
evaluation budgets gives 0.863767%, 1.227745% and 0.535307%. Signed net-flux
difference changes were 0.863190%, 1.227609% and 1.402098%; their conservative
comparison also remained below 5%, with B2 at 1.992563%.
At M32/K8, radial G2-to-G4 changes were 0.00000863%, 0.0291164% and 3.2407054%.
Physical-response pixel refinement from 256 to 512 remained below 1.116%.
Raw-arrival histograms still failed pixel refinement; at M64 their changes
were approximately 115.6% to 123.2%. The raw and physical-response conclusions
remain separate.

The M64 G4 source retained 19248 positive-weight nodes and 3072 zero-current
diagnostic nodes, with original probability summing to 0.9999999999999999.
There was no survivor renormalisation, image registration or removal of tilt,
defocus or centroid response. For example, B2's actual screen centroid changed
by +0.0308603 nm in X and -0.0247148 nm in Y. These source-base counts are not
unweighted particle counts.

An offline analyzer defect detected by the new records was also corrected:
adding identical channel budgets in a different grouping produced a one-ULP
total difference. The analyzer now uses the collector's channel order and
retains strict equality. Tests still reject a one-ULP change to an archived
channel or total, and changes to the original source measure. This changed
only postprocessing; propagation and the numerical acceptance gates were frozen.
The final related regression set contains 390 distinct passing tests.

These are passing independent comparisons within the recorded configuration,
not overall numerical qualification. Finite boundary scans do not exclude
unobserved intervals, and endpoint drift does not bound all internal transport
errors. Matched physical anchors across M still show batch-dependent gun
outputs. A replay retaining the complete first 32 source bases from M64,
scaling original probabilities by two before survival filtering, changed the
Cs/A1/B2 screen differences by 0.002266%, 0.015628% and 0.005259% relative to
the independent M32 run. This is a derived diagnostic, not fresh propagation
or a certified error bound. No third M level or new K/step axis was executed
for this implementation. Complete results, scope, raw arrays and provenance
are indexed in `outputs/source-sampling-improvement-20261017/README.md`.
