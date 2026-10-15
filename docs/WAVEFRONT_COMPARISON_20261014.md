# Actual and ideal-lens wavefront comparison

The Electron beam workspace can compare an executed coherent wave with an
ideal-lens reference at the same observation plane. This feature is layered on
the B-COR / Nanoprobe integration in local merge `8b1f409`; it does not change
the installed assembly or its saved operating controls.

## Using the view

On a fresh launch, first define the coherent source in **Electron beam → Tip
parameters**: select **Gaussian-Schell emission**, retain **Driven emission ·
non-paraxial near tip** for the default planar source, and click **Apply tip
parameters**. Then click **Calculate beam**. Merely selecting the model does
not apply it. A loaded coherent source can already have this step completed;
source admission still does not establish successful full-column propagation.

For the actual phase alone, calculate the beam and choose **Observe → Phase
plate**. Move observation Z (or enable **Follow Ray Z**) to read another supported
cross-section from the same calculation session. This readout requires neither
a saved target nor an ideal-reference calculation. It retains tilt, defocus,
aberrations and upstream aperture/specimen effects in the executed complex field.
Each coherent mode is inspected separately; the phase wraps to [−π, π].

After changing applied optical parameters, calculate the beam again (existing
live tuning also refreshes after edits settle). A notice beside the observation
marks a retained image as a previous calculation until a new result is accepted.
Switching observations or colour scales does not clear that notice. A failed or
cancelled calculation retains the notice and explicitly reports no new phase.
Only parameters affecting the selected plane's upstream wave field can change
its phase; a downstream component or a pure beam-current scaling need not do so.

For the optional actual/ideal comparison:

1. Calculate the actual beam at the desired Z in **Electron beam**.
2. Choose **Observe → Wavefront comparison**. For an overlay of independent
   source states, first select **Inspect state**.
3. Press **Compute reference**, then choose the coherent **Mode** if several
   modes are present. This performs an additional propagation from the same
   captured source and instrument.
4. Read the linked **Actual**, **Ideal lens reference**, and **Phase difference**
   panels. All use radians and the same cyclic colour scale from −π to π.
5. Change observation Z and calculate that plane before computing its reference.
   Returning to a retained plane can reuse its completed reference. Changing
   the selected mode or constant-offset option only recalculates the readout.

The **Observe → Phase plate** view displays the actual wave alone.
Neither view changes physical wave arrays. All three comparison panels blank
pixels that lack reliable common support. The status reports the plane, mode,
valid fraction, and phase-offset policy. Status and explanation text can be
selected and copied.

## Meaning of the ideal reference

The reference is the captured column's **first-order, energy-specific lens
reference**, not vacuum propagation. It retains main-lens focusing and rotation,
the selected C1 defocus, quadrupoles/stigmator controls, deflection drives,
physical apertures and bores, and the original axial sampling grid. It removes
declared lens Cs, intrinsic A1 and higher-order aberration terms, and nonlinear
hexapole actions. In particular, first-order relay focusing remains present in
the Probe Corrector and B-COR assemblies.

Both branches retain the same executed gun-exit source, reference current and
source-mode definition. Stored source identity uses the committed manifest and
payload checksums, so writing identical data to a different cache directory does
not change that identity. Downstream reference caches are distinct from actual
caches. Instrument, source ancestry, resolved scan identity and exact Z are
verified before a paired result can be returned.

Radial transport also hands active electric fields or finite dipoles to the
existing two-dimensional operator, preserving their energy and affine effects
instead of using the radial matrix-only shortcut.

This is a post-gun reference: gun optics and source preparation are shared, and
the reference does not remove source incoherence or energy spread. Each energy
mode keeps its own first-order focusing; an achromatic mixed electron beam is
not claimed. Material potentials and scattering are unchanged. A scattered mode
can be compared only when its mode ID, energy, electron reference and complete
scattering history match on both branches.

Aligned native **C3 + Probe Corrector + B-COR** plans are supported upstream and
downstream of the specimen. Posed or mapped magnetic fields are rejected when
their nonlinear residual cannot be reliably separated. Existing propagation
admission boundaries still apply, including unsupported observations inside the
gun or specimen and an energy filter without a coherent field operator.

## Phase and numerical safeguards

The displayed phase is the wrapped transverse complex envelope plus its
analytical tilt and curvature carrier. It includes the focusing curvature;
longitudinal action and time phase are not included. There is no aggregate
phase for an incoherent mixture.

The difference is `arg(psi_actual * conj(psi_ideal))`, evaluated on the actual
wave's physical affine lattice. Identical lattices need no interpolation.
Different lattices interpolate the ideal **complex envelope**, then evaluate its
analytical carrier at the actual physical coordinates. Wrapped phase is never
interpolated. The default mask requires both intensities to exceed `1e-6` of
their respective original peaks. Unlike grids also require a three-cell edge
margin and cubic/quintic interpolation agreement within `1e-4` relative envelope
error. These are readout acceptance checks, not a full propagation convergence
certificate.

By default, the common source's constant phase is retained. **Remove constant
phase offset** subtracts only the argument of the complex overlap from the
**difference panel**; the two individual phase panels stay unchanged. It does
not fit away tilt, defocus or higher-order structure. The fitted offset in
radians is recorded; near-zero overlap makes that optional operation undefined
and is reported explicitly.

Only selected modes are loaded from disk after a working-memory preflight.
Computation can be cancelled, and generation/identity checks reject results from
an older plane or mode. Starting a new readout clears the former three images,
so a failed new mode cannot leave its predecessor's phase on display.

## Validation scope

The dedicated tests cover physical plan preservation and aberration removal,
energy-specific focusing, analytical C1 curvature, radial/Cartesian/segmented
routes, source and cache identities, cancellation, known complex-phase fields,
affine-grid readout, support and memory gates, and GUI plane/mode transitions.
The real-geometry checks include the combined Probe Corrector/B-COR assembly.
Pipeline and GUI routing tests use explicitly labelled synthetic source fixtures.

Validation receipts are stored locally under `outputs/wavefront-comparison/`.
They are software and bounded numerical evidence, not experimental microscope
validation or a converged full source-to-camera coherent calculation.

Executed checks on 14 October 2026:

- Final frozen-source integration: **774 passed, zero failures or skips**,
  across 29 related test files in 555.30 seconds (`integrated.xml` and
  `integrated.log`). This includes the 58 dedicated new feature cases, existing
  coherent/material propagation, observations and state controllers, and
  B-COR/Nanoprobe regressions. The final run supersedes the preliminary
  `new-feature.*` run interrupted by concurrent source edits.
- Native Windows Qt: **25 passed**, including the 11 new comparison GUI cases
  and 14 existing observation cases (`native-gui.xml`). The rendered preview
  (`preview.png`) uses a labelled analytical source with common quadratic
  focusing and an added quartic phase; it is display validation.
- Local B-COR reference on RTX 5090: CPU NumPy and required CUDA CuPy both
  executed without fallback. Relative full-complex-field error was
  `4.32e-16`; curvature, lattice, tilt and weight differences were zero. The
  maximum wrapped phase difference on the probe's supported pixels was
  `5.57e-11 rad` (`ideal-gpu.json`). This separate probe uses an amplitude
  threshold of `1e-6` of each peak; it is not the GUI's intensity threshold.
  Unchanged physical hexapole arrays and original source were verified.

`validation-summary.json` records the frozen solver identity and final counts.
Compilation and Git whitespace checks passed. Non-failing CUDA occupancy,
JUnit-property and Qt teardown warnings are retained in the full log.

### Direct actual-phase readout follow-up

The subsequent Phase plate wording and persistent previous-calculation notice
were checked separately. No propagation formula or phase detrending was added.

- Initial focused sweep: **211 passed** (`actual-phase-readout.xml`), including
  independent analytical tilt/curvature changes, same-Z checkpoint replacement,
  old-worker rejection, live refresh, saved states and reference-view regressions.
- After fixing notice clearing when returning to the same valid cached plane:
  **133 passed** (`actual-phase-refresh.xml`), covering the affected observation
  and calculation controls plus saved-state return/admission paths.
- Native Windows Qt: **4 passed** (`actual-phase-native.xml`), checking actual
  phase replacement, persistent stale-input notices, and cached-plane return
  after either failure or cancellation.

All runs exited successfully. The phase examples are analytical fixtures, and
GUI control tests substitute propagation; this follow-up does not assert a new
full-column physical or experimental validation. The existing non-failing
PyQtGraph teardown warning remains in the offscreen logs.

## CPU memory and strong-corrector diagnostic follow-up (15 October 2026)

The production Cartesian solver now bounds additional FFT/operator temporaries
and releases obsolete refinement grids before the next retry. Its configurable
`WaveGridNumerics.multipole_spectral_tail_fraction` changes the bandwidth
admission quantile only: it does not delete spectrum bins, crop the physical
wave, or renormalize probability. The default remains `1e-12`; explicitly
relaxed values are recorded separately from physical support losses.

Relaxing that tail test alone did not resolve the strong HP2 field in the
executed Nanoprobe Normal case. An experimental CPU diagnostic therefore uses
coherently summed Gaussian packets after an actual stored upstream wave, with
local quadratic approximations of cubic/quartic actions. This avoids carrying
a finite Cartesian window through the corrector. Complex coefficients still
interfere; this is not an incoherent ray-density image or a fitted replacement
source. Small canonical matrices integrate the quadratic fields with an
independent fine axial step, and a fourth-order signed composition advances
the nonlinear actions. The approximation is **not installed as an automatic
fallback in the GUI**, and is not a certified full-wave solver.

The local study uses C3 + Probe Corrector, Nanoprobe Normal, 300 kV. C3 changes
by +/-0.1 percentage points; the common upper/lower objective control changes
by +/-0.01 percentage points. All other controls and the specimen entrance
remain fixed, with no autofocus. The plane is Z=1599.1999975 mm, 2.5 nm before
the specimen centre. The nine upstream energy modes are retained, but the
phase comparison follows one fixed coherent E0 component at
300.0003593443496 keV. There is no single phase for the incoherent mixture.

Prepared cases in `outputs/beamlet-case-inputs-20261015/` bind each instrument
snapshot to a checksummed executed wave. C3 cases continue all nine modes from
the retained C2 checkpoint. Unchanged upstream plans are checked before reuse;
this study does not rerun the complete tip-to-specimen source chain. The
experimental runner rejects unsupported hard apertures, posed/mapped fields
and finite dipole drives. Its Gaussian bore-tail bound applies at the original
physical axial nodes, not continuously between them.

With those local prepared inputs available, a fresh baseline can be run using:

```powershell
.venv/Scripts/python.exe scripts/replay_gaussian_corrector.py --case-directory outputs/beamlet-case-inputs-20261015/baseline --output outputs/beamlet-reproduction/baseline --pixels 360 --axial-order 4 --maximum-step-mm 0.5 --quadratic-step-mm 0.0625
```

Use new output directories to preserve prior receipts. The runner forces CPU
and one thread per numerical runtime. Repeat with `--maximum-step-mm 0.25`
and, separately, `--pixels 512` to check the axial and packet discretizations.
`scripts/experimental_gaussian_gram.py` supplies whole-plane coherent norms,
moments and unadjusted complex-field comparisons; the readout and plotting
scripts in the same directory retain absolute intensity and phase. No phase
piston, position or normalization is fitted away.

The current evidence is under `outputs/beamlet-hybrid-20261015/`. Earlier
second-order results under `outputs/beamlet-prototype-20261015/` failed the
axial comparison and must not be used as qualified lens comparisons. The
conservative accumulated local Taylor bound is too loose to certify full-wave
accuracy; successful numerical refinements establish convergence within this
experimental model only. Generated wave arrays and figures remain local and
are not included in the source repository.

The completed hybrid baseline checks retain absolute complex phase:

- Outer step 0.5 to 0.25 mm: whole-plane relative complex L2 difference
  **0.18415%**; RMS radius changes from 0.220738665 to 0.220719313 nm.
- Packet grid 360 squared to 512 squared: whole-plane relative complex L2
  difference **0.11197%**. The tail-sensitive RMS radius changes to
  0.222313819 nm (**0.7136%**), which is reported separately from field error.
- Final diagnostic/helper/provenance/acceptance regressions: **212 passed**,
  no failures or skips (`diagnostic-regressions-final.xml`). The independent
  obsolete-grid lifetime tests also confirm unchanged complex output.

These are baseline-only empirical checks, not a convergence claim for every
lens setting. The earlier second-order axial comparison failed despite its
apparently good packet-width comparison; it is explicitly superseded.

All five 360-squared hybrid continuations completed on CPU. Their isolated
process peaks were 290--308 MiB; the separate 512-squared validation reached
384 MiB. These are process measurements, not the total GUI working set. The
fixed-plane whole-space RMS radii were:

| Excitation change | RMS radius (nm) |
|---|---:|
| Baseline | 0.220739 |
| C3 -0.1 percentage points | 4.560803 |
| C3 +0.1 percentage points | 4.440682 |
| Common objective -0.01 percentage points | 65.091012 |
| Common objective +0.01 percentage points | 64.915020 |

The five coherent norms differ by at most 1.78e-10: the large radius changes
are redistribution at the fixed plane, not substantial loss. Figures use a
shared absolute logarithmic intensity scale and explicitly label their central
2 nm crops; wide X/Y cuts show the broadened beams. The local
`figures/report.html`, `figures/comparison-summary.json`, and
`global-moments-five-cases.json` contain the plots, numerical controls, audited
nonnumerical version changes, and exact packet/readout/source identities.
