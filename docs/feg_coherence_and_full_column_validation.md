# Physical FEG coherence and full-column hybrid validation

The physical FEG geometry, electrode voltages, installed column and absolute
emission current remain the instrument inputs. A coherence length is an
additional **source-model assumption**, not a measured gun calibration or a
replacement beam at the gun exit. The default classical Flat tip remains
unchanged unless the user explicitly applies a coherent boundary.

## Spatial correlation at the physical tip

`TipCoherence.transverse_coherence_length_nm` optionally specifies a fixed
transverse length L at **tip Z = 0**, using the barrel X/Y coordinates. At each
kinetic energy E the cross-spectral density is

\[
W_E(r_1,r_2)=\sqrt{I(r_1)I(r_2)}
 \exp[-|r_1-r_2|^2/(2L^2)]
 \exp[i(\phi_E(r_1)-\phi_E(r_2))].
\]

The existing source FWHM defines I. Existing phase tilt and quadratic curvature
define the deterministic phase. The modulus of the normalized correlation is
exp(-1/2) at separation L. This is an ensemble mutual intensity; independent
electrons need not have the same absolute phase. The spatial correlation is
not a random phase attached independently to every classical ray.
The length is specified only at the emitting plane. Subsequent focusing,
acceleration and apertures evolve the modes and their correlations; the
downstream correlation length is not reset to the entered value.

For the circular Gaussian-Schell model, the additional per-axis canonical
momentum RMS divided by total momentum is

\[
\alpha_{inc}(E)=\lambda(E)/(2\pi L).
\]

A fixed L therefore has different angular RMS at different energies. The
legacy fixed-angular-RMS parameterisation remains supported, but a nonzero
angular RMS and an explicit length cannot be specified simultaneously. Its
derived L is energy-dependent. Length must be finite and positive; the
fully coherent limit remains the legacy zero-incoherent-RMS selection.

For intensity RMS sigma, the existing Hermite-Gaussian decomposition uses
occupation `sqrt(1 + 4*sigma**2/L**2)`. Each energy has its own wavelength,
phase and mode weights. Truncated mode probability is reported, never restored
by scaling the transmitted current. Temporal coherence continues to come from
the original positive energy distribution; no independent longitudinal length
is added that would conflict with that distribution.

The mode-sum test reconstructs a mutual-intensity matrix element independently
and compares its normalized magnitude with the specified Gaussian correlation.
It also checks energy scaling and retained absolute mode weights.

This implementation follows the Gaussian-Schell coherent-mode framework of
[Starikov and Wolf, JOSA 72, 923 (1982)](https://doi.org/10.1364/JOSA.72.000923).
The need to retain acceleration and the emitter boundary in electron-source
wave modelling is discussed by
[Krecinic and Ernstorfer, Physical Review Applied 15, 064031 (2021)](https://doi.org/10.1103/PhysRevApplied.15.064031).
These sources support the modelling choices, not the particular simulator
coherence length, current, electrode dimensions or an OEM reconstruction.

## Controls and persistence

- Physical Layout > Tip parameters > Gaussian-Schell emission > Specify
  transverse coherence length.
- Coherent beam source controls expose the same optional length. Applying
  either interface updates the one instrument source. An unapplied draft
  cannot change an executed beam. The initial 20 nm editor value is an
  uncalibrated model assumption and is inactive until explicitly selected.
- Profiles retain the length in `[gun_source_model.tip_coherence]`.
  Historical profiles without it retain their angular parameterisation.
- Particle emission, source admission, coherent modes and optical support
  probes all use the same energy-dependent angular conversion.

The current 5 nm, 0.3 eV Flat tip requires the driven, non-paraxial near-tip
boundary route. A source that fails its forward-support or near-field accuracy
guard is rejected. Its size, energy floor and current are not changed to make
it pass. The driven particle reference represents geometrical phase-gradient
rays; quantum diffraction and reflection are additional wave effects, so
agreement with that reference is not exact quantum equivalence.

This length currently applies to the supported planar Gaussian boundary.
The separate Curved tip surface reservoir is not silently replaced by it;
extending spatial correlation to that curved surface needs a surface-specific
mode model and solver. The Flat tip remains the user's selected default.

## Reproducible physical workflow

`scripts/validate_physical_hybrid.py` records a fresh input profile and complete
instrument snapshot, emits at least 1000 original positive-current particles,
and traces them through the actual FEG and full finite-field column. Originals
stopped inside the gun retain their identities and weights. Column transport
never resurrects their finite exit placeholders. Installed apertures, bores,
liners and inserted receiving planes remain active.

The final exact accounting condition is
`emitted_weight = intercepted_weight + surviving_weight`, with absolute
residual <= 1e-12. Counts, effective counts and currents are separate fields.
Sample-plane D95/RMS/alpha95 are observations of fixed controls, not focus or
working-point qualification. A source-count comparison alone does not certify
field, step, geometry or coherent-grid convergence.

Example commands (each output directory must be new):

```powershell
.\.venv\Scripts\python.exe scripts/validate_physical_hybrid.py --output outputs/physical-hybrid-2026-11-09/classical1024-v2 --source-particles 1024
.\.venv\Scripts\python.exe scripts/validate_physical_hybrid.py --output outputs/physical-hybrid-2026-11-09/classical2048 --source-particles 2048
.\.venv\Scripts\python.exe scripts/validate_physical_hybrid.py --output outputs/physical-hybrid-2026-11-09/length20nm-static1024 --source-particles 1024 --coherence-length-nm 20 --hybrid --static-acquisition --sample-width-nm 1000000
```

The coherence cases keep the original 5 nm emission width, 0.3 eV mean energy,
0.3 eV RMS-equivalent FWHM spread and 10 microampere injection reference.
Selecting coherence explicitly replaces the historical truncated 1 mrad
classical angular law with the declared source boundary. It does not claim to
preserve that old brightness or infer coherence from it. The unchanged
classical source is retained as a named comparison.

By default these trials preserve the configured vacuum specimen and inserted
receivers. An inserted fluorescent screen legitimately blocks a downstream
camera. A nonzero camera image must not be manufactured by clearing that
screen implicitly. Actual material diffraction and an explicitly configured
camera acquisition are separate subsequent cases.

The default AC/DP12 and Descan drives have active raster scanning. Their
time-dependent fields, including induction, remain in the classical baseline.
The current hybrid phase transport rejects those drives instead of sampling
and freezing them. `--static-acquisition` explicitly creates a separate
stationary operating input: scan/wobble controls are disabled, while physical
coils, DC settings, lenses, apertures and receivers remain. Every changed
control is recorded in `report.json` and in the saved input profile. This does
not implement time-dependent coherent transport.

The example 1 mm specimen lattice is a numerical envelope for the untuned
beam. At 256 pixels it is not an atomic-resolution material calculation. The
lattice does not adjust optical controls or focus the beam.

CUDA is required and verified with actual device execution before every new
hybrid run. Reports retain the original source arrays and current/interception
ledger even if a later coherent stage rejects its numerical representation.
Only a completed `hybrid/report.json` establishes executed hybrid transport;
a successful classical baseline or coherent source preflight does not.

The near-tip angular Robin solve, FFTs, spectral transforms, gun canonical
maps and masks execute with CuPy in the physical hybrid path. Field-provider
queries and small real matrices remain host operations. There is no coherent
CPU fallback. Pure numerical CPU/CUDA comparisons are operator tests, not
full-source physical qualification.

## Executed evidence on 2026-11-09

All rows below are the classical, geometrical-optics reference. Each emits
the stated number of original positive-current particles and retains its
10 microampere injection current, original identities and absolute weights.
Every original reaches the gun exit in these particular cases; this is an
observed result, not a bypass of gun interception. The full-column absolute
weight balance residual is zero in all four rows.

| Source and acquisition | Original particles | Specimen survivors / effective count | Specimen current (microampere) | Specimen D95 (micrometre) | Alpha95 (mrad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical angular law, configured scan | 1024 | 1024 / 1024 | 10 | 53.4613 | 9.15089 |
| Historical angular law, configured scan | 2048 | 2048 / 2048 | 10 | 53.4687 | 9.16589 |
| L = 20 nm assumption, stationary | 1024 | 289 / 289 | 2.82227 | 311.183 | 51.3051 |
| L = 20 nm assumption, stationary | 2048 | 584 / 584 | 2.85156 | 312.355 | 51.4247 |

The two source laws are different physical assumptions. Only rows within a
source/acquisition pair form a source-count comparison. For the 20 nm pair,
doubling the originals changes specimen current by 1.038%, D95 by 0.377%,
radial RMS by 0.0313%, and alpha95 by 0.233%. The C2 aperture intercepts
7.17773 and 7.14844 microampere respectively. No lost current is restored.
These are untuned envelope observations, not an atomic probe, a best-focus
measurement or a fully converged operating point.

Only 5 and 8 originals respectively reach the HAADF receiving plane in that
pair. They are all intercepted there; this is too little sampling for a
detector-current convergence claim. The remaining electrons stop at physical
column boundaries. All four cases have zero camera current with their actual
receivers and mechanical interception retained. In the historical-source
cases the inserted fluorescent screen intercepts 1.39648 and 1.39160
microampere before the camera.

The source mode preflight at 20 nm uses three positive energy samples and
48 coherent modes, with omitted occupation probability 2.95376e-8. This is
only a source representation: it does not certify near-tip scattering,
gun/column propagation, specimen interaction or detector intensity.

Evidence directories under `outputs/physical-hybrid-2026-11-09/`:

- `classical1024-v2` and `classical2048`: completed configured-source baselines.
- `length20nm1024-v3`: completed classical scan baseline; hybrid correctly
  refused active AC/DP12 and Descan waveforms.
- `length20nm-static1024` and `length20nm-static2048`: completed stationary
  classical references. The former's additional hybrid attempt rejected the
  default near-tip transverse phase estimate of 0.400688 rad against 0.001 rad.
- `coherence-tests.xml`: 264 source, persistence and offscreen GUI tests passed.
- `gun-backend-tests.xml`: 54 focused hybrid/GPU/near-tip tests passed.
- `hybrid-admission-tests.xml`: 6 hybrid tests passed after adding the active
  scan rejection regression.
- `near-tip-domain-tests.xml`: 59 near-tip/GPU/policy tests passed, including
  the new numerical-domain validation. These overlap the previous 54 tests.

The numerical endpoint and checked transverse radius are now explicit gun
wave controls. Changing them never changes the physical source or electrodes.
Both the original transverse-phase budget and the executed 1e-8 outside-domain
and paraxial-continuation criteria remain mandatory. A smaller checked domain
is admissible only if the propagated wave independently fits inside it.

### Completed physical-gun wave and remaining column limits

The follow-up `length20nm-static1024-domain1` used the same 1024-original
source and stationary controls, with an explicitly selected 1000 nm near-tip
endpoint and 0.4 micrometre checked transverse radius. Its command was:

```powershell
.\.venv\Scripts\python.exe scripts/validate_physical_hybrid.py --output outputs/physical-hybrid-2026-11-09/length20nm-static1024-domain1 --source-particles 1024 --coherence-length-nm 20 --hybrid --static-acquisition --sample-width-nm 1000000 --near-tip-end-z-nm 1000 --near-tip-radius-um 0.4
```

All 48 modes completed the non-paraxial accelerating prefix and the installed
gun wave path to Z = 450 mm. Both stages record `compute_backend: cupy`,
complex128/float64 precision and no fallback. The emitted current reference is
10 microampere; the retained gun-exit wave current is 9.979200252 microampere.
This includes the executed source reflection and gun losses, without restoring
lost current. It is a result of the existing quadratic transverse-field gun
continuation, not field/grid/axial-step convergence or OEM qualification.

Maximum checks across all 48 near-tip modes were:

| Quantity | Executed value | Limit |
| --- | ---: | ---: |
| Transverse electric phase estimate | 7.81101e-5 rad | 1e-3 rad |
| Probability outside the checked domain | 2.00028e-9 | 1e-8 |
| Absolute injection/reflection/transmission balance residual | 1.29896e-14 | 1e-8 of injected modal flux |
| Paraxial-incompatible transmitted fraction | 7.34512e-15 | 1e-8 of transmitted modal flux |

The gun checkpoint is committed under that run's `wave_cache/`; its complex
mode arrays, near-tip spectral amplitudes/normal derivatives and implementation
identity remain intact. `gun_checkpoint_inspection.json` records a separate,
checksum-verified inspection of this saved result. Subsequent GUI-only controls
and provenance wording changes do not rewrite the historical checkpoint.

The complete hybrid run **did not reach the specimen**. The current boundary
planner needs 11,604 observation planes for this gun grid and installed column,
exceeding the requested 10,000-plane budget. Read-only planning with a larger
budget established this count; it did not execute column propagation.
Inspection of the first saved gun mode also rejects expansion of its analytic
phase carrier: the combined bandwidth is 34,001.1 rad against a 0.8*pi sampling
limit, requiring a nominal linear refinement factor of 13,528.6. Raising only
the boundary budget cannot resolve this second limit.

The next numerical work is a coherent packet representation that retains the
analytic curved phase, together with bounded geometry scheduling and explicit
intermediate lattices. Brute-force expansion would be impractical; cropping
the wave or removing boundaries would change the requested calculation.
No full-column hybrid image, material diffraction, or downstream wave-derived
particle current is claimed by this trial. The configured specimen is vacuum.

The two near-tip numerical controls are also exposed in the Coherent beam
advanced controls. Five focused GUI checks passed in
`final-gun-controls-tests.xml`, including invalidation of a prior result after
either numerical control changes and preservation of all source parameters.

### Subsequent nanoprobe tuning and saved physical evidence

The later Si[110] round deliberately retunes scalar lens/corrector drives on
the current installation while retaining the original FEG, 100 um C2 aperture,
absolute current and physical interception. Fresh 2048- and 4096-particle
references reduce the specimen D95 to about 9 nm; source-count and integration
step comparisons are recorded separately. This supersedes the untuned beam
width for that new working point, without changing the historical runs above.
See [the saved round, input profiles and result images](development/si110_hybrid_20261109/README.md)
for its exact execution status. A completed geometric nanoprobe or gun wave is
not a completed global Si diffraction/camera result.
