# Unified architecture: implementation and validation map

The unified-field work keeps physical inputs, numerical approximation and
product admission separate. The legacy production route retains its saved
settings. New potential executors and reductions are explicit APIs with bounded
capabilities; they are not an automatic replacement of every source-to-screen
calculation.

This map makes the proposal's deliverables and their evidence discoverable.
Tests qualify their declared fixtures. They do not establish agreement with an
experimental microscope, undisclosed OEM fields or an arbitrary input wave.

## Proposal stages

| Stage | Implemented responsibility | Evidence and scope |
| --- | --- | --- |
| P0 | Immutable physical identities, field/geometry dependencies, effect owners and replacement conflicts. | Seven column assemblies; shared Descan/Image-Diffraction hosts; duplicate fields, pupils and boundary actions rejected. See [ownership](P0_PHYSICS_OWNERSHIP.md). |
| P1 | Retain native analytical, electric and FEM potentials; derive E/B consistently; preserve gauge, mesh and known/unknown domains. | Manufactured and native fields, axes, poses, interfaces, composition and mutation checks. B-only coherent imports remain rejected. See [potential snapshots](P1_POTENTIAL_SNAPSHOTS.md). |
| P2 | High-order round-lens fields and an independently resolved FEM reference, with natural Cs/Cc measured from propagation. | Independent analytical/Lorentz and air-core FEM comparisons; field order, mesh, exterior boundary, ray step and fixed-field energy perturbations are separate checks. See [round-lens benchmarks](NATURAL_ROUND_LENS_BENCHMARK.md). |
| P3 | Gauge-consistent fixed-grid and canonical-carrier wave propagation, full-potential cross terms and absolute flux. | Independent complex-field, gauge, Hermitian, narrow-packet, support and actual CPU/CUDA checks. The carrier residual supports bounded CUDA execution; its field sampling, geometry and canonical chart remain on the CPU. Separate electric-tail, retarded native-scan and kinetic-order checks are described in [dynamic execution](UNIFIED_DYNAMIC_EXECUTION.md); historical magnetic qualification remains scoped in [wave operators](POTENTIAL_WAVE_BENCHMARK.md). |
| P4 | Static C3/Probe pilot in Normal and Large Angle, actual sources and aperture definitions, canonical aberrations and physical drive responses. | Canonical rays and both controlled whole-wave modes pass their bounded campaigns. The whole-wave tests cover three axial steps, independent grid/window/field/support checks, magnetic thickness and drive response. A separately declared HPC functional stress drive passes paired refinement; smaller unresolved changes remain explicit sensitivity evidence. See [canonical diagnostics](CANONICAL_PROBE_DIAGNOSTICS.md) and [wave qualification](POTENTIAL_WAVE_BENCHMARK.md). |
| P5 | Device capabilities, authoritative finite geometry, material/energy handoff and receiver observations through the retained potentials. | Seven-assembly geometry, zero-excitation hardware, tilted finite boundaries, real atomic material and energy-changing conditional modes. See [geometry](POTENTIAL_GEOMETRY_BOUNDARIES.md) and [material handoff](POTENTIAL_MATERIAL_HANDOFF.md). |
| P6 | Optional qualified quadratic reductions and bounded design on private captured states. | Coefficient proof, support/energy restrictions, source invalidation, replacement ownership and full-complex native-stigmator phase matching. See [reduction and design](QUALIFIED_QUADRATIC_REDUCTION.md). |

The three calculation categories are a separate user-facing organization:
[Ideal Optics, Finite-Field Optics and Coupled Multiphysics](CALCULATION_MODES_AND_NUMERICS.md).
The last remains disabled. A faster thin approximation is not automatically a
qualified surrogate of the finite-field model.

## Validation gates

| Gate | Main regression families | What a passing result means |
| --- | --- | --- |
| G0: preservation | `test_physical_scene`, `test_plan_physics_contract`, `test_physics_contract_persistence`, `test_simulation_modes` | Shared hardware and saved settings retain their meaning; model switches do not retune currents. Separate seven-assembly numerical captures compare the legacy route. |
| G1: fields | `test_analytic_potential`, `test_axisymmetric_potential`, `test_electric_potential_snapshot`, `test_high_order_round_lens`, `test_potential_fem_qualification` | The declared potential representation, Maxwell-region assumptions, field derivatives and domains agree with their independent references. |
| G2: dynamics | `test_potential_ray`, `test_potential_aberrations`, `test_round_lens_aberration_benchmark` | Full Lorentz and declared truncated dynamics meet their separate conservation and convergence checks. |
| G3: aberrations | `test_potential_fem_aberrations`, `test_potential_probe_canonical`, `test_potential_aberrations` | Fixed-snapshot, explicit-coordinate Cs/Cc responses converge; fits retain lower-order and held-out information. A local nonlinear round-lens fixture also compares independent ray action and cubic canonical gradient with the actual wave phase, without fitting away piston or focus. That local bridge does not qualify a whole-column coefficient-to-wavefront reconstruction. |
| G4: ray/wave limit | `test_potential_wave`, `test_potential_wave_carrier`, `test_potential_wave_carrier_reference` | Quadratic full-complex solutions and a narrow packet's mechanical motion agree in the declared regime. |
| G5: phase/flux | `test_potential_wave_mode`, `test_potential_wave_support`, `test_potential_phase_budget`, `test_potential_probe_whole_wave` | Absolute weights and phase references survive; full-field comparisons do not fit away translation, piston, tilt or defocus. Accepted kinetic truncation bounds consume an optional cumulative budget. Independent numerical refinements remain required. |
| G6: boundaries/materials | `test_potential_geometry`, `test_potential_specimen_transport`, `test_potential_support_lineage` | Finite obstruction and actual material losses are distinct from numerical support projection; conditional energies reach the downstream executor without a second legacy pupil. |
| G7: ownership | `test_interaction_registry`, `test_plan_physics_contract`, `test_qualified_field_reduction` | A field and its replacement, repeated aperture or shared circuit cannot execute twice. |
| G8: identity/restart | `test_physics_contract_persistence`, `test_potential_wave_observation`, `test_potential_geometry`, `test_potential_support_lineage` | Relevant state changes invalidate execution; exact boundary events and loss/error ledgers survive continuation. A changed source implementation intentionally invalidates captured state. |
| G9: execution parity | Actual-device cases in `test_potential_wave`, `test_wave_support`, existing specimen/backend regressions | The exercised CPU/CUDA paths agree for the same frozen model. Missing hardware is reported separately, never counted as a successful execution. |
| G10: admission/display | `test_potential_capabilities`, `test_potential_capture`, `test_potential_field_vector_scene` | Unsupported paths fail explicitly; the optional field display samples the same retained E/B and preserves unknown regions. Source-specific support metadata remains necessary for display sampling. |

All names above refer to files under `tests/`. The acceptance registry in
`src/temsim/acceptance.py` is authoritative for execution scope membership.

## Running the bounded campaigns

Use the repository environment and offscreen Qt. The normal `lens-geometry`
scope contains the short potential, geometry, material and design regressions.
The more expensive canonical and whole-wave campaigns have separate scopes:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -X utf8 scripts/validate_classical_scope.py --scope column-field-pilot --output outputs/agent-validation/column-field-pilot --timeout-seconds 5400
.\.venv\Scripts\python.exe -X utf8 scripts/validate_classical_scope.py --scope column-wave-qualification --output outputs/agent-validation/column-wave-qualification --timeout-seconds 20400
```

The whole-wave campaign is registered in the manual
`unified-column-qualification.yml` workflow because independent full-column
refinements take substantially longer than ordinary regression. It is not
replaced by a skipped test or a short local operator test. Both workflows retain
failed evidence. A workflow definition is not evidence that remote CI ran.

Keep implementation files stable during captured-state tests. Their source
identity guard deliberately rejects a restore if another process edits the
implementation. Rerun after changes stop; do not relax the guard to accept a
concurrent editing race.

The observation regression also launches a fresh Python interpreter from a
temporary directory. It restores serialized potential and wave records, resumes
nonempty support and kinetic ledgers, and compares the absolute complex result
with continuous execution. Budget resets and a changed gauge/source are rejected
in that child process. This exercises actual disk/process boundaries rather than
only returning an object from an in-process cache.

## Actual default-source development checks (2026-10-25)

These checks use the unchanged physical FEG and STEM Nanoprobe Normal optical
parameters. Explicit numerical settings and frozen implementation identities
remain part of each record; they are not a default nine-mode whole-column
qualification.

- The public particle calculation completed 3000 emitted trajectories to
  2987.15 mm. At the 1599.2 mm specimen entrance, 2657 survived, with absolute
  incident-reference weight 0.8856666666666668. Apertures, detector interception
  and finite geometry retained their actual losses.
- Two fresh public executions produced all nine FEG energy modes and
  propagated every mode to 450.4 mm through the captured electric tail and
  native dynamic snapshot. They used desktop-default gun steps (0.05 mm field,
  1 mm bore), 128 source pixels, nine energy samples and a 50 nm near-tip step,
  with explicit H8, Chebyshev and controlled support selections. The host
  limit was 8 GiB. After the retained-memory accounting repair, all committed
  wave arrays, analytical carriers, energies, clocks and absolute weights
  were bitwise identical to the preceding public execution. Final total
  weight was 0.9984474768831042; the separately recorded column numerical
  projection loss was 1.15962e-8 per entrance electron. Independent source-grid
  and near-tip axial refinements remain separate: source axial convergence
  has not been established. This short interval does not exercise the beam
  at the downstream AC coils or establish whole-column completion.
- One actual mode reached 513.8095 mm with H8, controlled support projection
  and the original physical gates. Declared host/device resource planning
  selected a 5346-square lattice instead of a device-inadmissible larger
  candidate. Its arrays, analytical carrier and all cumulative histories were
  exactly equal to the independent smaller-host-budget control.
- At the expensive C1 interval 502.8571 to 503.3333 mm, grouped Chebyshev and
  Taylor used the same entrance and full operator. The raw full-complex
  relative difference was 1.42e-15 and the absolute weight difference was
  1.11e-16, with identical analytical carriers and no fitted phase or
  normalization. Operator wall times were 273.15 and 85.69 seconds in this
  sequential run; the first Taylor run may include warm-up. All original
  intermediate check times remained active. This local speed comparison does
  not predict whole-column runtime.
- A continued actual mode reached 519.047619 mm on a 6250-square lattice.
  At the next 519.523810 mm step, the original envelope guard required
  refinement. The admitted-size candidate was 6600 square and required
  27,878,400,000 device bytes, exceeding the declared 24 GiB
  (25,769,803,776 bytes); the unresolved optical operator was not applied.
  The host limit remained 32 GiB. This continuation added no support
  projection events and retained absolute weight 0.11098136568960873.
  At the largest accepted grids, the additional Chebyshev workspace did not
  fit and the recorded method reverted to Taylor on the required GPU. It
  never changed to CPU or silently raised the resource budgets.

The complete default-source coherent route to the specimen and detector has
not completed. The fast-growing post-focus envelope bandwidth is a remaining
computational limitation. Resource limits, weak spectral tails, source
convergence and nonquadratic kinetic phase must not be hidden by changing
optical parameters, discarding modes, fitting away phase or reporting a
controlled low-NA fixture as the actual source.

A subsequent explicit weighted spectral-projection continuation crossed the
old resource limit and reached 520 mm on a 5544-square lattice, retaining the
original 24 GiB device and 32 GiB host limits. The original source, fields,
physical losses and spent phase/support histories remain intact. This result
qualifies that continued source mode and interval, not all nine modes or the
complete column. See [weighted coherent segments](WEIGHTED_COHERENT_SEGMENTS.md)
for projection mathematics, immutable history, segmented storage and limitations.

The explicit exploratory branch subsequently completed all nine original
source modes at 506.6176470588235 mm with a 2048-pixel column cap. Its cumulative
numerical support loss was `1.2424097245540268e-5` per reference electron and
total weight `0.9984350643820183`. It preserved every original mode and history;
this is an intermediate preview checkpoint, not nine-mode specimen or screen
completion. The wider declared preview budgets remain separate from the
earlier stricter single-mode evidence. A further continuation completed all nine
modes at 520 mm, each on a 1944-square lattice, with weight
`0.998173638036137`, cumulative support probability loss
`0.00027385044310326395` and support L2 bound `0.028170267610324085`.
Its original `0.001 / 0.1` allocation remained unchanged. The subsequent wider
`0.01 / 0.5` preset is a separate numerical branch from the actual gun output;
it cannot relabel that executed history or claim whole-column qualification.

## Limits that remain explicit

- The coherent operator is a forward scalar model. Admitted electrostatic
  acceleration and retarded native scan fields retain separate amplitude,
  phase and clock budgets. Spin, reflection, arbitrary pulse wavepackets,
  dynamic material scattering and curved coherent energy-filter propagation
  remain outside the implemented scope.
- The whole-column coherent fixture is a controlled low-NA entrance. It does
  not reconstruct the phase of an executed FEG particle cloud or qualify an
  arbitrary high-NA physical source. Native compact multipoles retain their
  declared zero exterior, not a newly calibrated fringe-field model.
- Numerical support projection has a cumulative loss/error record and never
  renormalizes the physical result. Its local projection bound alone does not
  bound the error of subsequent propagation on a changed domain.
- The optional kinetic budget sums accepted operator-truncation bounds. Its
  phase units are not a guarantee of pointwise phase accuracy, and it does not
  cover field-model, spatial sampling, splitting or material-model errors.
  A saved calculation cannot reset its consumed allowance at a restart.
- A control response exceeding five times the largest independently observed
  refinement difference passes an empirical resolution criterion. This is not
  a rigorous combined error bound or a statistical five-sigma claim. Paired
  response tests retain both original complex fields and their absolute
  convergence checks; they do not fit away common physical phase or motion.
  A resolved functional stress drive does not establish the smallest usable
  fine adjustment, an OEM range or a calibrated linear response interval.
- Mechanical boundary tests qualify an absorbing finite geometry model over
  specified refinements. They do not add atomic wall diffraction or arbitrary
  holder material from a display mesh.
- Geometry-driven and native Gaussian fields are different physical models.
  Their disagreements must not be hidden by changing the saved optical inputs.
- The existing legacy camera-wave intermediate-mask fixture has a known raw
  canonical-map rejection. At the recorded baseline and current model, the
  maps and rejection agree bitwise. Its symplectic guard remains active; this
  task does not claim that the entire repository test suite is green.

