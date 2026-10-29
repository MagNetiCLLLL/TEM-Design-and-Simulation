# Calculation models and independent numerical controls

The **Simulation** menu separates the column model from numerical resolution.
Changing a label does not reinterpret a saved model key or reset the electrical
operating point.

| Menu category | Saved key | Meaning |
| --- | --- | --- |
| Ideal Optics | `ideal_thin` | Fast particle transport with optical actions at declared planes instead of resolved field thickness or magnetic materials. Component positions and apertures remain meaningful. |
| Finite-Field Optics / Shared potentials and geometry | `unified` | Particle and coherent column execution consume one immutable potential and finite geometry definition. Natural field aberrations replace empirical screens in that interval. Unsupported configurations fail explicitly. |
| Finite-Field Optics / Parameterized fields (legacy) | `ideal` | The existing finite analytical fields, configured aberrations, multipoles and actual electron energies. The historical key retains its original behavior. |
| Finite-Field Optics / Analytical Field (compatibility) | `analytical` | Existing profile alias for the parameterized finite-field laws. The saved key remains unchanged. |
| Finite-Field Optics / Linear Geometry Field | `linear_geometry` | Existing linear FEM field recipes, with explicit geometry, current and material inputs. Unconfigured recipes are unavailable. |
| Finite-Field Optics / Nonlinear Material Field | `nonlinear_material` | Existing static isotropic B-H field recipes at the selected currents. This is not coupled circuit, thermal or structural simulation. |
| Finite-Field Optics / Custom / Per-lens Models | `custom` | Existing explicit per-lens analytical, generated or imported field selection. |
| Coupled Multiphysics | `coupled_multiphysics` | Future coupled electromagnetic, vacuum, thermal and mechanical model, including explicitly defined specimen drift. Disabled. Existing independent vacuum scattering is not this coupling. |

Linear, nonlinear and custom field sources belong to the finite-field category;
the menu grouping does not replace their current solvers or qualify additional
field models.

Ideal Optics retains the saved empirical Cs/Cc inputs but does not execute their
screens. Energy-dependent thin magnetic power and explicit integrated multipole
actions remain part of this approximation. Select Finite-Field Optics to study
distributed finite-thickness response. Configured empirical aberration screens
remain available in the explicit legacy execution entries.

Legacy finite-field execution preserves the existing explicitly selected equivalent-image
reduction as well as the distributed field route. The category name does not
turn that optional reduced interval into a resolved finite lens. Its ownership
remains recorded separately; disable the equivalent-image approximation when
studying distributed projection fields. Existing profiles are not rewritten.

Fresh desktop windows select `unified`. Its actual gun exit is handed to the
shared field/geometry runtime; the gun itself retains its source and acceleration
model. The CPU wave adapter includes the retained static electric gun tail in
its electromagnetic forward operator. Native axial scan coils use an explicit
retarded reference-wavefront approximation, including induced electric work;
this is not a time-domain pulse calculation. Unknown field domains, B-only maps,
unqualified temporal drives and raster flybacks remain explicit errors. An exactly
constant grounded-downstream potential can be referenced to the executed gun
kinetic energy while preserving its radial validity domain. Admission messages
identify unsupported configurations and retain previous plots as stale.

The standard source state has nonzero time-dependent AC/Descan scan commands;
these are not parked automatically. Both particle and coherent runs capture an
emission epoch and advance their own flight-time references. Automatic scan
calibration measures the finite-coil response of the shared-potential executor.
Exactly zero dynamic amplitudes are
admissible as static fields even if the enable flag remains set. Finite-volume
particle material transport, raster products, legacy coherent imaging and the
dispersive filter still have unmigrated internal consumers and are rejected by
the shared-potential product gate. Their existing High accuracy, Sample,
Scanning Image and Illuminating Image workflows remain in explicit legacy
execution. An inserted sample does not block a specimen-free optical ray
reference. The new Electron beam adapter has its own potential/material gate.
Direct Alignment, canonical conjugacy calibration and the DF geometry design
tool also require explicit legacy execution until their diagnostics are
qualified against the shared potentials. They do not fit or retune the new
model using an old equivalent Image map.

See [electric tails and native scanning](UNIFIED_DYNAMIC_EXECUTION.md) for
the energy and raw-phase convention, source-wavefront assumption, independent
error budgets and remaining material/acquisition boundaries. Numerical grid
refinement preserves the represented complex wave; it does not establish
convergence of the underlying continuous electron-source model.

An optical edit invalidates both representations. When an existing coherent
calculation is active, automatic optical edits wait for the latest particle
frame and a quiet edit interval before updating that wave. Particle-only use
does not silently create a coherent source. Numerical sampling remains separate.
The status bar shows current-step counts and completed workflow stages on
separate bars; total progress is not an elapsed-time estimate. Unknown work is
indeterminate, and only accepted publication completes a task.

## Particle calculations

**High-accuracy rays** and **Step (mm)** remain directly accessible in the
calculation toolbar. **Simulation > Particle: count and integration step**
focuses these controls. They use the existing emitter particle count and
instrument integration step. An explicit Tip quadrature still owns its complete
position, direction and energy population; its derived count is not replaced by
an unrelated requested number.

The step editor accepts scientific notation and preserves the complete stored
floating-point value through restoration and request capture. It does not
quantize a loaded step to a fixed number of decimal places. Existing resource
and numerical admission checks still apply. Explicit CDF tables also retain
their fixed population; editable Sobol counts are validated before changing the
source, including when a request is triggered before the editor loses focus.

For new GUI **High accuracy** requests, the executed maximum step is
`min(requested_step_mm, 0.05)`. The detached execution snapshot, memory estimate,
cache identity and manifest all use that applied value. The original requested
value remains visible and is recorded alongside the applied value in the result
and job log. Preview, Medium, explicit offline experiments and historical
restoration retain their own declared numerics. A step cap is an execution
policy; refine the step and source count to establish working-point convergence.

The shared-field particle executor retains complete history when it fits its
2 GiB history budget. Otherwise production requests retain existing integration
nodes at the requested display spacing, all scientific checkpoints, physical
faces and receiver-adjacent nodes. Every integration step and original particle
is still executed. If this retained history also exceeds the budget, the request
is rejected. Plot-node waist markers remain sampling dependent; use explicitly
executed observation planes for precise probe focus measurements.

**Run high-accuracy once** continues to calculate trajectories and retain the
incident beam used by Sample, Scanning Image and detector workflows. Numerical
settings do not silently change the particle seed, readout selection or sampling
model. Existing source and material settings remain separate controls.

Finite-volume material histories must not combine Ideal Optics plane actions
with continuous optical fields inside the specimen. The thin local route is
straight transport between separately owned material interactions. A nonzero
thin optical action intersecting a finite material envelope or its reference
matching path requires **Finite-Field Optics**: inverse action replay through
random scattering and backscattering is not implemented. In particular, an
excited objective at the specimen centre usually requires that finite-field
route. This is separate from particle count and does not disable the existing
high-accuracy finite-field specimen and detector workflows.

The overlap guard checks both each action's posed physical plane and its
global-Z execution plane against the specimen and physical-support bounding
boxes. A support opening does not establish disjointness when its enclosing
box intersects the plane. Zero excitations, empty material and zero-length
reference matching remain available. The auxiliary **Projected atoms (fast
approximation)** response performs angular redistribution at the executed
sample reference plane, followed by downstream thin transport without replaying
that plane's lens. That independent response is compatible with thin optics;
it is not an executed finite-volume collision history. A complete high-accuracy
request that also requests finite material histories must still pass the
material-overlap guard and may require Finite-Field Optics.

Thin first-order observers and Direct Alignment take their source coordinates
immediately after that plane's optical actions, matching an executed particle
checkpoint. Their source-to-source map is identity. Continuation from the
specimen therefore excludes its already executed objective action; a new
gun-exit source still enters the full column with its initial actions enabled.
Native millimetre action planes retain their exact reference values, including
when a segment is split at that plane.

## Coherent calculations

The **Coherent settings** toolbar button and **Simulation > Coherent: grid,
energy and memory** open the Electron beam page. Its **Coherent numerical
settings** group is visible without opening advanced options:

- Initial grid pixels and maximum refined grid pixels per axis.
- Energy samples for the selected coherent source.
- Column propagation step.
- Working memory per stage and the separate GPU working-memory limit.

These controls construct the existing captured `TipWaveRequest`; they do not
create another numerical state or translate particle count into wave cells.
The prescribed source width, emission energy distribution and coherence model
remain physical source inputs. The physical grid extent follows that source
and the existing support-admission rules; a larger numerical grid does not
silently shrink the electron source. Wave input is a complex field for each
coherent mode, with energy, phase reference and absolute weight. Classical
particle positions alone do not determine that phase.
Gun field/bore/energy stepping, material histories, segment length and cache
budgets remain in **Advanced calculation options**. Changing numerical settings
marks the existing wave result stale; it does not relabel its physical plane or
alter its stored complex field.

**Driven near-tip maximum step** independently controls axial slabs of the
driven Robin source solution, in nanometres. Its default remains 50 nm. A finer
source spatial grid retains larger transverse wave numbers and can require
smaller near-tip steps to resolve evanescent transmission. Refining the source
grid therefore does not replace axial-step convergence. The numerical solver
rejects unresolved slabs rather than deleting their evanescent channels.

For the unified column, **Advanced calculation options** also exposes the
kinetic expansion order (2, 4, 6 or 8), controlled support projection, its
relative complex-field L2 allowance, and its probability-loss fraction.
Budgets accept scientific notation without rounding small values to zero.
Zero disables numerical projection; physical apertures still act normally.
These limits apply to original mode weights and do not renormalize a clipped
beam. They bound recorded projections, not the whole propagation error.
Kinetic order, source/grid refinement and axial-step convergence are independent.
Compatibility modes leave these unified controls inactive while retaining
their selected values. Numerical changes make existing observations stale.
The residual exponential has a separate Taylor/Chebyshev selection and a
maximum Chebyshev group. Taylor remains the default. Chebyshev evaluates the
same Hermitian operator with a bounded polynomial, retains every original
substep check, and records resource-driven group reduction or Taylor fallback.
This numerical choice does not alter optical parameters, permit empirical
aberrations to be counted again, or establish whole-column convergence.

When the envelope spectrum fails its existing Nyquist guard, the complete
retained spectrum supplies a candidate refinement scale from cumulative
per-axis tail power. The original wave is then Fourier-embedded and the
operator is rerun with every admission check; no frequency is filtered out.
The candidate keeps the existing growth and sampling margins. It is a retry
suggestion, not evidence that frequencies missing from the source were resolved.

Spatial domain extension independently sizes each lattice axis for every
represented Fourier group's displacement plus the existing two edge cells.
It retains all executed input cells at their original coordinates, spacing and
phase carrier. Each candidate must pass the original full-band and output-edge
checks; a failed edge check grows the candidate by at least 1.25 before retry.
The receipt records attempted shapes and both physical padding margins. This
reduces unnecessary empty padding without changing the field, normalizing
intensity, relaxing a guard or establishing unknown exterior source tails.

The unified carrier can use the selected CUDA policy for its complete residual
exponential; geometry, canonical charts and field sampling remain on the CPU.
Only resource failures permit a complete retry from the same entrance under a
policy that allows CPU fallback. Explicit Require GPU, cancellation and physical
admission gates retain their meaning. Checkpoints remain host arrays.

Coherent propagation is unavailable in `ideal_thin`. The interface reports
**Select Finite-Field Optics** and leaves the chosen mode unchanged. Selecting a
finite-field model does not bypass existing source, field-domain, sampling,
memory or image-workflow admission checks. A coherent energy sample is a mode
contribution, not one classical particle.

## Field display

In Ideal Optics, Magnetic Field still shows the underlying parameterized
finite-field sources and labels this distinction in its status and model
details. Thin particle transport executes integrated plane actions, so the
display is not presented as its distributed propagation field. In finite-field
modes the existing propagation-provider display remains unchanged.
