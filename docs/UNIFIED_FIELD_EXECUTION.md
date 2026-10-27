# Shared-field execution and qualification boundaries

TEMForge keeps an immutable physical description separate from the numerical
method that consumes it. The existing production calculations retain their
saved optical settings and source-admission rules. The `unified` Finite-Field
Optics GUI entry uses shared potential-based column executors, while historical
keys retain explicit legacy execution. Executors report their qualified scope; adding an adapter does
not silently promote an entire microscope column.

## Shared authority

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| Physical scene | `physics/physical_scene.py` | Hardware and logical-channel identities, registration, materials, geometry and parameter dependencies. |
| Effect ownership | `physics/interaction_registry.py`, `physics/plan_contract.py` | Fields, reductions, empirical residuals, material events and boundaries have explicit owners. Reject overlapping replacements and duplicate actions. |
| Retained potentials | `physics/potential_snapshot.py`, `physics/potential_capture.py` | Immutable contributions, common gauge, actual source selection, domains and numerical identities. E and B come from the retained potential representation. |
| Field display | `potential_field_vector_scene.py` | An explicit prepared scene sends the same retained E/B sources to the existing Magnetic Field page, without another solve or the legacy independent B interpolant. |
| Potential rays | `physics/potential_ray.py` | Forward relativistic Lorentz trajectories with actual energy, static electric energy balance, raw action and declared stop losses. |
| Potential waves | `physics/potential_wave.py`, `physics/potential_wave_carrier.py` | Declared spinless forward electromagnetic operators, full-potential overlap terms, phase reference, absolute weight and independent numerical budgets. |
| Native dynamic execution | `physics/dynamic_potential.py`, `physics/retarded_wave.py`, `physics/unified_scan_calibration.py` | Captured scan laws, induction, actual flight-time references, bounded coherent reference-wavefront approximation and measured finite-coil calibration. |
| Observation | `physics/potential_wave_observation.py` | Exact requested planes, explicitly declared apertures, executed checkpoints and existing specimen/readout interfaces. |
| Diagnostics | `physics/potential_aberrations.py` | Read-only canonical transfer and scalar-gradient fits, independent holdout rays and energy scans. |
| Optional reduction/design | `physics/qualified_field_reduction.py`, `physics/bounded_optical_design.py` | Coefficient-proven quadratic interval maps and bounded private-state optimization; no automatic application to live settings. |

Shared Descan and Image/Diffraction channels retain one physical host. A joint
nonlinear magnetic circuit is one solved source, not a separate field to add
again for each drive. The same rule applies to diagnostic coefficients: Cs or
Cc measured from a field is an output, not a second kick to apply to that field.

Unknown field regions remain unavailable. A B-only import cannot acquire a
coherent phase merely by being wrapped in a potential interface. The explicit
Gaussian all-Z continuation evaluates the declared analytical formula; it
does not authorize zero-filling an unknown FEM exterior.

The explicit potential display adapter takes a display box and the executed
potential snapshot. The box does not enlarge source validity. Magnetic and
electric validity are retained separately; electron force is available only
where both selected fields are known. Joint sources are selectable atomically,
and a repeated selected key does not duplicate its contribution. The existing
Magnetic Field worker accepts that prepared vector scene only with its matching
captured instrument state. Ordinary production results continue to display
their own original providers; the new adapter does not silently relabel them.

## Model order and numerical resolution

The three user-facing model categories are described in
[Calculation models and numerical controls](CALCULATION_MODES_AND_NUMERICS.md).
Historical `ideal` files display **Parameterized fields (legacy)** while retaining
their original behavior. **Ideal Optics** uses the distinct `ideal_thin` key.
Coupled Multiphysics remains a disabled future category.

The thin executor uses exact drifts and integrated plane actions. It is a fast
approximation, not a qualified reduction of every finite-field preset. Strong
lenses can change focus or transmission when represented as thin elements;
TEMForge does not retune currents, recenter the beam or normalize away losses
to hide that difference. Its source, specimen and detector models retain
their separate ownership.

Potential-based natural-aberration benchmarks are also distinct from the
historical parameterized finite-field route. The latter retains its configured
empirical aberrations and explicitly selected equivalent-image reduction.
Only a qualified field interval may replace its own reduced actions. No global
Cs/Cc deletion or saved-profile migration is performed. Fresh windows select
`unified`; its field-owned interval does not execute the legacy residuals or
equivalent Image maps a second time. See the mode guide for runtime support
gates and the actual-source handoff.

Particle count, integration step and solver tolerances are numerical inputs.
Coherent modes instead require a complex input field, energy and phase
reference, physical grid, axial discretization and memory/error budgets.
The two representations preserve their original weights. Incoherent modes
sum intensities and do not define a single combined phase plate.

## Independent evidence

- Seven physical column configurations remain in ownership and thin-event
  regression coverage. Legacy output comparisons use unchanged saved settings.
- Native and manufactured potentials have axis, pose, gauge, field consistency
  and domain checks. Resolved FEM and high-order analytical round lenses have
  separate field-order, mesh, outer-boundary and ray-step comparisons; see
  [Field-derived round lenses](NATURAL_ROUND_LENS_BENCHMARK.md).
- Fixed-grid and carrier wave tests compare complete complex fields, not
  recentered intensities. They retain scalar phase, tilt, defocus and absolute
  flux; actual CPU/CUDA parity is tested where that executor supports CUDA.
  See [Potential wave operators](POTENTIAL_WAVE_BENCHMARK.md).
- The optional [strict native sampler](NATIVE_POTENTIAL_SAMPLING.md) changes
  evaluation overhead only. Its parity checks do not replace whole-trajectory
  convergence or symplectic checks.
- [Finite mechanical boundaries](POTENTIAL_GEOMETRY_BOUNDARIES.md) retain the
  authoritative vacuum bores, aperture plate thicknesses and rigid placement,
  including zero-excitation obstructions. Their absorbing-boundary model has
  independent step, window and opening-sampling checks.
- [Finite material handoff](POTENTIAL_MATERIAL_HANDOFF.md) executes the existing
  atomic potentials and conditional energy channels with the retained static
  electromagnetic field between material half-steps. Each outgoing energy is then
  propagated through the same fields and boundaries to detector readout.
- The C3/Probe pilot separates a small executed FEG particle population from
  controlled coherent entrances and canonical pupil families. Local raw-phase
  response is not evidence of complete physical-source propagation.

Canonical pupil diagnostics use `P_perp/p_ref = (p*u_perp - e*A_perp)/p_ref`
at both reference planes. Their energy scan holds the canonical input family,
gauge, field and reference momentum fixed. The reported `Cc_response` is
therefore explicitly distinguished from conventional field-free,
fixed-geometrical-angle Cc. Raw chief motion and C1 remain in the report.
Symplectic defects, Lagrangian consistency, held-out gradient errors and
energy-step convergence are separate gates; no matrix is projected onto a
symplectic one to force acceptance.

## Remaining admission boundaries

The carrier operator includes the captured static scalar potential and native
axial time-dependent coils, with separate forward-branch and temporal error
budgets. Its source-exit reference-wavefront assumption does not establish the
gun's transverse arrival-time distribution. Spin, back reflection, full timed
pulse wavepackets, dynamic material scattering and curved-coordinate coherent
filter transport remain unsupported. Existing gun, specimen and receiver
executors retain their explicit capabilities. Other temporal field bases and
curved coherent-filter extensions require their own operators and evidence.
See [electric tails and native scanning](UNIFIED_DYNAMIC_EXECUTION.md).

The low-level observation bridge owns explicitly listed thin aperture events.
Its finite-geometry wrapper instead owns captured vacuum bores and physical
aperture plates; the two policies cannot be applied to the same plate. Neither
route infers arbitrary CAD surfaces or unrepresented holder material. An
occupied numerical grid edge, unresolved spectrum, insufficient kinetic
expansion budget or unknown field region rejects the calculation. Optional
controlled support projection has a separate explicit loss/error ledger; it
does not replace physical absorption or repair intensity. Controlled-pilot
evidence and failed admission probes must both remain visible when deciding
whether a future source-to-observation route is ready.

New numerical tests are registered in the existing acceptance scopes. Scope
membership means they run in regression; it is not a claim of experimental
microscope validation or unrestricted production admission.

The [implementation and validation map](UNIFIED_ARCHITECTURE_VALIDATION.md)
connects proposal stages P0-P6 and gates G0-G10 to the relevant regression
families, long-running acceptance scopes and remaining capability limits.
