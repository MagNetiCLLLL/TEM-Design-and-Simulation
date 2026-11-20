# Project language, communication and execution policy

This policy applies throughout the repository.

## Git workflow

Work directly on `master` for all project changes. Create or switch to a different
branch only at the user's explicit request.

## Optical tuning contract

- Follow [optical control roles and illumination constraints](docs/OPTICAL_CONTROL_ROLES.md)
  when changing illumination controls or fitting operating presets.
- Preserve the microscope meaning of C1 spot size; do not replace it with a
  literal specimen-plane spot radius. Keep C3 Normal/Large angle selection
  distinct from Mini Condenser Nanoprobe/Microprobe selection.
- A tuned Nanoprobe working point must satisfy the selected C2-C3 and C3-ADL
  crossover requirements as well as convergence, focus and current criteria.
  Stored reference values or tests of another field model are not qualification.
- For Finite-Field Nanoprobe tuning, Mini Condenser and Upper Objective must
  focus directly onto the specimen upper surface. Reject a distinct additional
  pre-specimen crossover; a small final probe does not make that branch valid.
  Keep C2/C3 as the primary convergence-angle controls and distinguish the
  final focus tolerance from an extra upstream waist.
- Maintenance fixes must not silently retune lenses or refocus deliberate
  perturbations. Keep physical interception and absolute particle weights.

## Scanning model policy

- New Scanning Image work uses particle scanning or mixed particle-wave
  scanning. Do not add a separate pure coherent scan or revive coherent
  propagation from the physical Tip.
- Mixed scanning transports the actual finite particle source through the
  selected column fields to the specimen entrance, then introduces the
  declared near-specimen coherence model and propagates its wave components
  through the specimen. Preserve particle identities, weights, energies,
  clipping and captured optical inputs at that handoff.
- Coherent wave operators remain part of the mixed calculation; they do not
  make it a pure coherent source model. This distinction does not waive CUDA
  requirements, field-model admission or numerical qualification.

## Fixed physical winding counts

- Treat photo-constrained, measured or explicitly assumed winding turn counts as
  fixed physical inputs unless the user explicitly authorizes a bounded model
  adjustment. Preserve the original reference and distinguish a changed model
  count from measured regions or symmetry assumptions. The current nominal-area
  adjustment permits at most 20% change from its saved original counts.
- In ideal prescribed-current coil calibration, change current or its electrical
  reference to match a chosen field-strength condition. Do not increase turns,
  enlarge the winding envelope or invent a wire diameter to compensate for a
  mismatch. Report field-shape and focus differences separately.
- A split winding's explicit per-interval turn counts remain fixed when its
  geometry changes. Unknown wire size may retain envelope CAD; it must not
  prevent a known turn count from defining an ideal electrical source.
- Winding CAD accepts conductor metal cross-sectional area as the user-facing
  size input; wire diameter is a derived read-only value. Preserve legacy
  diameter records when loading old inputs.
- On an explicit turn-count or conductor-area edit, repack the conductors and
  fill the remaining allocated winding volume with insulation. Keep the winding
  ID, OD and axial envelope fixed; reject an impossible layout rather than
  shrinking conductors, dropping turns or enlarging the envelope.

## Internal signed currents in ideal coils

- Internal forward/reverse current groups apply only to Ideal prescribed-current
  coils in vacuum (`unified` with `round_lens_field_model = "copper_coil"`).
  Do not add resistance, constant-power, heating or magnetic-material constraints
  to this calculation. Other field models retain their existing behavior.
- Use the current default winding geometry as the baseline. Preserve its inner
  and outer diameters, axial intervals, physical turns, conductor area and poses.
  Partition its existing current-bearing cross-section instead of moving,
  shrinking or adding windings. Earlier resized candidates remain historical
  studies, not the baseline for this restricted calculation.
- A group must represent complete closed turns or closed subwindings. Do not
  assign different steady currents to successive pieces of one continuous wire.
  Keep physical turn counts separate from numerical integration nodes, and
  retain reverse turns in the physical inventory even if signed NI cancels.
- Preserve the source distribution, beam boundaries and solver tolerances.
  Check actual specimen-surface probe focus after field fitting; matching an
  axial field alone does not qualify focus or aberration correction.

## Minimum sampling for new physical trials

- As requested on 2026-10-29, every new finite-source physical calculation,
  including quick tests, sensitivity trials and tuning prescreens, must use
  at least 1000 original positive-current source particles. Use 1024 as the
  usual starting count when compatible with the selected source model.
- Check the actual emitted population, not just a command-line default. Do not
  count zero-current diagnostic rays, duplicate particles or artificial weight
  changes toward this minimum. Preserve the physical source law.
- When continuing a low-count candidate, explicitly prepare and record a new
  numerical input snapshot. Do not silently change historical inputs/results.
  Explicit source tables and structured sampling must remain internally valid.
- Report the emitted, surviving and effective particle counts separately.
  Physical aperture losses remain valid; increase source sampling if too few
  particles reach the observation plane. Do not rescale or restore lost current.
- The minimum is not proof of convergence. Compare increased source counts
  before accepting probe size, convergence angle, focus or current stability.
- Small chief-ray response calculations and pure-array/mocked software tests
  may remain small because they do not measure a finite source. They must not
  replace the required finite-source physical tests or qualify a working point.

## Nested source growth and reuse

- For fixed emission parameters, increasing the original particle count must
  append particles without changing existing particle IDs, initial positions,
  directions, energies, emission times or source boundaries. Particle count is
  a numerical sampling input, not a change to the physical source distribution.
- Generate added source samples by interpolation of a fixed, monotone source
  distribution or inverse-CDF representation. Its calibration and boundaries
  must depend only on the declared emission parameters and sampling schema,
  never on the requested batch size. Do not recenter, rescale or refit existing
  particle energies when adding samples.
- Reuse compatible completed low-count particle transport when increasing the
  count. Validate the source prefix, physical fields, boundaries, solver and
  other transport dependencies explicitly. Preserve the old trajectories,
  clocks, energies, clipping and provenance; propagate the added particles.
  Source interpolation must not substitute for transporting new particles.
- Preserve earlier raw results. Publish the larger population with explicit
  source-current normalization and an unchanged physical loss ledger; do not
  restore intercepted current or silently rewrite historical particle weights.
- Version source-sampling changes. Historical non-nested sampling must remain
  identifiable and reproducible; do not relabel it as the new nested source or
  reuse incompatible results after a source, field or boundary change.

## Temporary CUDA requirement for coherent calculations

- Until the user explicitly lifts this restriction, new coherent wave
  calculations require a usable CUDA device, driver and CuPy execution backend.
  An installed CUDA package alone is not evidence that execution is available.
- Enforce this requirement at public calculation entry points before expensive
  preparation or propagation, including GUI, scan and command-line workflows.
  Reject unavailable CUDA and explicit CPU execution clearly; do not fall back
  to CPU after a CUDA failure.
- Do not run coherent pipelines, long coherent benchmarks or broad coherent
  regression suites on this workstation without verified usable CUDA. Prefer
  bounded, targeted checks. Small CPU numerical-operator tests and mocked
  execution-policy tests remain allowed; they are not full coherent runs.
- Classical particle calculations may still use CPU. Editing inputs, inspecting
  settings and reading already completed results do not require CUDA.
- Preserve numerical formulas, physical inputs and existing results when
  applying this execution restriction. Report actual validation limits.

## Project files

- Write all project-authored content in English. This includes documentation,
  README files, code comments and docstrings, interface text, configuration
  descriptions, test descriptions and fixtures, and maintained reports.
- Use English for new file names and descriptive identifiers. Preserve existing
  package names, public interfaces, file formats and compatibility contracts.
- Translate Chinese instructions into English when implementing them in project
  files. Do not copy Chinese chat text into project documentation or comments.
- Scientific symbols, mathematical notation, units and language-neutral Unicode
  test data are allowed. Keep Unicode robustness coverage when translating test
  fixtures; English does not require ASCII-only content.
- Preserve original external reference material, third-party notices, raw data
  and user-supplied data. Write accompanying project-authored explanations in
  English; do not alter original evidence to enforce a language preference.
- Language-only changes must preserve numerical values, formulas, optical
  parameters, program behavior, citations and the meaning of historical results.

## User communication

- Communicate with this user in Chinese in the conversation, including progress
  updates, clarification questions, explanations and final responses.
- Identify condenser lenses explicitly as `C1 lens`, `C2 lens` and `C3 lens`
  in user communication. Identify apertures as `C1 aperture`, `C2 aperture`
  and `C3 aperture`. Do not use bare `C1`, `C2` or `C3` ambiguously for these
  different components. Preserve existing machine-readable component keys.
- Accept and follow instructions written in Chinese.
- Keep project file content in English even when the corresponding discussion
  takes place in Chinese. Retain exact identifiers and paths when mentioning them
  in the conversation.
