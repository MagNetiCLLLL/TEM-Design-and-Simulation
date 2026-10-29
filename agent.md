# Project language, communication and execution policy

This policy applies throughout the repository.

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
- Accept and follow instructions written in Chinese.
- Keep project file content in English even when the corresponding discussion
  takes place in Chinese. Retain exact identifiers and paths when mentioning them
  in the conversation.
