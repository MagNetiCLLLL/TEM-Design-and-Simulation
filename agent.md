# Project language, communication and execution policy

This policy applies throughout the repository.

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
