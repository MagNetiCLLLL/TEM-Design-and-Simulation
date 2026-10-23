# Code cleanup verification record

Date: 2026-09-29. Review baseline: `8c522d03a6f7f1398daa629be476e1a001ed8039`.
The workspace was clean when inspection began; the external audit's fixed commit matched local HEAD. This document records workspace changes on that baseline and does not imply that they were committed or pushed.

## Verification and treatment

| External candidate | Local verification | Action |
| --- | --- | --- |
| A1 | `state.py` does not use `restore_recording_system` and still imports `ensure_recording_system` | Removed the unused import; retained JSON read/write |
| B1/B2 | Position setting, initialization, installation switching and anchor resolution in the two diffraction components have identical ASTs and matching referenced constant values | Extracted `InstallationReferencedPosition`, retaining each component's field model, geometry type and validation |
| B3 | Position setting, initialization and reference-plane application in five tip-referenced components have identical ASTs | Extracted `TipReferencedPosition` without adding dataclass fields |
| B4 | Five round-lens helper methods are identical; magnetic-field normalization, validation and field sets do differ | Shared the helper methods while retaining the distinct magnetic-field implementations, validation and fields |
| B5 | Hidden controls are still accessed by main-window working-point restoration and some tests | Migrated all consumers to catalog-validated `AssemblySelection`; removed three hidden controls and signal-blocking adapters |
| C1 | Searches of source, tests, scripts and text references found only method/signal definitions and main-window connections, with no trigger entry point | Removed the old request chain as a group; retained `load_assembly` and Configure instrument |
| C2/C3 | Repository runtime code does not use the old entry points, but `state.py` does provide JSON read/write and `builder.py` provides a public import path | Kept these entry points in this round; absence of repository consumers was not mistaken for absence of external use |

Inspection searched not only names such as `assembly_panel.gun`, but also panel aliases and control usage in tests. Tests now check actual assembly selection and configuration operations instead of requiring hidden controls.

## Equivalence and verification

- Captured 35 scenarios across seven component types before and after the change: initial state, mechanical movement, optical reference-plane editing, direct Z editing, deactivation, installation switching and anchor resolution. Magnetic/multipole field arrays, layout readouts, focal lengths and field sets were captured as well.
- The default-state serialization summary, the above samples and column-transfer diagnostics from the specimen plane to downstream planes were byte-for-byte identical in their combined output. Both before/after JSON SHA-256 values are `0b5db563dd0bbb63b25a8d9354ce2e76f9436ba2041c68f7dc7c212e2c2ed186`.
- Detailed local data are under `outputs/removal-audit/`; these are generated verification artifacts and do not enter Git.
- 169 change-related regressions passed: position contracts, assembly state, working-point restoration and rollback, configuration apply/cancel, component persistence, navigation, assembly identity, archives and the current profile format.
- All application Python files passed syntax parsing, and `git diff --check` passed.
- The wheel build passed; the new position module, round-lens module and assembly panel were byte-for-byte identical between the package and workspace. Generated files remain in the local output directory above; installation in an independent environment was not repeated in this round.
- Magnetic-field, virtual-electron and hardware-tuning UI scope (`field-ui`): 433 passed, no failures or skips, 120.18 seconds; acceptance status PASS.
- Classical particle scope (`classical`): 272 passed, no failures or skips, 274.18 seconds; acceptance status PASS.
- Section continuation, material and EDS reuse, exact archives and result-file UI scope (`particle-continuation`): 126 passed, no failures or skips, 314.83 seconds; acceptance status PASS.

All three scopes used this project's `.venv`, Qt offscreen and 1 thread per numerical process; the source fingerprint remained unchanged during execution. The scopes overlap with the preceding targeted regressions, so these counts are not simply added as a total of distinct tests. The acceptance script retains `full_simulator_qualification=UNQUALIFIED`: this is a pass for bounded software scopes, not full-instrument physical qualification.

Local acceptance receipts are at `outputs/removal-audit/<scope>/report.json`, with the following run IDs:

| Scope | Run ID |
| --- | --- |
| classical | `6302519eeb8b4c90af8d27f7b5a8a464` |
| field-ui | `d564d9f205274a2281a4d6066e7bdab9` |
| particle-continuation | `c7da2fe23f74438d8bb2ca42e8957338` |

Runs used `scripts/validate_classical_scope.py --scope <scope> --output outputs/removal-audit/<scope> --timeout-seconds 900`. Logs still contain existing dependency Pydantic deprecation notices and Qt/ViewBox signal-disconnection warnings on exit; these warnings were not counted as functional test failures, and third-party libraries were not changed in this round.

JSON read/write checks compare using JSON array semantics: a Python tuple restored as a list from JSON does not imply a numerical or orientation change. When the mode catalog offers no preset for an assembly, the UI should clear the inapplicable preset and disable the apply button.

### Pre-existing test issues

The first classical-scope acceptance run found 7 failures. All 7 were reproduced in a separate test process loading the original application modules from the pre-change HEAD, so they were not introduced by this deduplication. Fixes were limited to tests:

- Rejection tests for retired specimen modes now match the current error message and still require an error to be raised.
- Two specimen-exit cache-routing tests explicitly insert the specimen; the current default is uninserted vacuum.
- The TEM-wave memory-rejection test explicitly imports a test CIF to reach the memory-estimation branch under test. This checks pre-execution rejection only and does not start wave computation.
- GPU fallback tests previously intercepted only the magnetic-field kernel, whereas the actual default gun-exit path already traverses an electric-field kernel. They now separately verify electric-field and purely magnetic branches, simulate GPU resource errors, invalid inputs and forced-GPU policy, and check actual CPU retries; these simulated-fault tests are not claimed as hardware-GPU numerical-equivalence tests.

All 35 updated calculation-control and backend-policy tests passed. No production source, specimen or resource restrictions were relaxed to make tests pass.

## Scope

This round did not change the gun, magnetic fields, integrator, specimen scattering, detector calculations or cache identity rules, and did not resume paused coherent-wave development. Numerical comparisons are component and column-transfer regressions within a defined scope, not claims of complete microscope physical qualification. UI evidence comes from offscreen automation; no claim is made of manually operating every control in the user's running desktop window. Not all external scripts on the user's computer were scanned, and the two public entry points noted above were retained.
