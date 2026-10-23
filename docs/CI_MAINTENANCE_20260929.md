# CI Maintenance Record (2026-09-29)

## Requirements and baseline

Following the updated AGENTS.md, repair current GitHub Actions failures, synchronize test coverage and verify standalone installation. Preserve the classical Tip emission chain, stopping conditions, cache/archive semantics, the pause on coherent development and the CPU budget.

Before changes: branch `master`, HEAD `37b15119bfc9dad461a43fa3a29900b9f8ab635b`; AGENTS.md was the only user-modified uncommitted file, and the index was empty. That change is retained.

Actions run `36621810661` for this commit (push, run 20) was inspected:

- `cpu-acceptance`, job `109588803114`: wheel build and installation succeeded; installed smoke reported `Nonfinite x in installed optical branch 1`, and subsequent classical checks did not run.
- `feature-acceptance (electron-execution)`, job `109588803604`: 480 passed, 1 failed; compiled integration of the long on-axis trajectory took 9097 steps versus 9061 for reference integration, while the old test required equal step counts.
- The other five active feature scopes passed remotely for this commit; `full-scope-status` was skipped on push as designed.

Original GitHub logs, both failure artifacts and local reproduction evidence are retained in ignored `outputs/ci-repair-20260929-2153/`, not committed.

## Scope and validation plan

First inspect valid trajectory intervals in installation checks and criteria for physical equivalence of adaptive integration. Prefer minimal fixes; do not remove failures by relaxing tolerances, deleting tests or changing physics. Change runtime algorithms only with evidence of implementation error.

Affected entry points: installed smoke, compiled-trajectory tests, existing acceptance-scope manifests and CI installation commands. Previously added position-linkage and assembly-state tests must enter formal scopes. Retain existing acceptance tools and report formats.

Validation plan: run minimal failing cases first, then all seven active scopes after stabilizing changes, using dedicated Windows/Python 3.12, locked CPU dependencies and single-thread numerical budgets. Also build a fresh wheel and run installed smoke with `-I` outside the source tree in a new non-editable environment. Real GPU, native desktop interaction and coherent imaging are outside this round's validation.

## Root causes and fixes

### Installation checks misclassified unavailable coordinates after stopping

The original check required finite values throughout stored arrays without reading `alive / blocked_z / blocked_key`. Runtime retains downstream-branch populations and upstream stop records, so unavailable coordinates after stopping may be NaN. Actual reproduction with the original smoke's 9 particles, CPU and 5 mm step/history sampling found 1790 nonfinite samples per downstream phase-space coordinate, but 0 before or at stopping.

The check now requires finite coordinates before and at stopping, and validates increasing axial coordinates, array shapes, consistent survival/stop markers, stop positions and component identities. Infinity is still always rejected. Regressions cover valid upstream/downstream stops, anomalies before/exactly at stops, missing stop information, future stop positions and malformed shapes. Transport and aperture physics are unchanged.

### Adaptive step count is not a criterion for cross-backend physical equivalence

The remote report showed 9097 versus 9061 steps. The original test passed locally in both the development environment and a new locked environment; the specific remote step-count difference was not reproduced. Local passes therefore do not prove the remote issue has disappeared. When adaptive error estimates approach floating-point rounding, accepted time grids may differ. Physical states must be checked rather than requiring index-by-index array correspondence.

This on-axis case without transverse excitation has an independent reference: transverse magnetic-force components vanish, and transverse position/momentum remain zero; electrostatic energy conservation gives `K(z)/e = K0/e + phi(z) - phi(z0)`. Within each linear-interpolation interval of the same frozen potential, let `k=K/(mc²)`, `u=sqrt(k(k+2))`; the independent clock integral is:

```text
Δt = Δz/c × (2 + k_left + k_right) / (u_left + u_right)
```

This expression does not call the particle pusher and also applies to constant potential. New checks compare the axis, path length, energy, relativistic momentum and independent clock at each backend's own sample points, then compare clock errors at identical physical sections. Original endpoint and cross-backend time tolerances remain: position `rtol=2e-7, atol=2e-12 m`; momentum `rtol=2e-7, atol=1e-29 kg m/s`; time `rtol=2e-10, atol=1e-17 s`; path length `rtol=2e-10, atol=2e-12 m`. Other off-axis, backward-emission and hardware-stop cases retain their strict comparisons.

An additional absolute independent-clock error limit of `1e-16 s` (0.1 fs) does not relax those backend comparisons. Over the 3.0264 m on-axis reference/compiled runs, maximum clock errors were both approximately `7.112e-17 s`. Tightening relative integration tolerance from `1e-4` to `1e-6` increased compiled steps from 9061 to 9203 and reduced maximum error to `3.642e-17 s`. Endpoint error is affected by cancellation and is not monotonic; no global-convergence claim is based on it. A new half-step case actually produces different sample counts while passing original-accuracy comparisons and independent-clock checks. This reference checks propagation in a frozen field, not experimental accuracy of the entire microscope field model.

### Acceptance coverage and CI installation workflow

- `classical` adds shared-component position and persistence tests.
- `electron-execution` adds tests that real emission and virtual electrons share the production column.
- `field-ui` adds assembly selection, navigation, configuration and working-point restoration tests.
- Acceptance policy still requires complete file inventories and execution receipts; no new skip, xfail or filter exclusions.
- CI builds in a fresh directory and requires a unique `tem_design_simulation-*.whl`, without hardcoding the versioned filename. It checks package location/configuration resources after installation and runs with `-I` outside the source tree in a fresh non-editable environment.
- Every installation command checks its exit code; failure remains red. Independent classical checks still run and upload results if installed smoke fails. Responsibilities of push / pull_request, the matrix and manually paused-scope reports are unchanged.

## Actual validation

Environment: Windows, Python 3.12.3, `requirements/validation-cpu-lock.txt`; NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0, PySide6 6.8.3. Dedicated validation and fresh non-editable wheel environments both passed `pip check`; the user's original environment was not modified. Remote Python is 3.12.10, retained as a platform limitation. Numerical pools were limited to 1 thread, with at most three bounded validation jobs concurrently, below half of the process's 32 available logical CPUs.

Formal scope command: dedicated interpreter running `scripts/validate_classical_scope.py --scope <scope> --output outputs/ci-repair-20260929-2153/scopes/<scope> --timeout-seconds 1800`. Local classical used an 1800-second limit; CI's original 900-second limit was unchanged. These budgets must not be described as identical.

| Validation | Status | Passed cases | Runner wall time (seconds) |
|---|---|---:|---:|
| acceptance-policy | PASS | 99 | 7.20 |
| performance-observation | PASS | 88 | 20.10 |
| classical | PASS | 295 | 304.37 |
| gun-fields | PASS | 121 | 118.31 |
| particle-continuation | PASS | 126 | 320.57 |
| field-ui | PASS | 466 | 203.66 |
| electron-execution | PASS | 488 | 111.19 |

Total: 1683 cases, still 1683 after node-ID deduplication, 0 failed, 0 skipped; every pytest/runner exit code was 0. Receipt setup, call and teardown were checked individually and all passed; source-hash inventories in all seven reports match final source. SHA-256 of normalized manifest JSON is `f1cc6241122d4c75930a1fc82fdf23fd8b6f5746eda9ecf94410faa4fd7d43c0`. Local `completion-summary.json` only checks existing receipts; it neither runs nor substitutes for acceptance.

Standalone wheel smoke: PASS, exit code 0. Actual 9-particle CPU optics included Tip, extraction, acceleration, focusing, apertures and column, with gun exit at 450 mm, specimen entrance at 1599.2 mm and recorded column grid to 3026.4 mm. The branches had 5 / 9 stopped particles respectively (the endpoint is the calculation-grid extent, not a claim that all particles reach it). Independent uniform-magnetic-field trajectories, exact save/load of 8 array groups, and offscreen Hardware tuning/Virtual electrons checks passed. Installed smoke took approximately 31.18 seconds, including 18.34 seconds for optics; these record this run only and are not performance-optimization claims.

The report is `outputs/ci-repair-20260929-2153/installed-smoke/report.json`. Formal scopes retain `report.json`, logs with run IDs, JUnit, receipts, source hashes and environment information in their own directories. Generated caches, full logs, arrays, environments and diagnostic archives are Git-ignored.

Real GPU, native desktop/OpenGL interaction and long coherent imaging: NOT_RUN. Passing software scopes does not elevate full physical qualification; status remains `UNQUALIFIED / NOT_ESTABLISHED`.

Nonfatal notices remain in logs: archive tests using `record_property` trigger pytest's `xunit2` incompatibility notice; some Qt test processes report pyqtgraph failure to disconnect `destroyed()` on exit. All cases and receipts still pass. These notices were not hidden, nor was scope expanded to third-party UI lifecycle or JUnit-format changes. They are not the two confirmed repaired CI failures.

## Delivery status

All seven complete active scopes and standalone installation checks passed, as did syntax checks for four multiline PowerShell workflow steps and `git diff --check`. Validation is based on the dirty workspace and explicit hashes above, not claimed as the original commit baseline.

These changes are uncommitted and unpushed; GitHub has not run them, so remote green status cannot be claimed. The user's existing AGENTS.md edit is untouched and the index remains empty. Runtime integrators, field calculations, configuration, cache and archive formats are unchanged. Scope did not expand into repository-wide cleanup, and acceptance rules were not modified to revoke physical requirements.
