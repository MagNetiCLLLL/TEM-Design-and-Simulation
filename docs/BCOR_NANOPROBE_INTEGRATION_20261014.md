# B-COR and Nanoprobe integration — 14 October 2026

This integration retains both the local B-COR commit `df7f676` and the remote
Normal/Large-angle Nanoprobe commit `ce2fe15`, including its predecessor
`141fa25`. The common ancestor is `fcdbb3a`. Both original commits remain in
the merged history.

## Combined behavior

Select **C3 + Probe Corrector + B-COR** to use the new image corrector with
either Nanoprobe range. The assembly panel offers **Apply illumination preset**.
This applies the incident-column controls while retaining the B-COR settings,
projection mode, and D/I/P1/P2 drives. The same transaction works through the
background preset worker when the nanopulser is installed. A failure rolls back
the controls; cancelled background work is not adopted.

The range remains independent of the image-corrector design:

| Range | C2–C3 interval | C3–ADL interval |
| --- | --- | --- |
| Normal | At least one executed two-dimensional waist | No additional converging-to-diverging waist |
| Large angle | At least one executed two-dimensional waist | At least one executed two-dimensional waist |

The original topology gate, finite checkpoint requirements, independent specimen
focus check, and alignment transaction checks are retained. Selecting a range
does not itself certify an edited optical state.

**C3 + B-COR** without the probe corrector remains a valid B-COR assembly, but
does not acquire the probe-corrected Nanoprobe calibration. No new B-COR
projector calibration or Microprobe preset qualification is claimed.

## Configuration reconciliation

The B-COR combination had copied the older CETCOR combination's AC scan-coil
location. Commit `0b346eb` had already moved the reference probe column's AC
pair to local Z 950 mm for its calibrated incident optics. The B-COR combination
now uses that same centre, 930–970 mm envelope, and 937.5/962.5 mm interaction
centres. The metadata describes the independent coil placement correctly.

The combined column also inherits the reference CM startup magnitude
43.88646617222637%. Nanoprobe mode switching preserves the live magnitude and
selects its negative sign. Existing custom saved controls are not replaced by
these defaults when a state/profile is restored.

Regression checks compare all incident parts through the specimen against
`C3_ProbeCorrector.toml`, excluding display text and list ordering. The B-COR
extension remains downstream of the Objective; its extra relay and third
hexapole are retained. The two textual merge conflicts preserve both the
independent corrector-coil navigation groups and B-COR component identities, and
both State constructor validation blocks.

## Persistence repairs

- A temporarily uninstalled B-COR no longer overwrites its saved enable
  preferences with the dormant `enabled=False` values during State restoration.
- Complete profiles now retain each initialized image-corrector component's
  enable preference as strict boolean metadata. Physically uninstalled
  components stay disabled; reinstallation restores the user's individual
  choices and saved drives. Active device switches remain authoritative.
- Profiles written after restoring a B-COR state without a probe corrector no
  longer include manifest-owned deflector geometry as editable controls. Earlier
  erroneous geometry readouts are ignored on profile input while the valid
  operating controls are retained.

The regression matrix includes both B-COR combinations, both Nanoprobe ranges,
opposite per-component enable patterns, State/profile/snapshot round trips,
schema-79 State migration, and profiles predating the range field. State/profile
migration preserves optical controls rather than reapplying presets.

Older profiles that never recorded dormant enable preferences cannot recover
that absent information. Old working-point snapshot graphs also retain the
project's explicit current-model-only contract: schema-79 snapshots are not
silently migrated to schema 80. The original code and rebase state were backed
up before integration; this change does not overwrite historical snapshots.

## Executed incident-column validation

Both branches were executed on the actual **FEG / C3 + Probe Corrector + B-COR /
No Energy Filter** assembly, at 300 kV with 3,000 emitted classical particles,
using CPU/Numba CPU and integration steps 0.05 and 0.025 mm. Both steps passed
for both ranges. Static verification disables raster/wobble.

Fine-step results:

| Range | Executed C2–C3 waist Z | Executed C3–ADL waist Z | Alpha95 | Local waist offset at specimen |
| --- | --- | --- | --- | --- |
| Normal | 672.0 mm | None | 29.99998768 mrad | +0.33520461 nm |
| Large angle | 668.0 mm | 1017.52643678 mm | 30.00673044 mrad | −0.02884026 nm |

Waist Z values are executed checkpoint locations at bounded axial sampling,
not exact continuous roots. The angle target was 30 mrad with 1% tolerance;
the independent focus tolerance was 1 nm. This is fixed-particle-budget,
two-step classical evidence, not particle-sampling convergence, wave resolution,
experimental calibration, or OEM qualification.

Detailed reports:

- [Normal](development/bcor_nanoprobe_normal_20261014.json)
- [Large angle](development/bcor_nanoprobe_large_angle_20261014.json)

The separate CUDA B-COR position-relay diagnostic passed for both B-COR
assemblies with implementation and input files unchanged. Its acceptance is
`PASS_POSITION_RELAYS_ONLY`; nonlinear and off-axis correction performance is
not established by that check.

## Regression and GUI evidence

The focused integration set covers 38 test files and 831 distinct cases. The
initial run finished with 830 passing and one stale assertion: the assembly
catalogue test still expected 60 combinations, while the two added B-COR columns
increase the complete catalogue to 84. That assertion now checks the complete
Cartesian product, uniqueness, and explicit coverage of both B-COR columns.
The entire affected test file was rerun: 36 passed. The resulting latest evidence
is **831 passed, zero failed, zero skipped**; production source did not change
between the broad run and this focused rerun. This is an integration regression
set, not a claim that the whole repository test suite was executed.

The native Windows Qt run independently passed 14 GUI cases, including actual
mode-button activation on the combined B-COR column. These cases overlap the
831 cases and are not added to that unique count. The broad run also exercised
real CUDA B-COR ray and third-hexapole wave comparisons. Non-failing warnings
concerned small CUDA grids and a pyqtgraph shutdown signal disconnection.

The local receipts are `integration-tests.xml`, `assembly-coverage-tests.xml`,
`native-qt-tests.xml`, and `validation-summary.json` under
`outputs/safe-integration/`. Source hashes bind the verification to the reviewed
Python and TOML files. The original 41-file working state and paused rebase
metadata were archived separately alongside a verified complete Git bundle
before any integration edits.

## Reproduction

Use the project Python environment with `PYTHONPATH=src` and the project root as
the working directory. The independent condenser CLI supports the combined
column explicitly:

```text
python scripts/validate_nanoprobe_condenser.py --column "C3 + Probe Corrector + B-COR" --range normal --rays 3000 --step 0.05 --threads 2 --output outputs/safe-integration/bcor-nanoprobe-normal.json
python scripts/validate_nanoprobe_condenser.py --column "C3 + Probe Corrector + B-COR" --range large_angle --rays 3000 --step 0.05 --threads 2 --output outputs/safe-integration/bcor-nanoprobe-large_angle.json
python scripts/validate_bcor.py --output outputs/safe-integration/bcor-relays
```

The new integration/mode tests are registered in the appropriate field-UI and
particle-continuation acceptance scopes. Detailed local test receipts and
diagnostic inputs are retained under `outputs/safe-integration/`.
