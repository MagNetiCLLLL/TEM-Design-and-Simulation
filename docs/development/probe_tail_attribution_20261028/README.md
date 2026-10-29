# Compact probe-tail investigation archive

This bundle preserves the reproducibility code and selected evidence for the
fixed-control investigation described in
[PROBE_TAIL_ATTRIBUTION_2026-10-28.md](../PROBE_TAIL_ATTRIBUTION_2026-10-28.md).
It contains historical records, not a newly qualified optical preset. No
scientific calculation was rerun while preparing this publication bundle.
The repository's `.gitattributes` marks this bundle `-text` so Git does not
convert line endings and invalidate the preserved byte-level hashes.

The tested model retains a physical 100 um C2 aperture, complete original
source IDs and absolute weights, actual clipping, and the native finite
electromagnetic fields. There are no additional empirical aberrations,
coordinate rescaling, or hidden removal of source tails. The former 0.7 nm
size cap is no longer an acceptance requirement. D95 defines the reported
waist; D90 and RMS remain separate diagnostics.

## Contents

- `manifest.json`: original relative paths, SHA256 checksums and byte sizes;
  archived-file locations; and a separate inventory of omitted local raw
  arrays. Its Git base is the pre-publication checkout base, not a claim that
  that old commit alone contains this turn's changes.
- `reproduction/outputs/...`: byte-preserved diagnostic drivers and their
  low-cost tests, using their original repository-relative layout.
- `executed_drivers/`: driver snapshots copied from completed runs and
  deduplicated by SHA256. These preserve the code actually captured by a run
  even if its working diagnostic driver was subsequently improved.
- `records/outputs/...`: selected English summaries, compact analyses,
  checksums, complete input snapshots and complete execution reports. Larger
  JSON records use lossless deterministic gzip; their uncompressed hash is
  recorded in the manifest. No fields were stripped from these records.
- `figures/outputs/...`: supporting plots. Numerical arrays and receipts
  remain authoritative; plots are not independent validation.
- `verify_bundle.py`: read-only archive/hash validation. It does not restore
  files, import the simulator, or execute any physical calculation.

The core evidence covers source emission reconstruction, poly/mono source
controls, constant-energy controlled injection, native gun-step and
electrode-grid comparisons, round-field radial-order sensitivity, held-out
launch-momentum fits, and the native 17-probe local-linear response.

## Deliberately omitted local evidence

All NPZ particle arrays remain local, including the large full histories and
the smaller mathematical-prediction arrays. The manifest identifies 17
principal raw arrays totaling 245,235,010 bytes. Additional batch duplicates,
older optimization histories, detailed field caches and repeated multi-MB
runtime captures also remain local and are not exhaustively inventoried by
that principal-array list. This is not an archive of the entire `outputs/`
tree. No external experimental/user dataset is included.

The completed-run reports therefore document historical measurements but do
not make their raw data available from Git alone. A checksum identifies the
original bytes; it does not reconstruct them or independently prove the
reported result. Replaying a saved-source injection or rerunning an offline
analysis requires the corresponding locally retained NPZ files. Do not
replace missing raw arrays with fitted or synthetic particles.

## Verification and future reproduction

From the repository root, verify the compact archive without running physics:

```powershell
.venv/Scripts/python.exe docs/development/probe_tail_attribution_20261028/verify_bundle.py
```

The JSON files compressed with `.gz` can be read using Python's standard
`gzip.open(path, 'rt', encoding='utf-8')` and `json.load`. The verifier checks
both compressed-file SHA256 and the uncompressed original SHA256.

For a future authorized rerun, restore only the needed diagnostic scripts to
their **original relative paths in a separate checkout**; the manifest maps
each archived file to that path. Some historical source-audit scripts rely
on their original directory depth. Use the maintained `src/` and `scripts/`
from the publication commit, the project virtual environment, and the full
saved snapshot. Respect each runner's new-output-directory requirement and
its resource/cancellation settings. Do not overwrite this archive or the
original completed outputs.

Original snapshots and receipts retain their implementation digests, input
identities and local configuration-root path. A different checkout or later
code version can legitimately fail strict historical identity checks. Such a
failure must be reviewed explicitly; do not bypass the identity gate or
relabel a new result as an exact replay. New full-source runs must generate
their own raw arrays and receipts. Historical report presence alone is not
recomputation evidence.

The sources include classical physical-run drivers as well as offline
analysis. Archive verification executes neither kind of driver.

## Interpretation boundaries

- Uniform column-entry energy is an explicitly labeled counterfactual with
  saved gun-exit phase held fixed; it is not a physical monochromator result.
- Full monochromatic emission changes gun-exit phase and convergence angle.
  Its response cannot be assigned solely to high-energy column chromaticity.
- The launch-momentum regression uses grouped held-out IDs, including ID306.
  A better fit is explanatory evidence, not unique physical causality. Its
  momentum areas are kinetic/mechanical, not asserted canonical emittances.
- The 17 calibration trajectories are mathematical differential probes, with
  no assigned physical beam current. Their local-linear prediction is kept
  separate from the actual 769-particle result. The residual is not uniquely
  a spherical-aberration coefficient, and its mathematical minimum is not an
  attained or demonstrated attainable physical probe.
- The electrode-grid series did not pass the existing axial-waist comparison
  budget. These bounded studies do not establish full model or experimental
  qualification, and no tolerance was relaxed to claim convergence.
