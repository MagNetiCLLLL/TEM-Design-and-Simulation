# Automatic startup results

After a current particle calculation completes, the application saves its
executed result and calculated settings in the background. On the next launch,
**Automatic — last completed particle result** restores that file before any
initial preview, gun propagation or optical preset calculation is submitted.
The first run still needs to calculate when no compatible result exists.

Choose the behavior under **File → Startup result**:

- **Automatic — last completed particle result** is the default. It follows
  successful automatic particle saves.
- A named result explicitly selected by the user takes priority over automatic
  restoration.
- **None — start with standard settings** disables automatic restoration.

Restoration includes the captured instrument parameters, particle arrays,
weights, executed section and continuation state, numerical settings, and
supported calculated products. Existing window and workspace layout preferences
remain separate. The observation initially returns to the calculated endpoint.
Uncalculated parameter edits and arbitrary plot zoom/pan states are not saved by
this feature. Coherent-wave results retain their separate caching workflow;
they cannot be restored through the classical particle archive codec.

## Validity and recovery

The existing `.temsection` / `.temresult` codec validates package checksums,
typed numeric arrays, the implementation identity, external inputs and the
captured physical state. Parameter or model changes must not reuse incompatible
results. Updating solver code can therefore require a new initial calculation.
Changing controls after a successful restore follows the ordinary calculation
and cache-invalidation paths.

A missing, damaged, replaced or incompatible automatic result is not installed.
The application reports the restore failure in its log and starts the ordinary
default calculation, leaving the old file intact. A failed explicitly selected
named result keeps the existing manual-open error behavior instead of silently
changing that selection. An unreadable automatic index is also retained for
diagnosis; it is not overwritten with fabricated metadata.

## Storage and completion

The existing automatic archive is reused without copying its large arrays into
a second cache. A small `automatic-result.json` index in the application's
`saved_results` directory points to the most recent accepted completed package.
Existing compatible application-storage locations continue to be respected.
These files are local application data and are not part of the Git repository.

The index is committed atomically only after the complete archive is saved and
validated. Its request token prevents a slow older save from replacing a newer
request. Failed or cancelled saves preserve the previous completed entry.
Updating the pointer happens inside the owned background file worker, so normal
shutdown can wait for it without relying on pending GUI callbacks. Manual
exports do not change the automatic pointer. An index-write failure is reported
separately and does not invalidate the already calculated result or saved file.

## Persistent AC/Descan response measurements

Finite-Field Optics retains complete AC/Descan response measurements separately
from particle-result archives. A later process can reuse these local derivative
measurements when their executed field and geometry dependencies, observation
planes, coil definitions, beam energy, integration nodes, perturbation and
implementation remain compatible. The numerical runtime and dependency versions
also participate in the identity. A component's nominal location alone does not
prove that its field is irrelevant to a cached response.

This cache does not store the user's final scan commands as a replacement for
calibration. Current field of view, raster and constraint settings still pass
through the existing drive solve and validation. An enclosing calibration only
publishes its response after all drive checks succeed. Receipts distinguish
fresh measurement, memory reuse and disk reuse while retaining the original
measurement evidence. Particle positions, energies, weights and integration
precision are not reduced by this optimization.

The disposable `scan_response_cache_v1` directory is under local application
storage. Records use lossless typed numeric data, checksums and atomic
publication, with a maximum of 32 entries and 64 MiB in total (8 MiB per entry).
Missing, corrupt, incompatible or unavailable storage causes fresh measurement.
Tests can isolate this storage with `TEMSIM_SCAN_RESPONSE_CACHE_DIR`; an empty
value disables disk persistence. Clearing this cache does not delete saved
particle results. A change inside the measured field region can still require
new calibration even when an upstream particle checkpoint remains reusable.

## Validation

Relevant regression entry points:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src;tests"
.venv\Scripts\python.exe -X utf8 -m pytest tests/test_automatic_result.py tests/test_automatic_result_worker.py tests/test_automatic_result_gui.py tests/test_result_files_gui.py tests/test_section_archiving_gui.py tests/test_acceptance_scopes.py
.venv\Scripts\python.exe -X utf8 -m pytest tests/test_unified_scan_response_cache.py tests/test_unified_scan_persistence.py tests/test_unified_scan_calibration.py
```

Tests cover atomic metadata, concurrent saves, stale requests, cancellation,
file corruption/replacement, implementation mismatch, named-result priority,
and a real particle archive restored in a newly constructed offscreen window
and a separate Python process. These checks forbid gun transport and new
simulation dispatch; the window check also forbids preset application. They
compare restored data to the executed result. This is backend/offscreen
verification, not manual operation of the user's desktop.
