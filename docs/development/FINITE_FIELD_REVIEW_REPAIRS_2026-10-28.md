# Finite-field review repairs, 2026-10-28

## Scan calibration responsiveness

Finite-field scan edits, **Calibrate and hold**, and **Descan-only** now submit
calibration to the application's coordinated background worker queue. Valid
requested raster settings and field-of-view labels update immediately. A STEM
request made while calibration is pending waits for a successful result.

The worker restores a complete instrument snapshot. It verifies solver and
external-input identities before publishing; the GUI checks the request
generation, live input graph, and dependencies again before installing only
calibration-owned drive values. New edits, cancellation, errors, state
replacement, and page deletion cannot install obsolete or partial results.
Installation failure restores the previous controls without replacing shared
physical coil hosts. Window shutdown participates in the existing bounded
cancellation wait; a postponed close leaves the page usable.

## Nanoprobe crossover acceptance

The Large angle branch now requires executed two-dimensional waists in both
the C2-to-C3 and C3-to-ADL intervals. Normal still requires the former and
forbids the latter. Incomplete coverage, unresolved candidates, or unresolved
population changes cannot produce a passing topology summary. Measured waists,
physical losses, and original particle weights remain available in the report.

These repairs do not retune illumination lenses, move apertures, or qualify an
operating preset.

## Validation

- 27 crossover/downstream-crossover tests passed.
- 62 scan calibration/control/held-drive regression tests passed.
- Three additional controlled tests passed for snapshot-capture failure,
  transactional installation rollback, and the actual MainWindow close path.

The GUI checks use controlled worker delays or simulated responses; they do
not constitute optical validation. Test receipts are retained locally under
`outputs/`. No coherent pipeline was run.

## Current and probe-size definitions

For specimen-plane D95, the denominator is the sum of original positive
particle weights that physically reach the specimen entrance. D95 is twice
the current-weighted 95th percentile of distance from that plane's weighted
beam centroid. RMS radius uses all arriving positive-current particles,
including the outer tail. Computing a conditional shape statistic does not
renormalize the physical source or restore intercepted current.

Report absolute current and loss at each physical stop separately, together
with total transmission relative to emitted current and gun-exit current.
The user confirmed that no hard transmitted-current lower bound is imposed.
After clarifying the two 95% quantities, the user selected **alpha95 = 30 mrad**
and **spatial D95 < 0.7 nm** at the specimen upper surface. This replaces the
previous physical-pupil-edge 30 mrad objective for the new strength search.
RMS radius and the physical pupil edge remain separate reported observables;
neither the old 0.2 nm RMS gate nor a 30 mrad physical edge is an additional
user-selected requirement for this search.

Any new strength candidate must retain its measured status until the joint
topology, pupil, specimen-probe, current, numerical-convergence, and driven-scan
requirements in the optical acceptance contract have been checked.

The new search uses C2/C3 as the primary angle controls and Mini Condenser /
Upper Objective for specimen-focus compensation. Executed measurements and
remaining optical limitations are recorded in the
[alpha95 search report](FINITE_FIELD_ALPHA95_SEARCH_2026-10-28.md).

## Final integrated validation

After integrating the direct-focus and finite-region acceptance repairs,
the actual project passes 526 distinct targeted regressions: 189 optical
observer/qualification cases, 199 scope-registry cases, 13 production/joint
entry cases and 125 scan/calibration cases. No failures or skips remain.
The registry's exact expected set now includes the newly registered tests
and the existing pupil/source/crossover files omitted by its older fixture;
the equality assertion is retained.

The complete final physical candidate has D95 2.199562 nm and alpha95
29.999866 mrad with 193 specimen arrivals and 10 microampere. Direct focus
and the required source-image regions pass at the recorded sampling, but
the diameter fails the requested 0.7 nm bound. No replacement default or
qualified scanning image is installed. See the
[direct-focus continuation](FINITE_FIELD_DIRECT_FOCUS_CONTINUATION_2026-10-28.md)
and its compact diagnostic evidence for the full numerical scope.
