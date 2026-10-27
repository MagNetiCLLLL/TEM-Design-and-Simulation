# Calculation progress

The status bar separates two kinds of progress:

- **Current** reports actual counts for the named solver step. Electron histories,
  axial steps, photon events, specimen slices and readout operations have different
  units. Finishing one step does not finish its enclosing calculation.
- **Overall** reports completed, explicitly declared workflow stages. Stage counts
  are not a time estimate. Stages can have very different computational costs.

An unknown local total is displayed as indeterminate and clears the previous
step's counts. Adaptive work may report its accepted count without knowing its
final total. A solver callback alone never establishes whole-job completion.

Particle pipelines retain their declared stages and reserve a publication stage.
Coherent requests have wave transport, intensity/phase readout and publication
stages. Paired requests additionally contain particle transport and, when active,
particle specimen continuation. Only accepted output completes publication.
Saved-state comparisons count the distinct executed states and a final complete
overlay publication. Accepting one member never completes the selected set.
When a coherent result is valid but its captured particle identity does not
match, the wave remains available while the paired comparison is marked failed.
Failure and cancellation retain the counts actually reached. Input checks that
reject a request before a worker starts are reported as failed requests, while
the previous completed scientific result remains available.

Each captured request keeps the same operation identity from its first
preparation update through execution and its terminal notification, including
an exact cache hit. Completing one request removes its active state without
hiding another running request. Stale queued updates cannot
revive a terminal request. A paired owner aggregates its child wave without
counting the pair and its child as two active jobs. Other concurrent requests keep
independent counters.

The existing `callback(done, total, label)` API remains valid for local progress.
`StageProgressLabel` adds explicit stage metadata without breaking consumers that
expect a string. Text-only optical transport callbacks can use
`report_transport_progress(label, completed=..., total=...)` to retain actual
counts. Neither interface parses numbers from display text or extrapolates
progress from elapsed time. These records are UI diagnostics, not physical state
or optical cache inputs.

Optical Transfer displays only canonical records already produced by the
background calculation. Publishing a Preview or Medium frame does not launch
another first-order solve on the UI thread. Missing or unsupported diagnostics
are labelled explicitly; older mechanical-basis records require recalculation
instead of being silently rebuilt while displaying a saved result.
