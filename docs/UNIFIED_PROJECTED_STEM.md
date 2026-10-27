# Projected particle STEM with shared potentials

The explicit High accuracy STEM page supports the `projected_atoms` particle
model in Finite-Field Optics. Run the optical Ray Diagram first with the same
source, lenses and numerical settings. The STEM calculation checks the exact
executed incident checkpoint before consuming it. It never substitutes a source
at the sample or silently switches to a compatibility field model.

## Field ownership

- Source-to-sample transport retains the actual particle positions, directions,
  energies, weights and arrival times.
- AC/Descan calibration and local raster derivatives use the shared immutable
  electromagnetic potentials. Calibration precedes request signatures, so an
  edited lens cannot retain stale scan drives.
- Conditional direct and scattered detector-response rays propagate through the
  same downstream potentials and mechanical geometry. Inserted detectors
  intercept particles in physical order, including detectors with readout off.
- Natural field aberrations remain owned by the Lorentz executor. No additional
  empirical objective Cc kick or equivalent image-lens transfer is applied.
- Responses are keyed by the field/geometry runtime, implementation, executed
  incident state including energy and time, and the projected scattering inputs.

## Explicit approximations

This is the existing independent-atom projected intensity model. A Gaussian
closure of the executed probe covariance illuminates the rotated and cropped CIF
sites. Projected optical depths determine direct/scattered probabilities, and a
bounded angular quadrature estimates physical detector collection. These
auxiliary quadrature rays do not replace the requested source particle count.

The projection is an instantaneous, zero-depth scattering screen at the sample
reference plane. It adds zero time to the actual incident arrival clock; it does
not claim a resolved material event depth or a finite-volume collision history.
The raster uses local finite-field derivatives around the reference trajectory.
It does not independently retrace the nonlinear column and upstream clipping at
every pixel. This approximation is identified in the result metadata.

Increasing the particle budget or reducing the integration step controls
numerical resolution within this model. It does not make projected scattering a
quantitative multislice calculation. Atomic channeling, coherent phase contrast,
Bragg interference, plural angular scattering and 4D-STEM diffraction are not
provided by this entry. A broad actual probe can produce very little atomic
contrast; the implementation does not refocus or normalize it to improve the
appearance.

## Supported boundaries

The `stem` page admits the explicitly selected classical `projected_atoms`
model. Coherent STEM, `material_paths` raster acquisition, diffraction sinks and
dispersive energy-filter transport remain rejected. Other finite-volume specimen
workflows are not enabled merely because this projected observable is supported.
Unknown event-depth timing cannot be used as a fabricated clock for dynamic
fields. Imported CIFs do not silently acquire the inelastic constants of a named
specimen preset.

A fully blocked beam returns a zero detector frame with no invented probe
centroid. Its raster coordinates are identified as requested positions. Raw
detector fractions preserve emitted-current normalization; absorbed, truncated
and uncollected contributions remain separately available.

## Baseline focus and controlled lens comparisons

A saved lens percentage preset is not evidence that the unified field model
forms a qualified probe. The historical parameterized-field calibration and
the unified Lorentz executor have different higher-order actions. Measure the
executed source at the specimen entrance before using a preset as an imaging
baseline.

`refine_unified_surface_focus` is an explicitly invoked, detached calibration.
It retains the source, current, apertures and physical fields and adjusts only
the declared round-lens controls. A covariance waist alone cannot pass: actual
spot size, convergence angle, retained source identities and independent
integration-step checks also apply. Failure returns no installable candidate.
This helper does not recalibrate the hexapoles or certify a complete instrument.

Once a baseline is accepted, an independent lens sweep fixes that baseline and
changes only the selected lens. Ordinary parameter edits and image acquisition
do not invoke the focus solver. A defocused or clipped perturbed probe is an
intended observable, not a reason to replace it with the baseline probe.

## Exact particle-response reuse and scheduling

Large AC/Descan response measurements may execute up to four independent CPU
workers. Each worker restores the captured instrument and verifies its field,
geometry and implementation identity. Every physical perturbation uses the
same scalar Lorentz RK4 algorithm and integration nodes as serial execution.
Results are assembled in their original order; cancellation or a failed worker
does not publish a partial cache. `TEMSIM_SCAN_RESPONSE_WORKERS=1` forces the
serial reference; values 1 through 4 are accepted. Frozen applications without
a packaged Python worker retain serial execution. Receipts identify the actual
scheduling policy and wall time; process concurrency is not a physical input.

Projected direct and vacuum branches can share an exactly matching prepared
trajectory within one request. The key binds the field/geometry plan, all
integration nodes, real entrance phase space, kinetic energy, flight time,
source identity and weights. Each branch still constructs its own probability
ledger and material metadata. The cache retains exact observation and stop
nodes; it neither interpolates a new scientific state nor aliases complete
responses based only on a zero-loss probability.

## Validation evidence scope

Focused tests cover public admission and workflow reuse, actual short-interval
shared-field downstream propagation, retained energy/time and source weights,
mechanical interception, readout-disabled interception, zero illumination,
unsupported products and explicit projected-model provenance. These checks
establish the implementation boundary, not experimental instrument validation.
