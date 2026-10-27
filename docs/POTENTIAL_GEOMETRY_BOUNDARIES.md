# Finite absorbing boundaries in the potential route

`potential_geometry.py` adds an explicit, opt-in boundary wrapper around the
shared-potential coherent operator. It does not change default propagation,
optical parameters, installed component dimensions or preset calibration.

## Authoritative geometry

`capture_potential_geometry(state, start_z_m, stop_z_m)` freezes the existing
resolved vacuum-bore inventory and the physical circular aperture plates. It
uses `placed_wave_bores`, including fixed tubes recovered behind moved lens
bores, and `PosedWaveAperture.from_component`. Plate thickness must come from
the actual runtime parameter or resolved part declaration; a CAD display
extent is not accepted as physical thickness.

The reduced coaxial vacuum profile can contain only the narrowest active
constraint. A wider coincident bore then contributes no additional opacity.
Posed constraints retain their independent material registration. Declared
physical hosts are unique; shared Descan/Image-Diffraction control channels do
not create a second physical body. Zero optical excitation does not remove
installed material. Aperture installation, insertion and enable selections
are captured explicitly. An equivalent objective pupil is rejected because
this route requires the real physical aperture plane and finite plate.

The immutable JSON snapshot includes its axial scope, primitive dimensions,
poses, host identities, aperture selections and digest. Instrument captures
also retain `PhysicalSceneSnapshot.geometry_digest`; a field snapshot from a
different scene geometry cannot be combined with them. Hand-authored snapshots
can explicitly omit this authority only for supplied-primitives experiments.
They do not claim to capture an instrument.

## Boundary operation and numerical policy

`apply_potential_geometry` computes the intersection of the finite local slab
with each global-Z characteristic and tests the swept cylindrical opening.
It catches tilted shoulders between planes that are both outside the plate.
This is a perfectly absorbing material boundary, not an atomic wall potential,
secondary-emission model or microscopic edge-diffraction calculation.

Before masking, every non-clear positive-radius opening requires at least 16
cells in radius. Automatic refinement interpolates the represented incident
complex field before absorption; a fixed or insufficient grid rejects the
operation. This minimum is an admission rule, not convergence evidence for
every sharp edge or nearly closed tilted passage. Independent pixel, physical
window and axial refinement remain necessary for a requested observable.

`observe_potential_geometry_checkpoint` marches the field operator and swept
boundary masks. Its axial step also respects the tilted bore's half-pixel
motion guide and a configurable minimum number of steps through a plate.
The guide is reevaluated after optical grid changes. If a candidate requires
a smaller geometry step, it is discarded and rerun from the unchanged input
checkpoint, including its original numerical-support ledger. Discarded
attempts do not consume a saved support budget. Exact native millimetre planes
and stored metre planes are reconciled by exact conversions, without a
proximity tolerance that could merge distinct observation planes.
When material transport supplies an explicit native-plane receipt, that
identity takes precedence even if a different adjacent millimetre value equals
the canonical metre-to-millimetre conversion. A same-plane result retains the
receipt. A successful forward step writes a canonical endpoint and retires
the input receipt; later geometry continuation validates the new endpoint.

The wrapper accepts the same complete captured geometry for specimen slices
and later detector propagation. Capture bounds, geometry identity and the
boundary step policy persist in the checkpoint. Changed geometry or boundary
policy requires restart from the entrance. Duplicate thin-aperture options
are rejected. Exponential and kinetic truncation budgets are distributed by
travelled distance within each wrapper call; accumulated accepted bounds are
recorded across continuations.

An optional `KineticPhaseLedger` instead retains one declared total kinetic
truncation budget across the incoming potential checkpoint and every accepted
geometry interval. It binds the potential identity, energy, expansion order
and entrance plane. Only the ledger returned by a finally accepted geometry
step is committed; a discarded retry spends nothing. A fully absorbed mode
advances with a zero-bound event. Same-plane observations retain the identical
ledger, and restored history cannot be reset or assigned a larger budget.
This option currently admits one controlled mode. The ledger measures a
kinetic-model error bound, not physical phase or lost probability. It does not
alter a field coefficient, phase carrier, numerical support threshold or the
declared total error limit. Material branching requires its own separately
admitted inheritance policy.

## Flux, phase and restart

Lost physical probability is summed directly from the masked cells. It is
never obtained by subtracting two nearly equal total intensities. The direct
sum retains, for example, a `1e-22` intercepted tail even when both displayed
total norms round to one. Host masks are applied sequentially: overlapping
opacity is attributed to the first intercepting host and is not counted twice.

`WaveMode` continues to represent a unit complex shape and its reduced
absolute weight per original reference electron. That factorization does not
restore physically lost current. Analytical curvature, tilt, coordinate
origin, constant phase and the axial action/time reference are retained.
Checkpoint records keep physical absorption separate from numerical support
projection. A saved support ledger cannot be reset on continuation. After a
conditional material branch, the inherited common-prefix support record is
read-only and no new projection budget can be opened by this wrapper.

## Bounded validation

`tests/test_potential_geometry.py` checks all seven installed column choices,
excitation-independent geometry, real plate declarations, immutable capture,
field/geometry mismatch rejection, tilted finite cylinder intersections,
direct tiny-loss accounting, incident refinement, physical absolute weights,
phase-carrier preservation, exact neighbouring planes, stored restart and
support-lineage restrictions.

An additional controlled 300 keV free-propagation case crosses a 2 micrometre
thick, 1 micrometre radius plate, offset by 0.2 micrometres and tilted by
5 mrad. It compares independent pixel, axial-step and physical-window changes.
Its sharp-edge study explicitly uses a 0.01 envelope spectral admission tail;
all Fourier coefficients remain present. This is not the default `1e-10`
spectral admission and does not qualify atomic edge diffraction or a full
instrument source. The test records raw absolute complex-field and flux
differences without phase alignment, recentring or intensity normalization.

The wrapper remains CPU-only and opt-in. Captured material outside the
authoritative bore/plate inventory, wall scattering, spin, a complete gun
source, general time-dependent fields and production admission are outside
this validation scope.
