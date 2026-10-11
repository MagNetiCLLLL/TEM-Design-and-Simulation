# Entrance beam steering and actual probe diagnostics

The entrance Beam Shift/Tilt pair owns the same four applied drives everywhere
in the instrument. Hardware tuning additionally offers a reference-plane
calculation. Requested X/Y changes are commands, not a second set of stored
hardware parameters. No separate compute selector is introduced; calculations
use the top-toolbar backend policy.

## Reference-plane tilt and shift

1. Calculate the current Ray Diagram to retain the source and upstream transport.
2. Open **Hardware tuning → Beam tilt** or **Beam shift**.
3. Set **Reference Z**, which initially selects the sample plane.
4. Click **Calculate calibration**. The status identifies the actual reference
   electron, numerical condition and execution backend.
5. Enter a relative X/Y change: mrad for tilt, nm for shift.
6. Click **Calculate and apply tilt** or **Calculate and apply shift**.

Only a forward-verified solution changes the four entrance drives. DPH, lens
strengths, source inputs and other hardware stay at their captured settings.
The usual result invalidation and recalculation controls then update the views.
Recalibrate after an applied change or any source/optical change. A stale retained
ray result must be recalculated before it can supply the upstream boundary.

The numerical state is `(x, y, dx/dz, dy/dz)` in the right-handed laboratory
frame, with electrons travelling along +Z. Tilt specifies changes of projected
mechanical angles, not canonical momenta. The solver targets

```
tilt:  delta x = delta y = 0; delta atan(dx/dz), delta atan(dy/dz) = commands
shift: delta x, delta y = commands; delta (dx/dz) = delta (dy/dz) = 0
```

The four-drive Jacobian is measured around the identified electron using the
finite fields with HP, Cs, posed components and hardware interception retained.
Finite requests are iterated and checked against the actual transported
endpoint. Singular responses, drive limits, interception, cancellation or input
changes reject the operation without applying partial values.

This is a reference-electron calibration. The finite beam centroid can respond
differently because of nonlinear aberrations and clipping. Raster and wobble
are sampled at the captured simulation time; this does not calibrate their full
time-dependent scan. Optical steering through an inserted specimen is not a
material-scattering calculation. The nominal first-order plane observer remains
a different diagnostic and must not replace the full-field local response.

## Actual incident beam

The aberration comparison view retains its reference-ring calculation, explicitly
labelled as independent of the actual probe. **Measure actual incident beam**
uses the published ray result and a recorded upstream checkpoint to measure the
beam through the corrector to the sample. It does not regenerate a centred ring.

Positions, directions, widths, current weights and shape measurements belong to
that captured result. Any measured non-affine deformation is not an independent
fit of Cs, A1 or coma. A new ray result is required to inspect changed illumination
or entrance drives. The wave intensity and finite-source ray statistics are not
interchangeable observables.

## DPH parts

DPH and the related corrector steering windings use the native instrument TOML
parts and ordinary dimension/edit/save paths. The coil insert geometry shares
its host placement with the field. Winding layer thickness, angular span and
end-band width are explicit dimensions; the previous profile name remains
readable for older files. These are declared engineering shapes, not verified
CEOS winding drawings or a measured current-to-field calibration.
