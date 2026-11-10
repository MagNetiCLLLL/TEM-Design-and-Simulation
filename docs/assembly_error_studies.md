# Assembly errors, aberrations and correction headroom

Open **Hardware Tuning → Corrector → Assembly error study**. This workflow
compares the captured probe column with independently perturbed copies. It
reuses the part editor, archived assembly reconstruction, placed fields,
fixed-pupil residual observer and validated probe-service electrical outputs.
Opening the page or changing its options performs no calculation.

## Inputs and execution

1. Add a placement owner, an error parameter and a signed delta. Multiple rows
   form a combined case. Select **Also compare each error separately** to add
   independent single-error cases against the same nominal instrument.
2. Select the service inputs permitted to compensate. For example, `AT_A2`
   upper is the HP1/HP2 differential input; its lower input is the TL21/TL22
   differential input. They are not laboratory X/Y controls. Inputs with no
   nonzero calibrated intrinsic weights cannot be selected. Configure missing
   weights explicitly in the existing service interface.
3. Select coefficient targets through third order, the diagnostic pupil and
   integration step. Declare tolerances for the target pupil displacement,
   chief position and mechanical direction. These are requested local study
   tolerances, not manufacturer specifications or numerical-convergence claims.
4. Declare the signed calibration increment, total route search bound and
   iteration budget. All service increments use percentage points of the
   existing output maxima; no screenshot currents are interpreted as amperes.
5. Optionally select **Compare with configured learning gains**. It starts off.
   This explicit comparison uses saved learning gains even if the live learning
   switch is off, without changing that switch or applying gains to the live
   instrument. Zero learning gains remain zero.
6. Run, inspect the stages and save the portable JSON study. Failed cases and
   unavailable learning stages are retained. Cancellation and stale live inputs
   prevent publication of a late result. No candidate is automatically applied.

## Placement and reference conventions

The captured instrument is the nominal reference; it is not assumed to be an
ideal or calibrated OEM assembly. Errors are additive to its existing poses.
Supported upstream main-column placement owners expose:

| Parameter | Unit | Existing model action |
| --- | --- | --- |
| `offset_x_mm`, `offset_y_mm`, `offset_z_mm` | mm | Translate the field and owned hardware |
| `assembly_azimuth_deg` | deg | Rotate owned CAD, windings, bores and fields about column +Z |
| `field_azimuth_error_deg` | deg | Rotate only the transverse field basis; preserve material assembly |
| `rotation_x_mrad`, `rotation_y_mrad` | mrad | Existing optical-axis tilt about transverse diameter axes |

Column +Z is downstream and positive azimuth turns +X toward +Y. Optical tilt
does not imply a newly modelled tilted material bore. Field-only azimuth is
not available for apertures. Shared channels have no second placement owner:
edit `ac_deflector` for AC/DP12. Gun and filter-branch placement are outside
this version's probe-column scope.

The input pupil, nominal first-order transfer, electron energy and observation
plane are captured once. They are not recaptured after each error or correction.
Only the selected service outputs change during compensation. No implicit
focus adjustment, source resampling, current normalization or fitted phase
screen is introduced. Axial errors do not move the observation plane.

## Intrinsic response and learning

The intrinsic route and learning route are independent, provenance-bearing
gain tables from `optics/probe_service.py`. Every intrinsic move is physically
propagated, including its effects on *all* measured coefficients, chief
position and direction. Off-diagonal response is natural coupling, not an
automatically generated learning coefficient.

Reports compare:

- The nominal captured instrument.
- The perturbed instrument at unchanged electrical controls.
- Compensation using only intrinsic service-route outputs.
- If requested, compensation with the same selected inputs and their saved
  learning feed-forward outputs. This solve starts from the same uncompensated
  case, independently of the intrinsic-only solution.

The report retains the intrinsic response, the separately measured
learning-only response, and the response of the combined command. Combined
responses are measured, not assumed equal to a finite-step sum. Identical
zero-learning commands reuse the intrinsic evidence with an explicit label.
No new learning matrix is fitted or written by this workflow.

## Local solve and reported limits

Each response column is measured at positive/negative full and half steps.
Step agreement and asymmetry are checked. Unavailable signed headroom or
unresolved nonlinear response disables that column and is recorded. A bounded
least-squares step uses at most half the checked increment; each accepted step
is independently propagated. Repeated local steps stay inside the declared
total route search bound. Shared outputs are summed and validated atomically,
including normal/skew magnitude limits and summed AC/DP12 coil loads.

The objective is the RMS ray displacement predicted by the selected Cartesian
coefficient differences over the fixed holdout pupil, together with chief-ray
position and direction differences. This weights aberration orders by their
effect at the declared pupil. All fitted coefficients, including unselected
terms, remain in the report. The fit is diagnostic and is never applied as a
correction field or wave phase.

Output and route headroom are reported separately in positive/negative
percentage points, with the native model maximum and final setting. These
are directional electrical limits with other drives held. They are not a
global correctable-aberration range. `LOCAL_SEARCH_LIMIT`, unavailable local
response, or a non-improving local step does not prove insufficient hardware.
`LOCAL_TARGET_MET` means only that the requested diagnostic changes were
restored within the stated tolerances.

Finite-source current, emitted/surviving/effective populations and aperture
transmission are explicitly `NOT_ASSESSED`, not zero and not inferred from
the diagnostic pupil. No finite-source calculation is launched here. Such
follow-up studies require at least 1000 original positive-current source
particles and independent sampling/step convergence. Coherent follow-up
calculations require usable CUDA under the current repository policy.

## Portable records and replay

The JSON contains a checksum-bound request, complete working point with
archived model inputs, explicit errors/options, geometry-edit evidence,
response matrices, residuals, physical drive changes and headroom. Replaying
requires the same solver implementation; changed-source records remain
historical evidence rather than silently acquiring new numerical semantics.

```powershell
.\.venv\Scripts\python.exe -m temsim.assembly_error_study study.json --output replay.json
```

The original study cannot be overwritten by this replay command. The Python
API exposes `capture_study`, `run_study`, `save_study` and `load_study` for
explicitly constructed `AssemblyErrorCase` and `AssemblyStudyOptions` inputs.
