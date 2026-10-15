# Physical A1/A2 channel calibration

The Hardware Tuning task **Corrector → A1 / A2 coarse and fine** measures the
current simulator's response and adjusts existing physical drives. It does not
write an aberration coefficient or add a compensation phase screen. These are
explicit simulator assignments, not calibrated CEOS or Thermo Fisher controls.

| Channel | Probe corrector | Image corrector, including B-COR |
| --- | --- | --- |
| A1 coarse | DPH1/DPH2 X/Y deflection | DPH1/DPH2 X/Y deflection |
| A1 fine | Condenser stigmator normal/skew | Objective stigmator normal/skew |
| A2 coarse | Main HP1/HP2 normal/skew | Main HP1/HP2 and installed HP3 normal/skew |
| A2 fine | Auxiliary HPC/HPol normal/skew | Auxiliary HPol normal/skew |

Coarse/fine therefore selects different physical hardware. It is not a change
of a GUI step multiplier, and the labels do not assert a manufacturer-specific
sensitivity ratio. A channel requires its hardware to be installed and enabled.
For a hexapole, Cartesian normal/skew drives are converted to the existing field
strength and orientation, including at zero initial excitation.

## Use

1. Select the installed probe or image corrector and one of the four channels.
2. Set the diagnostic pupil (initially 0.5 mrad) and maximum axial integration
   step (initially 0.05 mm), then select
   **Calculate response**. Calibration is an explicit background calculation.
3. Enter signed normal/skew **coefficient increments in nm**. The panel shows
   the physical drive group and individual-axis local limits. Simultaneous
   normal/skew requests may have a tighter combined limit.
4. Select **Calculate and apply change**. An independent propagated candidate
   must achieve the requested increment before any live hardware is changed.
   The panel reports the before/after A1, A2, C3 and C1 values. Recalibrate after
   each successful change or after modifying any physical input.
5. Recalculate the coherent beam, select the desired observation Z and read
   its phase plate through the existing section viewer. A stale wave cannot
   represent newly applied hardware settings.

Each calibration captures a new nominal first-order reference. Its before/after
pair shares that reference; absolute fitted coefficients from separate
calibrations are not directly comparable. In particular, a new reference can
absorb an already-applied stigmator's linear A1 action. The independent-pupil
fit RMS shown in the panel helps assess the polynomial diagnostic. The actual
wave retains the entire stigmator action. For later constraints on equal phase
plates, compare the propagated complex waves at matching physical coordinates
and source conditions, rather than comparing separately re-referenced fits.

The actual propagated wave retains tilt, defocus, higher-order coupling,
apertures and specimen effects supported by the existing propagation path.
No fitted low-order phase is subtracted from that wave. A mixed-state beam
still has per-mode phases; an incoherent sum does not acquire a single phase.

## Finite thickness and validity

All current hexapoles retain their distributed Gaussian axial field:
`effective_length_mm` is its FWHM, and `strength_m3` is the peak coefficient.
A shorter physical element remains finite. Editing field length at fixed peak
changes integrated excitation as well as the nonlinear interaction during
propagation. Editing length while preserving integrated excitation can still
change the third-order response. A zero-strength element has no field action
regardless of its length. Mechanical envelope length and effective field
length remain separate parameters; neither is a measured 3D magnetic map.

Hexapole and quadrupole inputs are checked at runtime edits, snapshot restore
and propagation-plan construction. Non-finite values, invalid effective
lengths and strengths beyond the declared limit are rejected. Signed physical
strengths remain valid within their bounds. Failed edits do not partially move
the associated hardware.

Calibration holds the source pupil, nominal transfer and coefficient
normalization fixed across every perturbation. Recalculating that reference
for each stigmator perturbation would absorb the first-order A1 response.
Signed full/half-step physical perturbations check response rank, conditioning
and local linearity. A normalized minimum-norm solve uses each drive's measured
step, bounds the requested adjustment, then propagates it again. Only the
measured physical settings are committed atomically; cancellation or changed
input identity prevents publication. Changes in field width, positions,
strengths and the rest of the captured physical state invalidate calibration.

The reported aberrations are finite-pupil ray diagnostics of this working
point. C1, C3, displacement and direction are coupled outputs, not constraints
silently held fixed. This diagnostic does not certify finite-source behavior,
interception, scattering, arbitrary pupil sizes, integration-step convergence
or an experimental microscope. Refine the integration step and pupil as
appropriate before relying on a numerical response for an optimization study.

## Regression coverage

`test_corrector_parameter_validation.py` covers rejected edits, restore and
propagation entry. `test_corrector_channels.py` covers calibration, bounded
candidate checks and state-safe application. `test_corrector_channel_gui.py`
covers explicit execution, stale inputs, cancellation and publication.
`test_hexapole_thickness_response.py` checks analytical finite-thickness
responses, and `test_corrector_channel_wave_transport.py` checks actual complex
wave and full-phase changes, signed normal/skew symmetry and axial refinement.
These enter the `field-ui` and `coherent-development` acceptance inventories;
passing bounded cases is not a full microscope qualification.
