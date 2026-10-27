# Finite-field Nanoprobe default

The normal Nanoprobe preset for `unified` (Finite-Field Optics) now uses the
working point measured on 2026-10-27. Standard GUI startup and recovery from an
unavailable automatic result select this model before applying the preset.
Explicitly applying normal Nanoprobe in this model restores these optical
controls. A model-menu switch alone continues to preserve the current controls;
saved results and explicit profiles retain their own inputs.

| Control | Value |
| --- | ---: |
| C2 excitation | 13.899040164772698% |
| C3 excitation | 15.68823681182136% |
| Mini Condenser reference magnitude | 43.88646617222637%, negative polarity |
| Objective excitation | 68.97810600833917% |
| HP2 strength | 4371028.449290636 m^-3 |
| HP1 strength | 2616320.416243901 m^-3 |
| HP1 orientation | -0.042012393991127016 rad |

C2/C3 match the convergence and corrector-entrance conditions. Mini Condenser
and Objective set the specimen focus; the hexapoles were rematched to reduce
the residual spot. The Objective control scales both assembly halves. Ordinary
illumination changes preserve the live Mini Condenser magnitude and only select
its sign, so a manually changed magnitude requires revalidation.

The final surface RMS radii were 0.15583140 nm at a 0.05 mm integration step and
0.15583240 nm at 0.025 mm. The measured alpha95 was 9.96935682 mrad. The original
49-particle emitted source and weights were retained, with all 49 rays passing.
The original saved optics gave approximately 203.054 nm RMS radius at 0.05 mm.
See the [recorded numerical evidence](finite_field_nanoprobe_20261027.json).

Standard startup sets the high-accuracy integration step to 0.05 mm. Fast
Preview retains its cheaper numerical settings and is not the fine-step focus
validation. Source sampling controls remain independent: the 49-particle
validation does not establish convergence at other particle counts, a qualified
condenser topology, or experimental hardware accuracy. The legacy/custom
presets and the large-angle branch retain their existing calibrations.

The subsequent Si[110] acquisition completed at 64 x 64 pixels and 0.02 nm per
pixel. Post-calibration RMS radius was 0.15639986 nm and D95 was 0.57736407 nm.
Raw DF fraction ranged from 0.00895366976844 to 0.02029088332969. TIFF and NPZ
matched, and the per-pixel probability ledger was conserved. This is expected
intensity from the independent-atom projected particle model with a Gaussian
footprint derived from the executed beam covariance. It does not include
coherent interference, channeling or Poisson noise. The default optics do not
automatically insert this specimen or replace the user's scan setup.

To start with these defaults instead of restoring an older calculation, select
**File > Startup result > None — start with standard settings**, then restart.
For an already open default assembly, select **Finite-Field Optics**, choose
normal **Nanoprobe**, and explicitly apply the lens preset. Recalibrate scanning
after changing the optical working point.
