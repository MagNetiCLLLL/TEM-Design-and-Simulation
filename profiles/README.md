# Operating working points

## Startup and explicit working points

Without a saved startup result, the desktop starts from the ordinary
**FEG / C3 + Probe Corrector** assembly and its Finite-Field Nanoprobe preset,
with 3,000 source particles and a 0.05 mm maximum integration step. It does not
automatically load a `default-nanoprobe.temwp` package. Saved startup results
retain their captured settings; the main default is not the historical
13.30 mrad corrected working point.

The physical **C2** two-condenser working point must be loaded explicitly from
the [accepted C2 profile](../docs/development/two_condenser_10mrad_20261112/projector/weak-c2-final-exact/final-exact-input.toml)
using **File > Open profile**. Its saved source count is 4,096, step is 0.05 mm
and source FWHM is 0.300 eV. For the C2 mixed-scanning experiment, explicitly
increase the original count to 16,384 and set Si[110] thickness to 10 nm while
retaining those lens controls and source law. Thickness edits preserve the
upper surface and extend along +Z. This choice does not change global C2/C3
defaults or install absent C3/probe-corrector hardware.

The [zero-FWHM Nanoprobe reference](feg-nanoprobe-300kv-zero-fwhm-20261110.temwp)
preserves the preceding 300 kV working point and its A1, D4 and A5 trim.

## Previous 0.300 eV Nanoprobe initial state

The [previous 300 kV FEG Nanoprobe working point](feg-nanoprobe-300kv-12p87mrad-initial-20261110.temwp)
retains the earlier 0.300 eV source setting. Load it through **Working
Points > Import... > Restore working point**. It restores the validated
Normal Nanoprobe inputs, including the 100 µm C2 aperture, the corrected probe
column, 16,384 original source particles and the 0.025 mm integration step.
The input-only package was verified against the executed field, geometry and
gun-source identities using the original project dependencies.

The [energy validation record](feg-nanoprobe-300kv-12p87mrad-energy-20261110.json)
preserves the current-weighted gun-exit and specimen-entrance distributions.
The saved setting has a nominal 300 keV beam and a 0.300 eV configured FEG
energy FWHM. The measured specimen-entrance RMS energy spread was 0.12775 eV.

## Detached particle gun design

`particle_tip_30mrad_20260915.temwp` is the complete **input-only** design that
passed the declared classical particle checks. It has no cached rays or waves.
Import it through **Working Points > Import... > Restore working point**.
The ordinary default assembly and extractor-relative gun voltage remain unchanged.

Use Preview to inspect the layout/rays. To reproduce the nominal numerical
budget, set **High-accuracy rays = 10369** and **Step (mm) = 0.025** in the toolbar.
Do not apply a lens preset or reload the default assembly after restoration.
This point is implementation-pinned and cannot be blindly restored after solver
changes. Full conditions, limits and scalar evidence are in
[the matching report](https://github.com/MagNetiCLLLL/TEM-Design-and-Simulation/blob/04e87584ecb88a802813e2f109d67a5719615399/docs/development/GUN_MATCHING_CONTINUATION_2026-09-15.md).

The two `particle_tip_surface_30mrad_769_*_draft.toml` files are older exploratory
drafts. They do not carry the final gun geometry or convergence qualification.
