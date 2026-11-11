# Operating working points

## Default Nanoprobe working point

[Default Nanoprobe](default-nanoprobe.temwp) is the fresh desktop startup
working point. It records the 300 kV Normal Nanoprobe after the A1, D4 and A5
trim, with zero configured source energy FWHM, 16,384 original source
particles and a 0.025 mm maximum integration step. It can also be restored
through **Working Points > Import... > Restore working point**.

The package contains instrument inputs. It starts ready for an explicit
calculation. Select **Default Nanoprobe working point** in the **Startup
result** menu to open these inputs while continuing automatic particle result
saves. Named startup results and automatic restoration remain available.

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
