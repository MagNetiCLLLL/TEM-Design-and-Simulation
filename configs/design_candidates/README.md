# Experimental input designs

`nanoprobe_large_angle_descan_position_angle_20261018.toml` and its matching
`.temwp` provide the corresponding Large Angle held working point at 300 kV,
Diffraction, 1 µm square FOV. The same relocated Image/Diffraction hardware
hosts Descan. AC is recalibrated for Large Angle; a held Normal calibration is
not silently relabelled. Both formats restore verified inputs, not retained
particle results. See `docs/development/descan-large-angle-2026-10-18.md`.

`nanoprobe_normal_descan_position_angle_20261018.toml` loads the held
position-and-angle Descan working point for the updated C3 + Probe Corrector
installation, STEM Nanoprobe Normal, 300 kV, Diffraction, 1 µm square FOV.
The matching `.temwp` contains complete archived inputs and no retained beam
results. Import it in Working Points to inspect or restore those inputs.
See `docs/development/descan-normal-2026-10-18.md` for finite-particle,
convergence, installation and shared-hardware evidence. Older held drives
retain their own accepted constraints until explicitly recalibrated.

The two 2026-10-18 Descan packages are regenerated from a clean checkout of the
scoped publication. Input-only restore verifies archived input integrity and
preserves the original implementation identity, while allowing a compatible
current implementation to read those inputs. All physical results must be
computed anew; this does not authorize reuse of historical calculations.
The fuller support-structure CAD reconstruction from the original study
workspace is outside this publication.

These are explicitly loaded, input-only Working Point packages. They are not
default operating presets, executed beam caches or qualified imaging results.
Import in **Working Points** and restore only after saving current settings.

`matched_tip_d95_20260915_r500.temwp` contains a 500 nm curvature radius and
0.5911968652 degree emitting cap, matched to the Flat source's projected D95 of
10.0568940617 nm. It retains the Flat Nanoprobe + Diffraction lens values as a
starting point, **not a working curved-source focus**. The paired `_flat.temwp`
keeps the reference inputs. Both contain only a JSON manifest, no numeric arrays.

See `docs/development/tip-curvature-comparison-2026-09-15.md` for measurements,
failed preset acceptance, numerical limitations and reproduction commands.
