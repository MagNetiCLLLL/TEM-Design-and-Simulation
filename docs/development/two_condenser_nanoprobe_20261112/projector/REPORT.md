# Independent two-condenser Nanoprobe diffraction reference

Only zero-current differential pencils were used. The emitter method was replaced with a rejecting guard before preparation and propagation. These probes cannot supply RMS, D95, specimen current, a scanning image, finite-angle performance or coherent-wave qualification. No project source or global configuration was changed by this observer.

## Working point and physical reference

The installed geometry is C1 + C2, not a three-condenser column with C3 switched off. The specimen upper surface is Z = 1025.1999975 mm; the catalog specimen centre is 2.5 nm below it. The detector-independent camera-length reference is the TOML main fluorescent-screen plane at Z = 2140.15 mm. The screen and camera are retracted, as required by the Nanoprobe insertion preset; the HAADF/DF/BF detector insertion states remain active. All nine pencils reach the main-screen reference through the physical geometry.

Upstream controls were C1 = 90%, C2 = 35%, Mini = 10% with Nanoprobe polarity, and Objective = 29.2235172%. The specimen-plane position response uses canonical transverse momentum: the conversion from mechanical slope was measured from the actual captured vector-potential gradient. Output positions/slopes remain physical laboratory coordinates.

## Problem in the inherited projector setting

The old D/I/P1/P2 setting gives a canonical position-response norm of 5.149866 at the main screen, and an effective camera length of 67.8653 mm rather than the requested 50 mm. The 35.73% camera-length mismatch and nonzero position dependence show that its older diffraction qualification does not apply to this Objective/geometry. The two baseline repeats used the same 0.1 mm mesh; they are not an independent step test.

## Independent proposal

Only D and I were adjusted. P1 and P2, upstream optics and all mechanical geometry were retained.

| Device | Excitation (%) |
| --- | ---: |
| diffraction_lens | 13.5899996093905 |
| intermediate_lens | 8.53204496401784 |
| projector_lens_1 | 32.1132572718703 |
| projector_lens_2 | 9.82687442281551 |

The proposal model integrates the axisymmetric Larmor equation using Bz sampled from the same captured static potential, rather than a fitted image map. The sampled axial potential is approximately 300000 V; its 1.16e-7 V variation is numerical noise. This first-order model is only a seed; acceptance below uses the complete production unified Lorentz executor.

## Production unified differential verification

| Step limit (mm) | Position / slope increments | Canonical position-response norm | Effective camera length (mm) | All pencils reach reference |
| ---: | --- | ---: | ---: | --- |
| 0.1 | 1e-10 m / 1e-07 rad | 5.25772395e-07 | 49.9999999864 | yes |
| 0.05 | 1e-10 m / 1e-07 rad | 4.87221413e-07 | 49.9999999531 | yes |
| 0.05 | 5e-11 m / 5e-08 rad | 4.87218024e-07 | 49.9999999531 | yes |

The 0.1 and 0.05 mm runs are independent mesh bounds; the last run halves both diagnostic increments. The maximum position-response norm is below 5.26e-7, compared with the existing 1e-3 differential criterion. The camera-length relative error is below 9.4e-10 in this diagnostic. The complete matrices, rotations, field/geometry receipts and input identities are retained in proposal.json.

## Qualification boundary

This establishes an independent, on-axis differential C2 Nanoprobe diffraction reference at the stated Objective and source/reference planes. It does not qualify the full finite-source angular distribution, detector acceptance, scattering pattern, scanning image, coherent resolution, changed Objective/Mini controls, another recording module or an experimental instrument. The incident finite-source focus study remains a separate qualification. Use a C2-specific diffraction override; retain the existing C3 projector values and their historical records.
