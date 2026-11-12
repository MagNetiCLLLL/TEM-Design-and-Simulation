# Exact final weak-C2 diffraction reference

The exact final upstream controls are C1 = 52.5%, C2 = 17.75%, Mini = -10% signed, and Objective = 29.292665862003528%. The physical C2 aperture diameter is 70 um. This observer does not measure the finite-source focus, alpha95, current, RMS or D95; those belong to the separate parent study.

| Projector device | Excitation (%) |
| --- | ---: |
| diffraction_lens | 13.5928121213767 |
| intermediate_lens | 8.51783304453163 |
| projector_lens_1 | 32.1132572718703 |
| projector_lens_2 | 9.82687442281551 |

The proposal D/I values were retained from the nearby provisional Objective point; P1/P2 remained fixed. A fresh production unified Lorentz response used nine zero-current differential pencils at the exact final point with both state step and maximum-step bound of 0.05 mm. Position and mechanical-slope increments were 1e-10 m and 1e-7 rad. The canonical input basis came from the actual captured vector-potential gradient. Emitter invocation was prohibited.

| Quantity | Exact final result |
| --- | ---: |
| Canonical position-response norm | 9.383691362701095e-06 |
| Effective camera length (mm) | 49.9999804682291 |
| Relative error against 50 mm | 3.906354181237859e-07 |
| Pencils reaching reference | 9 / 9 |
| Emitter calls | 0 |

The result passes the existing differential position-response criterion 1e-3 and camera-length relative tolerance 3%. The source plane is the actual specimen upper surface, Z = 1025.1999975 mm. The detector-independent target is the retracted main-screen plane, Z = 2140.15 mm. Physical HAADF/DF/BF insertion remains enabled.

The independent 0.1/0.05 mm and half-increment checks in ../weak-c2-provisional belong to Objective 29.29264%, not this exact final excitation. Those records remain unchanged. The present exact-point check uses only 0.05 mm.

final-exact-objective.json retains the full matrices, field/geometry/runtime receipt, exact control values and artifact hashes. final-exact-inputs.json.gz is the complete prepared input snapshot; final-exact-input.toml is the matching explicit profile; upstream-controls.json pins the parent-confirmed settings. The configured source count in this snapshot is 4096 but no source population was generated: only the nine differential pencils were traced. final-override.json is the final C2-specific Diffraction override suggestion.

This is a zero-current on-axis differential diffraction reference. Finite-angle detector acceptance, actual scattered/current signals, scanning images, coherent-wave resolution and experimental behavior remain unqualified. It must not be presented as a finite-source or RMS/D95 measurement. No src or global configuration file, old 2.2 mrad record, or provisional observation was changed by this observer.
