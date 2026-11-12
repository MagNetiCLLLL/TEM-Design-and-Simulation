# Provisional weak-C2 projector rematch

This is the explicitly authorized provisional working point: C1 = 52.5%, C2 = 17.75%, Mini = -10% signed, Objective = 29.29264%, and physical C2 aperture diameter = 70 um. The incident finite-source focus and alpha95 are not measured by this observer. The requested >=10 mrad convergence belongs to the separate >=1000-original-particle study.

Only D and I were adjusted; P1 and P2 were held fixed. Source emission was guarded against invocation.

| Device | Excitation (%) |
| --- | ---: |
| diffraction_lens | 13.5928121213767 |
| intermediate_lens | 8.51783304453163 |
| projector_lens_1 | 32.1132572718703 |
| projector_lens_2 | 9.82687442281551 |

The captured-field axisymmetric Larmor proposal model predicts that the prior 2.2 mrad working point's D/I setting has A = -0.023741626 and signed B = 0.0499473431 m at this new Objective. This baseline is a proposal-model calculation, not a new production zero-current baseline measurement. The new proposal was then checked through the complete production unified Lorentz executor.

The diagnostic starts at the specimen upper surface Z = 1025.1999975 mm. The detector-independent camera-length reference is the retracted main-screen plane Z = 2140.15 mm. HAADF/DF/BF insertion is retained; all diagnostic pencils reach the reference. Canonical input conversion uses the actual vector-potential derivative, with physical output positions.

| Step bound (mm) | Increment scale | Canonical position-response norm | Camera length (mm) |
| ---: | ---: | ---: | ---: |
| 0.1 | 1 | 5.264760677e-07 | 49.99999996552 |
| 0.05 | 1 | 4.879675883e-07 | 49.99999993395 |
| 0.05 | 0.5 | 4.87965174e-07 | 49.99999993395 |

All three differential checks pass the position-response limit 1e-3 and camera-length relative tolerance 3%. The 0.1/0.05 mm meshes are independent; the final check halves both position and slope increments. Each run uses nine zero-current differential pencils and creates no finite-source population.

proposal.json retains full matrices and receipts. The prepared snapshot, explicit profile, frozen upstream-control file and artifact hashes are stored alongside it. The old study directory is unchanged.

The Objective here is provisional. Perform a separate exact-final-Objective production check before writing final measured targets; retain these records without changing their working-point labels. This does not qualify finite-angle detector acceptance, current, RMS, D95, a scattering/scanning image or coherent resolution.
