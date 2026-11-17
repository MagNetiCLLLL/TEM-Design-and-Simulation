# Read-only fixed-envelope round-lens field comparison

This calculation restores two detached copies of the exact mixed-run-v3 boundary inputs. It compares all 13 enabled round-lens owners at identical percentages, polarity, geometry and winding counts. One detached copy selects the current Gaussian source; the other retains the captured ideal prescribed-current coil source. No source emission, particle transport, coherent propagation, focus tuning, or default-input edit was performed.

The reference is the current Gaussian coefficient set restored from the boundary. The 13 serialized lens records match the October 27 baseline records on all checked field controls and explicit Gaussian parameters (see historical-input-comparison.json). The old plain state does not retain every implicit component constant; complete historical geometry equivalence needs its original configuration evidence. The catalog's October 27, 49-particle result is not a qualification of this 3000-particle ideal-coil state.

## Main result

Matching the isolated Gaussian peak (12 round owners) or the Objective field at the specimen (Objective owner) did not match the axial field profile. The broad coil sources and their tails have substantially different focusing-power integrals. Current scales amplitude and leaves peak locations, normalized shape and FWHM unchanged. The two Objective lobes are physically separated from the old Gaussian peak near the specimen.

![Major axial profiles](major-lens-field-comparison.png)

## Candidate currents and irreducible shape residuals

These are reviewable field-only candidates, not recommended or applied working points. Signed currents include the existing polarity. LS minimizes integral (s C(z) - G(z))^2 dz over the entire real axis for one owner. The minimum relative L2 error is its one-dimensional projection lower bound. B2 matching uses s = sqrt(integral G^2 / integral C^2); it matches only the isolated paraxial-strength diagnostic. It does not reproduce thick-lens transfer, crossovers, spherical aberration or focus.

| Lens | Fixed turns | Archived current A | Best profile-LS current A | Isolated B2-match current A | Minimum profile error | Gaussian / coil FWHM mm | Archived B2 / Gaussian B2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 lens | 300 | 374.347808 | 45.7460645 | 105.494938 | 90.1090% | 10.110686 / 125.902685 | 12.591786 |
| C2 lens | 540 | 51.6999252 | 7.57273705 | 15.9510903 | 88.0122% | 17.693701 / 180.524797 | 10.505073 |
| C3 lens | 510 | 16.8955441 | 1.86921777 | 4.52485686 | 91.0686% | 12.739465 / 172.985936 | 13.942308 |
| Adapter lens | 23 | 34.529999 | 7.24234924 | 12.8430398 | 82.5835% | 9.908472 / 71.254289 | 7.2286546 |
| Probe TL22 lens | 19 | -1082.97947 | -202.370417 | -379.526099 | 84.5977% | 8.492976 / 68.808235 | 8.1424885 |
| Probe TL21 lens | 19 | 1016.15293 | 190.057775 | 356.276012 | 84.5827% | 8.492976 / 68.743136 | 8.1347644 |
| Probe TL12 lens | 19 | 675.200888 | 146.316275 | 255.552607 | 81.9871% | 9.908472 / 68.823226 | 6.9808029 |
| Mini Condenser lens | 17 | -387.888055 | -172.021033 | -217.040011 | 60.9771% | 11.323969 / 35.940934 | 3.1939875 |
| Objective lens | 1160 | 1097.80625 | 11.6761922 | 84.1582819 | 99.0329% | 34.160330 / 2 x 158.736416 | 170.16008 |
| Diffraction lens | 360 | 28.9628961 | 6.66250422 | 11.26792 | 80.6466% | 16.985953 / 109.768525 | 6.6068806 |
| Intermediate lens | 648 | 0.0618889428 | 0.0112152168 | 0.0212507349 | 84.9396% | 21.232441 / 171.242607 | 8.4816085 |
| Projector P1 lens | 432 | 32.8252695 | 9.85136222 | 14.6676598 | 74.0879% | 25.478929 / 124.039769 | 5.0083532 |
| Projector P2 lens | 648 | 7.5141373 | 1.80416324 | 2.97926671 | 79.5790% | 28.309921 / 171.242607 | 6.3612064 |

The actual archived Objective winding counts are 580 + 580, following the previously allowed CAD repacking adjustment. The photo-constrained original reference is 586 + 586. The archived NI remains 636727.622261 A turn per interval, so the archived 1097.806245 A is field-equivalent to 1086.565908 A at 586 turns each. This report preserves the archived 580 + 580 input. Candidate photo-reference equivalents are retained separately in results.json; neither count nor current was applied.

## Full round-lens sum and cross terms

All totals below use the signed on-axis sum of the 13 enabled round owners, including cross terms 2 sum(i<j) integral Bi Bj dz. They are not a claim about every three-dimensional deflector, multipole or gun field. The entire real axis is included.

| Candidate | Total integral Bz^2 (T^2 m) | Signed cross contribution (T^2 m) | Ratio to Gaussian | Relative profile L2 error |
| --- | ---: | ---: | ---: | ---: |
| current_gaussian_reference | 0.0413208097542 | -3.40168620471e-05 | 1 | 0.00000% |
| archived_current | 5.64217560524 | 0.0282270299 | 136.54562 | 1158.91527% |
| isolated_full_profile_ls | 0.00273426221021 | 4.9473834917e-05 | 0.0661715544 | 97.06185% |
| isolated_integral_b2_match | 0.0418740159136 | 0.000519189297273 | 1.01338808 | 126.98762% |
| whole_column_profile_nnls | 0.00251505007931 | -0.000381621807418 | 0.0608664277 | 96.90890% |
| whole_column_nnls_then_total_b2_match | 0.0413208097542 | -0.00626982429977 | 1 | 122.74272% |

Joint NNLS preserves each owner's existing polarity and minimizes the complete signed sum profile error. It is an alternative field fit, not a lens-by-lens preservation condition. Even the best joint fit has a 96.91% relative profile error. Its total integral Bz^2 is only 6.09% of the Gaussian value; rescaling it to equal total B2 increases profile error to 122.74%. Equal isolated integrals also do not exactly preserve the full sum because their cross terms differ.

The gun-exit to specimen interval (450 to 1599.1999975 mm) gives total integral Bz^2 2.857411844 T^2 m for the archived coils versus 0.02466718239 T^2 m for Gaussian. Both finite-column intervals and full Gram matrices are saved in results.json.

## Numerical evidence and limits

The field law retains the captured 12 x 12 loop quadrature and all coil tails. Axis specializations were checked against each captured public XYZ field sampler. Composite Gauss integration uses panels at most 20 mm and orders 32, 48 and 64, plus rational transformations of both infinite tails. Changes in isolated B2 integrals were below 6.5e-15 relative across these integration orders; Gaussian moments also match their closed-form integrals. This validates integration of the unchanged numerical field law. It is not an independent continuum-winding quadrature-convergence study or a focus/particle-sampling qualification.

The historical Gaussian model uses its own off-axis paraxial expansion and configured aberration terms. The ideal-coil model uses the vacuum coil field without pole material or an empirical Gaussian aberration kick. Even an approximate axial match would require separate transport and aberration validation before a working point can be called restored.

Files: results.json (metrics, all candidate currents and full Gram matrices); captured-field-coefficients.json (immutable inputs); axial-profiles.npz (z in metres, Bz in tesla and candidate scales); candidate-currents.csv; historical-input-comparison.json; major-lens-field-comparison.png/pdf; all-lens-field-comparison.png.
