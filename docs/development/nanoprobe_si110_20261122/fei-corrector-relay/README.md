# Nanoprobe Normal corrector relay

The comparison starts from the plotted +50 V gun, +6.696 A C3 and +40 A ADL working point. Both cases emit the same 10,000 positive-current tip particles at 300 kV with a 100 µm C2 foil. C1 carries −21.25 A and C2 carries −14.85 A. All four scan winding currents are zero.

| Observable | Before | Rematched |
| --- | ---: | ---: |
| HP2 effective image minus HP1 effective plane (mm) | +8.3675 | -0.0048 |
| First intermediate radial RMS waist (mm) | 1193.358 | 1217.770 |
| Second intermediate radial RMS waist (mm) | 1345.181 | 1440.003 |
| Specimen D95 (µm) | 175.0463 | 0.3489 |
| Specimen current (nA) | 6.765564 | 6.765381 |
| Current-weighted C2 rim angle (mrad) | 4.439 | 2.329 |

The effective hexapole reference uses the first moment of the magnitude of the six-pole harmonic sampled at 0.35 mm over ±24 mm. The new HP2 and HP1 centres are 1142.7651 mm and 1300.7672 mm. The two-axis ±1 µrad perturbation images HP2 at 1300.7624 mm with magnification 0.5819. The image of HP1 is at 1522.1493 mm. The independently solved objective-only radial-coma zero is at 1522.1239 mm. The signed skew-coma residual is recorded in the [objective reference](objective-coma-reference.json). The unit-magnification symmetry of the reference CESCOR design remains open. [Müller et al., Fig. 2](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)

The second image from HP2 is at 1522.1456 mm. Changing the symmetric integration window from ±19.5 mm to ±28.5 mm gives centres ranging from 1142.762 to 1142.829 mm for HP2 and 1300.651 to 1300.962 mm for HP1. Sub-window numerical agreement applies to the stated effective-plane definition.

Using the mechanical pole midplanes instead, 1144.500 mm images to 1301.355 mm, an offset of +0.505 mm from the 1300.850 mm HP1 midplane.

The specimen upper surface is at 1599.199997 mm. The measured D95 minimum is at 1599.2000 mm with offset +0.0000 mm. Restricting both cases to the same 186 arrival IDs gives D95 175.0436 → 0.3489 µm. All landing intensities use original source weights. The edge measurement uses the transmitted 47.5–50 µm annulus and its angle to the current-weighted chief direction. An AC pivot and an atomic STEM scanning image require independent measurements at these excitations.

The terminal paraxial source image is at 1598.8785 mm, -0.3215 mm relative to the upper surface. The finite-current D95 and RMS focus depths are reported separately.

The last step changed only the two series objective packs, from -14.519513 A to -14.517475 A. The same primaries reached the specimen in both fields. [Objective-only comparison](objective-focus.json)

The nonlinear field uses 6,841,872 tetrahedra and 94 installed material regions. Its relative residual is 6.32e-12. The gun and column carry ten matching field checks. The [measurements](verification.json) include both transverse directions, the half-step perturbation, the material peaks and the 19.5–28.5 mm harmonic-window sensitivity.

## Winding currents

| Winding | Before (A) | Rematched (A) |
| --- | ---: | ---: |
| probe_hp1_hexapole | +16.000000 | +27.323732 |
| probe_hp2_hexapole | +70.000000 | -15.121203 |
| probe_hpol_hexapole_x | -5.000000 | -10.729281 |
| probe_hpol_hexapole_y | -2.717050 | -12.922527 |
| probe_tl21_lens | +330.000000 | +356.532584 |
| probe_tl22_lens | -145.000000 | -178.071536 |
| condenser_lens_3/physical-pack-2 | +6.696000 | +6.379553 |
| mini_condenser/physical-pack-3 | +226.701546 | +358.913895 |
| objective_lens/physical-pack-4 | -1.477614 | -14.517475 |
| objective_lens/physical-pack-5 | -1.477614 | -14.517475 |
| adapter_lens | +40.000000 | -13.624616 |
| probe_tl12_lens | -320.006000 | -136.503290 |

The following values give the total signed on-axis field at the mechanical centres. The transverse six-pole harmonic and its effective centre are measured separately.

| Mechanical station | z (mm) | Before Bz (mT) | Rematched Bz (mT) |
| --- | ---: | ---: | ---: |
| ADL | 1088.200 | +46.144 | -20.593 |
| HP2 | 1144.500 | -0.693 | -2.286 |
| TL22 | 1166.350 | -117.918 | -136.272 |
| TL21 | 1249.950 | +163.965 | +175.566 |
| HP1 | 1300.850 | -0.395 | +0.045 |
| HPol | 1317.200 | -33.084 | -9.197 |
| QPol | 1331.750 | -43.178 | -16.398 |
| TL12 | 1366.300 | -159.858 | -91.028 |
| Upper AC | 1387.500 | -55.031 | -52.550 |
| Lower AC | 1412.500 | -31.087 | -60.251 |
| Mini / TL11 | 1522.500 | +150.440 | +169.298 |

## Replaying the working point

Select **Nanoprobe Normal | Corrector relay** on the joint FEM page. The desktop uses the [50,000-primary profile](../../../../configs/operating_modes/fei_50v_corrector_relay/profile.json) at the same excitation. To reproduce the original measurements in this report, run the archived 10,000-primary profile.

```powershell
.\.venv\Scripts\python.exe -m temsim.joint_fem outputs/relay-replay --profile docs/development/nanoprobe_si110_20261122/fei-corrector-relay/profile-10000.json
```

[Full column and corrector close-up](full-relay.png) · [Original full-column trajectories](before-column.png) · [Specimen landings](probe-detail.png) · [Matched-scale current maps](probe-comparison.png) · [Through focus](through-focus.json) · [Previous working point](baseline-corrector.json)
