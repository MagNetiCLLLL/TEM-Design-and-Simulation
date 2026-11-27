# Nanoprobe Normal specimen-plane D95 with HPC X/Y

The six corrector winding currents were adjusted together to minimize the current-weighted D95 on the specimen upper surface. The common objective-lens supply provided the only refocus adjustment. The starting point was `global-ce65fd820f587f73`, with a 1.743 µm D95. The source was the same 15 µA tip at +50 V gun-lens bias in every final comparison. The C2 aperture diameter was 100 µm. The relay constraints were the two radial RMS waists near 1220 and 1439 mm. The search had no convergence-angle target.

## Winding settings

The table gives current per physical supply. The two objective winding packs share the displayed current. Signs follow the winding polarity in the model.

| Supply | Reference (A) | Selected (A) |
| --- | ---: | ---: |
| HP1 | +0.308250 | −2.631749 |
| HP2 | −0.058000 | +0.036981 |
| HPC X | 0 | +69.869465 |
| HPC Y | 0 | −90.968633 |
| HPol X | +0.064510 | −1.145727 |
| HPol Y | −12.035705 | −11.582695 |
| Objective | −14.543190 | −14.420793 |

## Search and field checks

1. Add the independent HPC X and Y winding sources to the joint field solve with the installed poles and return paths. Setting both currents to zero reproduced the reference material solution at `global-d48c34fa2096ef94`. The mesh had 6,841,872 tetrahedra and 94 magnetic regions using the stored ARMCO pure-iron and PERMENORM 5000 H2 B–H curves.
2. Differentiate the loaded magnetic circuit with respect to the six corrector controls and common objective current. Screen signed current combinations using the same weighted source IDs and evaluate D95 at the physical surface after objective refocus. Re-solve the nonlinear material field at candidate biases and update the local response. Full-column checks at `global-56b347420269e6da` and `global-a9fb17635c7ea650` gave specimen D95 values of 1.288 and 0.387 µm.
3. At `global-89a9becc5914fff4`, the eight-layer production field gave a surface D95 of 0.2463 µm, with the D95 minimum 3 µm above the sample. Scan the common objective around that setting, holding the correctors fixed. The specimen-plane optimum called for a further −0.433 mA. Solve the coupled circuit at the resulting 24-channel setting `global-495554de9660806e`, build the gun and eight overlapping column field tables and propagate the original 100,000 tip particles through the physical stops. Check the focal plane using field propagation on both sides of the specimen at 1 µm axial intervals.

## Interpreting the probe

D95 is twice the radius enclosing 95% of the arriving current about its weighted centroid. The bright-core current uses a fixed 50 nm radius about the optical axis, with the original source weights. The specimen upper surface is at 1599.1999975 mm from the tip. Both settings delivered the same 1,780 primaries out of 100,000 and the same 5.8671 nA. Their current weights give an effective sample size of 269.7. The landing maps represent geometric current density. Diffraction, specimen scattering and a Si[110] scan require a separate wave calculation.

| Quantity | Reference | Selected |
| --- | ---: | ---: |
| Specimen D95 (µm) | 1.742885 | 0.242993 |
| Current inside the fixed 50 nm radius (pA) | 99.10 | 2,786.72 |
| Specimen radial RMS radius (µm) | 0.61645 | 0.06965 |
| First relay radial RMS waist (mm) | 1220.120 | 1220.154 |
| Second relay radial RMS waist (mm) | 1439.834 | 1440.826 |
| D95 minimum relative to the top surface (µm) | 0 | 0 |

The final D95 was 0.24499, 0.24299 and 0.24742 µm at planes 1 µm above, on, and 1 µm below the surface. The radial RMS minimum lay 9 µm downstream. An independent fit of the same material solution sampled the HP2, HPC and HP1/HPol bores at different azimuths. Halving its radial sample pitch from 100 to 50 µm gave D95 of 0.24314831 and 0.24314825 µm and 2,782.32 pA in the core in both cases. The 0.155 nm diameter change relative to the production table measures field interpolation sensitivity on this material mesh. The two relay waists are reported independently of the specimen focus.

The auxiliary winding model uses a 0.020 mm² metal cross-section. The selected HPC currents correspond to about 3,493 A/mm² and 4,548 A/mm². A continuous-duty operating point would require the actual winding cross-section, current limits and cooling performance. These data are unavailable for this instrument. The optimum reported here applies to the simulated current search.

## Replay

From the repository root with the project Python environment active, run the archived source and field tables through the production tracker. The destination receives the full paths and weighted sample landings.

```powershell
python -m temsim.joint_fem --profile docs/development/nanoprobe_si110_20261122/fei-hpc-d95/checkpoint/profile.json outputs/replay-hpc-d95
python -m temsim.joint_fem --profile docs/development/nanoprobe_si110_20261122/fei-hpc-d95/checkpoint/baseline-profile.json outputs/replay-hpc-d95-baseline
```

The final [material state](checkpoint/material-state.npz) shares its [material mesh](../fei-output-relay/field-replay/material-mesh.npz) with the earlier relay calculation. To reconstruct the native field archive, pass `corners_m` from the mesh first to `numpy.savez_compressed`, followed by every array in the state archive in its stored order. The native file SHA-256 is recorded as `solution_sha256` in the replay profile.

[Full tip-to-specimen ray diagram](figures/full-column.png) · [Specimen probe before and after correction](figures/sample-probe.png) · [Measurements and validation](evidence.json) · [Replay profile](checkpoint/profile.json)
