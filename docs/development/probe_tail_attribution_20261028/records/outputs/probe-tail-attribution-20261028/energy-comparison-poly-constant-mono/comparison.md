# Controlled column-energy comparison

Native current-weighted quantiles on exact saved planes; no interpolation, refocusing or source filtering.

Controlled injection changes entry energy at fixed original exit phase; full mono also changes gun-generated phase. No quadrature subtraction or additive attribution of D95 is valid.

| Case | Surface D90 (nm) | Surface D95 (nm) | Smallest observed D95 (nm) | D95 minimum offset (nm) | Alpha95 (mrad) | Current (uA) | ID306 radius (nm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| poly_baseline | 1.4961544 | 2.5628742 | 2.17137094 | 15.5 | 29.7696157 | 10 | 28.6946501 |
| poly_phase_changed_energy | 1.45143368 | 2.44206494 | 2.21178211 | 13.5 | 29.7696327 | 10 | 29.1742189 |
| full_mono_gun_and_column | 1.39224911 | 2.01238852 | 1.57612743 | 14.7000001 | 28.4968113 | 10 | 3.56749559 |

All minima are discrete observed minima within the saved window, not exact continuous argmins. RMS is auxiliary only. There is no probe-size acceptance threshold in this diagnostic.

- poly_to_controlled_injection: exact shared planes = 479; maximum raw paired displacement at the surface = 0.482287449 nm (source ID 306); changed survival at surface = 0.
- poly_to_full_mono: exact shared planes = 479; maximum raw paired displacement at the surface = 25.174409 nm (source ID 306); changed survival at surface = 0.
- controlled_injection_to_full_mono: exact shared planes = 479; maximum raw paired displacement at the surface = 25.6564758 nm (source ID 306); changed survival at surface = 0.

Raw particle and report SHA256 digests, every native through-focus row, and every exact shared checkpoint displacement are retained in comparison.json.
