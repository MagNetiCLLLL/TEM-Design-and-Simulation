# Continuous tube, local component bores and actual beam clearance

Review date: 2026-10-20. Scope: installed C3 + Probe Corrector, with backend verification. The saved dimensions below remain engineering/photo-scaled reconstruction, not newly established manufacturer measurements.

## Three different surfaces

| Saved geometry | Default dimension | Meaning |
| --- | --- | --- |
| Continuous tube `inner_diameter_mm` | 5.76 mm | Beam-accessible tube interior, radius 2.88 mm |
| Continuous tube `outer_diameter_mm` | 19.2 mm through the corrector | Outer material surface; not beam clearance |
| Corrector `mechanical_clear_bore_diameter_mm` | 20 mm | Local component opening around the tube; not the full installed clear beam aperture |
| Corrector row `vacuum_inner_diameter_mm` | 20 mm | Component-local bore constraint, intersected with overlapping tube/other local constraints |
| C1/C2 cartridge material ID / OD | 60 / 90.75 mm | Carrier annulus around separate pole members; neither value is the electron beam aperture |

There is no reason to replace all 20 mm values with 5.76 mm. In the corrector region, the 19.2 mm tube fits inside the 20 mm opening with 0.4 mm radial clearance. The actual beam passes through the tube's smaller interior. These numbers describe nested objects. Source metadata in the column geometry attributes tube OD and ID/OD ratio to the previous user-directed photo reconstruction; this review adds no OEM dimensional claim.

The material tube extends from module-local 0 to 1084.5 mm (global 450 to 1534.5 mm in the default assembly). Its saved outer profile reduces OD to 17.2 mm across C3's 18 mm pole bore, with explicit transition stations. This material clearance does not alter the 5.76 mm inner bore. The C2/C3 aperture insertion slots are separately saved under `mechanical_material_tube_*`; the actual slots follow the mechanical aperture position and are rebuilt from its current plate dimensions plus saved clearance.

## Consumers and boundaries

| Consumer | Authoritative input | Checked behavior |
| --- | --- | --- |
| Assembly resolver / CAD assembly edits | `_module_vacuum_segments` | Uses the minimum overlapping local bore and continuous tube ID; excludes nonmaterial control planes |
| Particle trajectory clipping | `column_wall.clip_column_wall` | Reads resolved bore segments; a corrector ray at radius 2.89 mm stops despite fitting inside its local 20 mm opening |
| Coherent wave transport | `column_wave._wave_bores` and `_expanded_profile_axis` | Uses the same resolved/posed bore inventory and radius 2.88 mm; no separate 10 mm wall is substituted |
| Grounded instrument electrostatics | Resolved liner → `bind_gun_field_environment` → `closed_gun_field._physical_liner_rows` | Uses tube inner radius 2.88 mm and outer radius 9.6 mm in the corrector region |
| Native/STEP tube material | `cad_vacuum_tube` | Uses saved outer profile and aperture slots, retaining the same inner cylindrical surface |
| Component analytic field domains and posed-field bounds | Component-local `mechanical_clear_bore_diameter_mm` / bore | Describes that source's local valid region; does not override the separately applied narrower column wall |
| Ray Diagram / bore diagnostics | `vacuum_bore_plot_points` | Plots resolved electron-accessible segments, not the carrier material ID |

The electric model remains an axisymmetric equivalent grounded boundary. Side-entry material slots are **not** a solved three-dimensional electrostatic opening, and no physical metal alloy or seal composition is inferred from their rendering. A rendering change must not silently enlarge the solver tube ID. If the tube ID itself is deliberately changed, resolution follows the new value while preserving any still-narrower local component constraint. Posed components retain their independent local bore and fixed tube constraints through the existing wall-placement path.

`test_continuous_tube_bore_contract.py` checks seven installed HP/DPH/DP units, particle and wave boundary agreement, grounded electric boundary capture, unchanged beam aperture under material-profile/slot edits and responsiveness to both independent bore inputs. Existing `test_cad_vacuum_tube.py` verifies saved slot geometry, no C3 pole/aperture intersection, correct insertion direction, placement and invalid-input handling. This establishes software contracts; it is not real-instrument bore metrology or a 3D field qualification.

## C3 radial clearance and shortest three-dimensional distance

The saved material tube has OD17.2 mm across local Z380–550 mm. The C3 pole bore is 18 mm. Their constant-radius sections therefore have radial clearance `(18 − 17.2)/2 = 0.4 mm`. No change to either diameter is required by the nominal coaxial geometry.

The tube's existing 10 mm ramps meet the ends of the C3 poles: radius falls from 9.6 to 8.6 mm between Z370 and380, and rises again between550 and560. At each outer pole edge the closest point lies on the sloping tube transition, slightly outside the pole's axial span. In an exact meridional section, the ramp slope is `s = (9.6 − 8.6)/10 = 0.1` and the shortest distance is:

`d = 0.4 / sqrt(1 + s²) = 0.398014876084 mm`.

This is a minimum Euclidean material distance, not a smaller beam-aperture radius. Both pole pieces give the same result. Native closed-mesh checks converge consistently:

| Angular samples | Minimum mesh gap, mm |
| --- | --- |
| 32 | 0.396117166 |
| 64 | 0.397540188 |
| 128 | 0.397896188 |

For aligned regular-polygon sections the independent faceted reference is `d_n = 0.4 cos(π/n) / sqrt(1 + [0.1 cos(π/n)]²)`. The tested meshes match it and remain disjoint. At128 samples the difference from the analytic conical surface is less than0.00012 mm. This discretization check is evidence about the configured nominal geometry, not a fabrication tolerance or an alignment allowance. Nonzero component placement can reduce clearance and must be assessed separately; the fixed tube must not follow a moved C3 lens automatically.

## Adapter Lens constant tube clearance

The Adapter Lens poles span module-local Z620–631.2 and Z645.2–656.4 mm. The saved tube has constant OD19.2 mm here, and the pole bores are 20 mm. The nominal shortest surface separation is therefore the radial difference `(20 - 19.2) / 2 = 0.4 mm`. Unlike the C3 section above, there is no adjacent conical tube transition that reduces this local minimum.

Both pole solids were compared with the actual continuous tube, including its saved profile and slots. At 32/64/128 angular samples the minimum distances converge as `0.4 cos(π/n)`:

| Angular samples | Minimum mesh gap, mm |
| --- | --- |
| 32 | 0.398073891 |
| 64 | 0.399518182 |
| 128 | 0.399879527 |

Every tested pair is disjoint; the final tessellation is within0.000121 mm of the analytic0.4 mm. The four added Adapter Lens end regions sit at radii39.165–45.525 mm, well outside the bore, and change neither the continuous tube nor effective beam-vacuum clipping. The saved tube ID5.76 mm through this column remains the stricter beam-space limit. Pole bore, tube outside clearance and beam-vacuum diameter continue to represent different physical surfaces.

TL22's poles, at local Z703.6–710.35 and Z722.35–729.1 mm, use the same 20 mm bore and constant OD19.2 mm tube section. Their minimum mesh gaps independently reproduce the same32/64/128 values and analytic0.4 mm limit. TL22 end hardware stays at radii39.165–46.75547125 mm and leaves both tube material and beam-vacuum clipping unchanged. Its zero-distance contacts with neighboring HP2/DP22 parts occur outside the beam tube and are separate material-interface limitations, detailed in [the TL22 construction ledger](cad-round-lenses.md#2026-10-20-tl22-transfer-lens-end-hardware-and-neighboring-contacts).
