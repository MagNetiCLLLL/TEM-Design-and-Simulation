# Installed part register

This index covers all 230 distinct part keys in the 13 installed-module definitions, including shared files expanded by the project loader. Each key appears once. Use the subsystem headings to jump to the corresponding recommendations in the [review](README.md).

The second column gives the number of independent native meshes for a key followed by the modules with that count. Wire guides are included. A count of zero identifies a grouping, control or datum. Where a part has several representations the third column describes the most detailed one sampled. A part can contain several meshes of one material. The counts do not rate optical accuracy.

## Module legend

| Module | Installed configuration file | Part keys |
| --- | --- | ---: |
| FEG | [gun/FEG.toml](../../../configs/instruments/gun/FEG.toml) | 11 |
| Mono | [gun/FEG_Mono.toml](../../../configs/instruments/gun/FEG_Mono.toml) | 13 |
| Therm | [gun/Thermionic.toml](../../../configs/instruments/gun/Thermionic.toml) | 8 |
| Pulser | [beam_blanker/ElectrostaticBeamBlanker.toml](../../../configs/instruments/beam_blanker/ElectrostaticBeamBlanker.toml) | 2 |
| 2L | [column/C2.toml](../../../configs/instruments/column/C2.toml) | 37 |
| 3L | [column/C3.toml](../../../configs/instruments/column/C3.toml) | 45 |
| 3L+P | [column/C3_ProbeCorrector.toml](../../../configs/instruments/column/C3_ProbeCorrector.toml) | 88 |
| 3L+I | [column/C3_ImageCorrector.toml](../../../configs/instruments/column/C3_ImageCorrector.toml) | 97 |
| 3L+PI | [column/C3_ProbeCorrector_ImageCorrector.toml](../../../configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml) | 140 |
| 3L+B | [column/C3_BCOR.toml](../../../configs/instruments/column/C3_BCOR.toml) | 110 |
| 3L+PB | [column/C3_ProbeCorrector_BCOR.toml](../../../configs/instruments/column/C3_ProbeCorrector_BCOR.toml) | 153 |
| Plain | [project_and_recording_system/NoEnergyFilter.toml](../../../configs/instruments/project_and_recording_system/NoEnergyFilter.toml) | 34 |
| Filter | [project_and_recording_system/EnergyFilter.toml](../../../configs/instruments/project_and_recording_system/EnergyFilter.toml) | 53 |

The seven column modules distinguish the two-lens column, the three-lens column, probe correction (P), CETCOR image correction (I) and B-COR image correction (B). The two recording modules are the unfiltered and filtered installations. Summing per-module entries gives 791 because several modules share the same key.

## Sources and gun apertures

21 keys. [Subsystem recommendations](README.md#guns-monochromator-and-blanker).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `feg_rear_insulator` | 1 (FEG, Mono) | Insulator | Show where the high-voltage feed crosses the rear insulating support. |
| `feg_cathode_holder` | 1 (FEG, Mono) | Electrode | Connect the rear collar to a schematic filament support and bias lead. |
| `feg_tip` | 1 (FEG, Mono) | W(310) | Magnify the emitting apex against the extractor recess and show illustrative support posts behind the tip. |
| `feg_emitter_enclosure` | 1 (FEG, Mono) | Electrode | Trace the front annulus around the emitter and the vacuum gap to the extractor. |
| `feg_extractor` | 1 (FEG, Mono) | Electrode | Section the front recess and the through-hole in the same inset. |
| `feg_electrostatic_lens` | 1 (FEG, Mono) | Electrode | Label the electrode bore and its bias connection in the gun section. |
| `feg_dpa_aperture` | 1 (FEG, Mono) | Platinum user identified unverified | Use the same live bore in both 3D views. Label the fixed gun aperture separately from the extractor. |
| `feg_accelerator` | 19 (FEG, Mono) | Ten annular electrodes and nine insulating spacers | Show the electrode and insulator sequence in section, with an external grading indication. |
| `feg_deflector` | 8 (FEG, Mono) | coil x winding, coil y winding | Use an end-on view for winding directions and a section for the stations and carriers. |
| `feg_stigmator` | 33 (FEG, Mono) | slotted carrier, coil, coil insulation, coil former, adhesive seat | Use an end-on view for winding directions and a section for the stations and carriers. |
| `feg_c1_aperture` | 1 (FEG, Mono) | Platinum user identified unverified | Magnify its live working hole. In Mono, show how the same mechanism houses the slit. |
| `thermionic_cathode` | 1 (Therm) | LaB6 cathode envelope | Add the LaB6 crystal, heater contacts and ceramic seat in a source inset. |
| `thermionic_wehnelt` | 1 (Therm) | Electrode | Open a section through the cathode recess and label its bias connection. |
| `thermionic_gun_lens` | 1 (Therm) | Electrode | Show the open electrode bore and its order after the Wehnelt. |
| `thermionic_anode_aperture` | 1 (Therm) | Platinum user identified unverified | Identify its working opening at the extraction stage. |
| `thermionic_accelerator` | 1 (Therm) | Accelerator envelope | Place stage markers at the ten saved centres. Mark electrode thickness as unknown. |
| `thermionic_deflector` | 8 (Therm) | coil x winding, coil y winding | Use an end-on view for winding directions and a section for the stations and carriers. |
| `thermionic_stigmator` | 33 (Therm) | slotted carrier, coil, coil insulation, coil former, adhesive seat | Use an end-on view for winding directions and a section for the stations and carriers. |
| `thermionic_c1_aperture` | 1 (Therm) | Platinum user identified unverified | Label the gun exit and its selected opening. |
| `feg_monochromator_wien` | 1 (Mono) | Wien envelope | Add a transverse inset for the crossed electric and magnetic actions and their supports. |
| `feg_monochromator_slit` | 1 (Mono) | Slit envelope | Show opposing selection edges inside the existing C1 aperture mechanism. |

## Optional blanker

2 keys. [Subsystem recommendations](README.md#guns-monochromator-and-blanker).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `nanopulser_deflector` | 2 (Pulser) | electrode negative, electrode positive | Expose opposing plates, insulating support and nearby pulse feed. |
| `nanopulser_aperture` | 1 (Pulser) | Limiting-aperture envelope | Place its limiting hole and the intercepted path in the blanker inset. |

## Condenser lenses and apertures

19 keys. [Subsystem recommendations](README.md#round-lenses-and-apertures).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `condenser_lens_1` | 0 (all columns) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `condenser_lens_1_excitation_coil` | 1 (all columns) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `condenser_lens_1_housing` | 1 (all columns) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `condenser_lens_1_yoke` | 1 (all columns) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `condenser_lens_1_lower_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `condenser_lens_2_upper_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `c1_c2_pole_piece_cartridge` | 3 (all columns) | c1 carrier unresolved, nonmagnetic separator assumed, c2 carrier unresolved | Cut away the common cartridge and label the two carriers and inferred separator. |
| `condenser_lens_2` | 0 (all columns) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `condenser_lens_2_excitation_coil` | 1 (all columns) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `condenser_lens_2_housing` | 1 (all columns) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `condenser_lens_2_yoke` | 1 (all columns) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `condenser_aperture_2` | 3 (3L+P) / 1 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | body, connection rod, fixing screw | Show the active hole, rod and selector at both instrument and enlarged scales. |
| `condenser_lens_3_upper_pole` | 1 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `condenser_lens_3` | 0 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `condenser_lens_3_excitation_coil` | 1 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `condenser_lens_3_housing` | 5 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | body, upper cooling plate, upper coil insulation, lower coil insulation, lower cooling plate | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `condenser_lens_3_yoke` | 1 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `condenser_lens_3_lower_pole` | 1 (3L, 3L+P, 3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `condenser_aperture_3` | 3 (3L+P) / 1 (3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | body, connection rod, fixing screw | Show the strip attachment and the selected opening on installed three-lens columns. |

## Steering and stigmation

8 keys. [Subsystem recommendations](README.md#correctors-and-steering).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `condenser_deflector` | 20 (3L+P) / 8 (3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | coil x winding, coil y winding, coil x carrier, coil y carrier, coil interaxis spacer, … | Use an end-on view for winding directions and a section for the stations and carriers. |
| `beam_deflector` | 20 (3L+P) / 8 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | coil x winding, coil y winding, coil x carrier, coil y carrier, coil interaxis spacer, … | Use an end-on view for winding directions and a section for the stations and carriers. |
| `ac_deflector` | 20 (3L+P) / 8 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | coil x winding, coil y winding, coil x carrier, coil y carrier, coil interaxis spacer, … | Show the two physical stations, both transverse winding axes and the channel connection. |
| `condenser_stigmator` | 33 (3L+P) / 8 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | coil, slotted carrier, coil insulation, coil former, coil adhesive seat | Use an end-on view for winding directions and a section for the stations and carriers. |
| `descan_deflector` | 0 (all columns) | Control channel | Identify the physical host `image_diffraction_deflector` on selection. |
| `image_diffraction_deflector` | 20 (3L+P) / 8 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | coil x winding, coil y winding, coil x carrier, coil y carrier, coil interaxis spacer, … | Assign image/diffraction and descan labels to the same physical stations. |
| `objective_stigmator` | 8 (all columns) | coil | Use an end-on view for winding directions and a section for the stations and carriers. |
| `diffraction_stigmator` | 8 (both recording modules) | coil | Use an end-on view for winding directions and a section for the stations and carriers. |

## Probe corrector and adapter

43 keys. [Subsystem recommendations](README.md#correctors-and-steering).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `adapter_lens_upper_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `adapter_lens` | 0 (3L+P, 3L+PI, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `adapter_lens_excitation_coil` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `adapter_lens_housing` | 5 (3L+P, 3L+PI, 3L+PB) | body, upper cooling plate, upper coil insulation, lower coil insulation, lower cooling plate | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `adapter_lens_yoke` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `adapter_lens_lower_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_qph2_quadrupole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_dph2_coils` | 13 (3L+P, 3L+PI, 3L+PB) | x winding, y winding, x carrier, x radial cover, interlayer bond, … | Expose the orthogonal windings in their local frame and label the electrical host. |
| `probe_dph2_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Identify the physical host `probe_dph2_coils` on selection. |
| `probe_hp2_hexapole` | 25 (3L+P, 3L+PI, 3L+PB) | return yoke, pole pin, coil, coil former, coil core bedding | Expose the six pole tips, copper packs and common return in an end-on section. |
| `probe_tl22_lens_upper_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_tl22_lens` | 0 (3L+P, 3L+PI, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `probe_tl22_lens_excitation_coil` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `probe_tl22_lens_housing` | 5 (3L+P, 3L+PI, 3L+PB) | body, upper cooling plate, upper coil insulation, lower coil insulation, lower cooling plate | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `probe_tl22_lens_yoke` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `probe_tl22_lens_lower_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_dp22_coils` | 13 (3L+P, 3L+PI, 3L+PB) | x winding, y winding, x carrier, x radial cover, interlayer bond, … | Expose the orthogonal windings in their local frame and label the electrical host. |
| `probe_dp22_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Identify the physical host `probe_dp22_coils` on selection. |
| `probe_hpc_hexapole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_qpc_quadrupole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_dp21_coils` | 13 (3L+P, 3L+PI, 3L+PB) | x winding, y winding, x carrier, x radial cover, interlayer bond, … | Expose the orthogonal windings in their local frame and label the electrical host. |
| `probe_dp21_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Identify the physical host `probe_dp21_coils` on selection. |
| `probe_tl21_lens_upper_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_tl21_lens` | 0 (3L+P, 3L+PI, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `probe_tl21_lens_excitation_coil` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `probe_tl21_lens_housing` | 5 (3L+P, 3L+PI, 3L+PB) | body, upper cooling plate, upper coil insulation, lower coil insulation, lower cooling plate | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `probe_tl21_lens_yoke` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `probe_tl21_lens_lower_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_qph1_quadrupole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_dph1_coils` | 13 (3L+P, 3L+PI, 3L+PB) | x winding, y winding, x carrier, x radial cover, interlayer bond, … | Expose the orthogonal windings in their local frame and label the electrical host. |
| `probe_dph1_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Identify the physical host `probe_dph1_coils` on selection. |
| `probe_hp1_hexapole` | 25 (3L+P, 3L+PI, 3L+PB) | return yoke, pole pin, coil, coil former, coil core bedding | Expose the six pole tips, copper packs and common return in an end-on section. |
| `probe_hpol_hexapole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_qpol_quadrupole` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `probe_dp11_coils` | 13 (3L+P, 3L+PI, 3L+PB) | x winding, y winding, x carrier, x radial cover, interlayer bond, … | Expose the orthogonal windings in their local frame and label the electrical host. |
| `probe_dp11_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Control channel | Identify the physical host `probe_dp11_coils` on selection. |
| `probe_tl12_lens_upper_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_tl12_lens` | 0 (3L+P, 3L+PI, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `probe_tl12_lens_excitation_coil` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `probe_tl12_lens_housing` | 5 (3L+P, 3L+PI, 3L+PB) | body, upper cooling plate, upper coil insulation, lower coil insulation, lower cooling plate | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `probe_tl12_lens_yoke` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `probe_tl12_lens_lower_pole` | 1 (3L+P, 3L+PI, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `probe_dp12_scan_deflector` | 0 (3L+P, 3L+PI, 3L+PB) | Optical or branch reference | Use a plane marker beside the two physical AC stations. |

## Objective and mini condenser

15 keys. [Subsystem recommendations](README.md#objective-specimen-and-analysis).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `objective_upper_electronics_sleeve` | 1 (all columns) | Objective electronics sleeve | Cut away a sector to reveal which inserts share this physical sleeve. |
| `mini_condenser_upper_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `mini_condenser` | 0 (all columns) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `mini_condenser_excitation_coil` | 1 (all columns) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `mini_condenser_housing` | 1 (all columns) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `mini_condenser_yoke` | 1 (all columns) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `mini_condenser_lower_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `objective_upper_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `objective_lens` | 0 (all columns) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `objective_lens_excitation_coil` | 7 (3L+P) / 2 (2L, 3L, 3L+I, 3L+PI, 3L+B, 3L+PB) | upper, lower, upper winding inner insulation, upper cooling plate, upper end insulation, … | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `objective_lens_housing` | 1 (all columns) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `objective_lens_yoke` | 2 (all columns) | upper, lower | Trace the configured magnetic return and identify any unresolved joint. |
| `objective_aperture` | 1 (all columns) | Platinum user identified unverified | Show the blade, side drive and insertion state beside the lower pole. |
| `objective_lower_pole` | 1 (all columns) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `objective_lower_electronics_sleeve` | 1 (all columns) | Objective electronics sleeve | Cut away a sector to reveal which inserts share this physical sleeve. |

## Specimen stage and EDS

4 keys. [Subsystem recommendations](README.md#objective-specimen-and-analysis).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `eds_detector_system` | 18 (all columns) | Six Si dies, six insulating supports and six cold shields | Expose sensor, support and cold shield in one head. Show sightlines from the specimen to all six. |
| `sample` | 1 (all columns) | Specimen envelope | Highlight the specimen upper surface in an inset and put a selected grid under the specimen. |
| `sample_holder` | 3 (all columns) | tip, shaft, handle | Show how a compatible clip and grid fit into the existing tip recess. |
| `sample_stage` | 3 (all columns) | port, flange, body | Section the port and guide, and mark insertion direction and holder-specific tilt axes. |

## Image correctors

65 keys. [Subsystem recommendations](README.md#correctors-and-steering).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `image_dp11_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dp11_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dp11_coils` on selection. |
| `image_hpol_hexapole` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `image_qpol_quadrupole` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `image_tl11_lens_upper_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl11_lens` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl11_lens_excitation_coil` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl11_lens_housing` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl11_lens_yoke` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl11_lens_lower_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_dp12_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dp12_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dp12_coils` on selection. |
| `image_tl12_lens_upper_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl12_lens` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl12_lens_excitation_coil` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl12_lens_housing` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl12_lens_yoke` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl12_lens_lower_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_dph1_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dph1_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dph1_coils` on selection. |
| `image_hp1_hexapole` | 13 (3L+I, 3L+PI, 3L+B, 3L+PB) | return yoke, pole pin, coil | Expose the six pole tips, copper packs and common return in an end-on section. |
| `image_tl21_lens_upper_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl21_lens` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl21_lens_excitation_coil` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl21_lens_housing` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl21_lens_yoke` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl21_lens_lower_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_dp21_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dp21_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dp21_coils` on selection. |
| `image_dp22_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dp22_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dp22_coils` on selection. |
| `image_tl22_lens_upper_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl22_lens` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl22_lens_excitation_coil` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl22_lens_housing` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl22_lens_yoke` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl22_lens_lower_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_dph2_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dph2_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dph2_coils` on selection. |
| `image_hp2_hexapole` | 13 (3L+I, 3L+PI, 3L+B, 3L+PB) | return yoke, pole pin, coil | Expose the six pole tips, copper packs and common return in an end-on section. |
| `image_adapter_lens_upper_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_adapter_lens` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_adapter_lens_excitation_coil` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_adapter_lens_housing` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_adapter_lens_yoke` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_adapter_lens_lower_pole` | 1 (3L+I, 3L+PI, 3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_ish_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_ish_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_ish_coils` on selection. |
| `image_dsh_coils` | 4 (3L+I, 3L+PI, 3L+B, 3L+PB) | x winding 1, x winding 2, y winding 1, y winding 2 | Expose the orthogonal windings in their local frame and label the electrical host. |
| `image_dsh_deflector` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Identify the physical host `image_dsh_coils` on selection. |
| `image_dstg_quadrupole` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Control channel | Show the channel in the optical overlay and mark the physical host as unverified. |
| `image_sad_plane` | 0 (3L+I, 3L+PI, 3L+B, 3L+PB) | Optical or branch reference | Label the SAD plane as an optional optical overlay. |
| `image_tl31_lens_upper_pole` | 1 (3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl31_lens` | 0 (3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl31_lens_excitation_coil` | 1 (3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl31_lens_housing` | 1 (3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl31_lens_yoke` | 1 (3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl31_lens_lower_pole` | 1 (3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl32_lens_upper_pole` | 1 (3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_tl32_lens` | 0 (3L+B, 3L+PB) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `image_tl32_lens_excitation_coil` | 1 (3L+B, 3L+PB) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `image_tl32_lens_housing` | 1 (3L+B, 3L+PB) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `image_tl32_lens_yoke` | 1 (3L+B, 3L+PB) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `image_tl32_lens_lower_pole` | 1 (3L+B, 3L+PB) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `image_hp3_hexapole` | 13 (3L+B, 3L+PB) | return yoke, pole pin, coil | Expose the six pole tips, copper packs and common return in an end-on section. |

## Diffraction and projection lenses

25 keys. [Subsystem recommendations](README.md#projection-detection-and-spectroscopy).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `selected_area_aperture` | 1 (both recording modules) | Platinum user identified unverified | Show the detachable selector at its recorded axial position, with the active hole enlarged. |
| `diffraction_lens_upper_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `diffraction_lens` | 0 (both recording modules) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `diffraction_lens_excitation_coil` | 1 (both recording modules) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `diffraction_lens_housing` | 1 (both recording modules) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `diffraction_lens_yoke` | 1 (both recording modules) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `diffraction_lens_lower_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `intermediate_lens_upper_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `intermediate_lens` | 0 (both recording modules) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `intermediate_lens_excitation_coil` | 1 (both recording modules) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `intermediate_lens_housing` | 1 (both recording modules) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `intermediate_lens_yoke` | 1 (both recording modules) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `intermediate_lens_lower_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `projector_lens_1_upper_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `projector_lens_1` | 0 (both recording modules) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `projector_lens_1_excitation_coil` | 1 (both recording modules) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `projector_lens_1_housing` | 1 (both recording modules) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `projector_lens_1_yoke` | 1 (both recording modules) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `projector_lens_1_lower_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `projector_lens_2_upper_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |
| `projector_lens_2` | 0 (both recording modules) | Optical assembly with physical children | Select the installed children together and open a longitudinal section. |
| `projector_lens_2_excitation_coil` | 1 (both recording modules) | Magnetic excitation coil | Expose the winding pack and give its current terminal a separate label from any cooling connection. |
| `projector_lens_2_housing` | 1 (both recording modules) | Magnetic lens housing | Use a removable section to reveal the winding, vacuum passage and adjacent interfaces. |
| `projector_lens_2_yoke` | 1 (both recording modules) | Magnetic lens yoke | Trace the configured magnetic return and identify any unresolved joint. |
| `projector_lens_2_lower_pole` | 1 (both recording modules) | Magnetic pole piece | Expose the pole nose, bore and face at its installed end of the gap. |

## Recording chamber and receivers

8 keys. [Subsystem recommendations](README.md#projection-detection-and-spectroscopy).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `projection_chamber_dpa_aperture` | 1 (both recording modules) | Fixed chamber-aperture envelope | Show the fixed restriction separating vacuum regions and expose its bore. |
| `haadf` | 1 (both recording modules) | Annular receiving slab | Expose the through-bore and side pickup. Distinguish the installed annulus from DF. |
| `flu_screen` | 1 (both recording modules) | Circular screen footprint | Show a screen frame, stowed outline and viewing direction appropriate to the selected chamber. |
| `post_projector_detector_chamber` | 1 (both recording modules) | Chamber envelope | Cut a window through the vessel to show receiver order, access and clear beam path. |
| `df` | 1 (both recording modules) | Annular receiving slab | Connect the annular face to a side carrier and a labelled signal outlet. Show its inserted state. |
| `bf` | 1 (both recording modules) | Circular receiving slab | Place the disk on a carrier and make its on-axis receiving face and readout clear. |
| `camera_recording_housing` | 1 (Plain) | Vacuum liner | Section the housing around the installed camera and mark the external connector. |
| `camera` | 1 (both recording modules) | Square active footprint | Distinguish the square active face, package and readout connector. |

## Optional energy filter

20 keys. [Subsystem recommendations](README.md#projection-detection-and-spectroscopy).

| Native part key | Mesh count by module | Current representation | Suggested schematic detail |
| --- | --- | --- | --- |
| `energy_filter` | 0 (Filter) | Optical or branch reference | Draw the installed branch direction and its entry point. |
| `energy_filter_bias_tube` | 1 (Filter) | Electrostatic bias tube | Show the bias lead and the tube bore in the same local coordinate frame. |
| `energy_filter_camera_deflector` | 2 (Filter) | electrode negative, electrode positive | Indicate how the electrodes select a downstream readout strip. |
| `energy_filter_dynamic_focus_electrostatic_quadrupole` | 4 (Filter) | electrode 1, electrode 2, electrode 3, electrode 4 | Give an end-on view of its four electrodes and their insulating mount. |
| `energy_filter_eftem_output_plane` | 0 (Filter) | Optical or branch reference | Use a switchable EFTEM output-plane outline. |
| `energy_filter_entrance_aperture` | 1 (Filter) | Platinum user identified unverified | Magnify the live entrance hole along the local incident path. |
| `energy_filter_multipole_01` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_02` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_03` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_04` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_05` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_06` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_07` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_08` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_09` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_multipole_10` | 1 (Filter) | Multipole carrier envelope | Locate the indexed carrier on the bent path. Label internal construction as unresolved. |
| `energy_filter_shutter` | 2 (Filter) | electrode negative, electrode positive | Mark the paired electrodes, insulating mount and electrical connection. |
| `energy_filter_slit` | 2 (Filter) | blade negative, blade positive | Show which edges set the selection width at the dispersive plane. |
| `energy_filter_tapered_prism` | 4 (Filter) | Two pole sectors and two wire channel guides | Section the pole gap on the curved path. Mark the bend and dispersion directions. |
| `energy_filter_zebra` | 8 (Filter) | Five scintillator strips, optical backing, housing and alignment guide | Enlarge the five strips and label the separate alignment region and optical backing. |

## Additional display geometry

The assembly synthesizes the following 14 vacuum-liner identifiers across the seven installations. A group can contain several annular sections. Each installation uses the applicable module drift groups and the shared liner where configured.

| Liner identifier | Native mesh count in the sampled assemblies |
| --- | ---: |
| `@vacuum_liner:@vacuum_drift:beam_blanker:nanopulser` | 3 |
| `@vacuum_liner:@vacuum_drift:column:c2` | 1 |
| `@vacuum_liner:@vacuum_drift:column:c3` | 1 |
| `@vacuum_liner:@vacuum_drift:column:c3_bcor` | 9 |
| `@vacuum_liner:@vacuum_drift:column:c3_imagecorrector` | 7 |
| `@vacuum_liner:@vacuum_drift:column:c3_probecorrector` | 1 |
| `@vacuum_liner:@vacuum_drift:column:c3_probecorrector_bcor` | 9 |
| `@vacuum_liner:@vacuum_drift:column:c3_probecorrector_imagecorrector` | 7 |
| `@vacuum_liner:@vacuum_drift:gun:feg` | 8 |
| `@vacuum_liner:@vacuum_drift:gun:feg_mono` | 9 |
| `@vacuum_liner:@vacuum_drift:gun:thermionic` | 6 |
| `@vacuum_liner:@vacuum_drift:project_and_recording_system:energyfilter` | 10 |
| `@vacuum_liner:@vacuum_drift:project_and_recording_system:noenergyfilter` | 9 |
| `@vacuum_liner:c1_c2_to_upper_objective` | 1 |

The optional specimen cell adds two material windows (`cell_window_upstream`, `cell_window_downstream`) and two reference guides (`specimen_cell`, `cell_sample_reference`) when inserted. The default installation has no active cell regions. The [support-grid catalogue](../../../configs/specimen_supports/catalog.toml) and [pumping diagram](../../../configs/environments/vacuum_map.toml) supply selectable specimen context and vacuum services outside the module part count.
