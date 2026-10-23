# DPH2 axial end-cover review (2026-10-20)

The default C3 + Probe Corrector `probe_dph2_coils` originally had 11 material regions: four copper winding packs, five radial support/cover/bond regions and two inter-conductor insulation regions for the X/Y layers. Volume-equivalent filling around conductor sides and windows was already complete; these materials were not added again. This round adds only upper and lower axial end covers, bringing DPH2 to 13 regions; the other four DPH/DP insert groups remain at 11 regions.

The two Probe + Image Corrector / B-COR combination configurations previously contained only DPH2's four copper regions and had not inherited the default configuration's radial supports. The same radial supports and end covers are added to those two configurations, so DPH2 has 13 regions in all three configurations, with existing copper shapes and original parameter values unchanged. Other DPH/DP structures retain their previous form in each configuration; the synchronization scope is not expanded here.

## Evidence and modeling limits

[Fig.4 and adjacent text in CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en) describe insulating cover layers on flexible conductors and full-surface bonding of stacked layers, and discuss avoiding interlayer air gaps. This supports the structural categories of covering and bonding, but **does not establish that CEOS DPH2 uses this model's annular axial end covers**. Representing coverage of the axial copper-pack ends as two complete rings is an explicitly labeled engineering inference for the structural family.

`winding_support_end_cover_thickness_mm = 0.04` is an engineering dimension saved in configuration; `winding_support_end_cover_geometry_status = "engineering_reconstruction_family_inference"`, the original source link and notes are saved with it. No resin or cover-film grade is specified; material colors are not treated as dielectric, thermal, mechanical, magnetic or vacuum properties. No unverified host fastening screws, wiring, vias, turn-by-turn insulation or winding seams were added.

## Actual material extents

| Item | Result in this review |
| --- | --- |
| Copper winding packs and original 11 regions | All original shapes, materials and positions retained |
| Copper / mounting allocation axial extent | Local Z 687.25–701.75 mm, length 14.5 mm |
| Upper end cover | Local Z 687.21–687.25 mm |
| Lower end cover | Local Z 701.75–701.79 mm |
| Full axial material extent | Local Z 687.21–701.79 mm; global Z 1137.21–1151.79 mm in the default assembly |
| Full material inner/outer diameters | 22.12 / 23.88 mm; the through-hole remains open |
| Actual vacuum-tube material outer diameter | 19.2 mm; minimum radial clearance 1.46 mm |
| HP2 actual material bore diameter | 24 mm; minimum radial clearance 0.06 mm |
| Upper end of nearest TL22 material | Local Z 703.6 mm; minimum axial clearance changes from 1.85 to 1.81 mm |
| DPH2 optical-field effective thickness | Original 4 mm, independent of copper and passive end-cover lengths |

The end covers lie outside the original copper allocation, so installation cannot be checked using the original `length_mm` alone. `full_material_axial_extent` provides the complete material extent, used by host installation and adjacent-component axial-overlap validation; HP2 still contains the entire DPH2 insert. The new end covers contact the copper ends without occupying copper volume. These clearances use actual material solids in the default assembly, including the real outer contour of the continuous vacuum tube, rather than treating vacuum beam-passage parameters as material tube diameters.

## Editing and compatibility

Old recipes without the new end-cover fields continue to generate the original 11 regions. When enabled, thickness must be finite and positive, and complete engineering status, sources and notes are required; booleans, overflowing integers, unknown prefixed fields and incomplete metadata are rejected before geometry construction. End-cover thickness is exposed through unified dimension editing, with undo, redo and save support. Native end-face selection includes the parameter provenance of this dimension.

Independent copied components can still use this recipe. With an explicit replacement CAD base enabled, the native winding/support recipe is saved but disabled; restoring the `existing` base triggers revalidation. If dimensions edited during replacement no longer accommodate the original winding, restoring native geometry fails atomically without leaving a partially modified draft. Optical control channels are not drawn as a second set of physical solids.

## Background evidence

Targeted regressions in `tests/test_dph2_end_insulation.py` and `tests/test_deflector_support_geometry.py` cover unchanged original 11 regions solid by solid, end-cover volumes/bores/contacts, actual tube-wall and adjacent-component clearances, full-material-extent validation, metadata, invalid inputs and editing transactions.

Analytical annuli and sectors acquire different circumferential chords in independently generated coarse triangulations, so generic CAD triangulation alone can produce spurious contact intersections. This cannot be declared a pass by relaxing the solid-overlap threshold. For the four flagged pairs of existing materials, actual global-coordinate OpenCascade surface intersection volumes were all zero; native display constructs corresponding insulation using the original copper mesh boundaries, with volume residuals no greater than approximately `1e-13 mm³` in checks at 32, 48, 72, 96, 144 and 288 segments. Both the original coarse-mesh flags and exact analytical rechecks are retained in this round's output, without changing the original copper shapes.

`outputs/probe-hp2-tl22-details-20261020/transport-verification.json` compares the saved complete pre-change configuration with the current configuration, using actual analytical field providers and CPU particle propagation: 300 kV, STEM Nanoprobe Normal, C3 + Probe Corrector, nonzero DPH2 deflection, 9 particles and 352 saved planes. Passive materials in DPH2, HP2 and TL22 all changed; 39 independent field sources, 6080 valid vector samples and all trajectory/slope arrays were bitwise identical before and after. An active positive control increasing DPH2 X deflection from 0.012 to 0.014 mrad produced approximately `1.05033e-6 T` maximum field-component difference and `1.36498e-7 m` maximum X-trajectory difference, confirming that the comparison did not mistakenly use static or zero-field data. Original configuration-file hashes remained unchanged.

These results establish that the passive model did not change the current solver's existing propagation inputs; they do not establish FEM fields, thermal or vacuum properties, or real-instrument spherical-aberration performance of the engineering reconstruction. New field solutions require materials, excitations and boundary conditions to be specified separately.
