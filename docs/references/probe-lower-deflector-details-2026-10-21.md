# Review of Probe Corrector deflectors downstream of TL22

> Subsequent update: this document preserves the inspection baseline and validation records before repair of three copper-end contacts. TL22–DP22, DP21–TL21, and DP11–TL12 were subsequently addressed using explicit copper winding lengths, insulating seals at both ends, and positive assembly clearances; see the [end-insulation and clearance repair record](probe-contact-isolation-2026-10-21.md) for current structure and evidence. “Unresolved contacts” and “copper retained” below describe only the earlier stage.

This round covers DP22, DP21, DPH1, and DP11. Coordinates are column-module local Z in mm, with +Z downstream; default installation gives global Z by adding 450 mm. The main corrector's downstream order is TL22 → DP22 → DP21 → TL21 → HP1 (containing DPH1) → DP11 → TL12. Optical reference channels do not generate duplicate material.

## Structural evidence and changes in this round

[CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en), Fig.1, 4, 6 and associated text support flexible carriers rolled into cylinders, conductor-cover insulation, full-surface bonding, and orthogonal two-layer windings; Fig.10, 11 support installation around vacuum tubes and inside optical-element bores. The source does not supply OEM mounting methods, layer thicknesses, fittings, or end-seal dimensions for this project's individual DP/DPH components.

The main Probe configuration already had four copper regions, two carriers, two radial cover layers, one interlayer adhesive region, and two conductor-gap insulation regions, totaling 11. The two combined Probe + Image Corrector / B-COR configurations lacked those seven passive regions; this round synchronizes them to the same declared laminate. Original copper shapes, positions, effective optical thicknesses, and channel ownership are retained.

This round reuses DPH2's engineering end-seal recipe but extends it only on sides with available space, rather than unconditionally copying double-ended seals to every component.

| Component | Original copper extent | Added end seal | Complete material extent | Material regions |
| --- | --- | --- | --- | ---: |
| DP22 | 729.1–743.6 | Lower end 0.04 | 729.1–743.64 | 12 |
| DP21 | 772.6–787.2 | Upper end 0.04 | 772.56–787.2 | 12 |
| DPH1 | 843.6–858.1 | 0.04 at each end | 843.56–858.14 | 13 |
| DP11 | 889.0–903.6 | Upper end 0.04 | 888.96–903.6 | 12 |

End seals are an engineering inference within the insulating-cover structural family, **not a DPH/DP end-cap design explicitly supplied by the patent**. Source, inference status, and notes are saved with thickness. Original recipes without end seals still generate original materials; old DPH2 entries without a declared side retain double-ended behavior. The new `winding_support_end_cover_side` field explicitly accepts `upstream`, `downstream`, or `both`, rejecting unknown, empty, or non-string values.

## Actual mechanical clearances and retained issues

Complete material ID/OD for all four groups is 22.12 / 23.88. Actual continuous-tube ID/OD is 5.76 / 19.2, giving nominal radial clearance **1.46** from tube exterior to carrier interior; the configured 20 clear-beam declaration cannot replace actual tube ID. DPH1 has **0.06** radial clearance to the HP1 material bore of 24, and **1.76** to each end of HP1's axial envelope; none is manufacturing-tolerance certification.

The following three existing interfaces remain end-face contacts between copper winding packs and pole pieces:

| Interface | Contact local Z | Status |
| --- | ---: | --- |
| TL22 lower pole piece—DP22 upper end | 729.1 | Zero axial clearance; copper end uninsulated |
| DP21 lower end—TL21 upper pole piece | 787.2 | Zero axial clearance; copper end uninsulated |
| DP11 lower end—TL12 upper pole piece | 903.6 | Zero axial clearance; copper end uninsulated |

These contacts cannot be described as “positive clearance for every component” or “electrical isolation complete.” With current copper allocations and optical positions retained, directly adding 0.04 seals on these sides would enter the pole pieces; production validation rejects such configurations. This round neither shortens nor moves winding packs and does not infer pole-piece recesses, insulating washers, or mounting shoulders from unknown OEM structures. Resolving these contacts requires separately establishing actual installation constraints before changing the mechanical design.

## Dimensions, display, and validation boundaries

All materials retain the original `SolidSpec`, native triangle-mesh, and analytical STEP formats. Changing seal thickness updates the complete material boundary only on the selected side, preserving copper and effective optical lengths. Radial dimensions still derive from original windings/support layers, with no separately saved per-side radii that could drift apart. Independent copying/displacement moves all materials with the new center; saving, undo, and restoration continue through the unified editing workflow.

Neighbor checks use full material extents, not copper envelopes alone; explicitly enabling a contacting side with `both` cannot bypass overlap checks just because original copper remains within bounds. `tests/test_probe_lower_deflector_details.py` covers consistency across three configurations, unchanged original copper CAD/meshes, closed seals and parameter provenance, actual tube-wall/HP1 clearances, three copper-end contacts, invalid sides, rejection of enabling contacting sides, thickness editing/saving, and independently copied/displaced parts. Original DPH2 and support tests retain unique assertions and update material-region counts for this round.

The 5 related test files for this section finished with **124 passed, zero failures, errors, or skips**, recorded in `outputs/probe-lower-details-20261021/deflector-tests-final.xml`; after tube-clearance checks switched to configured production `tube_solids`, the 45 dedicated cases passed again in `deflector-production-tube-final.xml`, without double counting. The first 10 atomic-save cases encountered sandbox file-permission restrictions; backend reruns with normal permissions passed without changing atomic-write logic. Production edit probes also exercised eight material dimensions per group: **all 32 changed corresponding material shapes and retained optical fields**; OD edits synchronizing mechanical-OD metadata in control records are expected coupling. Probe records are in `deflector-dimension-probes.json` in the same directory.

These validate reconstructed material geometry and software behavior, not OEM manufacturing drawings or real-instrument insulation/vacuum/thermal/magnetic performance. Added seals, carriers, and adhesive layers are passive regions and add no magnetic field sources.
