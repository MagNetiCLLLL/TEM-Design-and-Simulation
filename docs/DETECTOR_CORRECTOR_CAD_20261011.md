# Mechanical additions for EDS, EELS camera, and image corrector

The pre-change version for this round is `0b346ebab7e027f3c9b78ab380b1f25e0235bfcb`. The original 31 TOML files and SHA-256 values are saved in `outputs/detector_corrector_cad_20261011/originals/`; see `configuration-comparison.json` in its parent directory for field-by-field comparison.

## Models and evidence

| Object | Additions | Receiving/optical definitions and boundaries |
| --- | --- | --- |
| EDS | Schematic Si solids, insulating supports, and open cold shields for six heads; 18 material solids | Retains the old 2D heads' outer outlines and position algorithm. Six directions, 32.06°, and 4.45/4.04 sr still come from the sole EDS.toml; new mechanical disks do not redefine photon acceptance regions |
| EELS camera | Five indirect-conversion strips, shared optical backing, open frame; 7 material solids | Original strip front surfaces, widths/heights, spacing, and path coordinates are unchanged. Alignment remains a reference layer rather than an overlapping second sensor |
| image corrector | Eight separate winding hosts: DP11/12/21/22, DPH1/2, ISh, DSh; each has X/Y saddle pairs | Drive channels bind to actual hosts to avoid duplicate fields. DPH and HP share centers; thin straight runs lie between pole pins and end arcs lie beyond pin ends; HP bores, pole pins, and main coils are unchanged |

Detailed sources: [detector references and parameters](references/cad-detectors.md), [image-corrector coil references](references/cad-image-corrector-channels.md), and [original source library](references/cad-library/README.md). Sources include manufacturer datasheets, CEOS schematics, and original patent PDFs; OEM dimensions were not inferred from pixel proportions in undimensioned drawings.

Actual EDS wafer areas and distances are unpublished. New Si disks are mechanical representations inside the original engineering envelopes, **not calibrated photon-interception surfaces**. The old EELS housing definition had no solid dimensions; the new 30.672 × 6.8 × 6 mm outer envelope is explicitly labeled an engineering allocation. Material-layer detail retains the same simplification level as other modules; no circuitry, cooling lines, or unconfirmed brackets were added. HPol, QPol, and DSt lack sufficient evidence for separate mechanical hosts, so channel names alone did not create hardware.

## Consistency with optical calculations

Both image-corrector configurations were compared with pre-change records: round-lens, HP, and quadrupole field arrays at 809 axial samples are identical, as are original optical centers and effective field lengths. DPH has a mechanical allocation length of 33 mm and still a 30 mm effective field; DP11/21/22 copper windings are shorter, but original finite-field extents remain unchanged. Maximum relative difference of the full 4×4 first-order response is about 2.14e-9; reference-beam-center difference is about 1.35e-14 m. This comparison did not retune optical strengths or the electron source.

EDS material-thickness edits do not change angular acceptance. EELS housing-depth edits do not change strip front surfaces or `recording_mask`; increasing `path_center_mm` by 1 mm moves native geometry 1 mm along the branch tangent, and saving/reassembling increases actual `zebra_detector_d_mm` by 1 mm. Material positions and receiving-plane positions remain distinct concepts.

## In-application editing capabilities

| Operation | Findings in this round |
| --- | --- |
| Edit existing dimensions under Dimensions | New detector wall/depth fields are connected; shared material geometry rebuilds, with undo/redo and save/reload support |
| Physical position and tilt | Supported main-column optical hosts use the same coordinate transforms for display, persistence, and field registration; bound channels cannot move independently and separate graphics from fields |
| EELS receiving-path coordinate | Uses existing `path_center_mm`, synchronized to the actual computational plane on reassembly; no second receiving-position definition |
| Aperture selection | Fixed full-module 3D preview ignoring the current aperture selection, bringing it into agreement with standalone parts and STEP |
| STEP export | Fixed missing full-draft validation before export; drafts with valid dimensions but invalid assembly clearance are also rejected |
| Shape operations such as Box, scaling, holes, and slots | Native geometry editing/saving remain available; these do not automatically rebuild analytical electromagnetic fields. Unsupported custom features are currently explicitly rejected by STEP, not silently dropped |
| Add, copy, and move mechanical parts | Existing mechanical editing workflow retained; new mechanical solids do not invent optical fields or control channels |
| New material regions | Geometry parameters can be viewed/selected, but regions do not masquerade as new calibrated FEM material domains |

STEP and native 3D are generated from the same `SolidSpec` geometry. Current delivery is in `outputs/detector_corrector_cad_20261011/cad/`: `parts/` contains 187 files for all modules and optical units; `assembled/` and `assembled-energy-filter/` contain 82 and 100 files, respectively, for global full-instrument assemblies and their units without/with an energy filter. The total is 369 STEP files, including repeated module/full-instrument representations; each group passed actual reimport. The material catalog regenerated from final source matches export inputs solid by solid.

## Validation scope

Actual Qt-widget checks cover edits, undo/redo, saving copies, reassembly, aperture switching, STEP export validation, and receiving-plane synchronization. Native screenshots rendered by the software's `PartModelView` are saved in `outputs/detector_corrector_cad_20261011/preview/`; they are not hand-drawn schematics.

Collision validation used exact BRep Boolean intersections across 60 catalog assembly combinations and 5 holder extension positions, completing 3052 independent material-pair checks with 0 collisions among defined material solids. See `outputs/detector_corrector_cad_20261011/collision/report.json`. Old unresolved-material envelopes still have 40 positive-volume overlaps in two categories, FEG+Mono slit/C1 aperture and Thermionic accelerator/anode, unchanged from before; these uncertainties were neither added, removed, nor concealed. This is not full-instrument manufacturing certification and cannot guarantee collision-free arbitrary user edits.

In the first 330 editing/geometry regression tests, 7 failed due to old test assumptions or new-test registration. After correcting valid standalone-copy fixtures, current tab identities, aperture branch-coordinate/finite-plate-thickness expectations, and CI registration, all originally failing cases passed targeted reruns. Other already-passing cases were not repeated. Dedicated detector/corrector material and field-binding checks and actual Qt editing checks were also performed; all run records and optical comparisons are collected under the delivery directory's `verification/`, with initial failures retained.

This round validates mechanical solids, editing synchronization, and preservation of existing optics; it did not rerun long coherent propagation or claim new GPU performance or experimental quantitative calibration.
