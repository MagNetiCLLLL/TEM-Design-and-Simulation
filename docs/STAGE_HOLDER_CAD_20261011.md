# Stage, standard holder, and assembly interference checks

This work updates mechanical CAD. The software's 3D view, rotating section, holder/tube-wall sections in the main 2D view, STEP export, and material-interference checks share analytic shapes. Existing lens strengths, optical positions, source parameters, and axial clear beam diameters were not adjusted.

## Components and parameters

Five column variants reference the same `configs/subassemblies/sample_stage.toml`, positioned at their respective original specimen centers.

| Part | Main dimensions (mm) | Notes |
|---|---|---|
| Standard holder shaft | Diameter 7; X=18…235 | The 7 mm value is supported by a patent for an FEI-compatible structural family, not confirmation of an OEM model |
| Thin tip | Width 5, thickness 1.2 | Passes through the existing 5.4 mm pole-piece gap; the thick shaft begins beyond X=18 |
| Specimen seat | Diameter 3.1, depth 0.65; beam bore 2.4 | Includes an actual supporting step, without inventing specimen-grid thickness or clamps |
| Outer handle | Diameter 40; X=235…315 | Engineering envelope of a standard external operating handle |
| Stage guide sleeve | OD 12, ID 9; X=87…180 | Clearance of 1 on each side of the centered shaft |
| Flange and stage body | OD 70, 50 | Internal drives/bearings are unknown; simplified housings are retained |
| Side opening in the objective housing | Diameter 12.5 | Open only on the +X side to accommodate the guide sleeve; the opposite housing wall remains |

Positive X points outward toward the handle. `holder_retraction_mm` is a mechanical withdrawal distance of 0…400, edited through the same parameters in 3D Parts; it does not automatically change `sample.inserted` or crystal orientation. A screenshot of mechanical withdrawal does not establish that the specimen has been removed from electron-interaction calculations.

In Physical Layout, select **Sample Stage / Goniometer** or **Side-entry Sample Holder** under **3D Parts** to inspect, edit, and export each separately; **3D** and **Rotating section** show the assembly. The schematic/engineering-reconstruction labels remain in the images.

See the [reference record](references/cad-stage-holder.md) for dimension sources and their individual limits of applicability. Original PDFs, notes, figure pages, and SHA records are in `docs/references/cad-library/`. Unpublished dimensions of the shaft length, thin tip, outer handle, flange, and stage body are explicitly labeled engineering reconstructions, not Titan/Spectra OEM manufacturing drawings.

## Assembly issues actually found and corrected

1. The old stage had only an allocated envelope, with no independent holder or side opening in the objective housing. This work adds a solid side-entry structure.
2. `vacuum_liner_segments` previously expanded all axisymmetric electric-field boundaries into solids, causing duplicate material to penetrate electrode, pole-piece, and detector envelopes. Only explicitly defined continuous tubes and intermodule drift tubes are now drawn as independent material; the reason for retaining/omitting each segment is recorded in the audit, without arbitrarily subtracting every colliding solid to force zero overlap.
3. The 19.2 mm continuous tube genuinely interfered with the 18 mm C3 pole-piece bore. The material tube narrows to an OD of 17.2 mm at C3, with a 10 mm transition on each side and nominal radial clearance of 0.4 mm. The actual minimum distance including the conical surfaces is 0.398015 mm; the maximum outer envelope and ID of 5.76 mm retain their original values.
4. The C2/C3 aperture plates lacked side slots into the tube. Side entries follow the declared aperture-plate shapes, with engineering clearance of 0.2 mm per side, retaining the rear tube wall.

The continuous-tube corrections are stored in the `mechanical_material_tube_*` fields under each column's `[geometry]`. These define only mechanical material; existing electric-field solution boundaries remain axisymmetric equivalents. This work does not claim three-dimensional electric-field, vacuum-sealing, high-voltage insulation, or manufacturing qualification.

## Validation and delivery

Final execution records are saved in `outputs/stage_holder_20261011/`:

- `originals/`, `tube-material/originals/`: original TOML files and SHA records for the two stages.
- `configuration-equivalence.json`: complete parameter comparisons for pre-existing non-stage components in five assemblies; boundary floating-point rounding of up to approximately 1.14e-13 mm is recorded.
- `tube-material/material-geometry-equivalence.json`: added mechanical metadata for the continuous tube; all old configuration fields are unchanged.
- `collision/inputs.json`, `collision/report.json`: frozen inputs and an actual OpenCascade BRep audit; AABB overlap is not substituted for material intersection.
- `cad/parts/`: 11 module types and 158 component units, totaling 169 separate component and module STEP files.
- `cad/assembled/`, `cad/assembled-energy-filter/`: complete dual-corrector assemblies in the same global coordinates, with/without the energy filter; both main files are named `resolved_column_assembly.step`.
- `preview/`: actual Qt-view screenshots and old-TOML comparisons with the same camera, not generated concept images.

The checks cover catalog assembly combinations and declared mechanical states. Configured envelopes with undefined internal structures are listed separately; they are not treated as known hardware, nor does their presence justify claiming interference-free behavior for arbitrary new dimensions, stage tilts, or unmodeled accessories. STEP retains analytic surfaces, editable solids, material leaves, and component ownership; the complete original parametric rules remain in TOML and shape definitions, rather than an OEM-native CAD feature tree.

### Final results

- **60 catalog assembly combinations + 5 holder withdrawal positions, totaling 65 cases**: actual OpenCascade material-intersection checks took 177 seconds; positive-volume overlap between defined material solids was **0**, and small overlaps within the tolerance threshold were also **0**.
- Holder withdrawals of 0 / 50 / 160 / 320 / 400 mm had minimum clearances to other defined materials of **0.8 / 1.0 / 1.0 / 137.555 / 217.535 mm**, respectively. These are five discrete positions, not certification of continuous motion, arbitrary tilt, or manufacturing tolerances.
- **Two categories of old internal structures remain incompletely defined**: the slit and C1 aperture envelopes in FEG+Mono, and the accelerator and anode aperture envelopes in Thermionic. Each affects 20 configurations, totaling 40 unknown-envelope intersections; internal dimensions were not invented and subtraction was not used to conceal the issue. Standard FEG combinations and the holder states in this work have no such unknown-envelope intersections.
- There were 169 module/component STEP files, 74 complete-assembly and unit STEP files without an energy filter, and 91 complete-assembly and unit STEP files with an energy filter: **334 files in total**, all actually exported and successfully read back. Complete assemblies include both correctors; module positions and the energy filter's transverse installation transform follow the analytic assembly inputs.
- The initial comprehensive regression run for this work had 262 passes and 4 failures. After fixing path context at the shared-TOML save entry point and outdated migration comparisons, the original 4 cases plus adjacent save/undo regressions, **10 cases in total, all passed**. In addition, 3 actual CAD readback tests and 8 OCC audit regressions passed. Initial and rerun logs were retained; initial failures were not rewritten as successes.
- Software previews are offscreen screenshots of actual Qt widgets. The separate stage, holder, specimen seat, pole-piece gap, and old/new configuration comparisons were inspected; no claim was made of operating the user's running desktop window.

The only related implementation change after completion of the 65 audits was the shared-file save-path fix; it did not alter CAD constructors or TOML. Subsequently re-resolved inputs for 65 assembly states and 76 modules matched the frozen inputs field by field; evidence is in `collision-post-save-inputs/comparison.json`, and OCC was not rerun. The two types of unknown envelopes are nested allocations in the configurations, not confirmed material collisions; nesting also does not establish internal clearance, as explained in `nested-envelope-interpretation.json` in that directory. The original audit identity is retained, with the subsequent implementation-hash explanation in `collision/post-audit-source-note.json`. This work did not rerun expensive electron-wave propagation; equivalence comparisons of complete parameters and optical settings for old non-stage components are recorded separately.
