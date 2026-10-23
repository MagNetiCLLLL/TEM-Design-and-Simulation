# Conventional side-entry TEM holder, goniometer, and column entrance

Review date: 2026-10-11. Scope: room-temperature single- or double-tilt specimen holders in the Titan/Spectra-compatible structural family, for reconstruction of the current CAD assembly. Public sources establish side-entry topology, the 3 mm specimen format, and the order of magnitude of the slender shaft diameter; complete manufacturing drawings for a conventional Titan/Spectra holder were not obtained. The engineering allocations below must therefore not be presented as measured OEM dimensions or as vacuum/manufacturability certification.

## Sources and evidence levels

| Source | Verifiable location | Adopted here | Boundary |
|---|---|---|---|
| [Mel Build WO2025134741A1](https://patents.google.com/patent/WO2025134741A1/en) | Original Japanese PDF p10, printed p8, [0036]; [original page](cad-library/WO2025134741A1/page-10.png) | The text mentions both a slender shaft for FEI/now Thermo Fisher and a **φ7 mm holder shaft** | Patent for a specialized retractable/electrically biased tip; supports only shaft diameter within a compatible structural family, not full conventional Titan/Spectra dimensions |
| [US20100006771A1](https://patents.google.com/patent/US20100006771A1/en) | Fig.1/PDF p2; [0010]/p4; [0097]/p10–11 | Relationships among a conventional shaft, external handle, guide sleeve, column, and vacuum seal; **6–10 mm** shaft/front-end diameter design range | Not a specific Titan model; no shaft length, handle diameter, or thin-tip thickness; this diameter range does not mean the round shaft fits a narrower pole-piece gap |
| [Gatan Analytical Holders](https://www.gatan.com/products/tem-specimen-holders/analytical-holders) | 643/646 product table and tip description | Room-temperature single-/double-tilt analytical holders; tip preserves beam paths and EDS sightlines | The Be cradle is a local material in that product, not the material of the whole holder; required exterior dimensions are unpublished |
| [LBNL/NCEM physical inventory](https://foundry.lbl.gov/instrumentation/tem-holders/) | ThermoFisher/FEI entries 1–3 | Actual use of conventional FEI single-/double-tilt holders on TitanX and others; 3 mm specimens, clips, and spacers | Evidence of facility use, not OEM machining drawings |
| [Thermo Fisher CrystalPack](https://documents.thermofisher.com/TFS-Assets/MSD/Application-Notes/crystal-pack-module-application-brochure.pdf) | PDF p2, Figs.1–2 | The β axis may be noneucentric and compensated by other stage axes; holder-specific calibration is required | Does not justify assuming exactly intersecting tilt axes or supply mechanical dimensions |
| [Thermo Fisher Spectra Ultra](https://www.thermofisher.com/nl/en/home/electron-microscopy/products/transmission-electron-microscopes/spectra-ultra-tem.html/1000) | S-TWIN/Ultra-X product descriptions | A wide-gap objective can be combined with an analytical double-tilt holder; holder shadowing affects EDS acceptance | EDS solid angle cannot be inverted to obtain holder or insertion-port geometry |

Two easily misused original patents are also saved: the circular specimen carrier clamped by dual actuators in [FEI US20080173813A1](cad-library/US20080173813A1/summary.md) is not a complete long side-entry holder; [EP3843120A1](cad-library/EP3843120A1/summary.md) concerns another diffraction holder with an integrated goniometer and gives values such as 200–300 mm total length, but its architecture differs and those values are not transplanted into Titan defaults. They provide mechanism distinctions and scale comparisons only.

Original PDFs, a small selection of unredrawn whole-page images, authors/applicants, publication dates, URLs, and SHA-256 values are in the [CAD reference index](cad-library/index.json). The Gatan HTML download was rate-limited, two datasheet addresses returned HTML, and downloading the original Spectra page file was refused. Statuses are recorded; returned pages were not disguised as PDFs. Web facts were checked against search content from the original institutions' pages.

## Correspondence with the original TOML

The [research input snapshot](cad-library/stage-holder-config-comparison.json) retains SHA-256 values and original `sample`, `sample_stage`, and `objective_lens` rows for four `C3*.toml` files; it does not overwrite subsequent implementation records.

- `sample_stage.holder_insertion_axis = "x"`; the specimen is at the objective center. Local Z is 825.2 mm without a probe corrector and 1149.2 mm with one; this is not absolute Z for the full instrument.
- The original 650 × 500 mm transverse envelope and 5.4 mm axial span of `sample_stage` are assembly allocations, **not a solid plate filling that space**. The lateral entrance must be modeled separately; a broad plate or ring must not pass through both pole noses.
- The objective pole-piece gap is 5.4 mm and its current OD is 180 mm. A 7 mm round shaft cannot reach the central gap; the near-specimen section must become a thin tip, joining the shaft where the pole pieces flare outward.
- `sample.mechanical_outer_diameter_mm = 3` is the specimen disk OD; `length_mm ≈ 0.0001` is the 100 nm specimen thickness, **not the holder-tip or specimen-clip thickness**.
- `sample_stage.vacuum_inner_diameter_mm = 5` is the original axial main-beam-path dimension and cannot be used as the port diameter for a 7 mm side-entry shaft.

## Saved shared parameters and their status

These parameters have been saved to [`configs/subassemblies/sample_stage.toml`](../../configs/subassemblies/sample_stage.toml). Column variants reference them through `[[subassemblies]]`, anchored to the existing `sample` center, avoiding multiple independent dimensions. The table describes saved parameters and **does not replace separate collision-check results**. All X positions are in mm relative to the specimen center; positive X points toward the external handle, insertion is along −X, and retraction along +X. This is the project's coordinate choice, not manufacturer-specified handedness.

| Section | Saved value | Evidence/status |
|---|---:|---|
| Specimen disk | Diameter 3 | Actual FEI/Titan usage evidence agrees with the existing TOML |
| Long shaft | Diameter 7, X = 18…235 | Diameter supported by the FEI-compatible structural-family evidence above; endpoints are engineering allocations in the current assembly |
| Thin tip | Width 5, thickness 1.2 | Engineering reconstruction to fit the current 5.4 mm gap; no OEM tip-thickness evidence |
| Specimen locating seat | Diameter 3.1, depth 0.65 | Recessed from the tip's +Z face, with its floor at specimen-center Z − 0.05; all engineering values |
| Tip beam-through hole | Diameter 2.4 | Separate from the locating seat, leaving a radial support ledge; engineering value |
| External handle | X = 235…315, OD 40 | Represents only an ordinary operating handle, with no added heating, cooling, or electrical-control equipment; no OEM dimensional evidence |
| Guide sleeve | X = 87…180, OD 12, ID 9 | Engineering envelope based on shaft diameter and the current column-shell position |
| External column flange | X = 90…100, OD 70 | Engineering allocation; no claim about actual screw count, flange standard, or seal-groove dimensions |
| External stage body | X = 100…170, OD 50 | Simplified reconstruction of only the outer goniometer housing; internal bearings/drives unknown |
| Main-shell lateral entrance | Diameter 12.5 | Accommodates the OD 12 guide sleeve; a lateral opening separate from the Z beam bore |
| Mechanical retraction | Default 0, allowed 0…400 | `holder_retraction_mm` / `holder_maximum_retraction_mm`; purely mechanical display travel, an engineering allocation |

The current tip joins a round end of radius 2.5 mm to a flat arm 5 mm wide and 1.2 mm thick, transitioning to the round shaft at X = 18 mm. A solid ledge remains between the 3.1 mm locating seat and 2.4 mm through-hole; the entire seat is not cut through. No retaining plate, real TEM-grid thickness, or internal double-tilt drive was invented. A seat floor 0.05 mm below the optical specimen plane does not mean the software has modeled a 50 μm thick grid; actual clamping/support contacts or tilt collisions require user-confirmed grid/clamp definitions.

`sample_stage`'s `length_mm=5.4` continues to represent the original beam-path gap; the actual external stage Z range is given by `actual_z_extent_mm=70` and explicit lateral solids. The old 650 × 500 mm allocation is retained as `legacy_allocation_x_mm/y_mm`, not as solid-plate or new-housing dimensions. `sample_holder` uses a separate `side_entry_sample_holder` profile with `axial_vacuum_context_only=true`; its ±20 mm envelope must not create a new axial vacuum baffle.

Retraction translates only the mechanical holder; it does not automatically change `sample.inserted`, crystal orientation, or electron-propagation state. A retracted mechanical view therefore does not imply that the specimen has been removed from the physical calculation.

The current 315 mm allocation from specimen center to the outermost handle end does not directly contradict the acquired sources, but is not supported by the EP patent's 200–300 mm range either: reference planes and architectures differ. Unknown dimensions must not be completed as “real values” from schematic pixels or photographic proportions.

## Assembly orientation, entrance, and collision boundaries

Without tilt, electrons travel along project **+Z** through the XY specimen plane; the holder enters laterally along **X**, α rotates about the shaft axis X, and β rotates about a local axis perpendicular to the shaft inside the tip. The actual β eccentricity is unknown; any current eucentric simplification must be labeled rather than using software-compensation evidence to establish mechanical eucentricity.

A continuous X-directed insertion port is needed through the objective housing/vacuum boundary actually intersected, with the guide sleeve connected to it. A corresponding opening in the internal yoke is addressed only if that yoke actually intersects the path; real coils or entire magnetic cores must not be cut through for visual convenience. The main Z beam path must remain open and the specimen seat must retain visible support.

Subsequent collision review must at least distinguish the thin tip versus both pole noses, the 7 mm shaft versus the tapered gap, guide sleeve versus housing opening, handle/stage body versus main housing, and holder obstruction of EDS sightlines. A 9 mm sleeve bore around a centered 7 mm shaft provides only 1 mm clearance on each side, not guaranteed clearance under every stage translation or tilt; α rotation of the round shaft itself leaves its axis unchanged, while β must not incorrectly rotate the entire shaft.

This reference constrains CAD provenance and mechanical topology only. Adding these solids does not automatically establish integration of holder shadowing, EDS solid angle, actual specimen microstructure, or stage-motion calibration into particle/wave solvers; those computational interfaces require separate descriptions.

## Native-interface previews

[`scripts/preview_stage_holder.py`](../../scripts/preview_stage_holder.py) creates an actual Qt `PartModelView` and captures the widget; it changes only camera, section, and visible components, not meshes or postprocessed image compositions. Outputs under `outputs/stage_holder_20261011/preview/` include old/new assembly comparisons, near-specimen sections and standalone parts, and overall sections. `preview_manifest.json` records actual models, material categories, source-file and mesh SHA values, camera, and display omissions.

Old configurations are loaded independently from `outputs/stage_holder_20261011/originals/configs/instruments/column/`, without the new shared stage file; the new model does not fill in previously undefined holders or side ports. Comparison views omit the old large stage envelope without solid definitions and explicitly show its original allocation values. Previews hide other optical units and EDS components solely to inspect these mechanical relationships; they are not evidence of collision-free operation for every component in the full column.
