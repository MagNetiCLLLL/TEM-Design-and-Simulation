# Editable CAD and Software 3D Models

2026-10-11. This round delivers material solid models built to existing assembly dimensions. Public patents and manufacturer diagrams establish structural categories; proportions in undimensioned diagrams are not treated as measurements.

**Current full-library delivery:** `outputs/detector_corrector_cad_20261011/` adds EDS, the EELS camera and eight groups of image-corrector deflection windings. There are 187 component/module STEP files across 11 module types, plus global full-instrument assemblies with/without the energy filter. See the [detector and corrector update report](DETECTOR_CORRECTOR_CAD_20261011.md), including limits of unpublished dimensions, editing capabilities and interference validation. The text below retains the first CAD-stage record and is not the current version inventory.

**Later update on the same day:** The `outputs/cad_models_20261011` and full-library archive records below describe the first CAD delivery. Subsequently, image-corrector optical identities, local layout, default round-lens excitations and HP internal representation were corrected against CETCOR references. The two new image modules are delivered separately under `outputs/image_corrector_20261011/cad/`. Use that newer version for the image corrector; see the [CETCOR update report](IMAGE_CORRECTOR_UPDATE_20261011.md). Counts and the “optics unchanged” statement for the first full library apply only to that first CAD stage.

## Provided models

The delivery directory is `outputs/cad_models_20261011`. `step/` contains **154 STEP files**: complete assemblies for 11 configuration modules and their 143 optical units. There are 745 geometric solids, summed across alternative modules, not the component count of one microscope. Different column configurations, guns and recording systems must be selected separately, not all superimposed.

Evidence-supported pole shoes, bulk windings, yokes, housings, electrodes and insulators are retained separately within optical units. Unknown interiors retain clearly labelled configured envelopes only. Material categories are recorded through colour, names and STEP classification layers. Units are millimetres in configuration-module coordinates. Module STEP files preserve unit hierarchy and assembly positions. The software's complete assembly view then uses existing module installation positions; the energy filter retains its 90° branch at the column bottom.

| Component category | Model content in this round |
| --- | --- |
| Round magnetic lenses, including condenser, relay, objective, mini condenser and projector | Analytic revolved solids from existing material cross-sections, retaining upper/lower pole shoes, coils, yokes, housings and defined segmented structures |
| Main HP1/HP2 | Six pole pins, six bulk windings and one yoke confirmed by the original CEOS text; no unsupported housing or insulating supports |
| Other magnetic multipoles and stigmators | Bulk windings where coil topology is established; configured envelopes for unknown structures, without inferring physical pole count from optical harmonics |
| DPH, ordinary deflectors and AC scan coils | Shared component format; actual axial centres and radial nesting at the same Z retained, with coil material separate from drive channels |
| Tip, gun electrodes and accelerator | Actual Tip geometry, conductor electrodes of defined thickness and interstage insulation; modules lacking electrode thickness retain envelopes; emission width is not treated as metallic Tip radius |
| Aperture | Finite-thickness aperture plate using specified working bore and position; no invented value for an unknown bore |
| Electrostatic blanker | Two electrode plates retaining working gap and axial length, without unconfirmed insulating supports |
| Energy Filter | Bending poles, branch multipoles, electrodes, slit plates and drift tubes; detector active surfaces without specified thickness are not fabricated into fictitious solids |

## Viewing and exporting in the software

1. Restart the running software to load this round's code, then open **Physical Layout → 3D** for the full instrument.
2. Select a component and click **Open in 3D Parts**, or open the desired module directly in **Physical Layout → 3D Parts**.
3. Use **Section** to inspect interiors and material/component visibility controls to inspect individual items. Ordinary defined dimensions remain editable in the existing parameter interface.
4. Click **Export STEP…** to the right of **Save copy…** in the file bar and select an output parent folder.
5. The software creates a new `step-date-time` subdirectory and exports every optical unit plus the complete module in the background. Export includes the current unsaved draft and supplied runtime aperture/slit states; `catalog-input.json` preserves that input. Cancellation is possible before completion.

If no runtime slit gap is supplied, display/export uses the configuration's maximum mechanical-travel opening, not the current energy-selection width. Completion is reported only after STEP reread validation and input-identity checks pass.

New reconstructed material regions can be viewed but do not automatically become parameter domains for a material-field solver. The editor now supports STEP import, dimension scaling, holes and slots. These operations share parameters with native 3D, and export retains exact B-rep. See the [STEP editing instructions](CAD_STEP_EDITING.md).

## Editing and the meaning of synchronization

STEP contains analytic boundary solids editable in CAD software, not triangle meshes wrapped in STEP. Compatible CAD tools can select independent material solids, alter geometry or reassemble them.

STEP itself does not preserve the full modelling history. The project also provides `catalog.json` and generators storing cross-sections, Boolean constructions, placement transforms and material information for reconstruction. Software 3D views and STEP share component construction rules and existing parameters; edit supported parameters in the software and export again to maintain consistency. Externally modified STEP can be explicitly imported through **Import STEP…** as a new mechanical component or replacement for selected material solids. Source B-rep, original-file digest, name paths, colours and edits added in this program are embedded in TOML; lens excitation, material constitutive laws and detector acceptance are not automatically inferred.

STEP export uses a separate optional environment; the particle/wave environment neither installs nor loads OpenCascade:

```powershell
python -m venv environment/cad
environment/cad/Scripts/python.exe -m pip install -r requirements-cad.txt
```

Regenerate the complete model set from the current project configuration:

```powershell
.venv/Scripts/python.exe -m temsim.cad_catalog --output outputs/cad_models/catalog.json
environment/cad/Scripts/python.exe scripts/export_cad_models.py outputs/cad_models/catalog.json --output outputs/cad_models/step
```

## Evidence and model boundaries

Structural evidence is archived by component: [original literature, patents and figure-page catalog](references/cad-library/README.md), [common detail policy](references/CAD_DETAIL_POLICY.md), [round lenses](references/cad-round-lenses.md), [multipoles and deflectors](references/cad-multipoles.md), [gun, apertures and energy filter](references/cad-gun-filter-apertures.md). Each solid in `catalog.json` also carries `status`, `note` and `evidence`:

- `configured_geometry`: generated from currently defined dimensions; not manufacturer measurement certification.
- `engineering_reconstruction`: supported structural type, with some explicitly reconstructed internal dimensions.
- `configured_envelope`: only configured shape/envelope is reliable; internal structure remains unresolved.
- `mixed_material_cartridge_unresolved`: known shared-carrier shape retained, with internal material partitioning unconfirmed.

The final inventory contains 669 reconstructed solids, 29 configured geometries, 42 unknown-interior envelopes and 5 shared carriers without material partitioning. In particular, the latter three labels must not be interpreted as manufacturing accuracy or actual material volume.

No direct evidence sufficient to overturn specific existing external assembly dimensions was found, so CAD construction retains them. The electron source, excitations, optical centres, working aperture dimensions and propagation algorithms were not arbitrarily changed. Unconfirmed turn counts, cooling channels, screws, seals, material grades and B–H curves are not known inputs; these models are not manufacturing drawings or a basis for new material-field calibration.

For example, the `FEG_Mono` two-blade slit has no known working gap, so it retains a carrier envelope. Some detector packages, the specimen chamber and thermionic cathode likewise retain only configured external shapes. They are explicitly labelled in the inventory; this work does not establish all interior details.

## Validation results

All 154 STEP files were exported and reread, checking valid solids, positive volumes, preserved dimensions/volumes, name hierarchy, colours, material classification layers and millimetre units. Maximum boundary difference versus generated solids was approximately `3.81e-10 mm`, and maximum relative volume difference approximately `2.28e-9`; these are format round-trip errors, not physical dimensional accuracy.

Final manifests are `step/step-manifest.json` and `step-final-verification.json`. Following revisions to the common detail/evidence rules, all files were re-exported, reread and SHA-checked. Old interim acceptance manifests are excluded from the delivery archive.

Native previews are in `preview/native_material_previews.png`, rendered and captured offscreen by the actual `PartModelView`, not an image generator or hand-drawn illustration. Some outer components are hidden only to expose internal structure; see `preview/preview_manifest.json`.

Comparison with original TOML is in `assembly-comparison/README.md`, preserving original module/shared-definition text, SHA, per-component data and 18 same-scale two-column figures. It covers 11 modules, 493 component records and 204 unique keys: 362 native meshes are unchanged, 66 display geometries/representations change, 51 are nonmaterial channels, and 14 have no definable surface. Original inputs, identities, parent–child ordering and optical centres are retained; maximum full-instrument coordinate-transform error is approximately `4.55e-13 mm`, and applicable envelope/bore checks found no new negative margins. HP/DPH radial clearance remains `0.1 mm`.

Original objective coils are drawn over their parent's actual upper/lower material intervals, which already extend beyond their own equivalent axial segments. The report retains both initial numerical differences and correct physical intervals; coils were not changed to make comparison pass. These checks do not certify full-instrument 3D collision freedom or manufacturing tolerances.

Dedicated CAD geometry, GUI and CI-registration checks and related existing 3D/assembly regressions passed. The STEP backend was validated in a separate CadQuery environment and a separate GitHub workflow added; this does not claim a remote GitHub run. Unrelated long coherent-propagation calculations were not rerun.
