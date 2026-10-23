# AC Scan Internal Registration Structure Review (2026-10-21)

This round reviews the two AC Scan coil stages in default `C3_ProbeCorrector.toml`, proceeding downstream from the Probe Corrector exit. It does not move coils, change excitation, recalibrate the scan pivot or change optical propagation. The other two Probe Corrector combinations lack the same rigid-carrier configuration, so this structure is not installed automatically.

## Evidence and added detail

[Nikon US5994704A](https://patents.google.com/patent/US5994704A/en), Figs. 1–2 and the registration-hole description, supports concentric insulating carriers, spacers and radial locating pins. Outer carriers and spacers may use elongated registration holes, with round inner-layer holes; Fig. 4 shows an insulating-housing version. The text also permits all-round holes, so the original round-hole model is a valid simplification of this structural family, not a model lacking all supports.

Each stage retains its original four copper windings, inner/outer carriers, spacer, housing and two locating pins. Registration holes in the outer Y carrier and spacer of each stage now become circumferential slots with round ends. Four passive regions change across both stages; the other sixteen remain unchanged. Inner X-carrier holes stay round, with pin axes at 45° / 225°.

The patent's X/Y names reverse this model's inner/outer-layer names. This round maps radial construction only, without exchanging calibrated field directions. New `deflector_support_registration_slot_sweep_deg=6` means a total 6° slot-centreline sweep, not coil rotation or field-direction rotation. Missing or 0 restores the old round-hole structure.

**6°, all dimensions and material grades are not OEM measurements.** The model remains `engineering_reconstruction_not_oem`. Pins represent relative internal-carrier registration, not verified external attachment. Temporary mandrels used during patent manufacture are not finished internal hardware. Leads, actual turns, inter-turn insulation, external attachment and thermal design lack sufficient evidence and are not conjecturally added.

## Saved dimensions and actual materials

Coordinates are column-module local millimetres; default installation origin is global Z=450 mm.

| Item | Current value |
| --- | ---: |
| AC envelope | local Z930–970; global Z1380–1420 mm |
| Two action centres | local Z937.5 / 962.5; global Z1387.5 / 1412.5 mm |
| Single-stage physical winding length / interstage gap | 15 / 10 mm |
| In-stage carrier radial cover thickness | 0.2 mm |
| Outer insulating-housing thickness / pin diameter | 0.4 / 0.4 mm |
| Declared beam-clearance bore / outer envelope | ID20 / OD54 mm |
| Actual minimum carrier inner diameter / maximum material outer diameter | ID22.32 / OD33.44 mm |
| Local continuous vacuum-tube outer diameter | OD19.2 mm |
| Analytic radial material-to-vacuum-tube clearance | (22.32−19.2)/2 = 1.56 mm |
| Analytic radial clearance to upper electron-optics sleeve inner wall | (59.2−33.44)/2 = 12.88 mm |

The OD54 envelope is not a solid cylinder, and the ID20 beam-clearance constraint is not the actual material inner diameter. The nearest upstream TL12 lower pole shoe has saved key `probe_tl12_lens_lower_pole`; TL1 in old descriptions is a historical abbreviation for this model lens, not a different lens. Regressions check this lens's material subparts, upper sleeve, Condenser Stigmator, Mini Condenser material subparts, objective copper windings and continuous vacuum tube.

## Dimension editing and physical boundaries

Carrier thickness, housing thickness, pin diameter and slot sweep generate native meshes and CAD from the same saved fields. Both editing entry points retain undo, rereading and validation. Invalid values and elongated slots intruding into copper-winding windows cannot be saved. Copying retains structure and parameter ownership; explicit replacement with a custom model disables the native carrier recipe, which is re-enabled upon restoring native geometry.

These passive support parameters add no magnetic-field sources. Copper windings and centres are unchanged; Descan and Image/Diffraction still share their original physical host. Winding end-face dimension controls must target actual `mechanical_coil_length_mm`, not total length used only as a derived envelope.

AC physical winding length, however, **is not an independent passive parameter**: runtime centres depend on `coil_length + gap`; saving also requires `length = 2*coil_length + gap`, action centres symmetric about the mechanical centre, and finite-field thickness no greater than winding length. Changing only 15→14 mm may remain as a geometry draft, but saving must reject inconsistent field thickness/layout. Related fields must be adjusted together and scan calibration rechecked. This differs from this round's passive registration-slot addition.

## Background validation

`tests/test_ac_scan_structure.py` covers differences in four slotted regions, retained inner round holes, closure/nonintersection of twenty regions, actual vacuum-tube/neighbour clearances, dimension linkage through both editing entry points, invalid inputs, copying, model replacement, save/reread and runtime action centres. Mechanical overlaps are checked using actual material solids rather than empty optical-parent models. Exact STEP and full-instrument propagation consistency evidence is provided by this round's joint AC Scan / Upper Objective validation record.

These results establish consistency of geometry and software boundaries in the current engineering model, not OEM manufacturing reproduction or actual insulation-voltage, vacuum, temperature-rise or eddy-current testing. Passive detail is not presented as newly solved magnetic-field performance.
