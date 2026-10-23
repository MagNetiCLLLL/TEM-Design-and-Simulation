# Beam Deflector internal locating-structure review (2026-10-20)

This round covers `beam_deflector` in the default **FEG / C3 + Probe Corrector / No Energy Filter**
assembly. It retains the original two-stage BSh/BTlt physical host, without new optical coils or separate
Shift/Tilt mechanical components. Coordinates are column-module local millimeters; after installation, the module origin is global Z=450 mm.

## Sources and model boundaries

[Nikon US5994704A](https://patents.google.com/patent/US5994704A/en), Figs. 1–2
and the locating-hole description provide concentric insulating carriers, a spacer sleeve, and radial locating pins; circumferential slots
in the outer carrier and spacer permit angular assembly adjustment, while the inner carrier has round holes. The text also explicitly allows all-round-hole versions. Fig. 4 supplies an insulating
housing version. The old model therefore did not lack all supporting structure: it already contained four copper winding packs and six passive regions per stage,
for 20 material regions across both stages.

The [JEOL two-stage deflection-system description](https://www.jeol.com/words/emterms/20121023.075958.php)
supports the functional category of coordinated two-stage beam shift/tilt control, but cannot determine internal manufacturing dimensions for a particular microscope.

This model selects the supported slotted locating-hole structural family; the **6° total slot-centerline sweep is an engineering allocation, not
an OEM measurement**. Rounded-slot machining geometry, the 0.6 mm locating pins, carrier radial dimensions, and assembly positions are all
explicitly recorded model assumptions. The patent calls the outer layer X, whereas this model's existing outer layer is Y; the correspondence follows inner/outer structure
without swapping original winding packs, directional bases, or optical channels. Unknown external mounting, leads, actual turn counts, material
grades, and thermal/electrical performance remain uninferred.

## Changes in this round

| Item | Result |
| --- | --- |
| Physical stations | Local 577.5 / 602.5; global 1027.5 / 1052.5 mm, unchanged |
| Physical winding-pack thickness | Explicitly saves `mechanical_coil_length_mm=15`, removing fallback to effective interaction thickness; copper volume and existing effective thickness remain unchanged |
| New parameter | `deflector_support_registration_slot_sweep_deg=6`, meaning slot centerline ±3°; unit deg; passive geometry |
| Slotted regions | Outer Y carrier and interaxis spacer at each stage, four regions total; inner X carriers retain round holes |
| Locating pins | Existing 45° / 225° axes, diameters, and material regions unchanged |
| Material count | Still 8 copper + 12 passive regions; no duplicate host |

Slots combine round-ended radial holes with intermediate annular sectors; radial cutting tools extend beyond both material sides to avoid coplanar Boolean boundaries.
Tools are subtracted directly from carriers/spacers, preserving their geometric union, rather than first fusing them into a complex cutter. Actual
STEP reimport found that trimming seams from prefused curved surfaces caused volume-validation failures; the current construction passes
the original strict tolerances without relaxed checks. An absent slot parameter or explicit 0 preserves the old round holes.

Input validation rejects nonfinite values, booleans, negative sweep angles, and slot widths entering copper winding windows; save, undo, and
redo remain supported. Slot resizing is not interpreted as actual coil rotation or a new field excitation.

## Installation clearances and runtime validation

Checks use actual material solids and continuous-tube shapes, not nominal bores alone:

| Beam material to neighboring part | Minimum clearance |
| --- | ---: |
| Continuous vacuum tube, local OD19.2 | Analytical radial 2.9 mm; 144-segment mesh minimum distance 2.89931 mm |
| Current actual C3 Aperture hardware | 7.9 mm |
| Nearest Adapter Lens upper pole piece | 10.0 mm |

Boolean overlap volumes for these material combinations are all 0. The Beam nominal local Z570–610 and C3 Aperture
nominal envelopes touch at Z570, which does not mean their actual materials intersect. Material through-bore ID25 and the saved
local solver clear-beam ID5 are distinct constraints; neither changed.

The full instrument was reparsed separately from this round's saved before catalog and updated configuration, using the same 300 kV
STEM Nanoprobe Normal preset and nonzero deflection inputs. Values for 39 unique field sources, 13 finite deflection sources, 6080
valid vector samples, and 9 particles at 352 saved sections agree element by element; no recalibration was performed.
As a positive control, changing only upper-stage Beam X from 0.012 to 0.014 mrad gives maximum field difference
2.80088×10⁻⁷ T and maximum trajectory X difference about 266.976 nm, demonstrating that the check uses the actual response route.

Evidence: `outputs/post-c3-details-20261020/transport-verification.json`; reproduction script
`outputs/post-c3-details-20261020/verify_transport.py`. This checks consistency of current analytical magnetic fields, production electric fields, and
CPU propagation, not material eddy-current, thermal, or measured FEM validation.

Regression: `tests/test_beam_deflector_registration.py` (including an actual independent-CAD-environment
STEP write/reimport test for 20 material regions), plus existing ordinary-deflector, shared-field, and optical-propagation regressions.
