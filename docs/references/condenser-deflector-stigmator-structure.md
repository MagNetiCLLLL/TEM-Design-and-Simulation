# Internal structures of Condenser Deflector / Condenser Stigmator

Review date: 2026-10-20. Scope: the current FEG + C3 + Probe Corrector + No Energy Filter configuration.
The positive axial direction runs from the electron gun toward the specimen. Structural parameters use mm; positions below are resolved project values, not manufacturer measurements.

## Sources and limits

- [US5994704A, Fig.1–4](https://patents.google.com/patent/US5994704A/en): Saddle coils in two orthogonal directions, concentric insulating support cylinders, an intermediate insulating cylinder, locating pins and an outer sleeve support the existing two-stage deflector structural family. They do not establish this model's dimensions, winding turn counts or external mounting method.
- [JEOL stigmator description, Fig.2–3](https://www.jeol.com/words/semterms/20201020.111014.php): Four coils form one basis, with the second group rotated 45° from the first. Used only to explain the existing eight-coil, two-channel topology.
- [Philips US4902940A](https://patents.google.com/patent/US4902940A/en): A vacuum-tube passage is retained at the center of the quadrupole coils, and two bases offset by 45° are described. Field excitation and current scaling were not changed on this basis.
- [CN102509689B, Fig.5, 6, 12](https://patents.google.com/patent/CN102509689B/en): In an objective stigmator embodiment, the end of a cylindrical support has eight slots containing coil formers, and enamelled-wire windings are fixed with adhesive. Only the structural principles of support/former/insulation/adhesive are adapted here to the existing Condenser Stigmator copper coils; this is not proof of its OEM structure.

No magnetic shielding sleeve, iron core, thread, separate aperture, lead or material grade was transferred from the patents.
The new passive solids do not generate new fields; magnetostatic, dielectric, thermal and vacuum material parameters remain unspecified.

## Findings and changes

The original Condenser Deflector already contained two stages with eight copper saddle windings, and X/Y carriers,
inter-axis spacers, outer sleeves and two locating pins per stage, totaling 20 material regions. The coils and supports need no reconstruction.
This change only explicitly saves its existing 10 mm physical winding length as `mechanical_coil_length_mm`,
so the mechanical length is no longer derived from `effective_thickness_mm`. The current effective field width remains 10 mm;
future changes to effective field width will not automatically stretch the physical coils.

The original Condenser Stigmator had eight copper windings with radial windows, representing X/Y bases at the same axial position.
An eight-slot support, an equivalent insulating jacket for each coil, open formers and adhesive seats add 25 passive regions,
for 33 regions in total. The copper solids, channel angles, installation position, 30 mm overall envelope and 6 mm effective field width are retained.
Normal/skew quadrupoles are still supplied by the same device; the Probe A1 fine binding is unchanged.

| New configuration field | Engineering value | Meaning |
| --- | ---: | --- |
| `stigmator_support_insulation_thickness_mm` | 0.1 | Equivalent insulation jacket around the entire copper pack, not individual-turn enamel |
| `stigmator_support_former_wall_thickness_mm` | 0.3 | Former walls inside the winding window and flanges on both sides |
| `stigmator_support_adhesive_thickness_mm` | 0.1 | Axial allocation for the adhesive seat at the slot bottom |

These three values are saved together with `engineering_reconstruction_not_oem`, sources and notes.
Support slots and former windows derive from the same copper winding geometry; each support group rotates with its corresponding basis.
Edits producing invalid wall thicknesses, closed winding windows, intersecting adjacent supports or envelope overflow are rejected before editing and saving.

## Vacuum tube and actual material clearances

Checks use the actual outer-diameter polyline and aperture-slot solids of the same continuous tube as Physical Layout,
rather than simply comparing nominal device bores. This continuous tube has an equivalent inner diameter of 5.76 mm; its material outer diameter
varies axially between 19.2 and 17.2 mm. The solver also applies the common constraint with each device's local vacuum aperture,
so the entire electron beam-passage domain does not uniformly have a 5.76 mm diameter.

| Device | Global Z / mm | Actual material inner/outer radius / mm | Maximum tube OD in the same region / mm | Minimum radial clearance / mm |
| --- | --- | --- | ---: | ---: |
| Condenser Deflector | Envelope 790–830; material 795–805, 815–825 | 12.5 / 23.5 | 19.2 | 2.9 |
| Condenser Stigmator | 1436.5–1466.5, center 1451.5 | Support 10 / 28 | 19.2 | 0.4 |

There is a 10 mm axial material gap between the deflector's two stages; the configured 20 mm `mechanical_inter_coil_gap_mm`
currently represents the stage-center separation and must not be interpreted as the clear gap between copper solids. The deflector's `vacuum_inner_diameter_mm=5`
is an active local beam-passage constraint in the solver; the resolved 790–830 mm section still uses 5 mm for particle/wave clipping,
even though the continuous tube material inner diameter is 5.76 mm. This equivalent limit cannot be used to infer support-to-tube-exterior clearance.
The Stigmator's 20 mm material bore surrounds the tube with 19.2 mm outer diameter; after combining its local vacuum configuration with the continuous tube,
the actual solver beam-passage inner diameter remains 5.76 mm, not 20 mm.
Condenser Stigmator is downstream of the corrector, in the upstream nested space of the objective assembly,
not between C2 and C3. Neither device was moved in this change.

## Background verification scope

Regressions checked that all eight original copper geometries remain unchanged, the 33 regions have closed boundaries, materials have no positive-volume overlaps, all eight winding windows
and the central bore remain open, there is no intersection with the actual tube solid, conversion from millimeters to global position occurs once, and edits can be undone and saved as copies.
The insulating jacket has an outer shell and an inner cavity boundary around the copper; Manifold separates these two oriented surfaces,
which must not be interpreted as two disconnected solids. An independent OpenCascade check found 1 valid solid with 2 shells,
volume 96.4236 mm³; the negatively oriented cavity was not counted as a separate material.

These are geometry and program-consistency checks of the saved engineering structure, not OEM assembly, tolerance or field-calibration validation.
