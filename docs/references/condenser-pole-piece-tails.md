# Condenser pole-piece tapers and the C1/C2 cartridge

The operator identified the loose cylindrical component in the three supplied photographs as the shared C1/C2 pole-piece cartridge, described the cone as a magnetic part of the pole, and located the working gap in the side opening. The photographs locate an internal taper on the imaged end. The seven catalog columns reproduce the cylindrical outer form. Their pole solids taper on their **outer** diameters and have straight inner bores. This is a material-section discrepancy even where the apparent opening diameters are similar. [Current cartridge and pole definitions](../../configs/instruments/column/C3_ProbeCorrector.toml) · [Native pole section](../../src/temsim/part_model_3d.py) · [Native cartridge section](../../src/temsim/cad_condenser_cartridge.py)

## The loose component in the photographs

| Photograph | Feature visible or identified | Geometric constraint |
|---|---|---|
| 1, side view | The cylindrical wall has a round radial opening. The operator identifies the narrow span inside the opening as the pole gap. The edges bounding that span are the facing pole surfaces. | The opening in the outer wall and the axial separation of the pole faces require separate dimensions. |
| 2, oblique end view | A flat annular end and a short entrance rim surround a shiny internal surface that narrows towards a smaller throat. | At least this end has an internal taper. The far shoulder, the opposite end's contour and the iron-to-sleeve interface are not established by this view. |
| 3, assembly view | The smaller hand-held cylinder at the bottom has a narrow outer collar and visible radial openings toward the gold-colored end. | The long external envelope is cylindrical. The adjacent large cylinder and the cone on the separate disc were excluded from measurements. The finish and marker lines do not identify the magnetic alloy or its boundaries. |

The following are **projected image ratios** for this particular loose part. No length scale, column orientation or independently measured diameter accompanied the photographs. In photo 1 the narrow span occupies about **0.28** of the projected width of the side opening. In photo 2 the large opening spans about **0.69** of the outer end-face diameter, and the small throat spans about **0.35** of the large opening. The two openings lie at different depths. In photo 3 the centers of the visible openings lie roughly **0.16–0.20** of the projected cartridge length from the nearer end. Perspective, the curved exterior and occlusion limit all of these comparisons.

**Image traceability.** The photographs were read in the supplied order, with EXIF display rotation applied. Their SHA-256 prefixes are `0a18552afc4e783b`, `3521645ff6dd82d1`, and `a86a1fdef3f1ee9e`. The first ratio used six nearby horizontal profiles near the center of the round opening (0.271–0.284). The entrance opening and outside rim were measured across their widest region (0.684–0.691), and the deeper throat was measured separately (throat / entrance opening 0.335–0.356). These spreads describe the selected image edges, not machining tolerances. No absolute depth or physical gap in millimetres was derived from the photos.

## Current installed sections

Distances are along the beam axis. Each taper in the table is on the **outside of an individual pole body**, under the larger assembly envelope. All listed bores are straight. All eight pole keys omit an explicit nose length and inherit `0.38 × pole length`. These numbers describe the simulator's existing reconstruction. [Section rule](../../src/temsim/part_model_3d.py) · [Pole orientation](../../src/temsim/pole_geometry.py)

| Component | Occurrence in seven columns | Per-pole length (mm) | Outside taper / straight remainder (mm) | Shoulder OD → face OD (mm) | Bore ID (mm) |
|---|---:|---:|---:|---:|---:|
| C1 member, face downstream | 7 | 50 | 19.0 / 31.0 | 60 → 40 | 19.2 |
| C2 member, face upstream | 7 | 60 | 22.8 / 37.2 | 60 → 40 | 19.2 |
| C3, upper and lower | 6 pairs | 76 | 28.88 / 47.12 | 92 → 36 | 18 |
| Mini-condenser, upper and lower | 7 pairs | 4 | 1.52 / 2.48 | 32 → 26 | 20 |
| Adapter lens in the probe-corrected path, upper and lower | 3 pairs | 11.2 | 4.256 / 6.944 | 66.33 → 40 | 20 |

The C1/C2 carrier occupies module-local *z* = 0–300 mm with ID 60 and OD 90.75 mm. C1 occupies *z* = 40–90 mm and its exterior narrows over 71–90 mm. C2 occupies 110–170 mm and its exterior narrows from 132.8 to 110 mm toward the other face. The modeled gap is 20 mm, centered one third of the carrier length from its nearer end. In the native CAD, the first transition from the carrier opening into each pole's Ø19.2 mm bore is an axial step. The carrier itself comprises three axial annuli with no radial openings. It is a mixed-material mechanical component. The pole bodies are designated magnetic material regions. The separate vacuum tube and its beam passage have their own dimensions. [Carrier partition](cad-round-lenses.md) · [Vacuum-tube dimensions](cad-vacuum-tube.md) · [Magnetic material roles](../../src/temsim/magnetic_circuits.py)

The corresponding model diameter ratios are 60/90.75 = **0.661** and 19.2/60 = **0.320**. Similar end-view diameters can result from axial steps or a continuous taper. The visible narrowing in photo 2 and the side openings in photos 1 and 3 are the distinguishing features. Comparing the gap station with photo 3 also depends on registering the same physical end faces. The photographs do not identify which end is upstream.

## Axial length of the internal taper

For this part, record the entrance rim, the shoulder where the inclined **inner** surface begins, and the throat where that inclination ends as distinct stations, measured from the same physical end face. The taper length is the absolute difference between the shoulder and throat depths. Record the diameters at both stations. For a straight conical section, its half-angle to the beam axis satisfies `tan(angle) = (shoulder diameter − throat diameter) / (2 × taper length)`. A straight entrance land, throat land and the gap between opposing poles each have their own axial extent. The oblique end view establishes narrowing but cannot establish a constant slope, axial length, magnetic continuity, or a matching taper on the second side. A depth profile or calibrated axial section of this cartridge would resolve those quantities.

Philips' CM12/STEM and EM 300 sections provide separate examples of condensers with shaped material around the beam. Their illustrated tapers do not supply dimensions for the photographed cartridge. JEOL separately identifies the pole insert, magnetic yoke and nonmagnetic spacers in its double-gap first-condenser example. The one photographed C1/C2 component does not establish the internal contours of C3 or the mini-condenser. [Philips CM12/STEM, Fig. 5, p. 278](https://pearl-hifi.com/06_Lit_Archive/02_PEARL_Arch/Vol_16/Sec_53/Philips_Tech_Review/PTechReview-43-1986_87-273.pdf#page=6) · [Philips EM 300, Fig. 8, p. 377](https://pearl-hifi.com/06_Lit_Archive/02_PEARL_Arch/Vol_16/Sec_53/Philips_Tech_Review/PTechReview-29-1968-370.pdf#page=8) · [JEOL US4468563A, Fig. 4](https://patents.google.com/patent/US4468563A/en#description)

For verification, the catalog manifests, native CAD pole sections and axisymmetric magnetic material masks were read for all 46 installed instances in the table. A 195 × 139 axial/radial grid per instance gave 1,246,830 matching classifications. This verifies the modeled material contours. The geometry audit made no field or probe calculation. [Catalog](../../src/temsim/assembly_catalog.py) · [Magnetic material mask](../../src/temsim/physics/axisymmetric_magnetostatics.py)
