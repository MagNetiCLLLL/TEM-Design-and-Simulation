# CETCOR image corrector: optical path, conjugacy, and assembly evidence

Review date: 2026-10-11. The scope is the classical two-hexapole CETCOR image corrector; the design is not obtained simply by reversing a CESCOR probe corrector. This document records inputs before modification; it is not a calculation report demonstrating successful correction. The CEOS official materials found in this review did not confirm `CETCOR2` as an independent product/hardware model, so no additional component set is introduced on that basis.

See the [update report](IMAGE_CORRECTOR_UPDATE_20261011.md) for the actual changes, default parameters, GPU results, and revised CAD after this review. The pre-modification tables below remain as original comparisons and no longer represent the current configuration.

## Original sources

| Source | Directly supported facts | File/location |
|---|---|---|
| Martin Linck, CEOS, How do aberration correctors work?, 2025-08-31 | Ideal/actual two-hexapole imaging paths and additional alignment channels; finite-length HPs produce both second-order and negative third-order effects | [Original PDF](references/cad-library/CEOS-Linck-2025/original.pdf), p22, 24, 26, 29; [original p26 image](references/cad-library/CEOS-Linck-2025/page-26.png); [official source](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/01_How-do-aberration-correctors-work-pdf.pdf) |
| Peng-Han Lu, Forschungszentrum Jülich, CEOS workshop, 2025-08-31 | CETCOR/CESCOR comparison; objective pre/post half-fields, specimen/BFP, and two families of reference rays | [Original PDF](references/cad-library/CEOS-Lu-2025/original.pdf), [original p2 image](references/cad-library/CEOS-Lu-2025/page-02.png); [officially hosted source](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf) |
| P. Hartel, H. Müller, S. Uhlemann, M. Haider, CEOS, Residual aberrations of hexapole-type Cs-correctors | TL1 magnification/HP strength adjust C3; intermediate-image position at TL21/TL22 affects D4; tradeoff between C5 and off-axis coma | [Archived page](references/cad-library/CEOS-CETCOR-residuals/original.html); [official source](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR) |
| CEOS CETCOR product page, German | Product dimensions and supported instruments | [Archived page](references/cad-library/CEOS-CETCOR-product/original.html); [official source](https://www.ceos-gmbh.de/de/produkte/temkorrektoren/cetcor) |
| US8841630B2 | Typical negative-magnification imaging between two HPs; classical midplane-to-midplane −I; opposite polarities and rotation in a magnetic doublet; explicitly includes non-unit-magnification variants | [Original PDF](references/cad-library/US8841630B2/original.pdf), p14, 16, 20, 22; [patent](https://patents.google.com/patent/US8841630B2/en) |
| H. Müller et al., Physics Procedia 1 (2008), 167–178 | CETCOR model, two-HP symmetry, pole/coil/yoke structural family, and distinction between raw elements and combined alignment tools | [Bibliographic and availability record](references/cad-library/Mueller2008/source.json); [DOI](https://doi.org/10.1016/j.phpro.2008.07.093). The original PDF could not be downloaded during this review; only the public text of the authors' paper was checked, so original figures cannot be claimed as archived. |

Complete PDFs and unaltered full-page images of both original 2025 lecture documents were saved. Lu's optical diagram attributes its source to the CEOS residuals page, so these share one figure source and do not constitute two independent engineering measurements. The Linck lecture and original CEOS papers provide cross-support.

## Optical order along electron travel

After the specimen: **OL post half-field of the same objective → HPol/QPol/DP11 region → TL11 → DP12 → TL12 → HP1 (DPH1 at the same Z) → DP21 near TL21 → intermediate image → DP22/TL22 → HP2 (DPH2 at the same Z) → ADL → ISh → DSh/DSt → SAD image plane**.

This specifies effective-field order/overlap, rather than treating the outer edges of schematic symbols as mechanical end faces:

- `OL pre` and `OL post` are the upper and lower portions of the same objective spanning the specimen; the diagram does not show an additional second objective body in series.
- DP12 lies between the first transfer doublet's lenses and has a different effective plane from TL12. The diagram cannot establish their separation in millimeters.
- DPH1/DPH2 are drawn inside their respective HPs, supporting overlapping dipole/hexapole action regions. The public diagram does not uniquely establish independent DPH winding construction, reuse of pole pins, or specific radial dimensions.
- HPol/QPol/DP11 provide entrance alignment; adjacent symbols alone cannot establish a shared core or a separate housing for each. The classical image-corrector diagram lacks the probe-extension `QPH1/2` labels.
- DP21/DP22 lie on either side of the TL21/TL22 region; overlap in the detailed drawing does not establish which pole-piece slot contains a coil.

## Conjugacy conditions to validate

In the same transverse laboratory coordinates, mechanical slopes or normalized canonical momenta must each be used consistently. Write the first-order map at a given energy as

\[
\begin{bmatrix}r_2\\u_2\end{bmatrix}=
\begin{bmatrix}A&B\\C&D\end{bmatrix}
\begin{bmatrix}r_1\\u_1\end{bmatrix}.
\]

**Position-image conjugacy** checks `B≈0`; `A` gives magnification, rotation, and shear. **Angular/Fourier relationships** check `A≈0` from a specified starting plane to the observation plane; a few colored trajectories crossing the axis do not establish them. In thick fields, the coma-free aperture plane is near the focal plane but should not be identified exactly with a geometric BFP label without calculation.

The two plane families in the classical CETCOR diagram are:

1. Specimen-image family: specimen → intermediate image between TL11/TL12 → intermediate image near the TL21/TL22 symmetry plane → SAD. **The SAD aperture lies in a specimen-image plane here**; the word diffraction in its name does not mean that the aperture is installed in a diffraction plane.
2. Aperture family: objective coma-free aperture plane → HP1 midplane → HP2 midplane. TL11/TL12 match the objective and corrector aperture conjugates; magnification contributes to effective correction strength.
3. HP1→HP2: the midplane map is `−I` for the classical symmetric model with no net Larmor rotation and unit negative magnification. Report `B`, `C`, `A/D`, and net rotation separately, rather than only magnification. The patent also describes non-unit-magnification variants, so `−I` must not define every two-HP design.

An intermediate image at the TL21/TL22 symmetry plane is one condition for suppressing D4. Finite HPs, lens aberrations, field tails, and third-order combination terms must still be retained; satisfying a first-order relay does not establish that C3/A2/D4 targets have been met. Relative HP orientation, excitation, and transfer rotation must be matched together; assigning opposite signs to the two HPs while ignoring rotating coordinates is insufficient. C3-free, C5-free, and minimum off-axis coma settings do not always coincide.

## Pre-modification TOML comparison

Values below are in mm and use module **local Z**, not absolute instrument Z. In the combined module, all these image elements are shifted by +324 mm relative to the image-only module. Complete values and SHA records are in the [comparison record](references/cad-library/image-corrector-config-comparison.json).

| Element | Center in image-only | Mechanical/channel interval | Conclusion |
|---|---:|---|---|
| Specimen / objective assembly | 825.2 | 615.0–1035.4 | An objective spanning the specimen already exists |
| Additional image_ol_post_lens | 1062.9 | 1040.4–1085.4 | A separate downstream body cannot represent the diagram's OL post half-field; any independently supported relay/mini lens should be separately named and sourced |
| HPol / QPol / DP11 | 1092.9 / 1098.9 / 1104.9 | Each channel interval is 6 mm | Order is plausible for an entrance region; hosts are unconfirmed |
| TL11 | 1136.9 | 1113.9–1159.9 | Has an independent body |
| DP12 / TL12 | 1172.9 / 1172.9 | 1165.4–1180.4 / 1172.4–1173.4 | **Clearly inconsistent with the official optical order**: no public evidence supports coincident Z or TL12 being contained within DP12 |
| HP1 / DPH1 | 1202.4 / 1202.4 | HP 1180.4–1224.4; DPH 1187.4–1217.4 | Centers/axial containment are consistent with the diagram; actual windings/CAD are not established |
| TL21 / DP21 | 1256.4 / 1231.4 | TL 1238.4–1274.4 | DP21 is currently before the lens; the official diagram suggests proximity to the exit. Check effective fields/alignment responses rather than moving mechanical parts directly from symbol proportions |
| TL22 / DP22 | 1306.4 / 1281.4 | TL 1288.4–1324.4 | DP22 precedes the lens, consistent with the order |
| HP2 / DPH2 | 1346.4 / 1346.4 | HP 1324.4–1368.4; DPH 1331.4–1361.4 | Same evidence limitations as HP1 |
| ADL / ISh / DSh / DSt / SAD | 1386.4 / 1420.4 / 1444.4 / 1452.4 / 1476.4 | See comparison record | Main order is consistent; independent mechanical hosts for DSh/DSt are unconfirmed |

Only 20.5 mm separates the TL11 end and HP1 entrance; the original DP12 interval already occupies 15 mm. Retaining these end faces and DP12 leaves only 5.5 mm for an independent TL12, including all gaps. Other transfer lenses in this file are 36/46 mm long with a 20 mm pole gap; a TL12 body of the same specification cannot be added without rearrangement. A 1 mm effective plane is not a complete 1 mm round magnetic lens. First restore independent transfer degrees of freedom, then adjust local axial allocations under shared optical-path constraints. Any new envelope/coil dimensions must be labeled engineering reconstructions and imaging recalculated; millimeter dimensions must not be taken from CEOS schematic proportions.

## External dimensions and limits of solid detail

The manufacturer's German page explicitly states **H×B×T = 250×396×306 mm**, meaning height 250, width 396, and depth 306. It gives no interface/flange datum or explicit inclusion of external coupling lenses, so the complete OL→SAD optical path must not be forced into 250 mm. The current core from the HP1 entrance to HP2 exit is 188 mm; TL11 entrance to ADL exit is 290.5 mm; the additional OL post entrance to SAD is 436 mm. These use different boundaries and cannot directly be declared consistent or contradictory.

Müller2008 Fig5/6 supports finite-length HPs using pole pins, coils, yokes, and axial mirror plates for CETCOR/CESCOR designs; its example has a 154 mm yoke OD, 30 mm length, and 4.05 mm bore radius. The archived Müller2006 photograph of a six-pin body supports the same CEOS dual-hexapole structural family, but its 152 mm OD/4 mm radius belong to a different specific example. Public evidence supports choosing a generic six-pole-pin structure; exact pole faces, windings, material grades, and internal dimensions of current CETCOR hardware remain unsupported. The 2006 probe photograph must not be presented as an original image-corrector assembly image.

CAD can represent confirmed main structural families and TOML input envelopes; components such as DPH/QPol with only confirmed functions or field positions should be labeled generic reconstructions/unknown hosts. Unknown cooling holes, screws, brackets, independent QPHs, or new housings should not be invented to add detail. This document does not modify the calculation source, loaded specimen, field strengths, or geometry files.
