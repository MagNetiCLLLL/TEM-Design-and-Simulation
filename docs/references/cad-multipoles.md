# Multipole mechanical reconstruction: primary-source ledger

Reviewed 2026-10-11. This ledger supports generic, parameterized mechanical
representations. It is not manufacturer CAD, a measurement of the installed
Titan/Spectra/CESCOR, or a magnetic-material field qualification. Patent figures
provide construction alternatives; their proportions must not be used to infer
millimetre dimensions for the simulator.

The [2026-10-19 TEM stigmator audit](tem-stigmator-hardware-audit.md) adds
Philips/FEI section evidence, Tecnai/Titan/Spectra/Iliad applicability, and
assembly-specific positions. Declared stigmator winding groups now use the
same structural basis angles as the optical tensor: four X-group coils at
`channel_x_angle_deg + 90*k` and four Y-group coils at
`channel_y_angle_deg + 90*k`, all at one axial station. For a non-45-degree
independent basis, bulk winding widths are limited by the nearest angular gap;
this geometric allocation never rescales excitation or effective field length.

## Evidence

The adopted HP geometry is specifically supported by [Müller et al. 2006](cad-library/Mueller2006a/简介.md): p.443 identifies six ferromagnetic pole pins, and Fig.3 on p.444 is a real hexapole photograph identifying poles, coils and yoke. This supports the main probe HP1/HP2 construction family; the paper's numerical dimensions are not substituted for this model. See [the current uniform-detail policy](CAD_DETAIL_POLICY.md). The other patent alternatives below do not by themselves authorize extra components in this instrument.

The image HP1/HP2 model uses an explicitly labelled **CETCOR family inference**.
The author text of [Müller et al. 2008, *Aberration-corrected optics: from an idea
to a device*](https://doi.org/10.1016/j.phpro.2008.07.093), Figs. 5/6, identifies
the CETCOR pole-pin, winding and yoke construction. The six-pin count is supported
by the documented CEOS family in the 2006 photograph, not by a visual inspection
of the CETCOR2008 mechanical figure: the latter original PDF was unavailable in
this review ([availability record](cad-library/Mueller2008/source.json)). Thus
the 13 bulk solids—six joined pole pins, six bulk coils and one yoke—are an
engineering family reconstruction, not verified OEM internals. They retain the
configured 180 mm outer envelope, 8 mm bore, 44 mm axial envelope and 30 mm pole
allocation from the existing effective-length input. This geometric allocation
does not establish a measured field length. Neither the paper's 154 mm example
nor the 2006 152 mm example replaces the saved model dimensions.

Only the named main image HP1/HP2 components and their explicit geometry-template
copies receive that construction. An arbitrary hexapole harmonic still retains
an unresolved envelope. No image DPH winding, QPH, mirror plate, insulation,
mounting or cooling detail is created from missing dimensions. Image DPH1/DPH2
are bound optically to the corresponding HP plane; that is not a claim of a
particular winding location or material host. The official optical evidence and
the transfer-lens topology correction are recorded in
[the image-corrector source review](../image_corrector_reference.md).

| Primary source | Figures and supported facts | Limits for this project |
| --- | --- | --- |
| [CEOS, US20090079531A1, *Multipole coils*](https://patents.google.com/patent/US20090079531A1/en) | Figs. 1–4 show flexible printed-circuit conductors, insulation and rolled layers. Figs. 5–7 show dipole coils, including two dipoles in different concentric layers; Figs. 8 and 9 show quadrupole and hexapole arrangements. Figs. 10 and 11 place the insert in a lens passage or recess. Copper conductors, optional magnetic cores and an outer return yoke are described. | Direct CEOS evidence for compact coil construction and possible radial nesting. It does **not** identify the exact DPH winding, turn count or dimensions in a particular CESCOR. Its 120-degree dipole example and six-coil hexapole are construction variants, not universal requirements. |
| [US20240371597A1, *Multipole element, image error corrector and particle beam system*](https://patents.google.com/patent/US20240371597A1/en) | Figs. 1 and 2 show a surrounding magnetic yoke, wound radial supports and inward pole pieces; Figs. 3–7 vary pole-face and tube geometry. The described vacuum-separating tube can be ceramic, including alumina, or PEEK. Inner pole pieces and outer magnetic supports can be separated by that tube. Figs. 8–11 distinguish winding configurations. | Supports a possible separated-core/pole-piece topology and explicit insulation/vacuum boundaries. This is a specific architecture; it does **not** establish that existing project components or CEOS use that arrangement. Material grade and permeability curves remain unspecified for this simulator. |
| [JEOL, US6548816B2, *Stigmator assembly*](https://patents.google.com/patent/US6548816B2/en) | Fig. 1 uses eight circumferential coils; Fig. 3 uses twelve. The same assembly can produce different multipole harmonics through the coil-current combinations, including quadrupole and hexapole components. | A stigmator's quadrupole action does not prove that it has only four physical coils. The diagrams establish coil/drive topology, not a complete housing, turns or OEM dimensions. |
| [US5481164A, *Variable axis stigmator*](https://patents.google.com/patent/US5481164A/en) | Figs. 1 and 2 show two quadrupole-coil sections rotated by 45 degrees. The description explains that their drawn axial separation is for clarity; they normally share the axial location. | Supports normal/skew quadrupole construction with radial/azimuthal overlap. An exploded drawing must not be interpreted as two sequential lenses. It is not a CEOS assembly drawing. |
| [US8373137B2, *High resolution energy-selecting electron beam apparatus*](https://patents.google.com/patent/US8373137B2/en) | Fig. 1 locates multipole groups before and after the dispersing prism. The description permits twelve-pole elements with independently excited poles, or separate windings for different components, to supply dipole, quadrupole, sextupole and higher terms. | Supports several functional channels sharing a physical multipole host in an energy filter. It is an optical arrangement, not a dimensional mechanical section of this simulator's filter. |
| [JEOL, US7015481B2, *Charged-particle optical system*](https://patents.google.com/patent/US7015481B2/en) | Fig. 7A shows twelve electrostatic electrodes; Fig. 7B adds magnetic pole pieces with excitation coils. Fig. 8 relates excitations to the resulting multipole potentials. | Useful alternative for a combined electrostatic/magnetic host. It does not imply that every magnetic corrector contains electrodes or that the project has implemented all illustrated harmonics. |

## Modeling decisions supported by these sources

The CAD description should distinguish physical host geometry from the field
components driven on that host. A twelve-pole host can generate a quadrupole
field; a thin concentric coil insert can generate a dipole without a separate
axial housing. Neither the number of functional channels nor the harmonic order
alone determines the number of physical pole pieces.

For a conventional magnetic multipole reconstruction, separately represented
parts may include inward pole shoes, radial magnetic cores, copper windings,
electrical insulation, a vacuum liner and an outer return yoke. A coreless
printed-circuit insert is a different supported option. These alternatives
must remain explicit; adding all features of all patents to one assembly would
invent an unsupported machine.

Use the current instrument's explicit axial envelope, bore and outer dimensions
as model inputs. Preserve existing optical field centres and aperture clearances.
If a structural detail has no independent dimension, label it as a generic
construction choice, validate that it fits the declared envelope, and do not
mark it as exact OEM geometry. No longitudinal scale should be measured from
the cited schematic figures.

The analytical field coefficients remain the authoritative optical inputs.
Displaying poles, copper and a yoke does not mean their material response has
been solved or calibrated. CAD changes alone must not create duplicate dipole,
quadrupole or hexapole excitations, silently change field strength, or replace
an existing finite-field provider.

## Unresolved evidence

Exact OEM pole-face contours, magnetic alloys, B-H curves, lamination strategy,
winding turns, wire cross section, cooling paths and mounting tolerances are not
established by this ledger. A material-field solver or manufacturing claim would
require those additional inputs. No source here certifies the precise internal
construction of the installed CESCOR DPH1/DPH2, QPH or HP units.
