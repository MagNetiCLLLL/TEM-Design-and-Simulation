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

The probe HP1/HP2 shape and explicit mechanical-dimension update reviewed on
2026-10-18 is recorded in [Probe Corrector internal structure](cad-probe-corrector.md).
It refines the existing 13-solid construction using inward-tapered pole noses,
radial winding windows and saved independent dimensions. It does not change
the image-corrector geometry or claim a new field calibration.

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

## Ordinary TEM deflectors: rigid-carrier update, 2026-10-18

The subsequent [Beam Deflector registration review](beam-deflector-registration.md)
adds an explicitly saved angular-slot option to this same construction family.
Only the default column's Beam host selects it; the 20 regions, winding and
pin locations, field channels and other three hosts remain unchanged.

The four ordinary magnetic deflector hosts in the C3 + Probe Corrector column
now select `rigid_insulating_carriers_pinned` explicitly. This is a common,
source-backed engineering construction family, selected separately for each
saved host. It is not an identification of the manufacturer's internals.
Different optical functions alone do not establish different winding or
support architectures.

The existing two interaction centres and eight copper regions of each host
are retained. All positions below are column-local millimetres, not
source-absolute positions. Sizes are explicit engineering allocations within
the current physical envelopes; none were measured from patent drawings.

| Host | Two interaction centres (mm) | Support decision and allocation | Coverage |
| --- | --- | --- | --- |
| Condenser Deflector | 350 / 370 | Rigid insulating carriers in the existing larger winding envelope; 0.30 mm radial skin, 0.60 mm casing and 0.60 mm pin diameter | 8 copper + 12 passive solids |
| BSh/BTlt Beam Deflector | 577.5 / 602.5 | Same supported construction family, independently selected for the shared shift/tilt host; 0.30 / 0.60 / 0.60 mm allocations | 8 copper + 12 passive solids |
| AC Scan | 937.5 / 962.5 | Compact rigid carriers fitted to its existing winding envelope; 0.20 / 0.40 / 0.40 mm allocations | 8 copper + 12 passive solids |
| Image/Diffraction Deflectors | 1224.4 / 1249.4 | Compact rigid carriers; 0.20 / 0.40 / 0.40 mm allocations; the de-scan channel remains on this physical host. C3 + Probe Corrector placement updated by the Normal STEM finite-field study | 8 copper + 12 passive solids |
| De-scan control | Same as Image/Diffraction | Shared control, not a fifth physical host | 0 additional solids |

The FEG gun separately contains a two-station deflector at gun-local centres
402 / 418 mm, also eight copper solids. The EnergyFilter camera deflector
instead has two electrostatic electrode solids in its bent branch frame; it
must not receive a magnetic winding support by name alone. NoEnergyFilter
has no additional independently defined deflector in the inspected module.
The backend inventory is saved in
`outputs/probe_corrector_auxiliary_20261018/ordinary-deflector-inventory.json`.

Before this update, each station contained only four 120-degree bulk saddle
loops at fixed radial fractions of the saved envelope. Their region identities,
shapes, angular orientation and axial extents remain the comparison baseline.
The ordinary-deflector support profile does not apply automatically to the
gun, electrostatic filter parts, unselected instrument configurations or
Probe Corrector DPH/DP inserts.

| Primary evidence | Supported conclusion | Boundary for this model |
| --- | --- | --- |
| [JEOL, double-deflection system](https://www.jeol.com/words/emterms/20121023.075958.php) | Two axial deflection stages cooperate for alignment, illumination and STEM operations. | Confirms the optical layout class, not support dimensions or a universal winding construction. |
| [JEOL, de-scan](https://www.jeol.com/words/emterms/20160713.103817.php) | Downstream two-stage deflection can compensate the position/angle changes caused by illumination scanning. | The current software's shared Image/Diffraction host is a project configuration; the glossary does not prove that every microscope uses this sharing. |
| [Nikon, US5994704A, Figs. 1–4](https://patents.google.com/patent/US5994704A/en) | Concentric insulating carriers hold orthogonal saddle conductors, separated by an insulating cylinder. Two insulating radial pins register their relative angle; round holes are an explicit alternative to adjustment slots. Fig. 4 describes an insulating casing, whereas Fig. 2 prefers ferrite. | The model selects the insulating-casing variant. Carrier grades, dimensions and placement in this TEM are engineering choices. The Fig. 3 aluminium cylinder is a temporary manufacturing mandrel, not a retained part. |
| [Carl Zeiss, US6642525B2, Figs. 2–4, 8 and 12](https://patents.google.com/patent/US6642525B2/en) | Several winding arrangements and ferrite-ring stacks are described, including ceramic spacers. Fig. 12 supplies a TEM embodiment. | The TEM example uses a particular variable-axis arrangement. It does not establish that all beam-shift or scan coils use ring stacks. |

### Material representation and fit

Each station adds six passive regions: two rigid insulating carriers with
copper-shaped pockets, one insulating spacer, one insulating casing and two
insulating registration pins. The pins occupy 45-degree and 225-degree
radial directions at the station midplane, with matching round bores. Their
ends follow the carrier radial boundaries. These azimuths and all pin sizes
are allocated to avoid the existing copper and clear bore; they are not
manufacturer dimensions. No support extends the station's axial allocation.

`deflector_support_radial_skin_thickness_mm` sets the continuous radial
backing of each carrier. This keeps the pocketed carrier connected. It is
an engineering embedding of the existing bulk copper, rather than a literal
model of the patent's preferred through-wall electroplating process. The
old closed copper loops remain winding-package volumes: electrical terminal
gaps, individual turns, wire insulation and external leads are unresolved.

`deflector_support_casing_thickness_mm` and
`deflector_support_pin_diameter_mm` save the other two allocated dimensions.
Each selected host records
`deflector_support_geometry_status = "engineering_reconstruction_not_oem"`,
the primary URL in `deflector_support_source`, and its scope in
`deflector_support_source_note`.

With the saved values, Condenser and BSh/BTlt supports have a 25.00 mm material
bore and 47.00 mm outer diameter; AC and Image/Diffraction supports have a
22.32 mm material bore and 33.44 mm outer diameter. These dimensions describe
the added material assembly, not a replacement for the optical clear aperture
or the host's larger configured envelope. The four hosts contain 80 material
regions in total: the original 32 copper regions and 48 added passive regions.

The same TOML, `SolidSpec`, material-category, native 3D, edit/save and STEP
paths carry this structure. Insulating material categories do not establish
ceramic/polymer grades, breakdown strength or thermal performance. Internal
pins express the relative fixation of the concentric carriers; axial retention
of the complete assembly in the microscope remains undocumented. No extra
magnetic core, mounting screw, clamp, flange or field excitation is inferred.
Descan retains its existing Image/Diffraction ownership and receives no
duplicate coil or mechanical host.

### Backend verification for the saved configuration

The output location is `outputs/ordinary_deflectors_20261018/`. The frozen
baseline contains 32 copper regions, while the updated four-host catalog
contains 80 regions. `ordinary-deflector-comparison.png` shows the native
before/after sections; `ordinary-deflector-details.png` isolates the two
carriers and the spacer/pin arrangement. These are offscreen CPU renders,
without desktop automation.

- `focused-regression.xml`: 353 tests passed, with no failures or skips.
  Coverage includes ordinary and corrector supports, shared fields and
  dimensions, transactional editing, native geometry and parameter provenance.
  The old shared-deflector test fixtures were aligned with the existing
  Calibrate Descan (keep AC) contract; production calibration was not changed.
- `step/step-manifest.json`: four individual STEP files and one combined
  assembly passed analytic B-rep validity and STEP metadata/volume/bounds
  round-trip checks. All 80 material bodies are valid.
- `collision/report.json`: 89 bodies, including the four hosts, three adjacent
  material bodies and six scoped portions of the saved continuous tube;
  448 exact pair evaluations, zero volume intersections, 120 boundary contacts
  and no unresolved material pairs. The real tube profile and aperture-entry
  adapter are used. Tube material is included over each station's actual
  axial span and is not invented beyond the saved tube extent.
- `preservation-verification.json`: all 193 pre-existing full-column material
  records are unchanged. The only configuration changes are the 28 support
  fields on the four selected hosts; the 48 new passive bodies add no optical
  sources. Descan's shared host and all optical configuration values are intact.

These results validate the current engineering geometry and software paths,
not OEM construction, a material-field calibration or arbitrary future
mechanical poses. Reproduce the catalog and native previews with
`scripts/export_ordinary_deflector_models.py`, using the frozen
`before-catalog.json` as its baseline; the existing STEP exporter and exact
assembly validator consume its `catalog.json` and `collision-input.json`.
