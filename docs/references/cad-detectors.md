# EDS and EELS detector material reconstruction

Reviewed 2026-10-11. These are source-backed construction families with explicitly
declared engineering dimensions, **not an Ultra-X or Zebra manufacturing model**.
Internal electronics, cooling services, shutters and detailed semiconductor
structures are outside this reconstruction. All lengths below are millimetres.

## Primary evidence and limits

| Source | Supported structure | Not established for this simulator |
| --- | --- | --- |
| [US8410439B2](https://patents.google.com/patent/US8410439B2/en), Fig. 1–3, PDF pp. 3–5; [archive](cad-library/US8410439B2/source.json) | Sample-facing SDDs around the objective, a cold shield, a ceramic substrate/support, and a windowless option. The preferred illustrated arrangement has four heads. | Not the six-head Ultra-X drawing; no current head radius, distance, taper or packing dimensions. The printed assignee is recorded verbatim as PEI Company; this ledger does not certify ownership. |
| [KETEK VITUS product information](https://www.ketek.net/wp-content/uploads/2025/06/KETEK-VITUS_Product-Information.pdf), PDF p. 3; [archive](cad-library/KETEK-VITUS/source.json) | A commercial SDD family specifies 450 µm Si absorption depth for H7–H150. | Not an Ultra-X die measurement. Its windowed package is not transferred into the saved windowless array. |
| [Thermo Fisher DS0510](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf), p. 2, revision DS0510-EN-07-2025; [archive](cad-library/Thermo-Iliad-Ultra-DS0510/source.json) | Zebra uses indirect detection with five independent 2048-pixel strips, 14 µm pixels, and a 256×2048 alignment area. Camera deflection selects strips. | Does not give scintillator composition/thickness, optical coupling design, package depth, strip height/pitch or measured alignment-area position. |
| [US7642513B2](https://patents.google.com/patent/US7642513B2/en), Fig. 1–2, PDF pp. 2–3 and description p. 10; [archive](cad-library/US7642513B2/source.json) | An indirect EELS construction separates the scintillator, vacuum viewport, optical transfer and optical image sensor. | This is a different device, not proof of Zebra's specific lens/fibre arrangement or material. Its circuit/cooling layout is not copied. |

Original public PDFs and unchanged page renders are archived with retrieval time
and SHA-256. Public access is not a claim of open licensing; rights remain with
their respective owners. No length is measured from a schematic's page scale.

## EDS: unchanged angular reception, separate mechanical illustration

`EDS.toml` remains the only source of segment count, azimuths, takeoff angle and
solid angles. The physical photon model still uses those angular inputs.
It does **not** trace acceptance against a newly invented finite sensor disk.
In particular, the existing 4.45 sr aggregate cannot be used to infer both die
area and distance; the old equivalent-cone illustration is not a literal array
of independent circular sensor faces.

`eds_head_geometry()` lifts the pre-existing 2D head construction into six
azimuthal positions. Let `r_tip` be the greater pole-tip radius, bounded below
by 2.5 mm; `R_objective` the parent outer radius; and `R_pole` the larger pole
outer radius. The old drawing rules are retained:

```
r_face = max(0.45*r_tip, 0.025*R_objective)
length = max(0.8*r_tip, 0.05*R_objective)
r_front = 1.25*r_face;  r_rear = 0.82*r_face
distance = (R_pole + 1 mm + r_front*sin(takeoff))/cos(takeoff)
direction = (cos(takeoff)*cos(azimuth), cos(takeoff)*sin(azimuth), -sin(takeoff))
front_centre = sample_position + distance*direction
```

Negative column Z is toward the gun. The head's material extends away from the
sample along `direction`. No source proves that the resulting distance is the
real sensor distance. Each head contains a Si disk, insulating support and
open-front tapered cold-shield shell. The die drawing is explicitly mechanical,
not a second active-area definition. No entrance window is introduced.

| Persisted mechanical field | Current engineering value | Meaning |
| --- | ---: | --- |
| `mechanical_sensor_thickness_mm` | 0.45 | Illustrative Si depth, family-scale reference from KETEK; not measured Ultra-X thickness |
| `mechanical_support_thickness_mm` | 0.30 | Axial insulating substrate allocation behind the die |
| `mechanical_housing_wall_mm` | 0.25 | Radial shell thickness inside the retained tapered outer envelope |
| `mechanical_back_wall_mm` | 0.50 | Rear closing wall within that same envelope |

The present C3 ProbeCorrector dimensions give face radius 2.25, housing length
4.5, front/rear radii 2.8125/1.845, and a drawing distance of 59.57915.
The front centre is about 50.49289 radially and 31.62503 upstream of the sample.
These are calculated **display dimensions**, not inferred acceptance calibration.
Focused native closed-mesh checks included the actual split objective coil/yoke,
housing and both poles, rather than checking pole outlines alone. They found no
positive material overlap for this configuration. This is not thermal design,
service access, arbitrary edited assemblies or a mechanical tolerance certificate.

## EELS: retain the five strip faces and branch placement

The same `_zebra_meshes()` helper that owns the existing strip guides supplies
the planar corners of each material conversion layer. Its leading face stays
exactly at local detector Z = 0; positive local Z is downstream. The five strip
centres, width, height and pitch are unchanged. `energy_filter_frame_mm()` is
applied once, and the ordinary assembly branch-to-column mount is applied once
by the existing caller.

The old alignment guide overlaps several strip guides in the simulator, while
the public schematic shows a separate area without usable placement dimensions.
It therefore remains a logical reference overlay and is **not** extruded as an
additional coincident sensor. The five strips do not establish five independent
camera packages. A shared generic optical backing and an open-front rear housing
complete the simplified material representation. Optical-coupler and scintillator
composition remain unspecified; no claimed fibre plate, CCD board or cooling unit
is invented.

| Persisted mechanical field | Current engineering value | Meaning |
| --- | ---: | --- |
| `mechanical_scintillator_thickness_mm` | 0.01 | Conversion-layer allocation behind each unchanged strip face |
| `mechanical_optical_coupling_depth_mm` | 1.0 | Generic backing depth; not a measured lens or fibre plate |
| `mechanical_frame_margin_mm` | 1.0 | Transverse margin outside the complete strip stack |
| `mechanical_housing_depth_mm` | 6.0 | New declared downstream package allocation |
| `mechanical_housing_wall_mm` | 0.5 | Side and rear wall inside that allocation |

Unlike EDS, there was no previous material camera envelope to preserve. The old
definition explicitly marked it unknown. This new engineering envelope is
30.672 × 6.8 × 6 mm for the saved 28.672 mm strip width, 0.8 mm strip height and
1 mm pitch. It is not an OEM external camera dimension and does not move any
reception plane. Changing these mechanical fields must not change response,
spectral dispersion or pixel calibration.

## Persistence and validation boundary

Both families require the complete corresponding mechanical field set plus:

```
detector_material_geometry_status = "engineering_reconstruction_not_oem"
detector_material_geometry_source = "docs/references/cad-detectors.md"
```

Old documents with no such field set keep their earlier guide-only path. Partial
sets, non-finite/non-positive dimensions, unsupported provenance, or layers that
do not fit are rejected; no implicit thickness is assigned. The same `SolidSpec`
leaves feed native closed meshes and analytical STEP conversion. Tests preserve
the original reception geometry and immutable source dictionaries, check proper
frames and one-time placement, and reject material overlap between layer stacks.
