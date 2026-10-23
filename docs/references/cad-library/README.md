# CAD structural reference library

This library archives traceable sources only; the detail in a drawing is not equated with known accuracy for this instrument. Coordinates, boundaries and material definitions in the original TOML remain the current assembly inputs. A feasible topology in a patent does not automatically establish that this instrument has the same cooling pipes, supports, coils or hidden parts.

The repository retains each entry's `source.json` and summary. Original PDFs, downloaded webpages and full-page source images are retained locally only and are not distributed through Git or release packages. Index statuses such as “PDF saved” record historical local retrieval and parsing, not a promise that cloning the repository includes the originals; full-file SHA-256 values and original source links are in each entry's metadata. See the [local reference-material rules](../README.md).

The [original TOML component mapping](component-mapping.json) records the 11 module files inspected in this review and their SHA-256 values, and lists the actual component keys corresponding to each source. Unresolved natural-language categories in the table denote structural families only, not declarations of new or existing parts.

| ID | Applicable components/functions | Original-file status |
|---|---|---|
| [JP2967966B2](JP2967966B2/summary.md) | Conventional round magnetic lenses | Original PDF/photographs not obtained; see notes |
| [WO2021170762A1](WO2021170762A1/summary.md) | Objective | PDF saved |
| [US4306149A](US4306149A/summary.md) | Auxiliary mini condenser and objective | PDF saved |
| [US8841630B2](US8841630B2/summary.md) | Corrector transfer lenses | PDF saved |
| [US20090079531A1](US20090079531A1/summary.md) | CEOS multipole coils and DPH inserts | PDF saved |
| [US20240371597A1](US20240371597A1/summary.md) | General multipole cores/windings/yokes | PDF saved |
| [US6548816B2](US6548816B2/summary.md) | Stigmators | PDF saved |
| [US5481164A](US5481164A/summary.md) | Colocated normal/skew stigmators | PDF saved |
| [US9548182B2](US9548182B2/summary.md) | Gun and accelerating electrodes | PDF saved |
| [US6844548B2](US6844548B2/summary.md) | Wien filter | PDF saved |
| [US9966219B2](US9966219B2/summary.md) | EELS energy filter and slit | PDF saved |
| [US8373137B2](US8373137B2/summary.md) | Multipurpose energy-filter multipole elements | PDF saved |
| [Mueller2006a](Mueller2006a/summary.md) | CEOS CESCOR optical paths and physical hexapoles | PDF saved |
| [FEI-Titan-training](FEI-Titan-training/summary.md) | Titan mini condenser and illumination system | PDF saved |
| [CEOS-Linck-2025](CEOS-Linck-2025/summary.md) | CETCOR image-corrector optical path and alignment channels | PDF and full-page images p22/24/26/29 saved |
| [CEOS-Lu-2025](CEOS-Lu-2025/summary.md) | CETCOR/CESCOR optical paths, OL pre/post and reference rays | PDF and full-page images p1/2 saved |
| [CEOS-CETCOR-residuals](CEOS-CETCOR-residuals/summary.md) | Symmetry planes, intermediate images and residual aberrations | Original official webpage saved |
| [CEOS-CETCOR-product](CEOS-CETCOR-product/summary.md) | CETCOR height/width/depth and applicability | Original German official webpage saved |
| [Mueller2008](Mueller2008/summary.md) | Original CETCOR modeling paper and HP structural family | Original PDF download unavailable; bibliography and review record only |
| [legacy-user-photos](legacy-user-photos/summary.md) | Leads from earlier user photographs | Original PDF/photographs not obtained; see notes |

## Usage limits

- This library is not a set of Titan/Spectra/CESCOR manufacturing drawings. Comparisons must use the original TOML for the same module/variant, rather than forcing consistency using patent numbers from another model.
- The actual hexapole example in Mueller 2006 differs from the project's current HP dimensions; retain the differences and do not claim a complete match.
- Earlier photographs are available only as leads and must not be treated as evidence confirming aperture materials, actual thicknesses or hidden assemblies.
- Patent priority dates, publication dates and applicants are source metadata, not conclusions about current legal status or ownership.
- Public access does not imply an open license; rights to the originals remain with the authors/organizations, and figure pages must not be sold as this project's original work or as OEM deliverables.

Related notes: [round-lens evidence](../cad-round-lenses.md), [multipole evidence](../cad-multipoles.md), [gun/energy-filter/aperture evidence](../cad-gun-filter-apertures.md).

For the image corrector, also see the [CETCOR optical-path/assembly comparison](../../image_corrector_reference.md) and [pre-change TOML values/SHA](image-corrector-config-comparison.json). Local Z in that record differs from absolute Z for the entire instrument; later fixes do not overwrite this original comparison evidence.

## Side-entry specimen holders and stage

[Dimensions and assembly limits](../cad-stage-holder.md). These sources correspond to the ordinary side-entry structural family; special patented mechanisms have not automatically been added to this instrument.

- [US20100006771A1](US20100006771A1/summary.md): original_pdf_saved.
- [WO2025134741A1](WO2025134741A1/summary.md): original_pdf_saved.
- [US20080173813A1](US20080173813A1/summary.md): original_pdf_saved.
- [EP3843120A1](EP3843120A1/summary.md): original_pdf_saved.
- [Gatan-analytical-holders](Gatan-analytical-holders/summary.md): record_only_original_download_unavailable.
- [Thermo-CrystalPack](Thermo-CrystalPack/summary.md): original_pdf_saved.
- [Thermo-Spectra-Ultra-holder](Thermo-Spectra-Ultra-holder/summary.md): record_only_original_download_unavailable.
- [LBNL-TEM-holders](LBNL-TEM-holders/summary.md): original_html_saved.

## EDS and EELS camera materials

[Unchanged receiving planes, engineering mechanical dimensions and source limits](../cad-detectors.md). Four new original PDFs, figure pages, summaries and hashes have been added to the index:

- [US8410439B2](US8410439B2/summary.md): SDD, window choices, cold shield and insulating-support structural family.
- [US7642513B2](US7642513B2/summary.md): Indirect EELS conversion-layer and optical-transfer structural family.
- [DS0510](Thermo-Iliad-Ultra-DS0510/summary.md): Published specifications for five indirect-detection strips.
- [KETEK VITUS](KETEK-VITUS/summary.md): Order-of-magnitude reference for Si absorption-layer thickness in a related product, not an Ultra-X drawing.

For the image corrector's eight dipole winding packs, see the [CEOS figure/patent construction comparison](../cad-image-corrector-channels.md); original-file verification records are in the [CETCOR channel entry](CETCOR-channel-construction/source-entry.json).
