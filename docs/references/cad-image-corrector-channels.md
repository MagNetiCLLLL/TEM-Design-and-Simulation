# Image Corrector Control Channels and Mechanical Windings

This round only completes material representations for existing optical channels. Both image-corrector assemblies retain original optical centres, drives, round lenses, HPs, apertures, receiving planes and effective-field lengths. The shapes below are sourced **engineering reconstructions**, not CETCOR manufacturing drawings, and no new field strength is inferred from CAD.

## Evidence and unresolved scope

- [CEOS Linck 2025, page 26](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/01_How-do-aberration-correctors-work-pdf.pdf) explicitly shows DP11, DP12, DP21, DP22, DPH1/2, ISh, DSh and HPol/QPol/DSt. DPH occupies the same field region as its HP. This establishes optical position, not radial winding cross-sections.
- [CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en), Fig.5–7, supports two orthogonal dipole coils in concentric layers; Fig.10–11 supports compact coils inside channels or recesses. Four closed saddle windings from that structural family are used without assumed cores, yokes, housings, fasteners or cooling. The patent embodiment **does not prove identical windings in a particular CETCOR**.
- The same patent also gives Q/H coils and combined drives, but does not identify this assembly's HPol, QPol or DSt hosts, radial positions or wiring. These three channels therefore remain `physical_host_status="unverified"`, without false physical poles inferred from harmonic order or unverified attachment to nearby TL/HP units.
- Archived [Lu 2025](../image_corrector_reference.md) figures and CETCOR sources check overall topology; no new manufacturing dimensions establish those three unknown hosts. Original evidence files and SHA are in the [separate entry](cad-library/CETCOR-channel-construction/source-entry.json). This round's entry is included in the reference-library index.

## Currently selected assemblable reconstruction

Each completed channel keeps its original key and gains only one `mechanical_only` winding host, without a second optical drive. Every host has two layers with two closed copper windings each: 8 hosts / 32 material solids per variant. Table coordinates are local to `C3_ImageCorrector.toml`; add 324 mm throughout for Probe+Image.

| Channel | Local optical Z / mm | Original effective-field length / mm | Copper axial length / mm | Copper radial allocation / mm |
|---|---:|---:|---:|---|
| DP11 | 1037.9 | 6 | 4 | R 11.1–11.9 |
| DP12 | 1099.65 | 15 | 15 | R 11.1–11.9 |
| DPH1 | 1202.4 | 30 | 33 | R 26–27 |
| DP21 | 1277.9 | 14 | 3 | R 11.1–11.9 |
| DP22 | 1281.4 | 14 | 3 | R 11.1–11.9 |
| DPH2 | 1346.4 | 30 | 33 | R 26–27 |
| ISh | 1420.4 | 14 | 14 | R 11.1–11.9 |
| DSh | 1444.4 | 8 | 8 | R 11.1–11.9 |

Ordinary DP/ISh/DSh use two 0.32 mm-thick, 120° saddle layers with 0.2 mm end/side band widths. DPH uses two 0.08 mm-thick, 90° layers with 0.04 mm bands. These are this model's material-allocation parameters, not actual wire gauges, turn counts, PCB layer counts or validated current ratings.

DPH straight sections lie near 45°, 135°, 225° and 315°, avoiding the existing six radial pole pins. End arcs lie at HP centre ±16.5 mm, outside both ends of the original 30 mm pole-pin segment while entirely inside HP's original 44 mm outer envelope. HP pole pins, main windings, yoke, 180 mm outer diameter and 8 mm bore are neither cut nor enlarged. DPH host optical centres explicitly follow HP, so material and field positions share the same binding when HP moves.

DP11's 4 mm copper length fits the actual 5 mm gap. DP21/22 each have 3 mm copper at their existing centres, with 0.5 mm between them. Their original 6/14/14 mm lengths remain **effective-field allocations**, not solid housings or copper lengths. `winding_axial_length_mm` affects only copper CAD/native geometry; the solver still reads original `effective_thickness_mm`. No measured fringe-field model is added; existing uniform finite-field approximation limits remain.

## Binding, editing and validation

`channel.physical_host_key → coils` is the sole material/geometry owner; DPH is additionally positioned by `coils.optical_reference_host_key → HP`. Channels generate no materials, add no runtime components and do not duplicate dipole kicks. Windings without a declared copper length retain their historical full-axial-length shapes.

Local evidence from 2026-10-11 is in `outputs/image_corrector_channels_20261011/`:

- `candidate-material-report.json`: actual BRep intersections for 107 resolved solids, 0 material collisions; minimum clearance from 8 new hosts to other modelled material components is 0.5 mm. Parent–child relationships are not collision exemptions. This is a local static check, not full-instrument/OEM/manufacturing-tolerance certification.
- `optical-comparison.json`: 809-point round/HP/Q field arrays, optical parameters and nonzero finite-dipole records in both assemblies exactly match the previous state. DPH still covers original centre±15 mm. For the complete 4×4 numerical response normalized by 1 μm/1 mrad state scales, relative Frobenius difference is approximately 2.14×10⁻⁹ and reference-electron position differences no greater than 1.36×10⁻¹⁴ m. Numerical differences are retained, without claiming bitwise matrix identity.
- Comparison matrices come from the existing first-order observer, not a new nonlinear-aberration calibration. HP nonlinear-field arrays are compared exactly and separately; HP/Cs are not removed, and Tip is neither recomputed nor reworked.
- This round's main manifest handles final full-instrument combination audits. HPol/QPol/DSt internal materials remain unresolved, so “no collisions among modelled materials” does not establish clearance of unknown interiors.
