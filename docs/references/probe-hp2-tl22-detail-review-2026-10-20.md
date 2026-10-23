# DPH2, HP2, and TL22 internal-structure review (2026-10-20)

Continuing below the Adapter Lens, this round refines the Probe Corrector's DPH2, HP2 hexapole, and TL22 transfer lens. Inspection, geometry generation, STEP round trips, and propagation validation were all performed through backend code, without desktop interaction. Default column global Z is local Z plus 450 mm, with +Z downstream; all lengths in this document are in mm.

## Structural changes and evidence boundaries

| Part | Existing structure | Additions in this round | Current material-region count |
| --- | --- | --- | ---: |
| DPH2 | Four orthogonal saddle copper winding packs, radial carriers/covers/bonding, and conductor-gap insulation | Annular insulating edge seals 0.04 thick at each end; through-bore remains open | 13 |
| HP2 | Six pole pins, six radial copper winding packs, common return yoke | Six hollow winding formers and six equivalent core-bonding regions; former walls and end rims both 0.25 thick | 25 |
| TL22 | Housing, excitation winding pack, upper/lower pole pieces, yoke | 0.1 insulation and a 0.5 passive copper plate at each coil end; 0.675 space remains at each end | 9 |

All three Probe Corrector configurations save these structures. The two combined configurations previously lacked DPH2 radial-support declarations, which are also added here; all original parameter values remain unchanged. HP1, other DPH/DP components, and image correctors do not adopt these new recipes in this round.

- **DPH2**: based on flexible winding carriers, cover insulation, full-surface bonding, and orthogonal layers in [CEOS US20090079531A1](https://patents.google.com/patent/US20090079531A1/en), Fig.1/4/6. Axial edge seals are an engineering inference within this family; the patent does not provide OEM edge seals or dimensions for the modeled DPH2. See [DPH2 edge-seal notes](dph2-edge-insulation.md).
- **HP2**: existing hexapole topology follows [Müller et al. 2006, Fig.3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf). Added former/core-bonding principles follow [JEOL JP2004241190A](https://patents.google.com/patent/JP2004241190A/en), [0005], Fig.7(c,d). This adapts a generic multipole structure, not a CEOS HP2 manufacturing drawing. Rectangular end rims, thicknesses, and volume-equivalent adhesive fill are explicitly labeled engineering assumptions. See [hexapoles and auxiliary structures](cad-probe-corrector.md).
- **TL22**: based on coil end plates and insulation in [Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en), conventional Fig.8, [0005]. Space is allocated using TL22's own saved winding dimensions rather than copying ADL dimensions. See [lens ends and adjacency checks](cad-round-lenses.md#2026-10-20-tl22-transfer-lens-end-hardware-and-neighboring-contacts).

Screws, fittings, water circuits, individual wires, material grades, dielectric properties, or thermal performance were not inferred from insufficient photographs or schematics. New copper plates are passive structures, not additional coils that duplicate magnetic fields. Improvements concern internal-structure representation, parameter validation, and assembly inspectability, not improved spherical-aberration performance.

## Actual tube walls and neighboring clearances

| Interface | Current nominal result | Interpretation |
| --- | ---: | --- |
| Tube exterior—DPH2 inner layer | Radial 1.46 | Tube outer radius 9.6; support inner radius 11.06 |
| DPH2 outer layer—HP2 pole-tip bore wall | Radial 0.06 | Outer radius 11.94 versus bore radius 12; minimum nominal clearance, without manufacturing-tolerance evidence |
| Tube exterior—HP2 pole tips | Radial 2.4 | New HP2 formers lie away from this bore and do not change this clearance |
| Tube exterior—TL22 pole pieces | Radial 0.4 | Tube OD 19.2 versus pole-piece bore diameter 20 |
| ADL—DPH2 | Axial 30.81 | Includes new upper edge seal; original value 30.85 |
| DPH2—TL22 | Axial 1.81 | Includes new lower edge seal; original value 1.85 |
| TL22—TL21 | Axial 58.1 | Original positions unchanged |
| HP2—TL22 | **Zero-clearance end-face contact** | Local Z703.6; not positive clearance |
| TL22—DP22 | **Zero-clearance end-face contact** | Local Z729.1, including DP22 copper winding end faces |

The complete DPH2 material extent is local Z687.21–701.79; original copper winding packs remain at 687.25–701.75. Effective optical thickness remains 4; HP2 physical length remains 18.2 and effective field length remains 6. These lengths are not interchangeable.

The continuous vacuum tube's actual ID at these positions is **5.76** and OD **19.2**; the component's declared 20 clear-beam bore is not the tube ID. Existing particle/coherent clearance constraints still use actual continuous-tube rules; this round did not expand clear beam space.

**Structural gaps retained and explicitly recorded:** the HP2 yoke/TL22 upper-pole contact annulus is about 260.58 mm², but that does not establish an intentional OEM magnetic-circuit contact. Total contact area between the TL22 lower pole piece and four DP22 copper ends is about 30.82 mm² (128-segment mesh); DP22 axial insulation there is not yet modeled. The radial return connection between TL22's original yoke and pole shoulders is also not fully reconstructed. No parts were moved, pole pieces changed, or unsupported structures added to eliminate these existing contacts/gaps; zero volumetric interpenetration does not establish electrical isolation or a complete magnetic circuit.

## Unified editing and validation

New dimensions enter existing parameter editing, surface-parameter provenance, undo, save, and copy workflows. Invalid layer thicknesses, nonfinite values, unknown fields, compressed winding windows, and end seals beyond their hosts are rejected. Neighbor checks use complete axial extents including seals. Activating an explicit replacement CAD body disables the native recipe; restoring the native recipe reruns validation, with failures leaving draft/history unchanged. TL22 end changes also invalidate geometry fingerprints of existing imported field maps and require review rather than automatically generating a new field solution.

Evidence from this round is under `outputs/probe-hp2-tl22-details-20261020/`:

- `preservation.json`: compares 35 configuration files; only 75 structural/provenance fields were added, with no original values changed. Of 413 old default meshes, 412 are bitwise preserved; TL22 housing now uses the unified CAD mesh path, preserving its analytical shape, material, and bounds. Adding 18 regions gives a total of 431.
- `after/audit.json`: 130 installed entries, 431 regions; no closedness issues, CAD construction errors, or material omissions. Combined configurations also pass production-assembly generation checks; see `updated-variant-audit.json` for updated records.
- `assembly-clearance-verified.json`: checks 311 in-column material regions and actual continuous tubes in the same global coordinates, screening and calculating 214 bounding-box-overlap candidate pairs. Four coarse-mesh flags at DPH2 curved contacts have zero intersection on exact OpenCascade review; native display meshes from 32–288 segments also show no actual interpenetration. Original flags remain in `assembly-clearance.json`; overlap thresholds were not relaxed to conceal results.
- `step/step-manifest.json`: 48 selected material regions in 5 STEP files passed round-trip checks of solid counts, volumes, bounds, names, materials, and colors.
- `dph2-band-volume-verification.json`: when copper winding band width changes from 1.16 to 6, exact CAD copper-plus-insulation volumes of the X/Y layers remain 328.273812833028 / 342.267723149177 mm³, respectively, with maximum before/after difference about 1.71e-13 mm³; copper/insulation interpenetration and missing/excess volume relative to the complete annulus are all zero. Circumferential display-mesh sampling changes with band width, so coarse meshes cannot be subjected to equally precise volume-conservation assertions.
- `transport-verification.json`: at 300 kV, STEM Nanoprobe Normal, and nonzero DPH2 deflection, 39 independent field sources, 6080 valid field-vector samples, 9 particles, and 352 saved planes are bitwise identical before/after. A positive control increasing DPH2 X deflection from 0.012 to 0.014 mrad changes fields and trajectories.
- `verified-regression.xml` / `verification.json`: **final results for all 907 distinct tests passed, with zero failures, errors, or skips**. This is deduplicated by test name across 892 main regressions, 15 independent CAD regressions, and targeted post-fix reruns, not the count of one standalone run. Original failures are retained: restoration tests received missing version numbers, obsolete material-count and coil-category assertions were updated, and two backend CAD file-read checks restricted by the sandbox passed separate reruns.

These checks cover geometric consistency of the current model and preservation of existing CPU propagation; they are not OEM dimensional certification, material FEM field solutions, or validation of real-instrument thermal/electrical performance.
