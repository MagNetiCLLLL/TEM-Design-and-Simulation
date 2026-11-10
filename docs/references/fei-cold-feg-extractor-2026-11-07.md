# FEI-inspired cold-FEG extractor implementation

This change applies the tip-facing recessed extractor in FEI's
[US8736170B1, Fig. 5 and its description](https://patents.google.com/patent/US8736170B1/en)
to the simulator's FEG and FEG + Mono modules. One conductor both establishes
the extraction field and intercepts primary electrons which miss its opening.
It is a normal-emission engineering adaptation, not a dimensionally complete
reconstruction of an FEI commercial gun.

## Geometry and evidence

All positions below use the tip apex as Z = 0; electrons travel toward +Z.
Dimensions are millimetres. The patent gives approximate example dimensions,
not manufacturing tolerances. Its tip-to-extractor distance is interpreted
here at the small-hole entrance. That interpretation is explicit because the
text does not separately dimension every axial face.

| Feature | Implemented value | Evidence or assumption |
| --- | ---: | --- |
| Small-hole entrance / recess floor | Z = 0.75 | Interpretation of the patent's approximately 0.75 mm tip/extractor distance |
| Tip-facing counterbore depth | 0.45 | Approximately 0.45 mm in the Fig. 5 description |
| Outer upstream face | Z = 0.30 | Derived as 0.75 - 0.45 |
| Overall plate thickness | 1.00 | Engineering assumption |
| Downstream face | Z = 1.30 | Derived from upstream face and thickness |
| Residual aperture web thickness | 0.55 | Derived as 1.00 - 0.45 |
| Through-hole diameter | 0.50 | Engineering initial value; not disclosed by the patent |
| Counterbore diameter | 4.00 | Engineering initial value; not disclosed by the patent |
| Outer diameter | 20.00 | Retained simulator value; not an FEI measurement |

The previous default was a straight 4 mm bore from Z = 6 to 10 mm. Its
replacement is a 4 mm recess from Z = 0.30 to 0.75 mm and a 0.50 mm bore from
Z = 0.75 to 1.30 mm. The metal is one continuous, equipotential body.

`ExtractorContour` supplies the revolved CAD outline, rectangular material
pieces for the axisymmetric field solvers, and the two vacuum-bore intervals.
CAD exposes the recess floor and wall as individually selectable surfaces.
The adjacent field pieces may share their common face only as the same
electrode at the same potential. Unrelated touching electrodes still fail.

Electron interception uses the actual swept chord through the finite axial
intervals, including the recess shoulder and reverse crossings. A step whose
endpoints lie outside the whole plate can still hit it. The resolved column
geometry and frozen particle-contact geometry preserve the same stepped
opening and physical host key `feg_extractor`. The model absorbs incident
primary particles within the declared vacuum passage; it does not follow
scattered electrons or alternative passages outside the assembly.

There is no independent maximum-angle filter. Which particles hit the metal
depends on source positions/directions, the joint electric field, and the
physical contour. Changing the extraction voltage can therefore change
acceptance without changing the hole. Source weights and lost current are
never renormalized. The geometrical angle subtended by the aperture is not
treated as an electric-field trajectory acceptance angle.

## Rear hardware and the fixed source origin

The tip apex, gun entrance reference and emission origin stay at **Z = 0**.
Adding a negative physical extent does not change the gun exit or downstream
column placement. Layout distance conversion and mechanical-axis diagnostics
use the apex, not the centre of the extended shank.

| Component | Axial interval (mm) | Bore / outer diameter (mm) | Evidence or assumption |
| --- | --- | --- | --- |
| Tip cone/shank | -6 to 0 | Derived cone envelope, maximum OD 1.05004723 | Extended axisymmetric engineering support; apex radius and emission law unchanged |
| Emitter enclosure electrode (EEE) | -1.8 to -1.5 | 0.6 / 20 | Patent gives about 1.5 mm tip protrusion; thickness and diameters are assumed |
| Cathode support collar | -5 to -4 | 0.6 / 3 | Engineering replacement for non-axisymmetric support detail, electrically contacting the tip |
| Rear insulator | -6 to -5 | 2 / 20 | Engineering seat, in mechanical contact with the collar; no dielectric response |

The facing EEE/extractor gap is 1.8 mm, within the patent's 1.8-2.2 mm
example. The patent drawing shows a needle, V-shaped flashing filament and
pins; our collar is an explicitly axisymmetric adaptation, not a reproduction
of that support or a dimensioned OEM assembly. Cleaning hardware is omitted.

EEE normal-emission wiring is not established by the patent's cleaning
voltage description. Its configurable default is **tip reference + 0 kV**
(laboratory -300 kV at the default high tension). The contacting holder must
remain at tip potential. Edit `rear_voltage_reference` (`tip`, `extractor`,
`gun_lens`, `ground`) and `rear_bias_kv` in assembly parameters. The value is
an electrical design input, not a microscope current or a focusing multiplier.

The user-selected **Flat default is retained**. Its infinite planar cathode
idealization has no rear vacuum field; the field request explicitly reports
rear electrodes as omitted. Both supported Curved field paths consume the
EEE and holder as finite equipotential annuli. Their dimensions are shared
with CAD and interception. The insulator has geometry and interception but
no polarization, charging or assigned conductor potential. Unsupported CAD
overrides/tilts and electrically shorting configurations are rejected.

## Extractor angular clearance

For a point exactly on the axis at the tip apex, straight rays without any
field have the strict clearance condition

`theta < min_i atan(inner_radius_i / downstream_face_i)`.

The current 0.25 mm bore radius and 1.30 mm downstream face give a limiting
**half-angle of 10.885527 degrees**, or a 21.771054-degree full cone. Equality
grazes metal and is intercepted. Using only the hole entrance at 0.75 mm gives
18.434949 degrees, which overestimates passage through the finite thickness.
`ExtractorContour.straight_ray_acceptance()` and the source dialog expose this
geometrical result and identify the limiting face.

This is not a maximum initial electron angle in the extraction field. Actual
acceptance depends on the initial surface point, its local normal, energy and
electrode voltages. Emitting-cap polar angle describes the area on the sphere;
it is distinct from an electron's direction relative to local normal or global
Z. The audit reports observed initial angles for particles demonstrated to pass
the extractor, explicitly as finite sample extrema rather than acceptance bounds.

## Electrical connections

The numerical scalar potential is a rise from the tip. Laboratory ground is
the final accelerator anode. With total high tension H, extraction setting E,
and tip-relative gun-lens setting L, laboratory electrode potentials are:

- Tip: -H.
- Extractor, including recess wall/floor and aperture web: -H + E.
- Gun lens: -H + L.
- Accelerator stage with fraction f: -H + L + f(H - L).
- Final stage: 0, with f exactly 1.

Extraction is part of H, not another voltage added after H. Static particle
transport conserves kinetic energy minus charge-equivalent potential rise;
an electron's final kinetic energy also retains its actual launch energy.

| Default setting, H = 300 kV | Tip-relative rise | Laboratory potential |
| --- | ---: | ---: |
| Tip | 0 kV | -300 kV |
| Extractor | 4 kV | -296 kV |
| Gun lens | 5.2 kV | -294.8 kV |
| First accelerator ring, f = 0.1 | 34.68 kV | -265.32 kV |
| Final accelerator ring, f = 1 | 300 kV | 0 kV |

The default lens changes from 1.2 kV relative to the extractor to 5.2 kV
relative to the tip, preserving its initial physical potential. Subsequent
extractor adjustments are independent of that lens setting. The new
`voltage_start_reference = "gun_lens"` connects the acceleration divider to
the gun-lens potential. This connection and all operating voltages are
explicit simulator electrical design choices, not wiring values supplied by
the cited FEI patent. The joint Laplace solve includes field penetration and
fringe coupling; the divider formula only assigns metal boundary voltages.

Old assemblies without `voltage_start_reference` retain the historical
`"extractor"` divider base. Old assemblies without recess dimensions retain
their straight bore. Existing saved operating settings remain explicit;
selecting a fresh module applies the new module defaults. A saved old working
point is not automatically qualified for the new structure.

The field builders reject an invalid divider connection, a base outside
[tip, final anode), or stage fractions which do not increase to exactly 1.
The legacy analytic ramp field rejects the new recessed/divider geometry;
it cannot silently substitute for the geometry-based solution.

## Controls and boundaries of the adaptation

Edit `counterbore_depth_mm`, `counterbore_diameter_mm`, `bore_diameter_mm`,
and axial faces in the assembly geometry editor. Recess dimensions belong to
the assembly, not independently overridable runtime controls. The extraction
reference plane follows upstream face plus recess depth in runtime and layout.
The retained `optical_reference_local_z_mm` field is only a reference marker
for recessed extractors. Invalid recesses that consume the whole thickness,
close the hole, or exceed the outer surface are rejected before saving.
The retained `vacuum_inner_diameter_mm` marker does not independently narrow
the extractor: clearance follows its actual bore and recess dimensions.

Use extractor `voltage_kv` for the tip-relative extraction setting. Gun-lens
`voltage_reference` remains explicit (`tip`, `extractor`, or `ground`). Edit
the accelerator's divider connection in the assembly. The source dialog
displays electrode potentials relative to final-anode ground, including the
first and final accelerator rings and the divider start electrode.

The separate **Gun Aperture after Gun Lens** remains a separate interception
component; it is not the extractor's hole and has not been assigned a guessed
electrical connection by this change. Downstream lens positions, accelerator
positions, source laws and optical working points have not been retuned.

FEI's cleaning filament, thermal cleaning, backscatter confinement, desorption
and current stability mechanism are not implemented. The rear EEE adaptation
above consequently does not reproduce the full Fig. 5 enclosure and cannot
predict the patent's stability benefit. Edge rounding, material work functions,
space charge and tunnelling-current prediction also remain outside this model.

## Historical extractor-only trials before adding rear hardware

The following saved results used the earlier 1 mm shank with no EEE/holder.
They are retained as historical evidence, not results for the new rear gun.
Their original inputs remain in `outputs/feg-fei-extractor-20261107/`.

The initial 1024-particle, 10-degree emitting-cap run completed on 2026-11-07:
811 particles (effective count 811) reached the gun exit. Emitted current was
100 nA, exit current 79.19921875 nA, and the other 213 particles were intercepted
by the separate Gun Aperture at Z = 24 mm. This source did not lose particles
at the extractor; it does not establish wide-angle acceptance. The maximum
exit energy-invariant error was 1.30385160446167e-8 eV, within the existing
1e-3 eV budget. Final kinetic energies retained the launch energy, spanning
300000.01302407583 to 300001.31764620723 eV. The field grid was 303 by 1919
with electrode resolution 8 cells per bore radius; the linear residual was
1.697723443309466e-15. These figures do not establish mesh or sample convergence.

A second run used 1024 original positive-current particles and a 40-degree
**geometrical spherical-cap half-angle** (not a hard outgoing-angle cutoff).
It completed with 844 particles absorbed by the extractor, 122 by the separate
Gun Aperture, and 58 reaching the exit (effective count 58). Extractor contacts
were at Z = 0.75 to 1.2999152507725078 mm: the recess floor / narrow bore,
not the outer front face at 0.30 mm. The maximum exit energy-invariant error
was 7.916241884231567e-9 eV. All 1024 source IDs were unique; stopped as well
as surviving weights remained identical to their original weights.

The prescribed emission flux density was held fixed. Enlarging the emitting
area therefore raised emitted current to 1.5399667808785763 microamperes;
exit current was 87.22468094820061 nA. This is a separate source-support trial,
not a scan at fixed total current. It confirms functioning physical
interception, not a converged transmission percentage or a measured FEI
acceptance angle. In particular, 58 surviving samples are insufficient to
qualify a final probe. No coherent calculation was performed.

The current script can create a new 40-degree emitting-cap trial; it consumes
the current rear assembly and does not reproduce or overwrite the old result:

```powershell
.\.venv\Scripts\python.exe scripts/audit_feg_extractor.py --count 1024 --cap-half-angle-deg 40 --output outputs/feg-rear-20261107/wide-cap-1024
```

`tests/test_feg_geometry.py` checks field/metal/vacuum agreement,
recess-side and reverse crossings, matching frozen contact labels, voltage
gauges, final-anode energy reference, invalid connections, save/reload and
assembly ownership. CAD tests check the closed-shell volume and all six
physical surface groups. These small synthetic chords are geometry/software
tests, not finite-source physical trials.

## Rear-gun validation

The new 10-degree emitting-cap trial used 1024 original positive-current
particles (effective count 1024), emitted 100 nA, and preserved original IDs,
positions, launch directions and weights. All 1024 passed the extractor;
111 were intercepted by the separate Gun Aperture at Z=24 mm. The exit
contained 913 particles (effective count 913), carrying 89.16015625 nA.
The maximum exit energy-invariant error was 1.30385e-8 eV. Weights of stopped
particles also remained unchanged; lost current was not redistributed.
This is a new input and result, not a correction to the historical run.

The largest sampled initial global-Z angle among extractor passages was
94.85551044 degrees. This is possible for emission from off-axis positions on
the curved cap with tilted local outward normals. It is neither an on-axis
apex ray nor a maximum acceptance bound. It illustrates why the 10.89-degree
straight-ray value must not be used as an initial-angle cutoff.

Static Curved field validation changed only EEE bias from 0 to +250 V.
The same 321 by 2067 mesh obeyed the 0-to-250 V response bounds, with scaled
linear residuals 1.46e-15 and 1.40e-15. At (r=0.2 mm, Z=-1.65 mm), tip-relative
potential rose from 1.11154 V to 110.45898 V. At 10 nm ahead of the apex it
rose from 79.27339 V to 80.79644 V, and the axial electric field also changed.
Thus the rear electrode actually participates in the solve; these figures
are not a mesh convergence study or validation against an FEI instrument.

Reproduce current inputs in new output directories with:

```powershell
.\.venv\Scripts\python.exe scripts/audit_feg_rear_field.py --output outputs/feg-rear-20261107/static-field
.\.venv\Scripts\python.exe scripts/audit_feg_extractor.py --count 1024 --output outputs/feg-rear-20261107/reference-1024
```

This explicitly selects the curved classical reference source, verifies at
least 1000 original positive-current particles before tracing, and saves the
complete input snapshot, original particle table, exit/stop table and report.
It preserves each original source ID and weight. It reports emitted,
surviving and effective particle counts separately. `--cap-half-angle-deg`
and `--cells-per-bore` create explicitly recorded alternative inputs;
`--count` provides sampling refinement. Outputs are local generated evidence,
not bundled third-party reference files. A single run is not a convergence
study or qualification of the default flat source, a coherent beam or the
full microscope.

The same `tests/test_feg_geometry.py` suite checks origin preservation,
negative-Z CAD bounds, both Curved requests, Flat omission, fixed conductor
potentials, a static maximum-principle response, rear face/wall interception,
geometry editing and voltage-reference save/reload. It uses synthetic chords
and static solves, not low-count finite-source physical trials.
