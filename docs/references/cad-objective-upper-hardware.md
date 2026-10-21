# Upper Objective winding: material and clearance review

Research and backend implementation: 2026-10-21. Scope: the installed
`C3_ProbeCorrector.toml` profile, from AC Scan to the upper objective pole.
Other profiles keep their independent saved geometry. This is a mechanical
engineering reconstruction, not an OEM drawing or a magnetic/thermal solution.

## Evidence and limits

[Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en), conventional
Figs. 6–8 and paragraphs 0004–0005, identifies the magnetic yoke, winding,
pole pieces and cooling assembly. Its conventional Fig. 8 describes upper and
lower copper cooling plates with electrical insulation between the plates and
the winding. This supports the **material roles and end-layer topology**; it
does not establish this microscope's dimensions, insulation grade, coolant
path, fasteners or thermal performance. The later slotted embodiments are not
silently adopted here.

[WO2021170762A1](https://patents.google.com/patent/WO2021170762A1/en), its description
of existing TEM objectives and Figs. 2–3, supports opposed hollow tapered pole
pieces, a specimen gap and a supporting structure with access ports. The
current objective retains these roles and its existing ports. The patent's
later movable-pole invention is not implemented.

The winding was already declared `insulated_copper_winding`. The new inner
liner makes part of that insulation explicit beside the upper pole shank.
**The placement and 0.2 mm thickness of this liner are engineering choices,
not dimensions or a particular liner location demonstrated by these patents.**
The 0.2 mm end sheets and 1.0 mm passive plates are likewise saved engineering
allocations. They make the intended insulating interfaces inspectable without
claiming a measured Titan/Spectra construction.

## What the previous model represented

The objective is one physical assembly. Upper and Lower Objective are shared
optical references, not separate duplicated physical lenses or field sources.
The original excitation-coil row has an allocation envelope of
1068.2–1230.2 mm. That is **not its actual material**: the parent objective's
yoke endpoints and 5 mm winding inset produce two material intervals,
946.5–1091.5 mm and 1206.9–1351.9 mm. All coordinates here are module-local
millimetres; the default instrument installation adds its existing translation.

The upper winding's ID60/OD132 annulus had a copper-colored bulk surface at
radius 30 mm. The upper pole's OD60 shank touches that surface over
1090.5–1091.5 mm. There was no positive material intersection, but the CAD did
not separate the declared winding insulation from the copper at that interface.

The pole's vacuum connector occupies 1084.5–1090.5 mm and has ID5.76/OD19.2.
The real continuous vacuum tube ends at 1084.5 mm with those same diameters.
Their common end face is a continuous tube/connector joint, not a mechanical
penetration or a newly required empty gap. The existing 5.76 mm through-bore
and 5.4 mm specimen pole gap remain unchanged.

## Material subdivision and added end layers

The explicit recipe belongs to `objective_lens_excitation_coil`, using the
`mechanical_objective_upper_end_*` fields. It contains three independent
dimensions: axial insulation thickness, radial inner insulation thickness,
and cooling-plate thickness. It does not save duplicate contact coordinates
or duplicate coil diameters. Every surface follows the current parent-derived
material interval and the winding's current ID/OD.

| Region | Local Z, mm | Inner / outer diameter, mm |
| --- | --- | --- |
| Upper copper pack | 946.5–1091.5 | 60.4 / 132 |
| Upper inner insulation | 946.5–1091.5 | 60 / 60.4 |
| Upper passive cooling plate | 945.3–946.3 | 60.4 / 132 |
| Upper end insulation | 946.3–946.5 | 60 / 132 |
| Lower end insulation of the upper pack | 1091.5–1091.7 | 60 / 132 |
| Lower passive cooling plate of the upper pack | 1091.7–1092.7 | 60.4 / 132 |
| Existing lower copper pack | 1206.9–1351.9 | 60 / 132, unchanged |

The upper copper and inner insulation have exactly the previous upper winding
pack's union and volume. Copper now has a 0.2 mm radial separation from the
upper shank; the insulation is in contact with the shank. The end layers fit
inside the existing 5 mm axial inset, leaving 3.8 mm beyond each stack. Their
copper ID follows the inner insulation's outer diameter, so the lower plate
also clears the shank. There are no independently invented absolute material
positions, extra coils, magnetic pole pieces, magnetic gaps or new optical
references.

The upper electronics sleeve remains an open polymer carrier with
ID59.2/OD59.8. Its nominal radial clearance to the winding pack ID60 is 0.1 mm;
its end at 1083.5 mm remains 1 mm above the pole connector. It is not a vacuum
wall. Its 0.1 mm radial clearance to the OD59 mini-condenser is also unchanged.

## Dimension editing and installation constraints

Both the Part Model editor and Manifest Editor use the same three detail
dimensions. The parent objective's winding inset and yoke endpoints continue
to locate the material ends; changing the child allocation envelope cannot
resize those material intervals. Upper copper inner faces expose the radial
insulation thickness together with the saved winding ID. The lower winding
does not acquire that upper-only insulation control.

The existing objective installation contract requires the winding ID to equal
both pole mounting-shank ODs. A direct winding-ID edit updates its parent's
saved radial winding thickness and redraws the seven regions, but an unmatched
ID is a draft and cannot be saved. To change that installation diameter,
edit the winding ID and both upper/lower `pole_stem_outer_diameter_mm` values
together, then validate the resulting assembly. The regression uses 60 to
61 mm on all three values in a temporary copy; no default pole dimension or
optical parameter is changed. This constraint is preserved rather than
silently relaxing the defined mounting relationship.

`tests/test_objective_upper_integration.py` checks both real save/reload
routes, the parent radial-thickness linkage, native face parameter paths,
rejection of an unmatched ID, geometry fingerprints and imported-map impact.
Changing any of the three detail thicknesses changes the geometry identity;
editing the engineering source note alone does not. The existing homogenized
FEM material mask and analytical field remain unchanged by those three
subdivision dimensions, so an imported map requires revalidation without a
claim that these passive layers are newly solved FEM regions.

## Remaining reconstruction boundaries

The existing yoke ID140 and pole shoulder OD96 still lack a resolved radial
ferromagnetic return connection. The new copper plates are passive thermal
material and do not fill that missing connection. Exact magnetic returns,
coil turns and terminals, fastener paths, winding outer insulation, lower-pack
insulation and cooling details, tolerances, adhesives and coolant paths remain
unresolved. No guessed bolts or coolant passages are introduced into the
specimen access region.

The actual propagation model and field solver keep the established
**homogenized insulated-winding envelope**, current inputs, analytical field,
and shared objective field identity. The displayed upper copper/insulator
subdivision is not a new FEM conductor-domain solution. Consequently this
mechanical refinement does not demonstrate a change in lens strength, heating,
eddy currents, saturation or aberrations. Field-map identity includes the new
mechanical dimensions so a cached/imported map cannot silently survive a
geometry edit; documentation-only changes do not imply a changed field.

## Backend validation

`cad_objective_upper_hardware` produces the same seven `SolidSpec` regions for
native meshes and CAD. Validation rejects missing parent geometry, invalid or
unrepresentably thin dimensions, consumed copper thickness, conflicting radial
profiles, end stacks outside the parent yoke spaces and pole interference.
It checks the actual shank/tapered pole section rather than treating the pole's
OD96 allocation as a solid cylinder through its narrow connector. A saved
native radial profile takes precedence over the pole style or annular yoke
dimensions. Neighbors with replacement bases, enabled cuts or non-default CAD
transforms are left to the existing final posed-CAD overlap audit: an uncut or
untransformed reference section cannot certify their clearance. This local
detail check is not a substitute for that mechanical audit, and it does not
relax older module-wide objective sizing constraints on saved scalar inputs.

The regression suite checks closed meshes and disjoint material at 16, 32 and
64 angular divisions, original pack union, unchanged lower copper and pole/yoke
geometry, the copper-to-pole clearance, native/CAD equality and one shared
objective circuit. Parent inset and coil ID/OD edits propagate through all
dependent surfaces. A detached mechanical copy stores the already supported
two-interval snapshot, moves its complete geometry together, and carries no
field-source ownership. A native model with missing parent data never falls
back to the incorrect allocation envelope. Explicit user box/STEP replacement
continues to take precedence over the native recipe.

The joint AC/upper-objective checkpoint completed 731 distinct Python regression
cases and 8 CAD-environment cases. STEP export/readback covered 6 files and 40
material regions. The scoped exact installed-material audit checked 481 nearby
pairs in the default Probe column with no material penetration. Cross-component
contacts remain only at the upper pole's inner insulation, lower end insulation,
and vacuum-tube connector. The existing Probe+Image and Probe+B-COR geometries
were compared unchanged; their earlier copper/pole contacts were not modified.

Normal and Large Angle, each in ideal and analytical modes, retained all 22
recorded arrays bit-for-bit, including nonzero AC scan, field samples, scan
calibration and ray trajectories. A separate AC-kick positive control produced
nonzero magnetic-field and trajectory changes. This is bounded numerical
preservation evidence, not a new field calibration. Detailed local reports are
under `outputs/ac-upper-objective-20261021/`, including `final-validation.json`
and `independent-validation.json`.
