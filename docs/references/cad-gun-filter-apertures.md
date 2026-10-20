# Gun, filter and aperture CAD evidence

Reviewed 2026-10-11. The analytical CAD uses the saved simulator geometry as its
dimensional input. The sources below support construction principles, not the
claim that these files reproduce a specific manufacturer's complete machine.
Mechanical reconstruction does not recalibrate fields or change ray/wave inputs.

The final adoption rules are in [CAD_DETAIL_POLICY.md](CAD_DETAIL_POLICY.md).
In particular, this model does not acquire twelve physical Wien/filter poles
from an alternative patent, new blanker/quadrupole support bodies, or extra
accelerator entrance/exit rings. A gun without a defined electrode thickness
retains its original envelope instead of silently using a 1 mm default.

| Source | Figure and supported observation | Permitted interpretation and limit |
| --- | --- | --- |
| [Hitachi, US9548182B2](https://patents.google.com/patent/US9548182B2/en), [original PDF page 4](https://patentimages.storage.googleapis.com/13/8f/2a/0e3e6195b4f35a/US9548182.pdf#page=4) | Fig. 2 is a longitudinal gun section. It depicts accelerating electrodes separated by insulating tubes; the description distinguishes the beam-vacuum passage, interstage electrodes, insulators and surrounding high-voltage housing. | Supports alternating electrode/insulator topology. The project's stage centres, electrode thickness, bore and outer envelope remain configured values; radial thickness of added insulators is a declared reconstruction. Do not copy this embodiment's housing material, shielding electrodes, voltage division or dimensions into another gun without separate inputs. |
| [JEOL, US6844548B2, *Wien filter and electron microscope using same*](https://patents.google.com/patent/US6844548B2/en) | Fig. 1 is the transverse XY section; Fig. 2 is the longitudinal XZ section. This embodiment uses twelve magnetically active poles, with electric potentials and magnetomotive excitation illustrated separately in Figs. 3 and 4. Electric and magnetic dipole fields are mutually transverse. | Supports a combined electric/magnetic multipole construction around an open passage. It does not show that the project's simpler Wien provider has twelve independently calibrated electrodes, or establish its actual dimensions. Pole shoes, copper and insulating supports in a generic CAD must remain distinguished from solved electromagnetic boundaries. |
| [US9966219B2, *Electron energy loss spectrometer*](https://patents.google.com/patent/US9966219B2/en) | Figs. 1 and 2 distinguish a prior arrangement from the disclosed biased-drift-tube spectrometer. Fig. 3 shows a drift tube with a slit; Fig. 4 shows adjustable opposed energy-selecting edges. Post-bend optical elements and downstream detection are part of the arrangement. | Supports branch, drift tube and opposed slit-blade topology. It does not give this simulator's prism radius, pole thickness, carrier spacing or a commercial Iliad manufacturing drawing. Electrostatic elements must be represented as electrodes, not silently replaced by magnetic coils. |
| User-supplied objective-aperture top-view photograph, 2026-08-30; provenance preserved in native `aperture_mechanism_evidence_source` fields | The saved metadata records the user's identification of a perforated strip attached to a single connection rod by a screw. The material identification is explicitly unverified. | User evidence supports the strip/attachment class only. No scale calibration, strip outline, multiple-hole coordinates, platinum grade or measured thickness is established by that photograph. Use the saved plate thickness and explicitly selected/current opening; never substitute the carrier bore for an unknown working hole. |
| User-supplied accelerator side-view photograph, 2026-08-30; provenance preserved in `accelerator_electrode_stack_evidence_source` | The saved metadata records repeated external ring stages. | Supports repeated-stage appearance, not hidden internal geometry, dielectric composition, an OEM scale or the declared numerical electrode thickness. |

The aperture and accelerator photograph statements above are a record of the
existing user-supplied provenance in the project. This review did not remeasure
or independently authenticate the original photographs. The public patent
excerpts retained under [gun-electrode-patents](gun-electrode-patents/README.md)
also preserve their original document/figure identities.

## Geometry and ownership rules

- Accelerator electrode faces remain at each configured stage centre plus or
  minus half the configured electrode thickness, with the saved inner/outer
  radii. Added insulation occupies interstage space and cannot change those
  electrode faces.
- Tip apex radius, tangent cone and length come from the physical Tip geometry.
  Emission FWHM is not a metal dimension. Unknown supports or heaters are not
  filled in as if their dimensions were known.
- A thin aperture plate is centred at its optical reference and uses its plate
  thickness. The longer mechanism envelope is not plate thickness. Its
  transverse outline can remain explicitly schematic; an unspecified opening
  must be reported as unspecified, not assigned the carrier or vacuum diameter.
- A filter part's internal coordinates are transverse X/Y and longitudinal Z
  about its path centre. The filter branch enters along branch +X and bends
  toward branch -Z. For assembly, apply the established proper transform
  `(X,Y,Z) -> (-Z,Y,X) + (0,0,entrance_z)` exactly once. A 90-degree branch then
  enters along column +Z and exits along column +X.
- Reconstructed prism poles start outside the configured pole gap. The gap,
  radial channel and reference orbit are preserved; a guessed external return
  yoke is not a known physical boundary.
- Slit blades preserve the configured gap, centre, height and thickness. The
  outer blade width is a reconstruction, not an independently calibrated
  housing dimension. A finite camera/shutter electrode must preserve its gap
  and active length.
- A detector active area or virtual reference with no thickness is not a closed
  solid. Omitting an unknown package is more accurate than inventing one.
- Saved rigid placement must be retained in CAD. If a Boolean/shape/legacy
  transform has no analytical adapter, export must reject it explicitly rather
  than silently return an unedited shape.

See [the existing Energy Filter reference ledger](../PHYSICAL_LAYOUT_VIEWS.md)
for the manufacturer's public functional topology and the model's non-OEM
dimensional limits, and [the multipole ledger](cad-multipoles.md) for the
distinction between physical pole count and driven field harmonics.

## Gun electrode display audit, 2026-10-20

The native display previously showed the saved FEG extractor, electrostatic
gun lens and thermionic Wehnelt/gun lens as unspecified envelopes even though
CAD already constructed their configured annular electrode material. Both
routes now use the same analytical solid and electrode category. The upstream
face, downstream face, open beam bore and outer surface are separately
selectable, with dimension ownership pointing to the saved axial endpoints
and the actual bore/outer-radius fields used by the construction.

This adds material and surface detail without changing any electrode radius,
axial face, voltage or field input. The FEG/FEG + Mono display boundaries are
checked against the annular boundaries consumed by the grounded-tip field
request, including the millimetre-to-metre conversion. `configured_geometry`
means the displayed solid follows the simulator's saved dimensions; it does
not establish an OEM nose contour, support, insulator or manufacturing detail.
Those unavailable electrode internals remain unmodelled. Gun stigmators
without the explicit construction declaration below retain the unknown
envelope; an optical quadrupole name alone never supplies a winding count.

## Gun stigmator engineering adaptation, 2026-10-20

All three saved gun variants now explicitly select `eight_coil_slotted_carrier`.
This is a documented engineering design for this simulator, not identification
of the OEM gun's hidden hardware. The FEG and Thermionic envelopes remain
427–439 mm, centre 433 mm; FEG + Mono remains 477–489 mm, centre 483 mm.
Each retains the 12 mm body allocation, 10 mm open bore, 90 mm outer diameter
and independent 8 mm effective quadrupole length.

| First-party source inspected | Supported construction principle | Use and limit |
| --- | --- | --- |
| [JEOL stigmator glossary, Figs. 2–3](https://www.jeol.co.jp/words/semterms/20201020.111014.html) | Two four-coil sets offset by 45° allow twofold astigmatism correction in arbitrary orientation. | Supports eight bulk coil locations and two quadrupole basis groups. It supplies no gun dimensions, turns or current calibration. |
| [JEOL US6548816B2, description and Fig. 1](https://patents.google.com/patent/US6548816B2/en) | Regularly spaced eight-coil assemblies can generate a quadrupole field. | Supports the selected topology. Its optional higher-order control is not added to the gun propagation model. Eight coils are not eight new field sources or evidence of eight iron pole pieces. |
| [Beihang CN102509689B, description of Figs. 5–6 and claims 2–4](https://patents.google.com/patent/CN102509689B/en) | A cylindrical support has eight receiving grooves and separate coil frames; insulated winding is fixed with adhesive. | The support/former/adhesive principle is adapted from an **objective** stigmator. The model does not copy that device's magnetic shield, objective spacer, aperture, threaded stack or stated material grades. The original drawing files were unavailable through the retrieval service; this review used the patent's textual construction description and claims. |

The common `SolidSpec` builder produces 33 separate material regions: one
slotted carrier, eight copper winding allocations, eight equivalent insulating
coatings, eight insulating formers and eight adhesive seats. The carrier slots
open downstream and retain an upstream seating surface. The central bore stays
open throughout; no end cover, magnetic core or additional vacuum tube is added.
The formers have radial-axis windows and flanges. A bulk coating around each
winding is an insulation allocation, not evidence of individual-turn dielectric
coverage or electrical qualification. The adhesive seats give an explicit
relative mounting interface; fixing the entire carrier to an OEM host and
routing the leads remain unmodelled.

Every internal dimension is saved with the `gun_stigmator_` prefix: winding
radii 16/34 mm, tangential width 6 mm, physical axial length 8 mm, window width
3 mm and length 4 mm, insulation thickness 0.2 mm, former wall/flange thickness
0.5 mm, adhesive thickness 0.2 mm, and initial mechanical basis angle 0°.
These are envelope allocations, not values read from patent drawings. Slot
limits follow the same former/insulation/adhesive surfaces, so the components
meet without overlapping. The 8 mm physical winding default happens to equal
the existing active length; changing it does not change the optical field
length. All regions remain `engineering_reconstruction`, with unspecified
material grades and no magnetic, thermal or insulation calibration.

Validation covers positive closed solids, every pair's intersection volume,
the full open bore, seating-face contact, production vacuum-liner/adjacent-gun
clearance, all three gun variants, invalid inputs, and geometric response to
each saved dimension. Native meshes and analytic CAD share the same builder.
The existing gun quadrupole gradient, rotation, soft edge, effective length and
field evaluations remain unchanged; no separate winding field is superposed.

## 2026-10-20: C2 single-rod aperture attachment

The initial rollout enabled only `condenser_aperture_2` in the default `C3_ProbeCorrector.toml` for
`aperture_support_profile = "single_rod_nominal_screw"`. Previously both native
3D and STEP contained only a schematic thin strip; the rod and screw existed
only in the 2D drawing. The saved 2026-08-30 user-photo record identifies a strip
attached to one rod by a screw, but the original photograph was not located.
This review therefore does not claim to have inspected or measured it.

[JEOL JP2007329043A](https://patents.google.com/patent/JP2007329043A/en), the
description of Fig. 2 and the symbol list (holder 4/6, screw hole 6a, screw 7),
supports treating the working aperture and its mounting hardware separately.
Its descriptions distinguish a single opening, multiple separate plates and
multiple holes in one plate. None establishes this C2's hole count or spacing.
[JEOL JP2012146543A](https://patents.google.com/patent/JP2012146543A/en), the
holder-body/fixing-screw description, independently supports a movable holder
with mechanical fastening. Its differential-pumping rails, rollers and seals
are a different construction and are not added. These are textual primary
references, not a claim of visually matching their drawings to the lost photo.

The three shared native/CAD solids are the plate (`body`), one rectangular
connection rod and one nominal screw. The rod seats against the downstream plate
face. The screw head seats on the upstream face; its cylindrical shank occupies
matching fixing bores in the plate and rod. Threads, driver recess, clamp preload,
material grade, electrical/thermal contact, external drive and vacuum feedthrough
remain unmodelled. The simplified screw expresses a declared attachment rather
than an additional optical hole or a manufacturing-ready fastener.

All new sizes are explicit `aperture_support_` engineering parameters in mm:

| Feature | Saved dimensions and placement |
| --- | --- |
| Strip | Half-width 3.25; Y from -3.25 to +14.5, exactly the previous default schematic outline |
| Single rod | Width 3; downstream thickness 2; Y from +11.5 to +39 |
| Screw | Centre Y +13; shank diameter 1; head diameter 2.4; head thickness 0.6 |

These dimensions are assumptions within the unchanged 80 mm diameter / 20 mm
axial mechanism allocation. The original 0.2 mm plate thickness, module-local
optical plane Z=315 mm, 4 mm carrier bore, 5 mm vacuum passage and 2 mm maximum
working radius are retained. The current selected/runtime opening remains the
only working hole shown; an unspecified opening remains unspecified. The one
new fixing bore is labelled independently and never inherits the working-hole
control. Other column variants keep their previous strip-only geometry until
explicitly configured.

The unchanged tube OD of 19.2 mm leaves 1.9 mm to the rod start and 2.2 mm to
the nearest screw-head edge at the default position. Only the plate uses the
existing +Y side-entry slot, with its saved 0.2 mm clearance per side. Backend
intersection tests cover the inserted position and outward translations of
2, 8 and 20 mm through the same fixed slot, without cutting the opposite wall.
Physical offsets follow the existing placement convention; optical tilt does
not rotate the rod. Relative dimensions follow independent mechanical copies
and axial moves; tube clearance is checked only where their axial spans meet.

Validation checks closed positive material, pairwise non-overlap, shared seating
surfaces, the actual hole at zero/current/maximum radius, opening offsets,
selected-hole indices, production native/CAD agreement, tube insertion clearance,
invalid-draft rejection and copied-part placement. These checks do not establish
real vacuum sealing, mechanical stiffness or an instrument-specific structure.

CAD transverse scaling now updates the declared plate-entry slot before the
fixed millimetre clearance is added; the gap itself is never multiplied by the
displayed material scale. Scaled rod/screw clearance is checked before committing
the editor draft. Independent copies and replacement bases do not acquire new
tube slots: their fully transformed enclosing boxes must clear the tube in Z or
radius. A possibly intersecting envelope is rejected explicitly as an unsupported
insertion path. This is a conservative placement guard, not an exact clearance
qualification of arbitrary imported material. Tests cover nonuniform scaling,
single application of physical offsets, optical-tilt semantics, Boolean tube
intersection, and failure without changing the draft or undo history.

## 2026-10-20: C3 aperture attachment below the C3 lens

`condenser_aperture_3` in the default `C3_ProbeCorrector.toml` now explicitly
opts into the same single-rod/nominal-screw construction. Its existing saved
photo-topology statement already identified this attachment family; the old
3D/CAD geometry contained the plate alone. The implementation shares the C2
native/CAD builder and validators instead of introducing a second independent
representation. Other column definitions still use their legacy plate until
they explicitly opt in.

The primary evidence is the aperture/holder/screw distinction in
[JEOL JP2007329043A](https://patents.google.com/patent/JP2007329043A/en), Fig. 2
and the symbol list, together with the movable-holder construction in
[JEOL JP2012146543A](https://patents.google.com/patent/JP2012146543A/en).
These support the attachment class, not this instrument's C3 dimensions or
hole count. All added sizes are saved engineering assumptions. The absent
original user photograph has not been remeasured, and the nominal fastener
does not establish a thread form, torque, preload or vacuum-seal design.

The old strip outline is retained: half-width 3.25 mm and Y=-3.25..14.5 mm.
The single rod is 3 mm wide and 2 mm thick, from Y=11.5 to 36.5 mm; shortening
its outer reach relative to C2 fits C3's existing 75 mm mechanism diameter.
The screw remains at Y=13 mm, with a 1 mm nominal shank diameter, 2.4 mm head
diameter and 0.6 mm head thickness. Matching fixing bores prevent nominal
material overlap. The plate, rod and screw occupy Z=559.3..562.1 mm overall.

The optical plane remains Z=560 mm module-local, the plate remains 0.2 mm
thick, and the 20 mm allocation, 4 mm carrier bore, 5 mm vacuum passage and
2 mm maximum working radius are unchanged. No new working holes are inferred.
An unspecified working opening remains unspecified; the selected/runtime
opening and its offsets continue to control the same physical plate hole.
The fixing bore has separate face metadata and no working-aperture control.

At the current default placement, the 19.2 mm tube OD leaves 1.9 mm before
the rod and 2.2 mm before the nearest head edge. The plate uses the existing
+Y entry slot with 0.2 mm fixed clearance per side; support hardware is outside
the tube and does not enlarge this slot. Actual material intersections are zero
against the production tube and neighboring C3/Beam Deflector material. The
retained material extents leave 9.3 mm upstream to C3 and 7.9 mm downstream to
the Beam Deflector. The same fixed slot is clear during outward insertion-path
samples at 2, 8 and 20 mm, and its opposite wall remains present.

Dedicated backend regressions cover closed positive meshes, seating and
non-overlap, current/zero/maximum openings, native/analytic agreement, transverse
scale and physical-offset behavior, exact slot gaps, neighbor intersections,
invalid edits without history changes, copied-part placement and save/reopen.
Rod length edits affect the rod geometry alone. Optical-axis tilt retains its
existing meaning and does not rotate installed hardware. The additions create
no field source and change no propagation, clipping or operating-mode parameter;
they are material reconstruction, not an electromagnetic or instrument
qualification.
