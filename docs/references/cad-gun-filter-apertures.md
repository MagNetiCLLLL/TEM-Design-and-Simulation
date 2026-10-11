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
