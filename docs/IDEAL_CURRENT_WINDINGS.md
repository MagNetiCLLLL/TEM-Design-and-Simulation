# Conductor-area input, adaptive insulation and ideal prescribed-current fields

This guide covers round-lens windings. Calibrated deflector and stigmator
windings use separate closed-circuit geometry, fine-wire estimates and signed
current matrices; see [their input guide](TRANSVERSE_CURRENT_WINDINGS.md).

The mechanical winding and ideal field share fixed turn/current inputs. CAD
places the round conductor turns and fills the remaining allocated volume with
insulation. The ideal magnetic source homogenizes the original full winding
pack, including its insulation allocation. It does not shrink that envelope to
the metal area or substitute discrete CAD wires for the field quadrature.

## Inputs and ownership

| Input | Owner | Meaning |
| --- | --- | --- |
| `mechanical_inner_diameter_mm`, `mechanical_outer_diameter_mm` | Excitation winding | Original fixed pack ID/OD, mm |
| Existing axial geometry / occupied intervals | Winding and parent | Original fixed occupied axial envelope |
| `winding_turn_count` | Excitation winding | Positive integer conductor turns |
| `winding_interval_turn_counts` | Excitation winding | Optional fixed counts per occupied interval, summing to total N |
| `winding_wire_cross_section_mm2` | Excitation winding | Optional per-turn conductor METAL area, mm squared; canonical user size input |
| `winding_wire_diameter_mm` | Legacy winding record | Read-compatible old round-wire diameter, mm; read-only in the UI |
| `ideal_coil_current_a_at_100_percent` | Round-lens field owner | Nonnegative conductor current at 100% excitation, A |

Users enter metal area rather than diameter. The calculated round-wire diameter
is `d = 2*sqrt(A_metal/pi)` and is read-only. If both legacy diameter and area
exist, they must agree within binary64 conversion roundoff. Actual area edits
save the canonical area and remove the legacy alias. Unchanged legacy inputs
retain their exact original size and provenance. Unknown metal area remains
blank: no wire size, packing fraction or current rating is guessed.

Current retains its existing convention:

`I_actual = I100 * (excitation_percent/100) * polarity` (zero when disabled).

`NI_interval = N_interval * I_actual`.

One owner supplies the same prescribed series current to every owned winding.
Independent currents require independent owners. The historical 3000-A-turn
fallback remains stored but is inactive when explicit turns/current are present.
Gaussian and FEM field selections retain their separate calibration rules.

## Allocated area versus conductor metal area

The user authorized a nominal integer **9 mm2 per turn** for engineering turn
estimates. This is winding-pack allocation, including insulation and residual
space. It is distinct from the user-entered conductor metal area:

`A_pack = (OD-ID)/2 * sum(occupied axial lengths)`.

`N = round_half_up(A_pack / 9 mm2)`.

The authoring policy is saved in `configs/physics/coil_winding_estimates.toml`.
Integer rounding can make actual `A_pack/N` differ from nominal 9 mm2. This
policy is an explicit authoring estimate, not an automatic runtime geometry
binding or a minimum metal/cell area. Later manual input never silently
re-estimates turns or recalibrates current.

The subsequent explicit user permission allows this modelling adjustment to
vary original counts by at most 20%. All current catalog changes are at most
5%. Objective now uses **580 upper + 580 lower = 1160 turns**, a 1.02389%
reduction from the original adopted 586/586. The original upper photo estimate
and lower symmetry assumption remain in provenance; 580 is a model adjustment,
not a new measurement. At unchanged ID60/OD132 and occupied145 mm per region,
Objective pack allocation is now exactly `5220/580 = 9 mm2` per turn.

Current references were adjusted inversely with N during this explicit one-time
migration, preserving NI and the previous ideal fields. The standard Objective
I100 is now 1591.5285426144758 A, giving about 1097.806245 A at its unchanged
68.97810600833917% excitation. Its prior specimen-upper-surface condition and
full ideal field are retained. Other original Gaussian-peak matching conditions
also remain unchanged. This does not make the broad ideal field identical to
the historical Gaussian profile or qualify focus.

See the [current area/CAD report](development/wire_area_20261115/README.md).
The earlier [catalog calibration report](development/all_lens_current_20261115/README.md)
is a historical checkpoint with the earlier area ratio and turn counts.

## CAD adaptation at fixed envelopes

For a supplied metal area and N, the layout solver repacks every round wire
on a deterministic balanced orthogonal R-Z grid. Centre pitch, integer rows,
interwire gaps and boundary clearances follow the requested inputs. The original
ID, OD, axial limits and passive layers remain fixed. The remaining allocated
volume is an insulating solid with all wire cavities removed; explicitly
modelled end insulation, inner insulation and cooling layers remain separate.

A single occupied interval has the same count as total N, so editing N alone
updates that identical saved count. Split windings retain explicit regional
counts and require a valid sum; they are not silently redistributed. A requested
layout that does not fit the supported grid fails atomically, without dropping
turns, shrinking the conductor or enlarging the pack. The grid does not claim
maximally dense hexagonal packing. Tangency is a geometric limit, not certified
manufacturing insulation or a breakdown/current rating.

The readback records metal area per turn, derived diameter, total metal area,
remaining insulation area/volume, pitch, gaps, per-region turns and the distinct
original field envelope. Display meshes retain every turn with explicitly
reported tessellation. STEP, when available in the optional CAD runtime, uses
analytic tori and exact remaining-insulation solids.

## Editing and persistence

1. In **Physical Layout**, select the lens excitation winding and open
   **Configure winding...**.
2. Enter N, conductor metal area in mm squared, and owner I100. Leave area blank
   only when conductor size is unknown. Diameter is calculated and read-only.
3. Review automatic insulation/pitch/clearance readbacks. The duplicated regional
   input is hidden for a single interval; a split winding retains its explicit
   regional counts.
4. **Apply to draft** commits one validated Undo operation; **Save** is separate.
   Invalid or stale drafts retain the preceding document and live state.

The generic **Dimensions** panel exposes N and canonical metal area as editable
values, with diameter read-only. Both edit paths apply the same atomic CAD
validation, single-interval identity, provenance and fixed-envelope policy.

Manual N edits keep current numeric values unchanged and mark any old field
calibration obsolete. Area-only edits change CAD/provenance while preserving
the ideal field at fixed N, I and envelope. Neither path refocuses a beam.
Completely unchanged input preserves full binary64 values, provenance and Undo
history. Clearing optional area/legacy diameter survives save and reload.
Cached dependencies include the new area input, so changed CAD/provenance is
not silently reused as an old calculation input.

## Authoring APIs and resource limits

```python
from temsim.part_model_document import PartModelDocument

document = PartModelDocument("path/to/instrument-module.toml")
document.configure_winding(
    "condenser_lens_2_excitation_coil",
    winding_turn_count=100,
    winding_wire_cross_section_mm2=0.5,
    ideal_coil_current_a_at_100_percent=30.0,
)
document.save_copy("path/to/edited-definition.toml")
```

These are illustrative inputs, not default or OEM winding data. Omitted size
preserves its saved definition; explicit None clears area and the old diameter.
The batch `configure_windings` API shares the same transaction and current.

`coil_turn_estimation.estimate_winding_turns_from_allocated_area(envelope, 9.)`
returns a pure engineering estimate. The earlier reference-area estimator remains
available for historical reproduction. `coil_current_calibration.calibrate_round_lens_current`
returns a detached single-plane current recipe without changing live controls.

The pure detailed-layout budget remains 65,536 turns; display meshes retain
their four-million-triangle guard and STEP its 8,192-turn guard. Those are
explicit resource gates, not physical laws or permission to omit turns.
Verification distinguishes array/CAD/GUI checks from finite-source, focus,
material, STEP and CUDA coherent execution. See the
[calculation scope matrix](CALCULATION_SCOPE_MATRIX.md).

Corrector HP1/HP2 and the named auxiliary quadrupole/hexapole channels use
separate radial-axis winding CAD and fixed signed multipole current matrices.
See [multipole winding and current inputs](MULTIPOLE_CURRENT_WINDINGS.md) and
[Probe Corrector internal CAD](PROBE_CORRECTOR_INTERNAL_CAD.md). Their native
inverse-square/cubic-metre coefficients map explicitly to amperes; the
column-axis round-winding I100 convention does not define those gains.
Probe HP retains the original N=2 per radial coil from the earlier nominal
9 mm2 allocated-area estimate. Current fitting preserves that physical count.
