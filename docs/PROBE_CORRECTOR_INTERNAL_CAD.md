# Probe Corrector internal CAD and radial winding inputs

HP1 and HP2 retain their original poles, copper-pack envelopes, yoke, formers
and core bedding. Each now adds six transparent **0.15 mm outer insulation
jackets**, explicitly saved as engineering reconstruction rather than OEM data.
The earlier bulk construction had 31 material regions. Current detailed turns
and the explicit winding angular datum are shared with the ideal current
source; see [multipole winding and current inputs](MULTIPOLE_CURRENT_WINDINGS.md).

## Initial engineering turn counts

Following the user's size-based initialization request, HP1 and HP2 now each
have **2 estimated turns per radial coil**, or **12 turns across six coils**.
These are fixed engineering starting inputs, not measured or OEM counts. The
original count initialization left metal area and current undefined. The later
ideal-current recipes supply explicit metal area, insulation and calibrated
current gains while preserving N=2. A legacy record with unknown conductor
size continues to retain bulk copper and passive supports/jackets.

Use the existing nominal **9 mm2 allocated pack area per turn**, including
insulation and unused space. For a radial-axis rectangular winding every turn
crosses each straight side; use radial pack length times the **narrowest side
band**, not the column-axis annulus, full frame area or six-coil sum:

| Host | Radial pack length (mm) | Minimum band (mm) | Limiting section (mm2) | N per coil |
| --- | ---: | ---: | ---: | ---: |
| HP1 | 6.885 | 1.991 | 13.708035 | 2 |
| HP2 | 6.885 | 2.002 | 13.783770 | 2 |

Round section/9 to the nearest integer with halves upward and a minimum of one.
Actual allocated area after rounding is 6.8540175 and 6.891885 mm2/turn; neither
number is conductor metal area. This coarse common-density assumption does not
determine actual wire size or an OEM winding prescription. Saved N stays fixed
on later geometry edits. The explicit current calibration is a separate,
fixed-N operation against the native m^-3 field reference; the CAD-only
estimator does not perform it. ADL retains 23 turns; TL22, TL21 and TL12 retain
19 turns each.

## Detailed CAD inputs

In Physical Layout, select **HP1 Hexapole** or **HP2 Hexapole**, then choose
**Configure winding...**. Enter turns **per radial coil** and conductor **metal
cross-section in mm squared**. Diameter is calculated and read-only. Known N
may be saved without metal area, retaining bulk CAD. Metal area without N is
invalid; clearing both removes the winding definition. Supplying both inputs
generates detailed turns.

The recipe places the exact requested round-wire rounded rectangular loops
around each pole stem and fills the remaining original winding frame with
insulation. Outer radial/tangential/axial bounds, core window, pole/yoke and
support interfaces remain fixed. Invalid packing is rejected without shrinking
wire, dropping turns or enlarging the frame. Closed turns do not infer leads,
series connections, current ratings or an OEM winding prescription.

| Persistent field | Meaning |
| --- | --- |
| `hexapole_winding_turn_count_per_coil` | Positive integer N for each of six coils |
| `hexapole_winding_wire_cross_section_mm2` | Per-turn conductor metal area |
| `hexapole_winding_geometry_status` | `engineering_reconstruction_not_oem` |
| `hexapole_winding_source_note` | Explicit input or assumption note |

The generic Dimensions table also edits N and area, with derived diameter
read-only. Both paths preserve atomic validation, source records, no-op values,
Undo and save/reload. Replaced/imported CAD disables the native recipe; restoring
native geometry validates the saved inputs again. Unconfirmed physical control
hosts receive no conjectured new body.

The headless document API is:

```python
document.configure_probe_winding(
    "probe_hp2_hexapole",
    turn_count_per_coil=6,
    wire_cross_section_mm2=0.10,
)
```

Those values are illustrative, not installed defaults. Passing None for both
values removes the recipe; a known N with area=None retains bulk CAD. Legacy
CAD-only records remain supported. For a calibrated ideal-current recipe,
the legacy API routes through the electrical transaction, retaining numeric
gains and marking changed turns stale; clearing also removes its current
definition. Native optical commands remain independent. The multipole
current convention is separate from a round lens's I100 input.

## Model and resource boundaries

ADL/TL22/TL21/TL12 keep their round-lens internals and metal-area editor.
Named HPC/QPC, HPol/QPol and QPH channels have opt-in engineering winding-only
CAD at their existing reference planes. Their control-channel transport role
and existing clipping ownership remain unchanged; no additional opaque
cartridge or verified OEM hardware host is inferred.
The same SolidSpecs feed native meshes and analytic STEP. A 4096 total-turn gate
applies across an HP's six coils; display complexity and physical STEP counting
have separate guards. No resource gate silently omits turns. Material-region
classification uses pure bounded layout rather than rebuilding Boolean solids.

The [implementation report](development/probe_corrector_details_20261115/README.md)
contains CAD/UI examples, independent clearance/closure checks, exact field
comparisons and bounded actual STEP verification. These are software/geometry
checks, not current calibration, manufacture, beam focus or coherent imaging.

The [earlier turn initialization study](development/corrector_windings_20261115/README.md)
records the original size-based assumptions. The
[multipole current report](development/multipole_current_coils_20261115/README.md)
records the subsequent explicit current calibration and its validation limits.
