# Deflector and stigmator windings and prescribed currents

Physical deflector hosts and material stigmators now have separate winding
geometry and electrical calibration inputs. The construction uses public
saddle-coil, flexible-coil and radial slotted-carrier families; saved dimensions,
wire sizes and initial turn counts remain engineering estimates, not OEM data.
Separate OEM cartridges are not inferred from unverified control names.
The named corrector multipoles use their own explicit engineering winding-only
recipes and calibration; see [multipole winding and current inputs](MULTIPOLE_CURRENT_WINDINGS.md).

## Edit a winding

1. In **Physical Layout**, select the material deflector body, the DPH/DP
   **steering coils** insert, or the material stigmator. Select **Configure
   winding...**. A logical control channel or replaced/imported CAD does not
   expose this native winding action.
2. Set the positive integer **turns per physical coil**, conductor **metal
   cross-section in mm2**, and **radial wire-insulation thickness in mm**.
   The displayed bare-conductor diameter is calculated from metal area and is
   read-only. Known turns may be saved with unknown metal area, retaining bulk
   CAD. Insulation size without a turn count is invalid.
3. The layout includes every requested closed turn and automatically fills the
   remaining allocated winding frame with insulation. The original bore,
   winding inner/outer envelope and axial bounds stay fixed. Impossible packing
   fails; conductors are not shrunk and turns are not silently dropped.
4. Where electrical calibration is available, edit the signed **additional
   current in amperes** for each physical coil channel. These biases add to the
   calibrated native-command current without changing the native commands.
   The original current matrix and reference energy are read-only.
5. **Apply to draft** creates one Undo operation. **Save** writes the definition;
   the live instrument changes only when it loads the saved definition. A
   changed draft cannot accept values from an older open winding dialog.

Clearing turns and size fields removes the native winding recipe and its
electrical definition. The old bulk geometry and native-field admission remain
available. Importing/replacing CAD instead retains the recipe inactive, so that
restoring native geometry validates the saved inputs again.

## Current channels and units

The electrical source is

`I = M @ u + bias`,

where `I` and `bias` are signed physical coil-channel currents in amperes.
For a two-station deflector, rows and columns are ordered station 1 X, station
1 Y, station 2 X, station 2 Y. The complete matrix includes coupling between
stations. A single-station component has two channels. Radial stigmator winding
groups and optical normal/skew directions can have a phase difference; the
saved matrix includes the signs and channel conversion.

| Component | Native command vector `u` | Matrix units |
| --- | --- | --- |
| Column deflector | Native X/Y deflection commands, in mrad | A/mrad |
| Column stigmator | Native X/Y strength percentages; 100 means 100% | A/percent |
| Gun deflector | Upper/lower transverse Bx/By field commands, in mT | A/mT |
| Gun stigmator | Normal/skew gradients from `gradient*cos(2*rotation)` and `gradient*sin(2*rotation)`, in T/m | A/(T/m) |

The current matrix matches the previous signed peak field or gradient at the
declared native reference planes and calibration energy. It is fixed after
calibration: changing acceleration voltage does not rescale the actual current
or magnetic field. Individual electrons respond using their own momentum.

The dialog's current readback is a preview of available runtime or saved native
commands. Shared scan contributions, blanking and time-dependent commands must
be inspected in the runtime magnetic-field diagnostics; the dialog does not
execute a scan or mutate live state.

## Source model and editing consequences

Electrical inputs drive **Ideal prescribed-current coils in vacuum**, whose
compatibility key is `copper_coil`. Other field selections keep their native
field providers. This model integrates closed current circuits over the
original homogenized winding-pack envelope. Magnetic cores, pole shoes,
permeability, resistance, heating, current limits and drive-electronics dynamics
are excluded. Material CAD remains distinct from active field sources.

Changing turns or the current-bearing envelope retains the numerical current
matrix and biases, marks the previous field-strength fit stale, and does not
perform a new fit or focus adjustment. A manual current-bias edit likewise does
not assert that the old field target still holds. Metal-area or wire-insulation
edits change detailed CAD and packing at fixed N, I and envelope; they do not
compress the homogenized current source. Calculation/cache dependencies still
include saved source definitions and implementation identity.

Matching a peak does not reproduce the old complete axial field profile,
integrated field, fringe field, focus or aberrations. Those differences belong
in the validation report rather than being hidden by optical retuning.

## Saved fields

- `transverse_winding_turn_count_per_coil`
- `transverse_winding_wire_cross_section_mm2`
- `transverse_winding_insulation_thickness_mm`
- `transverse_winding_geometry_status` and `transverse_winding_source_note`
- `ideal_transverse_current_matrix_a_per_unit`
- `ideal_transverse_current_bias_a`
- `ideal_transverse_calibration_reference_energy_ev`
- `ideal_transverse_calibration_kind`, `ideal_transverse_calibration_status`
  and `ideal_transverse_calibration_source_note`

Source notes and per-parameter provenance distinguish initial engineering
estimates, calibrated currents and later explicit edits. An unchanged Apply
preserves full saved values, provenance and Undo history.
