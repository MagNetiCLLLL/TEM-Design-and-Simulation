# Corrector multipole windings and ideal prescribed currents

The named Probe and Image HP1/HP2, QPol/QPC/QPH1/QPH2 and HPol/HPC channels
support explicit engineering winding recipes and current-to-field conversion
in **Ideal prescribed-current coils in vacuum** (`copper_coil`) under unified
execution. Monochromator and Energy Filter magnetic sources are outside this
change. A mechanical winding recipe and an electrical calibration are separate
inputs; neither establishes OEM geometry or experimental calibration.

## Winding ownership and physical boundaries

Main HP1/HP2 windings use one set of six radial closed coils with alternating
current senses. Probe HP retains its existing physical per-coil turn count and
canonical `hexapole_winding_*` size fields. Image HP uses the new multipole
fields when its count was previously undefined. Main HP uses one signed
electrical drive with a saved native orientation reference.

Auxiliary quadrupoles have two four-coil groups separated by 45 degrees;
auxiliary hexapoles have two six-coil groups separated by 30 degrees. They use
the existing channel's allocated annular envelope and optical reference.
Their explicit winding-only CAD is an engineering reconstruction, not a new
opaque cartridge or proof of a proprietary winding location. The pre-existing
transport role, cartridge ownership, clear bores and clipping boundaries remain
authoritative. No extra native field is added on top of the same current source.

The angular datum is shared by CAD and source geometry. A native optical
orientation is a separate value. Main HP records its calibration orientation;
later native orientation changes rotate the whole HP cartridge and source
relative to that reference, without changing current or inventing a second
winding. Auxiliary hexapole normal/skew commands rotate through the fixed
two-channel current matrix. Current readbacks and CAD refresh dependencies
retain that distinction.
An independent nonzero main-HP skew command remains unsupported by its single
six-coil basis and fails explicitly.

## Edit inputs

1. In **Physical Layout**, select the named multipole and open **Configure
   winding...**. Replaced/imported CAD does not expose the native action.
2. Enter positive integer **turns per physical coil**, conductor **metal
   cross-sectional area in mm2**, and **radial wire-insulation thickness in mm**.
   Bare-wire diameter is calculated and read-only. Known turns with unknown
   metal area retain bulk winding CAD.
3. Review fixed-envelope packing. Every requested closed turn is retained,
   and the remaining winding allocation is filled with insulation. An
   impossible layout fails; conductors are not shrunk or silently dropped.
4. Where a saved electrical calibration exists, enter signed **additional
   current in amperes** for each physical winding channel. These are biases,
   added to the current produced by the existing native commands. They are
   not absolute-current readbacks.
5. **Apply to draft** creates one Undo operation. **Save** persists the draft;
   the live instrument changes when it loads that saved definition. An older
   open dialog cannot overwrite a draft edited after it opened.

The Dimensions panel supports the same N, metal area, insulation and signed
bias edits. The winding angular datum is an explicit engineering mounting
input. Calibration matrices, reference energy and reference orientation are
read-only. Clearing turns and size removes the winding recipe and electrical
definition together, retaining legacy native-field admission. Undo, linked
catalog save/reload, optional-field deletion and provenance follow the same
atomic transaction. Existing Probe HP CAD-only inputs remain readable.

## Native commands, amperes and field calibration

The electrical contract is

`I = M @ u + bias`.

| Source | Native vector `u` | Matrix units |
| --- | --- | --- |
| Main HP1/HP2 | Signed main hexapole coefficient | A/(m^-3), one channel |
| Auxiliary quadrupole | Normal and skew coefficients | A/(m^-2), two channels |
| Auxiliary hexapole | Normal and skew coefficients | A/(m^-3), two channels |

The legacy Image QPol owner exposes only a scalar normal coefficient. Its
second physical winding group is an explicit engineering 45-degree extension
of that reference, with zero native skew command and an independently editable
ampere bias. It is not presented as a previously existing native knob or a
measured second-channel calibration.

Native percent controls are normalized operating controls; their percentage
must first resolve to the declared native coefficient. They are not amperes.
The saved matrix includes winding phase, signs and native basis conversion.
Physical coil-group labels do not directly specify the electron focusing axis.

Current matrices are fitted at the original reference plane to the signed
native quadrupole gradient or hexapole quadratic-field coefficient. The source
integrates closed radial circuits over the fixed homogenized winding pack,
retaining N and I. It excludes magnetic cores, pole shoes, permeability,
saturation, resistance, heating, current limits and lead routing. Detailed CAD
wires and homogenized field quadrature are distinct representations.

The numerical gains stay fixed after calibration. Changing beam energy does
not rescale actual current or magnetic field; particles still respond with
their actual momentum. A turn, winding-envelope, angular-datum or relevant
reference-field edit retains numeric gains and biases, marks the fit stale,
and does not repeat a current fit or refocus the beam. A bias edit likewise
does not assert that the old field target still holds.

Metal area and insulation repack CAD at fixed N, I and envelope; they do not
compress the homogenized electrical source. Cache dependencies include these
saved definitions conservatively, along with full all-Z source geometry,
native commands, poses, installation gates and implementation identity.
An upstream Gun or incident-probe result cannot reuse a stale multipole tail.

Matching a reference coefficient does not reproduce the previous Gaussian
axial width, integrated multipole action, focus or aberrations. A winding's
ampere-to-field calibration is not an ampere-to-A1/A2 coefficient calibration
at the specimen. That response also depends on the full installed optics.
The optical working point is not adjusted by this input workflow.

## Saved fields and APIs

Common geometry fields are `multipole_winding_insulation_thickness_mm`,
`multipole_winding_basis_angle_deg`, `multipole_winding_geometry_status` and
`multipole_winding_source_note`. Auxiliary and previously undefined Image HP
N/area use `multipole_winding_turn_count_per_coil` and
`multipole_winding_wire_cross_section_mm2`. Probe HP keeps its existing
`hexapole_winding_turn_count_per_coil` and
`hexapole_winding_wire_cross_section_mm2`; competing duplicate values are
rejected.

Electrical fields use the `ideal_multipole_` prefix: current matrix, signed
bias, calibration reference energy, native orientation reference, kind,
status and source note. Main HP's orientation reference is required.

```python
document.configure_multipole_winding(
    "probe_qpol_quadrupole",
    turn_count_per_coil=4,
    wire_cross_section_mm2=0.01,
    insulation_thickness_mm=0.005,
    current_bias_a=[0.02, -0.03],
)
```

These are illustrative edit values, not a calibrated default. Signed biases
require an existing saved matrix. Omit `current_bias_a` to preserve it.
`turn_count_per_coil=None` with no sizes clears both recipes. Legacy
`configure_probe_winding` edits of a newly calibrated Probe HP route through
the same transaction so they cannot leave stale unlabelled gains or phantom
electrical inputs.

Validation distinguishes metadata/CAD/field-array checks from finite-source
transport, focus, manufacturing and CUDA coherent qualification. The ideal
prescribed-current model admits local stationary mixed particle-wave STEM
subject to actual per-request field and numerical qualification. The general
column wave kernel and arbitrary downstream mixed observations remain
unqualified and rejected. No native multipole may be substituted, live drive
frozen or winding tail truncated to bypass a boundary. See the
[local mixed STEM admission](MIXED_STEM_SCAN.md#ideal-prescribed-current-coils-in-vacuum).

The [executed multipole current report](development/multipole_current_coils_20261115/README.md)
binds installed input values, reference fits and independent validation evidence.

## Internal opposing currents in fixed winding envelopes

Ideal vacuum coils can additionally retain signed internal closed-current
groups without changing their master current, geometry, physical turns or
normal/skew commands. Round lenses use the Model Inspector's **Internal coil
currents...** editor and `temsim.ideal_coil_segments.set_current_segments`.
Probe HP1/HP2 use the distinct
`temsim.ideal_transverse_segments.set_transverse_current_segments` API.
Their radial/nested pack fractions represent complete closed subwindings,
not independently driven sides of an open wire or additional normal/skew axes.
The parent physical turn count and full occupied winding domain remain intact.

These recipes run only in the ideal prescribed-current vacuum-coil mode.
They remain dormant in other modes. Saved inputs and runtime, Gun and scan
dependencies retain their identity; edited currents cannot reuse stale
transport. Equal ratios of one recover the original uniform source exactly.
Opposing regions are not deleted when their total signed NI is zero.
No resistance or constant-power constraint is imposed.

The [fixed-envelope current-fit report](development/ideal_current_fit_20261116/README.md)
records the failed bounded reconstruction and its numerical conditioning.
No fitted candidate was installed and no focused-probe or scanning-image
qualification follows from this electrical input feature.
