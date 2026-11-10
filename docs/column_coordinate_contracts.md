# Column coordinates at the blanker and energy filter

The microscope column is the fixed right-handed reference. The FEG tip apex
is Z=0; +Z points downstream. Positive azimuth turns column +X toward +Y
about +Z. A camera image or Larmor rotation does not redefine these axes.

## Electrostatic beam blanker (NanoPulser)

`azimuth_deg` is the shared plate-assembly and field azimuth in degrees.
The saved parameter name is retained for compatibility.

- At 0 degrees, the two plates are separated along column X. The positive
  plate is on +X, the negative plate on -X; their width is along Y.
- At 90 degrees, the gap and positive-electron-deflection direction are +Y.
- `voltage_v` is positive-plate potential minus negative-plate potential.
  For positive voltage, the electric field points toward local -X and the
  electron impulse points toward local +X. Negative voltage reverses the
  impulse without rotating or exchanging the physical plates.
- CAD, native part/assembly previews, exported solids, and the existing
  integrated electric kick use the same `blanker_rotation` basis.
- Editing the angle invalidates both part and assembly geometry caches.
  Standalone assembly reviews also capture the angle from their state.

This is an assembly rotation. It is distinct from a field-only azimuth
error or a tilt about a transverse diameter. The physics remains the
existing static paraxial kick with a downstream absorbing stop; this change
does not introduce a finite electrostatic boundary solve or pulsed/RF model.

## Energy-filter entrance and internal frame

The public ray entrance coordinates `x_mm`, `y_mm`, `tx_rad`, `ty_rad` and
the controls `alignment_x_mrad`, `alignment_y_mrad` use column X/Y.
Slopes are transverse momentum divided by column longitudinal momentum.
Alignment is added in the column frame before conversion to the solver.

The CAD and solver share one proper branch-to-column rotation:

```text
F = [[0, 0, -1],
     [0, 1,  0],
     [1, 0,  0]]

column position relative to entrance = F @ branch position
column vector                       = F @ branch vector
branch vector                       = F.T @ column vector
```

Add the actual entrance Z to the column position when absolute coordinates
from the tip are required. Vector conversion does not translate or change
units. The shared entrance helper accepts metres and dimensionless slopes:

```text
column position  (x, y, 0)   -> branch position  (0, y, -x)
column direction (tx, ty, 1) -> branch direction (1, ty, -tx)
```

This replaces the previous reflected entrance-X convention. The same
conversion is used by production Boris transport and the adaptive reference
validation. Existing branch magnetic fields, multipole frames and the sector
geometry retain their proper orientation; they are not mirrored or retuned.

After bending, each element uses a transverse frame following the beam:

| Location | Local X in column | Local Y in column | Local longitudinal axis |
|---|---|---|---|
| Entrance, M01-M03 | +X | +Y | +Z |
| After the default 90-degree bend, M04-M10 and downstream devices | -Z | +Y | +X |

The slit gap and detector spectral width use local dispersive X; strip
selection uses local non-dispersive Y. These remain explicitly local exit
coordinates, not column X/Y labels. Stored branch trajectory arrays retain
their documented internal frame. The branch side-view labels identify
column Z from the entrance and negative column X; the 3D assembly uses the
mounted column frame.

## Saved-state compatibility

New serialized filter records carry
`entrance_coordinate_convention = "column_xy_v1"`. New operating profiles
carry the equivalent `coordinate_conventions.energy_filter_entrance` tag.

For an untagged older filter record or profile, loading negates only the saved
filter `alignment_x_mrad`. This preserves that control's previous physical
direction. The source file is not rewritten. A subsequent save writes the
new tag, preventing repeated conversion. Y alignment, multipole strengths,
slit settings, and NanoPulser angle/voltage values are retained. Unknown
coordinate tags are rejected.

Complete working-point object graphs, including graphs with pinned array
assets, carry `coordinate_conventions.energy_filter_entrance = "column_xy_v1"`.
These graphs bypass profile migration and restore all values exactly. Both
entrance conventions existed in untagged graphs, so a nonzero untagged X
alignment is ambiguous and restoration is rejected with an explanation.
The record remains available for historical viewing. A zero untagged X
alignment is safe to restore; recapture adds the tag, while the original graph
and checksum remain unchanged. Explicit but unknown or incomplete coordinate
metadata is rejected. Disabled filter hardware follows the same rule because
its saved controls can later be enabled.

Direct current API assignments are already in column coordinates. Actual
incoming electron positions and slopes use the corrected transformation;
they are not reflected to reproduce historical results. Previously computed
off-axis filter results should therefore be recomputed with the corrected
model. No lens refitting or current rescaling is part of this change.

## Validation on 2026-11-07

The focused regression set covers 175 distinct tests. All have passed after
the final targeted rerun. The last broad run passed 174/175; the remaining
adaptive-reference test had initially requested precision far tighter than
its configured 1 pm absolute position tolerance. That comparison was set to
the declared 1 pm budget, and all 15 boundary/clock/reference tests passed.

Checks include actual CAD plate faces at 0, 90, 180 and -45 degrees;
voltage reversal; unchanged blanker-off transport; part/assembly cache
refresh; column/branch round trips; CAD frames for all ten filter
multipoles; signed entrance shifts and tilts; slit and detector arrivals;
Boris/adaptive reference alignment; and Qt controls and persistence.

One pre-existing blanker test compared cache/provenance hashes for an absent
versus an uninstalled device. Its failure was reproduced with the original
blanker implementation. The assertion now compares complete transported
coordinates and slopes and the interception mask, which directly checks the
intended physical invariance while retaining scene provenance.

GUI checks used Qt offscreen. Numerical checks used analytical/operator or
chief-ray fixtures; no new finite-source trial or coherent propagation was
run. The checks establish the coordinate contract, not OEM calibration or
new working-point qualification.

Coordinate tests are maintained in `tests/test_nanopulser_assembly.py`,
`tests/test_energy_filter_model_3d.py`,
`tests/test_energy_filter_current_contract.py`,
`tests/test_energy_filter_profile_model.py`, and
`tests/test_parameter_semantics_ui.py`, alongside transport tests in
`tests/test_energy_filter_time_of_flight.py` and the existing reference suite.
Detailed run records are under `outputs/assembly-axes-20261107/`.

## Snapshot repair and test consolidation on 2026-11-08

Complete object graphs now carry the same explicit column entrance convention
as profiles. Regression cases cover exact restoration, pinned-array identity,
safe untagged zero alignment, ambiguous untagged nonzero alignment (including
disabled hardware), and unknown metadata. Historical inputs are never rewritten.

Coordinate tests were consolidated into the module suites listed above; FEG
extractor and rear-assembly coverage now shares `tests/test_feg_geometry.py`.
The repair run passed 453 tests, with one full finite-source propagation test
deliberately deselected. Native rear CAD materials and face metadata were also
verified for FEG and FEG_Mono. Qt was offscreen; no new finite-source or coherent
propagation was performed. Records are in `outputs/project-repair-20261108/`.
