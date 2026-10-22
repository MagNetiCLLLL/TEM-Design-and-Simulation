# Component field vectors

Open the **Magnetic field** tab immediately after **Physical Layout**, then
choose **Component field vectors** after a particle calculation.
**Field contributions** selects **All fields superposed**, **Selected component
only**, or the advanced **Checked sources** sum. Selecting a component chooses
its observation envelope separately from which sources contribute.
Selection affects only the diagnostic; it never disables hardware or changes
the propagation calculation.
Use the always-visible **Observe** picker to select an installed physical part,
including passive structures. Advanced source and section controls start
collapsed to preserve plot space on small screens.
When expanded, those controls scroll while the canvas retains at least 220 px
of height so axes and annotations cannot consume the entire field area.

The display offers magnetic **B**, electric **E**, a separate-colour **B + E**
overlay, and electron force. The default **Longitudinal section (Z–U)** uses
Z to the right and `U = X cos(theta) + Y sin(theta)` upwards,
matching Ray Diagram and Physical Layout's rotating section. The viewing angle
is shared in both directions. Ray Diagram, this page and Physical Layout's
rotating section use the same **Angle / X-Z / Y-Z / slider** control at the
upper left. Left/middle drag pans; right drag scales the two display axes
independently, matching Ray Diagram. Dragging never changes the shared angle.
As in Ray Diagram, the
wheel over the bottom axis zooms only Z, and over the left axis zooms only U.
These display scales are independent; wheel zoom inside the plot changes both
while retaining their relative scales, with the same wheel sensitivity as
Ray Diagram. Captions, legends and corners do not zoom the plot. Double-click
resets both scales and the pan without changing the shared observation angle.
There is no transverse XY view or separate XZ/YZ slice selector.
**Rotation changes the actual longitudinal section**, with
`V = -X sin(theta) + Y cos(theta) = 0`. At 0° it samples the XZ plane; at 90°
it samples YZ, still with Z horizontal. The field and material section stay
aligned without a fixed XZ slice collapsing edge-on.
Position controls use millimetres, while the
provider queries use metres. +Z points downstream in a right-handed XYZ frame.

## Quantities and interpretation

- B: tesla. Arrows point along the magnetic field, not electron motion.
- E: volts per metre. E and B use independent display scales; they are never
  numerically added to one another.
- Force: newtons, `q * (E + cross(v, B))` with negative electron charge and
  relativistic velocity calculated from the specified local kinetic energy.
  The energy/direction controls define a test electron at every sampled point.
  They neither reconstruct a trajectory nor impose final gun energy throughout
  the accelerating gun. Changing them does not change the captured E/B values.
- For a section, an out-of-plane field is distinct from a zero field. Read the
  section-normal indication as well as the in-plane arrows.

Arrow scales and sampled extents are display settings. A visible sparse grid
does not establish spatial convergence of the underlying field solution.
**Auto scale** starts enabled and fits B, E and force independently to their
current sampled peaks, including after changing the observation window. Use
**Fixed scale** when comparing excitation changes: it retains the references
across snapshots and navigation. Editing a B/E/F reference selects Fixed scale;
**Use current peaks** updates the references deliberately. Arrows are capped at
four times the reference, and the legend reports samples exceeding that cap.
The hover values and maximum remain unmodified in either mode.
The legend identifies cyan B, amber E and violet force. Hollow squares mark
unavailable samples, hollow circles mark valid zero values, and diamonds mark
nonzero in-plane vectors too short to resolve at the current scale. A circled
dot/cross marks a positive/negative component along the viewing-normal axis.
Zero sampled values do not establish a zero field throughout the column.
Dashed boxes show
physical observation envelopes' enclosing boxes, not exact winding surfaces.

## Source ownership and physical limits

### Physical model overlay

Enable **Physical models** to overlay the selected sources' existing Physical
Layout geometry. **Model opacity** ranges from 0% (invisible) to 100% (opaque).
Selecting an actual component automatically enables its model and highlights
it and its declared children at at most 35% opacity, applied once to the whole
selected group so overlapping surfaces cannot become opaque. Other source contributions
may remain active without displaying their entire hardware around the selection.
Copper windings, magnetic poles and supports keep their material colours.
In a 3D volume the actual surfaces rotate with the field vectors; in the
longitudinal section the overlay is the material intersection at the shared
angle's V=0 plane, using the same section geometry as Physical Layout.
Vectors are drawn in front of the surfaces to keep internal fields visible.

**Fit models** frames the hardware and the current sample volume together.
**Reset view** returns to the sampled extent. These controls, visibility and
opacity never change the physical model or arrow reference values. With
**Auto section detail** enabled, changing the view can sample additional
locations within the existing section and physical envelope.
The near-axis field volume can be much narrower than the bore and winding;
fitting the hardware does not imply that the field outside the sample is known.

Meshes come from the same captured configuration as the field, not a newer
live Physical Layout edit. They share global XYZ coordinates and the existing
mechanical placement rules. Optical-axis tilts do not rotate the material CAD.
Shared hosts are drawn once. Descendant hardware, such as a DPH insert within
an HP assembly, supplies mechanical context without adding its field to the
source selection. Sources without modelled material hosts remain labelled as
unavailable; no body is invented. Geometry is built in the background on first
use and cached until the input snapshot changes.

Select a real body in the vector view, the **Observe** picker or a row in the
source list. Right-click it to jump to Ray Diagram, Physical Layout's rotating section, 3D Parts or the
Vacuum map. Physical Layout's right-click menu also offers **Go to Magnetic
field**, which opens this component view. Jumps use the captured component key
and axial position; a stale snapshot cannot offer a misleading hardware jump.

Only sample locations inside captured physical outer envelopes are displayed.
These include the central bore and internal voids, because fields act there.
A selected component supplies the observation envelope; **Whole column** uses
the union of declared physical envelopes. **Fit selected** frames that region.
For a selected field-producing component, the default sampling window intersects
its physical envelope with its field source's support. This resolves narrow
near-axis fields instead of spending every off-axis sample outside the model.
Use **Fit models** to see the full hardware around that same sample; material
and field positions remain at their true relative scale.
Mechanical placement and supported CAD affine transforms are retained; a global
axis-aligned bounding box alone does not decide membership in a tilted cylinder.
Missing geometry or unsupported imported/bent shapes do not borrow a field
provider's numerical support as a pretend mechanical outline.

The observation mask does not modify propagation. In **All fields superposed**,
every selected captured provider contributes at each retained location,
including fringe fields from other components. It is not a sum of fields each
artificially truncated to its own housing. Changing the contribution mode keeps
the sampling positions fixed. A passive sleeve or independent mechanical copy
can observe the total field, but has no independent source in component-only
mode. A shared nonlinear magnetic circuit or joint electrode solution remains
one complete source and is labelled accordingly.

The backend selects the production magnetic providers already captured by
`prepare_magnetic_scene`, rather than turning lenses on and off and solving a
different instrument. Gun E uses the canonical instrument electric capture,
whose mechanical solution boundary is independent of plotting cutoffs.

Descan and Image/Diffraction share physical coils. Their commands are already
combined by the production drive mapping. Upper/lower stages and their logical
aliases belong to one selectable physical component. Joint nonlinear magnetic
circuits are likewise inseparable sources: individual saturated channel fields
are not additive. The entire jointly solved map is counted once.

Gun electrostatics is a joint electrode-boundary solution, not a collection of
independent electrode contributions. Wien E and B are one selectable crossed
field source; its E is excluded from the separately displayed gun base so it is
not counted twice.

Each source reports its model and numerical support. Near-axis equivalent
multipoles, uniform finite deflector coils, analytic round lenses, and registered
field maps retain their existing approximations. A detailed physical winding
mesh does not turn an equivalent field into a coil-resolved Maxwell/FEM solve.
Passive structures participate only where the production solver includes their
boundary or material properties; they are not invented independent field sources.

Queries outside an active source's radial/map/electric validity remain unknown.
They are masked, not silently zero-filled or rendered as a partial sum. Explicit
disabled channels and modeled finite axial zero continuations remain known zero.
The active electrostatic blanker has no continuous E provider, and the bent
post-column energy filter lacks a straight-column field provider. Selecting these
entries marks their unsupported region, including the downstream diagnostic
continuation, as unknown. Selecting only supported contributions is allowed but
is labelled as a selected sum, not the complete instrument field.

## Update lifecycle

Changing instrument inputs immediately invalidates and clears obsolete vector
data. The next completed calculation publishes the captured state and automatically
refreshes an open vector view. With automatic ray preview enabled this follows
the normal preview lifecycle; otherwise calculate again explicitly.
An old asynchronous job cannot publish over a newer snapshot or section angle.
Zoom and pan first redraw cached vectors. While the angle changes, the last
accepted 2D section stays visible at its original angle, explicitly labelled
with the displayed and requested angles. After a 250 ms pause the latest plane
is sampled from the captured providers and replaces that display atomically.
Pending sections do not offer field hover values or component picking.
Cancelled base requests finish their current chunk before the next request
starts, so repeated angle changes cannot queue overlapping base sampling jobs.
This also applies with automatic viewport detail switched off: a new physical
plane requires new sample locations. Hidden pages defer sampling until shown.
The original field models and particle trajectories are not recalculated.
Section, source and sampling changes reuse the same
captured providers. Opening this page does not launch a magnetic FEM solve.
An **All** selection includes newly appearing sources after recalculation,
including newly active unsupported fields; a manual subset stays a subset.
Canonical gun E capture may reuse or create its normal cached solution in the
background worker; errors remain explicit and do not turn missing E into zero.

## Zoom and display detail

**Auto section detail** starts enabled. After a 250 ms pause in section zooming,
panning or resizing, the visible U-Z area is sampled from the same
captured providers. The grid stays inside the original sampling window and
physical observation mask, including its bore. It does not move the section's
normal coordinate or extend a provider's validity. The default spacing is about
32 screen pixels; the density control changes this spacing. Each request is
limited to 1600 locations, evaluated in chunks of at most 512 on one CPU worker.
Only the latest view can publish. A four-view cache avoids repeat sampling when
returning to a recent view; changing the physical inputs clears that cache.
Hidden pages cancel pending work and resume the current view when shown.

Turn this option off to restore the fixed section samples and make zoom/pan
use cached data only. Rotation still samples the new longitudinal plane.
A 3D volume always retains its bounded original grid
(at most 9 × 9 × 9 locations) when rotated.

Longitudinal grids reserve positions for U=0, nearby off-axis fields, thin
source centres and individual physical lenses in shared sources such as C1/C2.
They redistribute the sampling budget into a nonuniform U-Z grid instead of
increasing it without bound. The complete grid is capped at 1600 locations;
physical masking can reduce that number. Sampling hints use the complete
captured source catalogue, so changing the displayed contributions does not
move the sample locations. Source supports guide placement but never replace
physical masks or provider validity. Explicit grid axes preserve missing rows
and columns for hatching; the display never interpolates across those gaps.

Arrows use screen-space thinning to avoid overlap in the overview, prioritising
nonzero field samples so near-axis fields survive thinning. Zooming reveals
local arrows; Fixed scale retains their SI magnitude references.
Normal-direction dot/cross symbols and unavailable values retain their separate
meanings. Hover reports the actual sampled value of a displayed glyph. Mouse
wheel zoom keeps the physical point under the cursor stationary.

Envelope intersections and material-section paths are cached while panning or
zooming; accepted detail samples from the same captured configuration reuse
the geometry. Whole-column envelope guides are subdued so they do not compete
with the field vectors. Strength hatching uses a labelled, bounded texture
preview during navigation (at most 32,768 pixels), then restores the full
texture after 150 ms idle. This only changes texture resolution: sampled field
values, validity masks, section coordinates and fixed strength references remain
unchanged. The final texture may still take longer to render on complex views.

The **2D combined field** view has its own **Auto detail** switch. It preserves
the full-column 3000-point overview, then samples the visible Z interval after the
same 250 ms pause when the overview is too coarse. The budget is approximately
two samples per pixel, capped at 4096, with one CPU worker and four cached views.
It does not republish a field snapshot or reset electron trajectories, vector
sources, shared axis limits or the fixed field-line strength reference.

The **3D field lines** view retains its captured streamlines. At each view it
chooses up to 160 visible direction arrows at least 20 pixels apart, rather than
spending that budget on offscreen parts of the column. Zooming does not retrace
field lines. Electron paths retain their existing screen-error simplification.

These are display refinements, not solver remeshing or increased physical-model
accuracy. The approach follows the distinction in
[COMSOL plot quality](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_results.37.024.html),
the interaction/detail split described by
[ParaView rendering](https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html#rendering),
and configurable vector spacing in
[Ansys Maxwell vector plots](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v252/en/Subsystems/Maxwell/Content/ReportsandPostProc/CreatingVectorFieldPlots.htm).
The numerical budgets above are this simulator's CPU-oriented implementation
choices, not requirements or defaults of those products.

## CPU strength hatching in 2D sections

In **Magnetic field → Component field vectors**, **Display** switches between the default
**Arrows** and **Strength hatching (2D)**. The adjacent quantity selector chooses
magnetic B, electric E, or electron force. Hatching automatically selects B
when leaving the B/E arrow overlay: different SI units cannot share one
strength scale. Hatching uses the rotating longitudinal U-Z section; selecting
a 3D volume restores arrows. Z stays horizontal for every viewing angle.
The angle control, Ray Diagram and Physical Layout's **Rotating section**
share the same angle while its **Follow Ray Diagram** option is enabled.
Physical Layout's older **2D** schematic remains its existing fixed projection.

Hatching represents the full XYZ vector magnitude, including the component
normal to the section. The selected sources are summed as vectors first.
The display bilinearly interpolates the cached vector components and then
takes their norm; these are approximate display values, not new field solves.
Hover shows the interpolated magnitude and the physical location in SI units.
The original unmasked grid axes accompany each sample result, so interpolation
cannot bridge missing rows or columns. All four cell corners must be present
and valid. A geometric mask also clips the texture to the physical observation
envelopes, including curved and tilted boundaries.

The six nested crosshatch spacings are 128, 64, 32, 16, 8 and 4 logical screen
pixels. They correspond to positive strength below 1/8 reference, then
strength/reference ≥1/8, ≥1/4, ≥1/2, ≥1 and ≥2. Values at or above twice the
reference share the densest pattern; hover still reports their magnitude.
The swatches show these lower thresholds. Zero has a quiet neutral fill with
no hatching. Grey dots mean unavailable model values; fully transparent areas
are outside the observation envelope or lack complete sampled cells.
Missing data must not be interpreted as zero.

The reference controls under **Sources and section controls** apply to both
arrows and hatching. **Auto scale** follows current sampled peaks; **Fixed scale**
retains references across zoom, grid refinement and snapshots for comparisons.
Use **Use current peaks** to deliberately reset the fixed comparison scale.
Hatch spacing is independent of the numerical sampling grid and never indicates
field direction. This is a magnitude texture, not a finite-element mesh or an
isoline plot.

Rendering uses NumPy and Qt's CPU image painter, without CUDA/OpenGL. One
viewport texture is retained, capped at 524288 RGBA pixels (2 MiB) and 1024
pixels on its longest side. Larger views use a downsampled texture with small
area samples to reduce aliasing. Interpolation and physical masking use batches
of at most 8192 texels; physical regions are culled by bounds before their
local shape is tested. Ordinary repaints reuse the texture. Changing display
style or quantity reuses the same field samples; the existing optional
250 ms viewport refinement still governs zoom and pan.

`scripts/preview_field_strength_hatching.py` produces native Qt screenshots at
0°, 45° and 90° and timing/cache receipts using a single production
condenser-stigmator B provider, always in a horizontal-Z longitudinal layout.
It disables CUDA, limits CPU work to one thread, and does not run trajectories,
gun electrostatics, or a finite-element solve. The diagnostic temporary drive
is documented in its output and does not modify the installed instrument.

## Verification

The regression entry points are `tests/test_field_vector_scene.py`,
`tests/test_field_vector_production.py`, `tests/test_field_vector_page.py`, and the
vector integration cases in `tests/test_incremental_magnetic_scene.py`.
`tests/test_field_vector_lod.py`, `tests/test_field_vector_viewport.py`, and
`tests/test_magnetic_profile_lod.py` check visible-area sampling, bounded jobs,
cache reuse, cancellation and fixed references without optical/FEM solves.
`tests/test_field_density_texture.py`, `tests/test_field_density_canvas.py` and
`tests/test_field_density_page.py` check magnitude interpolation, masked gaps,
constant-field stability, texture bounds, cached rendering and the 2D controls.
`tests/test_field_display_regions.py`, `tests/test_field_component_inspection.py`
and `tests/test_magnetic_component_navigation.py` cover physical-envelope masks,
all/single-source inspection and cross-page navigation.
They belong to the existing `field-ui` acceptance scope. Qt checks run offscreen.
`tests/test_gui_signal_receiver_lifetime.py` verifies that plot helpers lose
their signal connections with their owning Qt object, even while Python still
holds the helper. The helpers are QObject children rather than plain Python
receivers, avoiding the multiple-signal receiver lifetime pattern fixed upstream
in [PYSIDE-3148](https://github.com/pyside/pyside-setup/commit/d580aae898e10e7b4491f2692f320d7706899c1e).
This is a compatibility precaution for the pinned PySide6 6.8.3, not evidence
that this upstream bug alone caused every native crash.

The Scanning Image controller remains a hidden Qt child of the workspace after
its parameter and result panes move into their visible tabs. Leaving this
controller parentless retained its signal callbacks, the workspace and all its
plots even after native window destruction. `tests/test_workspace_lifetime.py`
checks ownership, visible-panel routing and release of the workspace and views.

The Qt test teardown flushes deferred deletion, collects Python cycles on the
GUI thread, and flushes any new deferred deletion. A second sweep before the
next test releases fixture values that pytest retains beyond its teardown hooks.
Parent-widget destruction
can leave pyqtgraph menus, axes and ViewBox wrappers in Python cycles until
collection; retaining these across tests increases memory use and leaves dead
views in the shared registry. `tests/test_qt_test_cleanup.py` checks the actual
pytest-qt cleanup across successive tests in a small subprocess, including
plots returned by fixtures, with periodic GC
disabled so cleanup cannot pass accidentally because a timer ran. These checks
use tiny synthetic plots and do not lower numerical solver accuracy.

Opening the main window and selecting a compute policy do not probe CUDA or
CuPy. These queries can initialise a native driver
([Numba host API](https://numba.readthedocs.io/en/0.61.1/cuda-reference/host.html));
they belong to calculation admission, not tooltip construction. CPU integrators
also defer importing CUDA and constructing device kernels until GPU execution.
An explicitly empty or `-1` `CUDA_VISIBLE_DEVICES` returns unavailable before
importing either GPU library, while a previously quarantined context still
reports its original failure. Auto retains CPU fallback and Require GPU still
rejects unavailable devices. `tests/test_gui_backend_discovery.py` checks
window construction and every device-policy selection without a hardware probe;
`tests/test_cpu_backend_isolation.py` checks imports in a fresh process.
Synthetic backend-policy tests use fake CUDA modules rather than loading a
driver just to simulate its absence. `tests/test_lazy_cuda_kernels.py` executes
two rays over two steps using Numba's CPU CUDA simulator to check deferred kernel
construction and numerical consistency; it does not qualify actual GPU hardware.

These checks validate source reuse, equations, selection, validity and publication
behaviour; they do not constitute experimental magnetic/electric field calibration.

Recording-plane observation envelopes now include the saved rectangular camera
footprints and disk/ring detector footprints with their captured placement.
Selecting a camera or detector therefore samples its observation region without
creating a new field source. Unsupported EELS branch geometry and points outside
the modeled field domains remain unavailable, rather than extrapolated. Clearing
or replacing a sample also hides any tooltip from the previous numerical result.
