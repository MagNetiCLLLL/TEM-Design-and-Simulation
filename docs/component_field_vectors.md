# Component field vectors

Open the **Magnetic field** tab immediately after **Physical Layout**, then
choose **Component field vectors** after a particle calculation.
The source checklist selects one physical source or a sum of selected sources.
Selection affects only the diagnostic; it never disables hardware or changes
the propagation calculation.

The display offers magnetic **B**, electric **E**, a separate-colour **B + E**
overlay, and electron force. XY, XZ and YZ sections use global column coordinates;
3D volume mode permits rotation. Position controls use millimetres, while the
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
Each quantity initializes its arrow reference from its first nonzero sample and
then keeps that reference across snapshots. Edit B/E/F reference values or use
**Use current peaks** to rescale deliberately. Arrows are capped at four times
the reference for readability; the hover values and maximum remain unmodified.
Hollow squares mark unavailable samples, while a circled dot/cross marks a
positive/negative component along the named section-normal axis. Dashed boxes
show enclosing model support, not the physical winding geometry.

## Source ownership and physical limits

### Physical model overlay

Enable **Physical models** to overlay the selected sources' existing Physical
Layout geometry. **Model opacity** ranges from 0% (invisible) to 100% (opaque).
Copper windings, magnetic poles and supports keep their material colours.
In a 3D volume the actual surfaces rotate with the field vectors; in an XY,
XZ or YZ section the overlay is the material intersection at the sampled plane.
Vectors are drawn in front of the surfaces to keep internal fields visible.

**Fit models** frames the hardware and the current sample volume together.
**Reset view** returns to the sampled extent. These controls, visibility and
opacity never change field values, sample positions or arrow reference values.
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
An old asynchronous job cannot publish over a newer snapshot. Camera changes
only redraw cached vectors. Section, source and sampling changes reuse the same
captured providers. Opening this page does not launch a magnetic FEM solve.
An **All** selection includes newly appearing sources after recalculation,
including newly active unsupported fields; a manual subset stays a subset.
Canonical gun E capture may reuse or create its normal cached solution in the
background worker; errors remain explicit and do not turn missing E into zero.

## Verification

The regression entry points are `tests/test_field_vector_scene.py`,
`tests/test_field_vector_production.py`, `tests/test_field_vector_page.py`, and the
vector integration cases in `tests/test_incremental_magnetic_scene.py`.
They belong to the existing `field-ui` acceptance scope. Qt checks run offscreen.
These checks validate source reuse, equations, selection, validity and publication
behaviour; they do not constitute experimental magnetic/electric field calibration.
