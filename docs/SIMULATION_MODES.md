# Simulation model levels

The **Simulation** menu sits beside File and View. These choices select column
lens physics, not the number of rays, integration step, CPU/GPU backend or the
High accuracy product request. New desktop windows start in Finite-Field Optics;
legacy states and operating profiles without this setting use Custom so their
previous mixed models are not silently replaced.

| Menu choice | Implemented behaviour |
| --- | --- |
| Ideal Optics (`ideal_thin`) | Fast thin optical events and straight drifts. Native integrated strengths calibrate round lenses, stigmators, deflectors and hexapoles. No finite-thickness dynamics, material field solves, empirical aberration screens or coherent propagation. |
| Finite-Field Optics (`unified`) | Shared immutable electromagnetic potentials and finite absorbing geometry for particles and coherent waves. Field-derived aberrations replace empirical residual actions and equivalent Image maps in the owned column interval. Explicit support gates apply. |
| Parameterized fields (legacy, `ideal`) | Existing parameterised finite column fields, configured spherical/chromatic aberrations and corrector multipoles. Actual particle and wave-mode energies are retained. Pole shape and magnetic materials are not used to solve the field. |
| Analytical Field | Compatibility profile within Finite-Field Optics, retaining the same parameterised laws. Registered spatial maps and geometry recipes are retained but inactive. |
| Linear Geometry Field | All enabled column round lenses require explicit linear FEM recipes. Missing recipes or unsupported geometry fail; no automatic analytic fallback. |
| Nonlinear Material Field | Explicit sourced B-H recipes, jointly solved static axisymmetric fields. No hysteresis or thermal feedback; unsupported/missing inputs fail. |
| Coupled Multiphysics | Disabled. Circuit/thermal/structural feedback is not implemented. |
| Custom / Per-lens Models | Existing per-lens analytic, geometry and imported-map configuration, including legacy behaviour and its explicit approximation warnings. |

## Model boundaries

The legacy parameterized entry uses the existing magnetic-field description. It
retains finite-length focusing, rotation, dipoles, quadrupoles, HP hexapoles,
configured lens aberrations, physical apertures and detector interception.
DPH therefore changes the trajectory inside HP, including the nonlinear effect
of passing off-axis. It does not mean a perfectly corrected or aberration-free
instrument. Explicit operator switches used by diagnostic comparisons still
permit a separate linear reference; these do not define the new Ideal menu mode.

Pole-piece shape, permeability and B-H saturation do not solve a magnetic
boundary-value problem in the parameterized or compatibility field sources. Installed positions,
lengths, field-profile parameters, alignment and physical obstructions still
matter. Geometry/material modes select the field provider, not whether
correctors are active. Both particle and wave propagation retain each state's
energy instead of silently replacing it with the nominal energy.

The electron-gun calculation and Energy Filter's dedicated dispersive transport
are not replaced by this column-lens selector. It does not switch TEM/STEM modes,
enable wave imaging, change sample materials or erase scattering signals.
Configured manual wave-aberration coefficients remain active in the legacy parameterized entry and
Analytical Field. The existing Field-derived coefficient fitter requires a
registered or generated spatial field map; it is unavailable in these two
parameterised modes. An incompatible saved fit request is reported rather
than replaced with fabricated fitted coefficients. Editing manual coefficients
does not switch to Custom or activate an inactive field map. Coefficients not
fitted or measured stay labelled as such; choosing a model does not establish
calibrated microscope performance.

The historical `ideal` key changed to these finite-field laws on 10 October
2026. It is now displayed as Parameterized fields (legacy) and keeps those laws. The new
Ideal Optics uses the distinct `ideal_thin` key; old files are never silently
converted to thin lenses. Analytical Field also retains its existing key.

Fresh windows select `unified`; saved files retain their original keys. This
route preserves source, specimen and detector responsibilities while replacing
column actions with the shared potential/geometry execution. Stored Cs/Cc,
intrinsic residuals and equivalent-image settings are retained for legacy runs
and are not added a second time. Field excitation, pose, effective thickness,
apertures and retained-potential recipes control the new propagation.
Changing a lens model recipe keeps `unified` selected and rechecks capability.

The CPU coherent route includes actual static gun-tail acceleration and native
axial scan coils. It evaluates the latter in an explicit retarded reference-
wavefront approximation, retaining induced work and raw phase. Unknown domains,
B-only maps, unqualified temporal drives and raster flybacks produce explicit
errors. No numerical tolerance converts a nonzero gun tail into zero electric
field. An exactly
constant grounded-downstream potential has a separately recorded reference
conversion with unchanged domain and executed kinetic energy. The source is
always the executed gun wave. Source, sampling, memory and cumulative error gates
remain active; the interface does not substitute a synthetic source or migrate
to legacy execution. See [electric tails and native scanning](UNIFIED_DYNAMIC_EXECUTION.md)
for the scope of the source-wavefront approximation and the separate material
and acquisition limitations.

The thin executor collapses each calibrated field integral to its placed
reference plane. It retains signed magnetic rotation and actual-energy
scaling, but not propagation inside an element, overlap dynamics or empirical
Cs/Cc screens. Field widths can still change a calibrated integral at fixed
peak strength; that is different from resolving thickness during transport.
It uses exact straight drifts and event nodes, so the particle integration
step does not create unnecessary free-space steps. Plot spacing and physical
apertures still control observation nodes. Gas-collision transport and coherent
propagation require Finite-Field Optics. The gun source and acceleration,
specimen interactions and dedicated filter remain separate existing models.
Finite-field operating presets can lose focus or transmission under this
approximation; selecting a model never retunes their currents automatically.

See [independent numerical inputs](CALCULATION_MODES_AND_NUMERICS.md) for the
particle, coherent grid, energy and memory controls.

## Controls and state

- The menu is exclusive. Unsupported tiers are visibly disabled. Linear Geometry
  becomes selectable when every enabled round lens has a recipe and no declared
  saturation insert requires an unsupported solver. This is a readiness check,
  not a mesh-convergence or geometry-overlap certification.
- **Configure lens models...** opens Model Inspector. Explicit field/model edits
  there, or importing/clearing a map, select Custom and retain current operating
  values. This avoids editing a recipe that the active global tier would ignore.
- Unconfigured lens drafts use the sourced pure-iron [material defaults](LENS_MATERIAL_DEFAULTS.md):
  constant reference permeability for Linear and the full B-H table for Nonlinear.
  Coil ampere-turns remain explicit; defaults do not bypass setup requirements.
- Nonlinear Material Field similarly requires B-H recipes for all enabled round
  lenses. See [static nonlinear fields](STATIC_NONLINEAR_MAGNETICS.md) for reference
  materials, joint-current routing and the distinction from multiphysics.
- Each mode retains field recipes, probe/image coefficient options, the additional
  chromatic-kick setting and the existing equivalent-image-lens option. Switching
  models preserves the current lens percentages and polarities. Historical shelf
  excitation records remain readable; only an explicit operating-profile load
  applies saved excitation. No cross-model strength matching is performed.
- Geometry, installations, enabled hardware, source, apertures, specimen,
  detector controls and numerical resolution are shared, not copied into model
  shelves. A shelf cannot restore an old mechanical layout.
- The current model is shown in the status bar and completed-calculation status.
  Model changes invalidate pending workers immediately; previous plots remain
  explicitly stale until a new result is accepted. Other complete results stay
  in the existing bounded result cache.
- Operating-profile format 6 persists the selection and shelves with an explicit
  migration report; formats 1-5 remain readable. Format 4 also preserves nullable
  operating values, including automatic Cs/Cc estimation. Imported arrays are not
  embedded in TOML; their original file descriptors retain the normal provenance
  and geometry checks.

## Cache and numerical verification

The active model participates in stage/result identities and the propagation
plan signature. Inactive shelves are excluded from calculation identities, so
returning to unchanged settings can reuse their old result. Shared physical
geometry changes still invalidate affected products; retaining results never
means accepting an incompatible field map. The persisted solver identity is
`temsim-solver-2026-09-static-bh-v1`.

`tests/test_simulation_modes.py` checks selection, invalid-mode rejection, profile
and State round trips, inactive-shelf identities, map bypass/restoration, actual
linear FEM routing, active nonlinear corrector/Cs coefficients, actual-energy
transport, diagnostic linear references, and menu/cache behaviour.
Synthetic test inputs are not OEM dimensions, coil ratings or material data.
These tests do not establish real-microscope accuracy. Nonlinear numerical tests
and their boundaries are documented separately; multiphysics is not implemented.
