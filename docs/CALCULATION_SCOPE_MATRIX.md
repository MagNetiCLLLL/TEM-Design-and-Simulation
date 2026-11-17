# Calculation modes: physical scope and numerical cost

Reviewed against the executable entry points on 16 November 2026. This is a
source-level capability inventory, not a new simulation, mesh-convergence
study or real-instrument qualification. Historical operators and archived
results do not establish an available public calculation route.

## Independent selections

A calculation is a combination of these selections, not one ascending list
from Preview to a complete physical microscope:

| Selection | What it controls |
| --- | --- |
| Column execution model | Thin optical actions, compatibility ray equations, or shared electromagnetic potentials and finite geometry |
| Round-lens field source | Gaussian/saved maps, Ideal prescribed-current coils in vacuum, or explicitly configured geometry/material field recipes |
| Gun and source | Physical emission law, source distribution, extraction/acceleration field and particle transport |
| Requested product | Optical reference, material particle paths, particle raster, local mixed wave, mixed particle-wave raster, receiver or spectroscopy |
| Numerical quality | Original source population, integration mesh, retained history, wave grid/modes/slices and acceptance budgets |
| Execution backend | Supported CPU, Numba or CUDA operations; a device choice does not add physics |
| Optical operating point / assembly | STEM/TEM illumination, projector settings and installed condenser/corrector topology; these specify hardware/operation rather than model fidelity |

Changing the column field model preserves lens excitations but can change focus,
aberrations and transmission. Model selection does not match focal lengths or
retune the probe. A more detailed field source does not automatically select a
more general trajectory equation or enable specimen/wave products.

## Column model menu

| UI entry / saved key | Included system conditions | Main approximation or restriction | Main cost |
| --- | --- | --- | --- |
| Ideal Optics / `ideal_thin` | Thin-lens approximation: exact straight drifts and placed integrated round-lens, rotation, dipole, quadrupole and hexapole actions; actual-energy scaling; positions and physical apertures | No resolved propagation inside each optical element or overlapping finite fields; configured empirical Cs/Cc screens inactive; no coherent propagation. Integrated nonlinear multipoles can still produce aberrations. Gun and specimen responsibilities remain separate. | Lowest column transport cost; events and observation nodes |
| Finite-Field Optics, Shared potentials and geometry / `unified` | One captured A/phi definition and finite absorbing geometry; forward-Z relativistic Lorentz RK4 particles with executed kinetic energy and flight time; admitted prescribed scan drives include induced electric work | A shared execution framework, not a magnetic material solver. It uses a forward axial chart, not arbitrary turning/backward 6-D plasma transport. No duplicate empirical Cs/Cc or equivalent-image actions in its owned interval. Public product gates still apply. | Field evaluations at integration stages, geometry contacts and adaptive steps |
| Parameterized fields (legacy) / `ideal` | Finite analytical lens fields, rotation, correctors, configured Cs/Cc and actual particle energies; physical apertures and detector interception | Historical key is not thin optics. Native round fields use compatibility paraxial equations. An explicitly enabled equivalent-image reduction remains a reduced interval. Pole geometry and B-H properties do not solve the field. | Moderate finite-field ray transport |
| Analytical Field (compatibility) / `analytical` | Compatibility alias for the preceding parameterized field laws, including configured correctors/aberrations | Registered geometry/maps are inactive in this selection; not a new Maxwell solver | Similar to parameterized compatibility transport |
| Linear Geometry Field / `linear_geometry` | Explicit axisymmetric linear FEM recipes for enabled round lenses; declared coil current, pole/coil geometry, mesh and constant permeability | Setup required for every enabled round lens; no analytic fallback. Constant permeability has no saturation/hysteresis. The global compatibility execution is not automatically replaced by unified Lorentz transport. | Field solve/cache preparation plus mapped-field transport |
| Nonlinear Material Field / `nonlinear_material` | Joint static axisymmetric isotropic B-H field solution at the selected current vector, with explicit material and geometry | Setup required; shared material/mesh settings and compatible rigid placement required. No hysteresis, eddy currents, thermal feedback or arbitrary 3-D deformed pole solve. Compatibility trajectory routing remains distinct from the field solver. | Highest built-in static magnetic field preparation cost; nonlinear solve and mapped transport |
| Custom / Per-lens Models / `custom` | Explicit per-lens native, generated or imported field recipes; supported rigid placements | Mixed fidelity, not a uniform highest level. Compatibility B-only RZ/XYZ maps do not automatically become shared wave potentials. Actual map domain and recipe admission govern use. | Depends on each recipe and integration route |
| Coupled Multiphysics / `coupled_multiphysics` | Planned coupled electromagnetic, circuit, vacuum, thermal and mechanical response | Disabled; no executable coupled system model | Not available |

Compatibility mapped/posed vector-field branches can retain full Lorentz-slope
contributions for those fields while native round fields retain their paraxial
equations. A FEM field label therefore describes the field source, not a
guarantee that the whole instrument uses unrestricted Lorentz dynamics.

Native Gaussian round lenses captured by the production shared executor use
the third-order Aphi expansion. A separate potential API supports additional
orders, but these are not silently selected by High accuracy. Likewise, the
existence of general normal/skew multipole operators through twelve poles does
not enable every harmonic in every column component. Production column
capture retains the admitted native dipole/quadrupole/hexapole models; a
corrector's mechanical pole count is not an inventory of executed harmonics.

## Round-lens field source and mechanical dimensions

| Choice | Geometry/material conditions actually used | Not inferred |
| --- | --- | --- |
| Gaussian calibration / saved lens models, fixed calibration | Existing Gaussian amplitudes, widths and positions, or the explicitly selected saved map/recipe; native non-round sources remain | Mechanical pole/bore edits do not automatically reconstruct Gaussian B or Cs. A bore can still change interception and valid geometry. |
| Link Gaussian widths to coil length / `axial_length` | Explicit actual/reference current-bearing coil-length ratio scales native Gaussian widths and internal offsets; optical centres, amplitudes, excitation and polarity remain | An engineering scaling rule, not a winding/material solve; no focus or B-squared-integral normalization. Saved mapped recipes retain their own rules. |
| Ideal prescribed-current coils in vacuum | Declared owned winding envelopes, fixed turns, signed currents and rigid pose; axisymmetric annular windings and calibrated closed saddle/radial stigmator, quadrupole and hexapole winding packs; mu_r=1 | Pole pieces, iron, B-H saturation and discrete winding leads are not magnetic sources. Existing mechanical/clipping ownership remains active. Only unconfigured legacy sources retain native fields. General column-wave propagation remains unqualified; local stationary mixed STEM has separate per-request admission and qualification. |

Ideal prescribed-current coils in vacuum and Gaussian length linking require
`unified`. The prescribed-current model admits classical particles and the
local stationary mixed particle-wave STEM entry, subject to actual per-request
field and numerical qualification. General/downstream wave entry points still
reject it. The program does not switch to Gaussian automatically to obtain a
wave result, freeze live drives or truncate winding tails.

The **Ideal prescribed-current coils in vacuum** model computes the field
from the declared ampere-turns and winding geometry with permeability mu_0. No resistance,
conductivity, supply-voltage equation, Joule heating or magnetic-material
response is solved. This is neither a finite-conductivity material law nor a
superconducting-material or Meissner-effect calculation. The persisted key
`copper_coil` remains solely for compatibility with existing inputs, schemas
and archives; it is not the user-facing mode name.

Explicit winding definitions now connect CAD turn count, round-conductor
diameter and prescribed electrical current to that same ideal source. The
original winding envelope is retained while CAD resolves wires/insulation.
See [detailed winding CAD and ideal-field conversion](IDEAL_CURRENT_WINDINGS.md)
for parameter ownership, the existing percentage/polarity convention,
transactional editing, historical inputs and resource limits.
All catalog round-lens definitions now save user-authorized R-Z-area turn
estimates (current Objective 580 upper and 580 lower, original 586 references
retained after the explicitly bounded adjustment) and explicit current
references. The other lenses match their original Gaussian peak planes;
Objective matches its original specimen upper surface. This is isolated scalar
field calibration, not a whole-column field or focus equivalence. Legacy
3000-A-turn records remain stored but inactive when explicit N/current is used.
The current estimate uses nominal 9 mm2 allocated pack area per turn; canonical
user conductor size is separate metal cross-section. CAD adapts insulation at
fixed ID, OD and axial envelope. See the [current area/CAD report](development/wire_area_20261115/README.md).

Ignoring magnetic materials in this source does not remove the gun's ideal
conducting-electrode electrostatic boundaries, mechanical interception or a
separately requested specimen model. Those are independent system conditions.

The field method is magnetostatic Biot-Savart source integration: the circular
loop's azimuthal integral is evaluated through complete elliptic-integral
expressions, and the continuous winding cross-section is integrated numerically.
The default is 12 radial by 12 axial Gauss-Legendre nodes, or 144 weighted
current loops per winding segment. These are quadrature nodes, not the physical
turn count. The declared ampere-turns retain their original normalization.

The prescribed-current fields use finite Gauss-Legendre winding quadrature.
In the admitted near-axis region the native evaluator uses six Taylor terms (Aphi through
r^11, with axial derivatives through order 12). Away from that region it uses
the native elliptic-integral loop expressions. The compiled near-axis path
has narrower pose/range/roundoff admission and returns to that native evaluator
when necessary. Field tails remain at all Z. Winding-interior and unsupported
geometry evaluations fail explicitly. This is an ideal axisymmetric vacuum
winding model, not an arbitrary 3-D Maxwell solution of the CAD assembly.

Thus bore diameter has three different meanings: mechanical interception,
an input to an explicitly solved pole/material recipe, or winding geometry
in the prescribed-current coil model. A fixed Gaussian calibration represents
only the first of these automatically.

## Gun, vacuum and common boundaries

- The column selector does not replace the gun with a thin lens. Cold FEG
  particles start at the physical source and traverse extraction/acceleration
  and the admitted gun magnetic field. Flat, continuous-curvature and explicit
  surface sources select their respective geometry-based electrostatic models.
- The accelerating Gun magnetic provider is a frozen-time full-vector B
  capture. Column scan drives are frozen there; their induced electric field
  is retained by the unified column runtime, not a joint time-dependent Gun
  E+B acceleration operator.
- The closed flat-cathode model solves a classical axisymmetric electrostatic
  boundary problem with declared electrodes, planar cathode and grounded
  downstream liner. Tip/surface alternatives retain their own boundary
  approximations. Prescribed emission/current/energy laws are separate from
  solving many-electron dynamics.
- Cold-FEG current and Boersch broadening are prescribed source inputs. There
  is no self-consistent many-electron space-charge/PIC transport. Thermionic
  emission has Richardson-Dushman/Schottky and Child-Langmuir current-limit
  models; that source-current limit is not a collective transport field solve.
- Particle trajectories retain source IDs, absolute weights, actual energies,
  emission reference, flight times and physical losses. A clipped current is
  not restored by normalization. Particle statistics do not create quantum
  interference.
- Shared-potential apertures/bores use declared finite opaque absorbing
  geometry. Compatibility execution retains its plane aperture checks,
  finite bore clipping and admitted tilted finite-plate contacts. These are
  not automatic electron scattering from every CAD wall or a full material
  model for all mechanical parts. Rigid offsets/tilts/rolls are supported only
  by their admitted registrations; arbitrary nonrigid CAD edits are not
  field solutions.
- Residual-medium scattering, where supported and explicitly configured, is
  an independent-atom screened elastic approximation with separately supplied
  removal. It omits molecular bonding, recoil, ionization and stopping power.
  It is not enabled merely by a more detailed lens mode. The shared-potential
  runtime currently rejects residual-gas collisions and unsupported nanopulser
  or temporal configurations.
- Static nonlinear B-H and prescribed quasistatic scan fields do not imply
  circuit feedback, frequency-dependent eddy-current response, electromagnetic
  radiation, heating or mechanical drift. Coupled Multiphysics is unavailable.

## Preview, Medium and High accuracy

These are numerical/request policies. They do not change the selected column
field source, add magnetic materials or establish physical convergence.

| Entry | Normal numerical request | Executed product scope |
| --- | --- | --- |
| GUI Preview: fast rays | Normally 1024 original positive-current particles; column maximum step 1.0 mm; requested history spacing 2 mm | Physical particle section/current-pixel readout; wave products disabled; no complete STEM raster |
| GUI Medium: sampled beam | Normally 1024 original positive-current particles; step 0.25 mm; history 0.5 mm | Same particle-section intent, finer sampling; extra zero-current support probes where that source scheme uses them |
| GUI Run high-accuracy once | User-selected original N or complete fixed source table; applied column step min(requested, 0.05) mm; history at least 0.5 mm | Normally `workflow="rays"`: complete optical reference and exact incident prerequisites for page products; not automatically every material/image/spectrum |
| GUI High-accuracy page calculation / GUI explicit full request | Captured requested numerics and the same GUI step cap | Only the selected product, or explicitly requested complete pipeline, subject to that model's capability gates |
| Diagnostic optical-only Preview | Explicit worker/API branch `particle_tuning=False`; caller may override its step | Optical reference with specimen scattering deliberately omitted. This is distinct from normal GUI Preview. |

Explicit source quadratures retain their complete node tables and weights.
Sampling-schema constraints can change the total budget; zero-current support
probes do not count as original positive-current particles. New finite-source
physical trials require at least 1000 originals. Source count and integration
step are independent convergence dimensions. Gun transport has its own spatial
ceiling and adaptive error gates, separate from the GUI column step.
Direct/offline APIs retain their declared numerics; the 0.05 mm cap belongs
to GUI request preparation.

The normal GUI particle Preview/Medium path can include supported elastic,
aggregate Poisson inelastic branches and selected current-pixel EDS in a
material-capable compatibility model. It does **not** silently omit a selected
specimen. In `unified`, a full section extending beyond a non-vacuum specimen
reference plane is currently rejected because finite-volume particle material
transport is not connected. An explicit upstream cutoff is permitted. An
uninserted, vacuum or zero-thickness specimen permits a vacuum optical section.

Normal GUI physical previews and High requests can prepare scan calibration.
Automatic calibration can resolve different actual AC/Descan drive matrices at
different numerical settings; held calibration restores its recorded drives.
Equal independent lens inputs alone do not establish a fixed-field step
comparison. Compare actual field witnesses when studying step convergence.
The older optical-only Preview branch has different preparation semantics.

Retained history spacing is not an integration step. The shared executor keeps
all actual integration nodes when memory permits; otherwise it retains
selected existing nodes/checkpoints while still executing the full mesh.
Plot-node waist markers remain sampling dependent.

## Specimen, image and detector products

| Product / entry | Included conditions | Limits and compatibility |
| --- | --- | --- |
| Ray Diagram / `rays` | Physical gun and optical column, actual energy/clock, apertures and recording interception; incident optical reference | Does not include specimen scattering simply because a specimen is drawn/inserted |
| Sample interactions, detailed paths and X-rays / `sample`, `sample_region` | Selected material geometry, elastic particle transport, aggregate Poisson inelastic channel probabilities/representative-loss branches and requested detailed observables | Nonelastic channels here do not locate individual collision events. Classical particle paths do not include coherent interference; finite-volume material execution requires a supported compatibility route, not shared `unified` particle material transport |
| EDS / `eds` | Separately requested X-ray generation, material response and detector model over admitted specimen histories | Not implied by BF/ADF or an elastic wave image; needs configured material and EDS conditions |
| Ordinary Scanning Image, projected particle model / `stem`, `projected_atoms` | Executed incident population reduced to a Gaussian intensity closure and local affine raster response; auxiliary projected atomic detector response | Available shared-potential particle STEM route. No Bragg interference, phase contrast or channeling. Not finite-volume material paths, a coherent exit wave, 4D-STEM or a measured raster-coil waveform at every pixel |
| Other ordinary particle STEM choices | Selected compatibility particle/material response and readouts | Shared `unified` gate rejects finite-volume material-path raster, legacy coherent STEM, 4D-STEM and dispersive filter STEM |
| Camera / screen image / `receiver` | Executed particle arrival/interception and requested receiver response | A particle intensity image does not contain quantum interference; physical inserted recording devices can absorb current independently of whether a readout is selected |
| Mixed coherent plane observation / Electron beam or Transverse beam | Exact particle handoff at specimen upper surface; local mixed-state construction; admitted specimen and shared static downstream potentials to the selected plane; actual downstream recording absorption | Usable CUDA required. Propagation requires `unified` and admitted potentials; Ideal prescribed-current coils in vacuum and thin-lens optics rejected. Static downstream fields only; no arbitrary energy-filter continuation. The mixed beam has no single phase. |
| Mixed particles + waves (CUDA) Scanning Image | Same executed entrance source; finite-thickness static elastic multislice, admitted local axial field split, translated raster, BF/ADF exit-angle integration and one actual exit diffraction pattern | Separate local specimen kernel, not complete post-column wave propagation. Requires usable CUDA and supported finite-CIF geometry; no frozen phonons or configured finite inelastic/absorption channels. Static ideal-current sources have purpose-specific admission subject to actual per-request field and numerical qualification; dynamic current-coil inputs must be made stationary explicitly before particle generation. See below. |
| Illuminating Image / `imaging`, historical wave STEM / old wave flags | Historical Tip-wave-dependent products | New Tip-origin coherent production is retired. Retained code/settings/archive readers do not enable these old public wave routes. Use the separately admitted mixed-wave products. |
| Energy Filter / `energy_filter` | Dedicated admitted dispersive compatibility transport and detector model | Not automatically included in straight-column optics; shared-potential product gate rejects the unmigrated dispersive route |
| Complete calculation / `full` | Explicitly requested set of enabled products and their dependencies | Not permission to bypass any product gate; may reject a combination that `rays` alone supports |

Material-capable compatibility routes use CPU screened-Rutherford elastic
trajectories through finite specimen/support geometry alongside separate
aggregate Poisson inelastic branches. The elastic model is provisional, not
ELSEPA. EDS uses elastic Monte Carlo or explicitly selected straight-primary
tracks, Bote-Salvat shell ionization and direct xraylib relaxation;
bremsstrahlung, secondary fluorescence and resolved Auger transport are
omitted. The energy filter uses dedicated Boris particle transport with slit,
shutter and detector interception. Its EELS spectrum consumes the material
distribution separately from the actual ray-transmission counters; this is
not complete coherent or energy-resolved spectral transport. Parallel wave,
elastic, inelastic and photon observables must not be added as independent
electron-current branches.

### Mixed coherence is a declared local model

The handoff uses native float64 particle checkpoints at the exact upper face.
If needed it continues the original particles to that plane without emitting a
replacement source or interpolating a display plot. The lift assigns a
minimum-uncertainty Gaussian mixture, measured statistical correlations and
a fitted common angular phase; the default independent source-image
translation RMS is 10 pm per axis. That width and local phase construction are
model assumptions, not a measured Tip-to-specimen mutual-coherence function.
The magnetic vector potential contributes to canonical transverse frequency.

Within each mode amplitudes interfere; independent modes contribute weighted
intensities. A selected component has a phase image, but the complete mixture
has no unique phase. Actual per-particle energy spread affects the upstream
trajectories and is retained in ancestry. Current local wave propagation uses
one current-weighted stationary mean wavelength, rejects an actual energy span
above 2 eV and does not integrate separate energy modes. Selecting a 0.3 eV
source therefore does not imply energy-resolved temporal wave propagation.

### Mixed plane observation versus mixed raster

The arbitrary-plane path uses the general shared electromagnetic wave adapter,
including admitted static A/phi overlapping the specimen and subsequent
optical components. Its default material result is the coherent zero-loss
branch with configured inelastic removal recorded; additional material
history options require their own admission. It is not interchangeable with
the elastic-only scan kernel.
Configured frozen-phonon snapshots contribute independent material histories
in this general wave route. Its optional `trajectories` inelastic method is a
conditional Markov-history model, not an atom-resolved transition-potential
calculation. Partial-depth observations currently require the zero-loss
branch. The propagated wave is a scalar positive-energy branch, not a full
spinor/time-dependent many-electron solution.

The mixed raster uses a qualified symmetric split of the static near-uniform
axial magnetic field inside the specimen, with independently checked omitted
field terms. It does not execute every downstream lens or retrace the time
law of the scan coils at each pixel. The current material path requires actual
multislice, a proved rectangular finite-CIF repeat and supported transverse
sites. Unsupported geometry, frozen phonons, separate high-angle channels,
finite inelastic/absorption channels and phase-object substitution are rejected.
It is a finite periodic-grid model with source, sampling, slice, domain and
field-splitting qualification gates.

Default scan BF/ADF are ideal exit Fourier annuli of 0--15 and 50--150 mrad.
They are not automatically the currents of the physical downstream detectors
or evidence of detector diffraction conjugacy. ADF here contains the admitted
static elastic contribution, not a silently added thermal/inelastic channel.
The displayed diffraction is one actual selected raster position recorded in
the result, not a scan average or a complete 4D-STEM cube. Absolute source-current
denominators and lost-current channels are retained.

## Backend and cost

| Choice / stage | Current meaning |
| --- | --- |
| Auto (GPU first) | CUDA first for supported operations, including small jobs; otherwise Numba/CPU according to capability and size |
| CPU / Numba CPU | Requested classical CPU execution; supported compiled operators may still be used inside CPU-only preparation |
| CUDA GPU / Prefer GPU | Requested supported CUDA operations with policy-dependent, reported classical fallback; not a whole-pipeline GPU guarantee |
| Require GPU | Rejects an accelerated stage that is unavailable or unsupported; declared CPU-only preparation remains a separate stage |
| Shared-potential classical column | Currently CPU; Require GPU rejects this unsupported accelerated column route |
| Gun | CPU stage, with admitted compiled acceleration; not moved onto CUDA by the toolbar |
| New mixed coherent work | Usable CUDA device/driver/CuPy required; no CPU coherent retry after failure under the current project policy |

Cost grows with field evaluations times original particles and steps for
classical transport, with geometry/material meshes for field preparation, and
with wave modes, transverse grid, slices and raster positions for coherent
acquisition. These are cost drivers, not measured speed guarantees. Cache hits
can avoid compatible work but do not add physical conditions. Source, fields,
geometry, solver and numerical dependencies must agree; old solver results
cannot be retagged as current transport. Stable nested source prefixes permit
incremental reuse only in their explicitly admitted static-source scope.

## Scope of the 14 November copper benchmark

The completed 1024-original, 0.3 eV comparison used `unified` with Ideal
prescribed-current coils in vacuum for round-lens fields. Its Preview was the
explicitly requested optical-only worker at 0.5 mm, not the normal GUI's 1.0 mm
physical live-preview entry. High
used `rays` with the 0.05 mm cap. The independent fixed-field coarse/fine
comparison used the identical new Gun and actual calibrated fields. These
results validate that declared classical optical-reference scope; they do not
validate coherent propagation with Ideal prescribed-current coils in vacuum,
Si scattering, mixed STEM or full GUI material preview. See
[the executed benchmark](development/copper_gun_adaptive_20261114/README.md).

### Why the Gun was expensive in that run

The ideal-current winding source does not add electrical resistance, iron
response or a material update during acceleration. Gun electrodes still
define ideal equipotential electrostatic boundaries: removing those boundaries
would change the extraction/acceleration model. Their geometry is not a
conductivity, dielectric-polarization or thermal material calculation.

In the executed case the electrostatic solution was already cached. Its
reported cache load took 0.0340 s; the retained original solve record is
3.869 s. These are different timings, not a new solve at every particle step.
The long-running work was transport of 1024 particles through the accelerating
field while repeatedly evaluating the full electric/magnetic operators and
performing implicit full/half/fine updates. Vacuum winding tails remain in the
Gun even when their physical lenses are farther downstream.

Of the old 5942 rejected trials, 4875 started at 0.5--1.1 mm. The outer loop
discarded a successful small duration and proposed a large cap again, causing
repeated halving. Accepted-step feedback reduced total rejections to 183 under
the same four accuracy gates. This is measured integration overhead, not
evidence that a magnetic material solve was added to the Gun. See the
[executed Gun field report](development/copper_gun_adaptive_20261114/validation/preview/gun-electrostatic-model-report.json)
and [rejection audit](development/copper_gun_adaptive_20261114/validation/independent-preview-rejection-audit.json).

## Source map

- [Model registry and capability checks](../src/temsim/simulation_modes.py)
- [Round-lens choices](../src/temsim/round_lens_models.py),
  [vacuum winding field](../src/temsim/physics/annular_coil_potential.py)
- [Shared runtime preflight/capture](../src/temsim/physics/unified_field_runtime.py),
  [shared particle executor](../src/temsim/physics/unified_field_particle.py)
- [GUI quality requests](../src/temsim/gui/calculation_request.py),
  [tuning policies](../src/temsim/physics/optical_tuning.py),
  [worker dispatch](../src/temsim/gui/calculation_controller.py)
- [Product admission](../src/temsim/calculation_workflow.py),
  [page continuation](../src/temsim/simulation_workflow.py),
  [physical particle sections](../src/temsim/simulation_pipeline.py)
- [Local mixed source](../src/temsim/physics/particle_wave_source.py),
  [mixed plane propagation](../src/temsim/physics/mixed_wave_pipeline.py),
  [mixed raster](../src/temsim/physics/mixed_stem_scan.py)
- [Device selection and coherent CUDA gate](../src/temsim/physics/compute_backend.py)
- [Mixed coherence guide](MIXED_COHERENCE.md),
  [mixed raster guide](MIXED_STEM_SCAN.md)
