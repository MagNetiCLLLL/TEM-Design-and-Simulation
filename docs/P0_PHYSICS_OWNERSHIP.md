# Shared physics identity and interaction ownership (P0)

P0 adds an immutable provenance and ownership contract to the existing axial
propagation plan. Particle and coherent column-wave preparation use this same
compiler. It does not change the electromagnetic equations, lens settings,
aberration values, source populations, numerical steps, clipping or phase
operators. P0 alone admits the `legacy_compatible` route. The subsequent thin
executor also uses this contract with `ideal_thin`; bounded native-potential
executors use their own explicit capability and ownership checks. This legacy
plan compiler does not admit generic `field_native` or calibrated
`field_plus_residual` requests. See the current
[implementation and validation map](UNIFIED_ARCHITECTURE_VALIDATION.md) for
the separately implemented P1-P6 responsibilities and their limits.

## Identities and responsibilities

| Record | Responsibility | Boundary |
| --- | --- | --- |
| Existing `InstrumentSnapshot` | Complete working-point restoration and content-pinned external input records | Remains the authoritative restoration contract |
| `PhysicalSceneSnapshot` | Detached resolved geometry, declared optical inputs, component order, physical hosts, logical drives and magnetic circuits | Supplementary input metadata; not an executed field or complete source state |
| `geometry_digest` | Resolved parts, declared runtime dimensions/poses, vacuum bores and layout anchors | Geometry remains present at zero excitation |
| Frozen field providers and plan coefficient arrays | The field representation actually compiled for this segment | Existing approximations and support limits remain unchanged |
| `InteractionRegistry` | Which physical response is executed once, by which owner and representation | Observations do not execute; numerical partitions are not new sources |
| `AxialPropagationPlan.physics_contract` | Scene, registry, interval, physical reference planes and independent compiled numerical digest | Included in the new plan signature and current archives |

Coordinates retain explicitly named units. Assembly and action intervals use
mm, physical provider buffers retain SI units, and geometry fields retain their
existing unit suffixes. No geometry is inferred from excitation. Unknown input
objects are marked unresolved; `identity_complete` refers only to the declared
scene metadata scope. It does not qualify the physical model. Source populations,
external asset bytes and solver implementations remain covered by the existing
working-point, field and calculation records.

Shared Descan and Image/Diffraction controls resolve to the same physical host.
The combined drive is recorded once per physical coil, with both logical drive
keys. Corrector channels with explicitly modelled inserts use those hosts;
unbound control channels retain unresolved physical ownership rather than
inventing a separate mechanical body. Different multipole responses on one host
can legitimately coexist.

Native source digests conservatively identify the complete compiled numerical
plan. They are not independent per-source cache keys. Imported and posed field
providers retain their existing frozen content fingerprints. A scene digest
alone is never sufficient to reuse a numerical calculation.

## Current action inventory

| Existing action | P0 ownership | Preserved behavior |
| --- | --- | --- |
| Native round lenses | Distributed `round` field response per optical key | Existing axial profiles and first-order transport |
| Stigmators/quadrupoles and finite hexapoles | Distinct field response per multipole family | Existing thickness, orientation and polynomial coefficients |
| Finite deflector coils, including shared channels | One field action per host coil | Signed integral, thickness and captured drive unchanged |
| Posed analytic providers | The same response in the existing placed frame | No extra scalar copy at the old position |
| Imported magnetic maps | Field ownership includes round transport and its aberration responses | Native Cs/Cc and intrinsic actions remain excluded |
| Joint nonlinear magnetic solve | Total field claims each participating response; included member maps are observations | No summation of independently saturated solutions |
| Image equivalent lenses | Reduced linear round response replaces distributed round transport | Additional declared Cs/Cc remain separate as before |
| Native or posed Cs | Explicit legacy additional residual | Existing Cs/f^4 action and phase operator unchanged |
| Cc and intrinsic C1/A1/B2/etc. | Named additional residual with provenance | Main-field energy dependence retained; no new chromatic term |
| Gun E and overlapping gun magnetic support | Existing captured field ownership | No new emission or coherence model |
| Anonymous external tuple events | Separately identified reduced actions with an explicit identity limitation | No inference that identical coordinates imply a shared source |
| Apertures, bores and passive geometry | Captured in the physical scene; existing boundary executors retain application ownership | Clipping remains active independently of excitation |

Source emission, specimen scattering/absorption, energy-filter transfer,
receiver imaging and detector response retain their existing stage executors.
They are listed outside the shared column operator and are not falsely declared
to have migrated to field-native propagation. The generic registry supports
boundary, material and detector roles for subsequent stage integration.

## Rejection rules

- Duplicate interaction IDs or overlapping execution of the same atomic response
  are rejected before a newly compiled plan can execute.
- A reduced map must claim the responses it replaces. A resolved field plus a
  second equivalent action for those responses is rejected.
- A joint circuit total and a separately executing member cannot claim the same
  response twice. Distinct linear basis fields may still overlap physically.
- A finite coil consumed from the legacy tuple-event list cannot leave an exact
  nonzero copy behind as a thin slope kick. Unidentified independent probes are
  not silently merged.
- An explicit residual needs provenance and its own response coverage. Calling a
  duplicate total contribution a residual does not bypass the check.
- Numerical half-step records may share one action with matching metadata and
  fractions summing to one. Adjacent distributed intervals are allowed. A point
  action at either endpoint of an already owned response is checked for overlap.

Discrete-event metadata uses the existing executor's node assignment, including
mm/m roundoff at endpoints. Physical reference planes are retained separately.
No node or physical plane is moved by the ownership layer. Sliced wave plans
retain the parent scene and omit already executed entrance actions. The ideal
first-order reference removes the same Cs, higher intrinsic and hexapole records
that its existing operator removes; it keeps C1 and additional Cc.

## Persistence and compatibility

New plans freeze the contract recursively. Current particle record graphs retain
it; current incident seeds use `incident-simulation-seed-v8-electromagnetic-energy-ownership`.
Restore verifies nested content digests and duplicate ownership. A current seed
missing its ownership field is rejected for continuation. Historical seed codecs
remain readable through historical array viewing and are not silently upgraded
into current executed plans. Hand-built test/legacy plans may explicitly carry
`physics_contract=None`; this does not claim a captured P0 scene.

P0 metadata is not a replacement for the existing executed-array integrity and
working-point checks. It describes the compiler's decisions, not an independent
proof that an arbitrary caller-modified array implements those decisions.

## Validation scope

The regression set covers all seven shipped column configurations: C2, C3,
C3 Image Corrector, C3 Probe Corrector, C3 Probe + Image Corrector, C3 B-COR,
and C3 Probe + B-COR. It checks shared hosts, native/reduced exclusivity,
particle/wave ownership, immutable identities, zero-field geometry, map exclusion,
endpoint events, first-order references, checkpoint slicing and archive damage.

A separately captured before/after CPU baseline covers 21 plans: seven columns
times full-column, native post-specimen and reduced Image post-specimen paths.
All 483 plan arrays and 126 full-precision result arrays were bitwise identical;
all 35 configuration TOML files retained their hashes. The four-ray, coarse-step
fixture proves compatibility for that bounded comparison, not optical
convergence, experimental agreement or instrument qualification.

The tests are registered under acceptance criterion `particle-continuation/PC-11`.
No FEM/potential upgrade, field-derived Cs/Cc extraction, or expanded coherent-wave
support is claimed in P0.
