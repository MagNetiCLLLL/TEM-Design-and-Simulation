# Native potential snapshots

This implementation extends the P0 physical scene and ownership boundary with
opt-in, immutable potential records. It is a representation foundation; it
does not replace the existing particle or coherent propagation operators.

## Representations

| Source | Retained representation | Derived field | Limit |
| --- | --- | --- | --- |
| Axisymmetric magnetic FEM | Original `r`, `z`, triangular P1 `A_phi` coefficients | Curl of that same basis | Piecewise derivatives; no global smooth Hessian |
| Native analytical round lens | Captured Gaussian axial terms and rigid registration | Curl of the existing local `A=(-y B0/2, x B0/2, 0)` | Existing near-axis law, not a new high-order Maxwell solution |
| Native analytical multipole or deflector | Existing polynomial `A_z`, envelope, support and registration | Curl of the same polynomial | Hard support edges excluded from derivative queries |
| Native gun electric field | Original scalar interpolation/cut-cell/axis-regular coefficients | Negative gradient of the same scalar law | Only supported native providers and their declared extensions |

Positions are metres, magnetic vector potential is T m, magnetic field is T,
scalar potential is V, and electric field is V/m. The electric scalar uses the
original cathode-reference potential rise. Metadata records any conversion to
the absolute grounded reference; this offset must accompany future energy and
wave-carrier definitions.

The FEM adapter deliberately differs from the legacy separately interpolated
nodal B field. Both come from the same solve, but their off-node values need not
be equal. Existing propagation retains its original B interpolation. No
transport result is silently changed to use the new curl interpolation.

## Explicit capture and composition

`capture_round_lens_potential(state, key)` resolves the selected native provider
and retains its potential, geometry, material/solver settings and solve
identity. A linear FEM source captures its current scale. A nonlinear circuit
captures the complete operating point and total solution once; it cannot be
scaled as independent saturated member fields.

`capture_selected_column_potentials` accepts explicit round-lens and auxiliary
component keys, and an explicit electric-field option. A shared logical drive
must be selected through its physical coil host. Selection is not a declaration
of complete instrument coverage. Gun magnetic alignment, blankers, filters,
sample interactions and unknown imported maps are not implicitly zero fields.
Configured FEM resolution can perform a solve on a cache miss, so capture
belongs in a numerical worker, not a plot refresh.

`ElectromagneticPotentialSnapshot` rejects repeated source IDs and overlapping
declared atomic ownership of the same field kind. A joint nonlinear total and
one of its member contributions therefore cannot both enter a sum. The generic
composition interface validates explicit coverage declarations; it does not
prove arbitrary caller-supplied IDs describe the hardware. The native capture
entry points construct these declarations from existing providers.

Every selected source must admit every query point. Unsupported regions remain
unknown: the snapshot does not extrapolate or fill exterior regions with zero.
An explicitly empty selection means no selected sources, not verified vacuum.
Consequently this API alone is not a whole-column propagation domain manager.

Static gauge changes are applied once after the selected sources are summed:
`A' = A + grad(chi)`, `phi' = phi`, and
`psi' = exp(i q chi / hbar) psi`, with electron charge `q = -e`.
The supported polynomial gauge preserves E and B. This algebraic check is not
evidence of gauge covariance of a numerical wave propagator.

`physical_digest` describes the declared scene and ownership only; it must not
be used as a solved-field cache key. `numerical_digest` also includes all
retained coefficients, provider metadata and the common gauge. Serialization
checks both content and schema, and snapshots retain no mutable provider arrays.

## Qualification boundary

The focused tests check native electric-law parity, analytical curl and
derivatives, triangular-basis magnetic fields, rigid transforms, axis limits,
strict domains, immutable persistence, ownership and actual provider capture.
Synthetic manufactured fields are numerical evidence, not OEM or experimental
validation. Existing optical parameters, finite-field operators and phase
screens remain active as before.

New potentials remain marked `wave_admission=False`. B-only imported maps are
rejected rather than assigned an invented gauge. Dynamic drives can be captured
at one stated time; induced electric fields and coupled time evolution are not
implemented by freezing such a drive.

The later proposal stages require evidence separate from P1: high-order
round-lens field and Lorentz convergence; natural Cs/Cc fits at frozen physical
fields; a gauge-consistent coherent operator and classical limit; C3/Probe
Corrector Normal and Large Angle pilots; remaining source/sample/receiver
coverage; and qualified reduced models. These now have their own bounded
implementations and regression families, described in the
[implementation and validation map](UNIFIED_ARCHITECTURE_VALIDATION.md).
Passing the P1 representation tests alone does not certify those stages or
remove configured aberrations.
