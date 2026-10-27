# Optional compiled native potential sampling

`compile_native_sampler(snapshot)` captures an independent CPU sampler for
the opt-in potential-based benchmarks. It reduces Python and temporary-array
overhead; it does not replace the physical fields by fitted lenses, reduce
field order, loosen integration tolerances, or qualify a full instrument.
The default snapshot sampling path remains available and unchanged.

## Supported representations

- `HighOrderRoundLensPotential`, including the declared odd radial orders 1
  through 9, strict native support or explicitly selected Gaussian all-Z
  continuation, and rigid placement.
- `AnalyticPotentialDomain` around a static `FrozenPosedMultipole`: the
  original degree-one to degree-three Az polynomial, uniform or Gaussian
  compact envelope, rigid placement and closed-face sampling convention.
- The snapshot's common static polynomial gauge, applied once after source
  summation. The constant gauge action remains owned by the original snapshot;
  it is not part of the vector-potential gradient.

Electric, FEM, arbitrary mapped, dynamic and unrecognized providers are
rejected. An unsupported contribution is never silently omitted. Every
selected source keeps its original radial and axial domain. Outside a compact
multipole slab the original known zero is returned without evaluating an
inactive radial polynomial. Derivatives on its hard axial faces are rejected.

## Interface and identity

```python
from temsim.physics.potential_native_sampler import compile_native_sampler

sampler = compile_native_sampler(snapshot)
sampler.require_snapshot(snapshot)
values = sampler.sample(positions_m)
jets = sampler.sample_jet(positions_m)
```

`sample` returns `phi_v`, `E_v_per_m`, `A_t_m` and `B_t`, preserving input
shape. `sample_jet` returns `A_t_m`, `jac_A_t` and `hess_A_t_per_m` in global
Cartesian coordinates. Optional `source_ids` follows snapshot ordering,
rejects unknown or duplicate selections, and retains the common gauge even
for an explicitly empty selection.

The sampler owns detached, read-only packed coefficients and the exact
snapshot numerical digest. A changed field, pose, domain, gauge or source
representation fails `require_snapshot`; callers must capture a new sampler.
Direct construction from packed arrays is rejected; only the validating
`compile_native_sampler` factory is supported.
Joint sources are sampled once per contribution, regardless of the number of
atomic owners they cover. No live state, field-solver callback or mutable
coefficient cache is retained.

Numba kernels use `fastmath=False` and ordered source summation. The values
kernel avoids inner-loop temporary matrices. Repeated **exactly equal** local
Z coordinates reuse the same Gaussian derivatives within one sampling call.
There is no Z rounding, field interpolation, tail cutoff or cross-call cache.
General poses and noncoplanar queries use the same formulas and domain tests.

## Validation and performance boundaries

The dedicated tests compare all admitted radial orders, mixed rigidly placed
sources, gauge transformations, full first and second derivatives, source
selection, compact faces and rejected domains against the original adapters.
An independent finite-difference curl check separates derivative truncation
from coefficient parity. Actual C3 + Probe Corrector captures exercise all 29
sources (9 round fields and 20 auxiliary source contributions) at orders 3, 5
and 7. These are numerical model comparisons, not hardware measurements.

Initial JIT compilation and snapshot capture have separate costs. Sampling
speedups depend on batch size, common local Z, hardware and field order; they
are not whole-trajectory or whole-application speedups. Full ray, action and
canonical-reference comparisons must use the same inputs, DOP853 tolerances,
support faces, stop events and symplectic qualification gates. A faster sampler
does not justify weakening any of those gates.
