# Finite material and potential-field handoff

`potential_specimen_transport.py` is an explicit controlled-entrance workflow.
It does not change the default source or column mode. It reuses the existing
atomistic potential preparation, finite multislice half-phase actions, frozen
phonons, material attenuation and conditional Markov collision channels.
Between material half-steps, each wave is advanced with the selected immutable
electromagnetic potential at that history's actual energy. The same snapshot continues
downstream; no legacy column, empirical Cs/Cc kick or image pupil is appended.

The combined `propagate_potential_specimen_to_detector` entry requires a captured
physical geometry interval from the exact specimen entrance through the receiver.
Finite bore/aperture constraints remain active inside the material and downstream.
The receiver consumes the executed complex modes and absolute weights. Conditional
histories contribute intensities, never an invented aggregate phase. Detector
phase is a labelled wrapped point readout with retained analytical carriers;
it is not an unwrapped or Fourier-resolved reconstruction.

When a material transport callback receives multiple modes, finite geometry
executes once for the whole beam. The checkpoint owns one aggregate physical
loss history, while each mode retains only its own loss details. Continuation
preserves the original entry weight and previous losses; it neither repeats the
prefix per mode nor replaces the total with the final mode's contribution.

The admitted material is an untilted finite axial slab with a vacuum holder
support. The field operator admits its existing static magnetic analytical
providers and native static scalar-potential interpolation. Dynamic drives, unsupported FEM carrier
representations, oblique slabs and non-vacuum supports fail explicitly. This
bridge executes its field stages on CPU, including a local CPU choice under
Auto; it leaves the material backend policy unchanged. The existing material model
remains a local Markov approximation with representative losses and angles,
not an atomic transition-potential or resolved EELS calculation.

Material slice thickness, atom-potential sampling, wave lattice and longitudinal
field splitting require independent convergence. The exponential and kinetic
budgets are apportioned across the complete interval rather than restarted at
each slice. Grid, working memory and detector-readout memory budgets remain
separate; enlarging them does not relax boundary or sampling thresholds.
An optional cumulative `kinetic_phase_ledger` retains the exact executed common
prefix in every environmental history. The prefix is reported once. Each history
continues its unchanged total allowance independently; history totals are never
summed into a common phase. An actual recorded energy-changing collision closes
the current energy-bound segment and gives the next segment only the remaining
allowance. Saved receipts, energy/order/source binding and native material-plane
authority survive continuation; missing or changed history cannot restart a
budget. Existing calls without this ledger retain their interval-based policy.

Static electric overlap additionally requires an executed electric energy
history and cumulative kinetic ledger at the entrance. Between actual
collisions, the original stationary total energy determines the changing local
kinetic energy; continuous acceleration does not allocate a new kinetic budget.
A recorded collision must match the actual local energy before and after its
loss. Only then is the scalar-potential reference rebased at that exact plane.
The accumulated clock, longitudinal action, complex wave and absolute branch
weight remain intact. Completed electric forward-error bounds compose as
`product(1 + bound) - 1`; the next segment receives only the remaining original
allowance under that composition. The common prefix is counted once per
conditional history. It is not multiplied by the number of branches or treated
as a shared phase. Dynamic material histories are rejected explicitly: this
stationary handoff does not qualify time-dependent scattering or energy sidebands.

This cumulative bound applies to kinetic operator truncation on unnormalized
linear or contractive histories. It is not a pointwise phase-accuracy guarantee
or a relative error bound on a normalized conditional state. Zero-loss attenuation,
physical absorption, phase actions and spatially constant conditional channels
are admitted. A spatially varying Kraus mask followed by state-dependent
conditional normalization is rejected only on this new ledger path, because it
can amplify an earlier error. The historical material workflow is unchanged.

An upstream `NumericalSupportLedger` becomes an immutable common prefix before
conditional branching. Its original projection loss is counted once at the
specimen, final checkpoint and readout. Its scope remains the executed upstream
projections; it is not promoted into an error bound on normalized conditional
histories. Descendants cannot allocate a new projection ledger or spend the
parent's unused budget. A future branch-aware error contract is separate work.
The schema, source/snapshot identity and digest survive storage; altered records
or attempts to reopen projection are rejected by both observation entry points.

Regression coverage includes actual finite Si atomistic slices with a controlled
100 eV-per-slice collision instrument, nonzero magnetic overlap, actual-energy
time/action, unchanged legacy-route admission, same-field downstream geometry and
absolute detector readout. A separate manufactured smooth material fixture checks
three conditional histories inheriting one actual nonzero projection receipt.
Two unequal-weight modes also cover complete and partial geometric absorption,
with the same absolute balance and complex field after segmented continuation.
Additional cases check common kinetic prefixes, actual Si energy changes, cache
restoration, false energy edits, duplicate prefixes and forbidden budget resets.
Static electric cases distinguish continuous acceleration from actual Si
collisions, retain both original budgets, and reject a collision recorded using
the old launch energy instead of the local kinetic energy.
These tests qualify the declared interface and numerical cases, not all materials,
whole-column instrument accuracy or manufacturer geometry.
