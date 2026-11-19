# Existing 1024-particle Gun rejection audit

This is a read-only audit of the completed baseline. No new source, field query,
transport or physical trial was run. Original optics, 0.3 eV source, boundaries
and tolerances were not edited.

The observer retains all 9,444 attempts: 3,502 accepted and 5,942 rejected.
Recorded attempts took 2961.117 s; rejected attempts occupied
1998.585 s (67.49%).
There were no implicit-convergence failures or nonfinite gate trials.

The outer loop proposes dt from the original spatial/field/impulse caps at
every accepted step. The successful smaller dt is not remembered. Of 1,103
steps needing retries, all following proposals exceed the previous accepted
duration, and 1,088 remain within 5% of the previous rejected initial duration.
The median restart ratio is 64.025; the worst is 290.303. This is direct
evidence of repeated reject-and-halve sawtooth work.

The dominant rejecting gate is transverse momentum (5,663 trials, 95.30%);
XYZ position dominates 172 and XY position 107. These exclusive dominant-gate
counts differ from the nonexclusive failed-gate totals in the JSON report.
The 0.7-0.8 mm and 0.8-0.9 mm start-Z bands each consume about 555 s. Most
steps there halve 7-8 times, selecting 1/128 or 1/256 of the initial proposal.
The electrostatic-lens neighborhood at 14-25 mm is a smaller secondary hotspot.

The extractor nose changes bore radius from 2 mm to 0.25 mm at 0.75 mm. The
retained beam maximum radius near 0.788-0.804 mm is only 53.6 nm; this is not
evidence of near-wall interception. The axial mesh is about 15.6 micrometres
there, the bunch spans about 16.3 micrometres, and all particles are inside the
first 15.625-micrometre radial cell. Interpolation-interface effects are a plausible
contributor, but this observer does not identify each trial's internal cell
crossings, so no causal percentage can be assigned.

A controller can remember a conservative successful dt and cautiously grow
it while still applying every original physical cap and all four gates. The
existing half-update cache already avoids one full update per rejection.
The exact call identity is 22,390 = 3*3,502 + 2*5,942; removing retries from the
same accepted sequence would require 10,506 calls. This is bookkeeping, not a
runtime promise: direct accepted-dt proposals need a fresh full update and a
new controller can change the accepted sequence. Preserve safe halving and
do not assume a globally cubic normalized-error law.

See `baseline-rejection-audit.json` for fine axial bins, timings, every failed
gate, retry depth, dt ratios, limiting original IDs, beam snapshots and explicit
evidence/inference separation. New full 1024 Preview/High runs must validate
any later implementation; this audit itself makes no new convergence claim.
