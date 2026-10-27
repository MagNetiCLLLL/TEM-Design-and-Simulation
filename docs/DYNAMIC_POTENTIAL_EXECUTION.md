# Native dynamic potentials and particle transport

The unified runtime retains one static potential snapshot and explicit native
time-dependent physical coils. Shared Descan commands belong to the two
Image/Diffraction coils; they are not additional magnetic hardware. A changing
drive is never represented by a magnetic field alone.

All coordinates are metres, times are absolute laboratory seconds, potentials
are volts and tesla metres, and the electron charge is `q = -e`. The common
definitions are

\[
\mathbf B=\nabla\times\mathbf A,\qquad
\mathbf E=-\nabla\phi-\partial_t\mathbf A.
\]

The retained electric gun solution supplies static `phi` and `-grad(phi)`,
including its downstream tail. The native coil law supplies the same compact
linear `A_z` basis as its original finite dipole model, with its actual raster
or sinusoidal drive. Its ordinary time derivatives also supply the induced
electric field. Contributions are summed before transport. The static gauge
is applied once. This prescribed native field is not a coil-circuit,
retarded-radiation, fringe-field, or material Maxwell solve.

The runtime stores the emission epoch. Each ray retains its executed gun and
column flight time, and samples the fields at
`emission_time_s + flight_time_s`. A time-dependent calculation rejects unknown
entrance clocks. Negative emission epochs are permitted; elapsed flight times
remain nonnegative. The frozen dynamic snapshot accepts absolute time directly
and does not add another epoch.

For static scalar potential and time-dependent vector potential, the forward
Lorentz integrator uses kinetic energy in electronvolts as

\[
K=K_0+\phi(\mathbf r)-\phi(\mathbf r_0)+W_{\mathrm{ind}},\qquad
\frac{dW_{\mathrm{ind}}}{dz}
=\partial_t\mathbf A\cdot(x',y',1).
\]

This is electric work for an electron. It preserves the static potential
invariant without integrating the difference of large energies, while retaining
induction work separately. The full electric field also enters the Lorentz
direction equation. Neither empirical Cs/Cc nor equivalent Image maps act in
the interval owned by this executor. A resumed section starts with its executed
kinetic energy and flight time; it does not restart the physical clock.

The native raster has a discontinuous flyback. Ordinary derivatives are valid
only on open line intervals. Point evaluation on a flyback or a transport step
crossing it is rejected because no temporal impulse operator has been admitted.
There is no implicit freezing, smoothing, or zero-field substitution. Native
line membership and the event guard use one right-continuous clock partition:
rounded integer multiples of the retained line period. Quotient rounding is
corrected against those same boundaries, including negative epochs and adjacent
floating-point times. The serialized `rounded-line-interval-v1` policy is part
of the field identity. Ordinary non-boundary values agree with the native raster
law to floating-point precision; independent frame/line arithmetic cannot hide
a flyback between two adjacent times.

Native
compact axial faces are inserted into the particle mesh. Force quadrature uses
the interval's interior side. Explicit one-sided potential jets are available
only on exactly aligned compact faces; tilted and unknown interfaces keep their
original strict admission rules.

Mechanical clipping remains separate from field validity. A strict potential
domain failure is not an absorbing aperture. Existing finite bores and apertures
can absorb a particle exactly once; recorded stopped rows are inspection states,
not downstream flux. No intensity normalization compensates for these losses.

`trace_unified_response` measures finite-coil response from explicitly supplied
differential probes. It changes the retained coil's DC `A_z` coefficients on a
private immutable snapshot and returns positions, mechanical slopes, energy,
clock and vector potential. It does not substitute a source, add a thin kick,
mutate the working state, or grant instrument calibration. Its receipt identifies
both the parent runtime and the diagnostic field increments.

The focused regressions compare native time laws, curl and induction, an
independent full-momentum DOP853 solution, static electric and magnetic limits,
clock/energy continuation, physical-owner uniqueness, strict domains and
serialization. Actual default-source and coherent propagation qualifications
are separate from these mathematical and implementation checks.

The potential conventions follow [David Tong's electromagnetic coupling notes](https://www.damtp.cam.ac.uk/user/tong/aqm/solid1.pdf).

## Certified remote Gaussian domain

The native near-axis radial radius remains unchanged. Applying that radius to
the complete infinite axial Gaussian extension can otherwise reject a particle
far downstream even when it remains inside the actual local mechanical bore.
The unified runtime explicitly requests `gaussian-binary64-remote-roundoff-v1`;
the adapter's default remains strict, and the policy changes its frozen digest.

This policy admits an additional point only when both the exact Gaussian
axisymmetric vacuum continuation and every retained finite radial-expansion
term have conservative absolute bounds below `2^-1075`. The bounds include the
Cartesian vector potential, its first and second derivatives, and the magnetic
field, with conservative rotation norm factors. This is half the smallest
binary64 subnormal, expressed separately in each quantity's SI units. It is
not a freely chosen physical tolerance or a claim about measured OEM fields.

For one on-axis term `b exp(-(z-z0)^2/(2 sigma^2))`, the continuation admits the
disk-average representation `F=A_phi/r`. With `d=z-z0`, `r=sqrt(x^2+y^2)`,
`M=abs(b) exp((r^2-d^2)/(2 sigma^2))`, and
`g=(abs(d)+r)/sigma^2`, conservative bounds are
`abs(F)<=M/2`, `abs(grad F)<=g M/2`, and each Cartesian second derivative is
bounded by `(g^2+1/sigma^2) M/2`. Polynomial coefficient bounds and positive
Hermite recurrences separately bound the retained radial terms. All Gaussian
terms are bounded by absolute values; cancellations cannot grant admission.
The axisymmetric expansion is documented in
[McDonald's electromagnetic field exercises, Problem 9, equations 16, 17 and 19](https://kirkmcd.princeton.edu/examples/ph501set4.pdf).

The implementation uses outward-rounded elementary operations and integer
binary exponents. It never creates an underflowed floating-point value for
`2^-1075`, and does not use an underflowed sampled exponential as its proof.
Nonfinite prefactors, unresolved denominators, non-Gaussian sources, and
unqualified near-field points still reject. After admission the original
polynomial is evaluated unchanged; its naturally rounded result must also be
finite and zero. No tail or coefficient is replaced by zero.

The union of the original cylinder and remote qualified regions is nonconvex.
`require_global_box` certifies the complete enclosing box using outward-rounded
coordinate intervals, the maximum transverse radius, and each Gaussian's
minimum and maximum axial distance. Two admitted remote end planes therefore
cannot authorize a trajectory or swept grid crossing an unqualified interior.
This guard is separate from mechanical absorption and from quantum-grid
convergence requirements.

## Executed particle scope

A frozen implementation run exercised the public particle-section pipeline
with the actual `nano_probe` preset, all 3,000 default FEG emission samples,
the original 5 nm source FWHM, 0.3 eV emission energy and energy spread, and
1/3 mrad angular RMS/cutoff. Native dynamic scan remained enabled. The original
optical parameters and 0.5 mm requested integration step were retained. The
CPU run completed to 2987.15 mm in 567.174 seconds, including gun transport and
automatic finite-field scan calibration.

At the 1599.2 mm specimen entrance, 2,657 particles survived and 343 were stopped
by the C2 aperture. The absolute surviving reference weight was
0.8856666666666668. Downstream, the configured HAADF, DF and BF surfaces recorded
1,128, 148 and 8 particles respectively; other mechanical stops were retained
individually, and 10 particles remained at the final plane. Source weights were
not renormalized. This is a vacuum-specimen particle execution result; it does
not qualify coherent-wave convergence, raster-image generation, material
interactions, or an experimentally calibrated instrument.
