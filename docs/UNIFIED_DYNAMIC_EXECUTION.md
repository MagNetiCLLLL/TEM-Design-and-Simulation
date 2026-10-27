# Electric tails and native time-dependent scanning

Finite-Field Optics captures one immutable geometry and electromagnetic
potential definition for particle and coherent column propagation. Changes
to physical controls, source definitions, emission epoch, native scan laws,
or the solver implementation invalidate the corresponding execution identity.
Actual backend status and calibration evidence are outputs, not physical
inputs. Backend policy and resolved coil commands remain captured inputs.

## Source and interval ownership

The installed gun owns the interval from the physical tip to its exit.
The default graphical source configuration uses the explicitly selected
`driven_gaussian_schell` boundary. A classical source without a coherence
definition does not supply a complex wave automatically. The driven boundary
retains the non-paraxial near-tip solution and the existing variable-momentum
quadratic gun continuation. The latter remains a declared approximation; a
shared input identity does not qualify omitted higher-order gun dynamics.

Gun support inspection returns an immutable, non-executable description.
It cannot accidentally run the post-gun column executor over the accelerating
source interval. Unsupported overlapping non-quadratic components are rejected.
The exact gun-exit energy, retained weights, complex wave, longitudinal action,
and axial reference flight time form the column entrance state. Acceleration
before that face is not added again downstream.

## Common fields and dynamic drives

The electric tail remains in the captured scalar potential. There is no
amplitude threshold that replaces the tail by zero. For native scanning
coils, the retained finite-thickness vector potential follows the captured
raster or wobble law at absolute laboratory time:

`t = captured emission epoch + executed elapsed flight time`.

Both `B = curl(A)` and `E = -grad(phi) - partial(A)/partial(t)` are evaluated.
A shared Descan and Image/Diffraction coil is represented once, with its
logical commands combined by that physical owner. This is a prescribed
native uniform-coil model, not a circuit, hysteresis, fringe-field, or
radiation simulation. An ideal raster reset is a discontinuity; a trajectory
crossing a reset cannot silently be propagated with zero induction or a
frozen magnetic field. Unsupported temporal intervals are reported explicitly.

The particle route advances mechanical momentum, energy, and each ray's
elapsed clock. Static scalar-potential energy changes and work by the induced
electric field are accounted for separately. Mechanical interceptions retain
their actual losses and positions. No empirical Cs/Cc screen, fitted Image
map, or additional centre-plane deflector kick is appended to a field-owned
interval.

The declared high-order Gaussian round-lens model has an optional, identified
remote-domain certificate. Outside its native radial cylinder, admission
requires conservative bounds for the exact current-free Gaussian extension,
the retained polynomial, and their required derivatives below half the
smallest binary64 subnormal. The implementation still evaluates the original
finite-order formula; it does not truncate a tail by an amplitude tolerance.
This only qualifies numerical insignificance of a remote engineering field.
It does not widen a mechanical bore, qualify an unknown FEM exterior, or
claim a measured manufacturer field. A swept region needs a whole-box proof:
remote qualified corners alone cannot establish its interior validity.

## Scan calibration

Automatic AC/Descan calibration measures central differences of the actual
finite-coil vector potentials using the same Lorentz executor and geometry.
These local DC chief-ray derivatives are calibration probes, not a source
distribution or a substitute for the propagated beam. The local nominal beam
energy and the upstream AC support face are recorded as the response reference.

Canonical transverse momentum is evaluated from the executed mechanical
momentum and the actual vector potential, `P_perp = p_perp - e A_perp`.
The solver determines AC coupling and requested specimen FOV, then the selected
Descan position or position-and-angle constraint. User pivot offsets, static
alignment, lens excitation, source parameters, and mechanical positions are
retained. A failed solve restores all derived controls and the shared physical
host relationship. Explicit held drives are restored with their provenance;
restoration is not described as a newly executed field calibration.

Calibration does not establish a finite-beam waist or image/diffraction
conjugacy. Its physical response, perturbation size, integration step, and
resolved commands are recorded. Progress covers the complete sequence of
response traces; completing one trace cannot finish the whole calculation.

## Coherent time reference and qualification boundary

The coherent source is a set of stationary energy modes, not a longitudinal
pulse envelope. A time-dependent column calculation therefore requires an
explicit retarded, slowly varying reference-wavefront approximation. A
per-mode axial reference time is not a measured transverse arrival-time
distribution, and it is not a complete time-dependent Klein-Gordon or Dirac
solution. The physical gun's transverse group-delay distribution is not
qualified by a post-gun delay budget.

Electric acceleration changes the reference momentum and its longitudinal
action. The scalar field must enter the same Hermitian square-root expansion
as the magnetic covariant momentum, retaining their cross terms. Independent
thin electric phases would not implement that operator. Forward-branch,
spatial sampling, accumulated truncation, and temporal approximation budgets
remain distinct. Phase belongs to each coherent mode; an incoherent mixture
has no single complex phase.

The implemented positive-branch expansion uses electron charge `q = -e`,
`D = Pi_perp^2 + p_axis(z)^2 - p_local(x,y,z)^2`, and
`H_N = sum(c_n D^n / p_axis^(2n-1), n=1..N) - q Az`, with
`c_n = binomial(2n,n) / (4^n (2n-1))`. The explicit kinetic orders 2, 4,
6 and 8 retain powers through `N = 1, 2, 3, 4`; their coefficients are
`1/2, 1/8, 1/16, 5/128`. Operator powers act repeatedly on the wave,
retaining mixed covariant derivatives and scalar-potential terms. The
order is part of the saved execution policy and cannot change while
resuming an existing history. With static phi,
`K_local = K_entry + e (phi_local - phi_entry)`;
the entrance reference remains fixed across numerical segments. The
represented square-root defect is kept below 0.1 and its omitted series
phase has a non-resetting per-mode 0.01 rad allowance. Higher orders
reduce the omitted kinetic series; they do not relax that allowance or
establish spatial-grid, source, or axial-step convergence.

The added scalar forward-branch connection is bounded on the complete
represented swept cylinder using the original r-squared/Z interpolation
nodes, including radial potential differences already present at entrance.
For the positive operator `Q`, the flux variables use `Q^(1/4)`; the
fractional-power derivative gives an integrated connection majorant
`Gamma <= kappa(Q)^(1/4) Var(Q_E) / (4 lambda_min(Q))`. The admitted added
amplitude error is `expm1(Gamma)`, accumulated with a 1e-6 per-history
allowance. This bounds the added scalar connection, not every previously
omitted magnetic quantum correction. Independent two-way scalar-slab tests
compare reflected flux and full transmitted complex amplitude.

The retarded axial clock integrates the actual induced axial work and scalar
energy law. Its conservative column-added delay includes both the full
represented mechanical-momentum cone and the speed changes caused by
transverse electrostatic and induced-energy variation. Native time-derivative
bounds control delayed vector-potential sampling and the one-energy
transverse approximation, with a separate accumulated 0.01 rad raw-phase
allowance. The source-exit plane is explicitly assumed to define one common
reference wavefront; an initial zero *additional column delay* does not claim
that the physical gun has zero transverse group delay.

The actual complex envelope includes the stable added longitudinal action
`integral (p_axis - p_original_entry) dz / hbar`. Dynamic induced work also
contributes `-integral (H_axis - H_entry) dt / hbar`. The canonical `q Az`
term already belongs to the propagation operator and is never added twice.
The large constant entrance-energy free carrier remains factored out;
`AxialWaveReference` separately retains the total spatial action and clock
as bookkeeping. Readout never reconstructs a tiny tail phase by subtracting
two enormous total actions. A true material collision changes the energy
reference at its physical face; ordinary checkpoint segmentation does not.

These native engineering-field calculations are development models. Passing
regressions or a bounded Normal-mode example is not OEM field calibration,
complete-instrument validation, or qualification of pulsed wavepacket dynamics.

## Strong-field numerical splitting

The carrier kernel has an explicit `quartic_scalar_splitting` numerical option.
It requires the fourth-order kinetic operator and ordinary scalar splitting;
its low-level default is disabled. Writing the discrete covariant operator as
`K = L + U`, where `U = m_perp^2 + electric_defect` is real multiplication,
gives `K^2 = L^2 + L U + U L + U^2`. The option evaluates `U^2` together with
the existing real scalar phase. The remaining differential operator evaluates
`L(L psi) + L(U psi) + U(L psi)` directly. It does not subtract two large
nearly equal arrays or omit derivatives of the variable coefficients.

This changes the numerical factorization, not the retained electromagnetic
Hamiltonian. The full kinetic-domain and omitted square-root remainder bounds
remain unchanged. Scalar phase sampling and spectral checks still apply.
The differential exponential uses its smaller proven operator bound, while
the additional Strang splitting error needs independent axial-step convergence;
it is not included in the exponential-series bound. Saved continuations retain
the selected operator policy and reject an unannounced change. Independent
numerical comparisons start from the same executed entrance and preserve its
producer and cumulative error/loss history separately.

For kinetic orders 6 and 8, `higher_order_scalar_splitting` extends this
same factorization to all retained powers. It is an explicit option,
disabled by default in the low-level kernel, and requires scalar splitting.
It cannot be combined with the fourth-order-only option. Each real `U^n`
term is exponentiated as a scalar phase; the differential remainder uses
`R_1 psi = L psi` and
`R_(n+1) psi = K(R_n psi) + L(U^n psi)`. This preserves the full
noncommuting `K^n - U^n` without subtracting large nearly equal arrays.
The bound uses positive binomial terms in `||L||` and `||U||`, while the
kinetic remainder still uses the full represented `K` bound. The splitting
remains second order in distance regardless of the retained kinetic order.

## Device execution and numerical controls

The coherent advanced controls separate kinetic order, axial step, source
sampling, and optional controlled support projection. Projection limits are
relative to each original entrance mode: an initial weight `w_i` receives
absolute allowances `epsilon * sqrt(w_i)` and `delta * w_i`. The allocation
is fixed at entrance, never redistributed after physical clipping. Projection
receipts remain numerical losses, distinct from mechanical absorption; they
bound the projection itself, not subsequent different-domain propagation.
Changing a numerical request invalidates its observations even at the same Z.

An entrance must pass its original occupied-edge check. If lossless Fourier
preparation exposes additional weak edge samples, an explicitly supplied
support ledger may admit a same-grid two-cell C2 window. This is a separately
recorded `numerical_boundary_taper`, using the same remaining and per-event
allowances as other support projections. It preserves sample coordinates,
analytical carriers and interior complex samples; it neither repairs flux nor
claims a lossless extension. The candidate must pass all original boundary,
spectrum and field-domain checks before its receipt is committed. Original
occupied entrances, unknown field domains, and propagation boundary failures
are not converted into this retry. The new receipt uses ledger schema v2;
historical crop-only v1 records retain their original meaning.

The explicit `step_boundaries_and_retry` policy considers support projection
at step entrance, after the executed quadratic canonical map and before the
upper residual, and on sampling retries. Historical `sampling_retry` and
`step_start_and_retry` policies keep their original timing. The selected
policy is bound to continuations and cache identity.

For the post-canonical projection, the complete unprojected output first
passes its lattice, pixel limit, boundary, spectrum and field-domain checks.
A conservative 256-bytes-per-cell diagnostic/window reservation also counts
the retained entrance buffer. This does not reserve the 640-bytes-per-cell
workspace of a residual that has not yet run. A candidate projection must
then pass the original full residual-memory and physical checks before its
loss is committed. The unprojected radius remains in electric and retarded
error bounds. With no admissible projection, the original memory requirement
still applies; the new ordering never expands a declared resource limit.

Support windows can broaden a previously resolved envelope spectrum. At step
starts, after canonical propagation, and during sampling retries, a compacted
wave therefore remains a proposal while the ordinary lossless Fourier
refinement checks its boundary, spectrum, field domain and resource limits.
Only an admitted candidate commits its projection receipt. If the optional
candidate cannot meet its sampling, boundary or resource limits, execution
keeps the unchanged valid input and unspent ledger. Unknown field domains and
cancellation remain errors; this mechanism never repairs an invalid original
entrance, removes Fourier coefficients, or relaxes a physical gate.

The optional CUDA path accelerates the complete carrier residual exponential
in complex128/float64. Field sampling, physical-domain admission, canonical
propagation, and checkpoints remain on the host. Fixed coefficients are
uploaded once per residual; all retained kinetic powers and every exponential
substep's boundary and spectral checks execute on the selected device. The
unchanged full-domain kinetic bound and scalar phase-sampling gate are evaluated
before execution, and final host flux accounting still applies.

Device admission follows the existing compute policy and memory limit. A
permitted unavailable-device or out-of-memory fallback restarts the entire
residual from its unchanged host entrance. Require GPU never falls back.
Residual grid planning checks a required device's declared workspace before
allocating a Fourier refinement. When the usual 25% candidate growth would
exceed that partition, it tries the actual requested scale with the existing
5% resolution guard. Optional acceleration may prefer this smaller admitted
candidate, while retaining the original allowed CPU fallback if neither
candidate fits the device. Host and device partitions remain separate;
unused host memory cannot enlarge a required device's limit. Availability and
live device memory are still checked at execution. Every candidate reruns the
complete optical operator and its original sampling and physical gates.
Cancellation, physical or numerical admission failures, and unknown device
errors do not become CPU retries. Cleanup preserves the original failure.
The executed backend and any fallback reason are recorded separately from
the shared electromagnetic-field identity. CPU/CUDA parity establishes this
device implementation on tested inputs; it does not establish source, grid,
axial-step, or whole-instrument accuracy.

The optional `column_exponential_method="chebyshev"` request selects a bounded
Hermitian polynomial recurrence for the same complete residual operator.
`column_chebyshev_group` limits concurrent intermediate results (1 to 32,
default 8). The method remains explicitly selected; the request default is
Taylor. All original Taylor substep times retain their boundary and spectrum
checks, including intermediate times within a group. Resource admission can
reduce the group or retain Taylor on the selected backend. Require GPU still
cannot become CPU execution.

The polynomial certificate separately bounds exact-arithmetic truncation and
binary64 coefficient conversion at each original check time. It does not bound
FFT or recurrence roundoff. The original Taylor-based flux tolerance is kept
unchanged; fewer operator applications do not weaken a physical check or
justify normalizing the output. Executed receipts distinguish planned Taylor
applications from actual applications and record any method fallback.
Operator-policy schema v3 binds the method and group to continuation. Valid
v1/v2 receipts retain their historical Taylor meaning and cannot silently
adopt Chebyshev. Independent algorithm comparisons preserve the original
entrance, producer, absolute weights and already spent physical/numerical
history.

## Public theoretical references

- [Khan and Jagannathan, quantum charged-particle beam transport](https://doi.org/10.1103/PhysRevE.51.2510): paraxial and aberration descriptions for magnetic lenses; this is not a blanket qualification of time-dependent acceleration.
- [Halvorsen and Kvaal, gauge-invariant discretization](https://arxiv.org/abs/0905.4177): electromagnetic gauge consistency in discretized quantum propagation.
- [Op't Root and Stolk, one-way wave propagation](https://staff.science.uva.nl/~cstolk/Papers/OptRootStolk2010.pdf): one-way factorization and the distinction from the full two-way equation.
- [Sang et al., dynamic STEM scanning](https://link.springer.com/article/10.1186/s40679-016-0020-3): physical scan dynamics that are not reproduced by merely prescribing an ideal raster waveform.
