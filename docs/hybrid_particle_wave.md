# Hybrid particle and wave transport

> **Retired Tip-origin execution.** The Tip-to-observation calculation described below
> is retained as historical documentation and evidence, not an executable workflow.
> New work uses [particle transport and near-specimen mixed coherence](MIXED_COHERENCE.md).
> Existing inputs, numeric results and archives remain unchanged and readable;
> Tip-origin propagation and continuation are rejected before CUDA or cache reuse.

This opt-in development workflow uses the existing reduced-action ray integrator
to propagate declared coherent modes, executes the configured finite specimen
as a wave, and samples weighted outgoing particles for the physical receivers.
It does not replace the normal instrument calculation or change an operating
point. The external electric and magnetic potentials remain fixed: specimen
feedback onto those fields is omitted.

## Interfaces and execution

`temsim.physics.hybrid_pipeline.simulate_hybrid(state, request)` accepts a
`HybridRequest` with an explicit detector, specimen entrance lattice and
numerical budgets. `HybridGrid` defines a numerical observation plane; its width
is not a probe-size constraint. Ordered intermediate grids may be specified for
large changes in beam extent. The existing instrument must select Finite-Field
Optics, a static field configuration and an inserted downstream receiver.

The physical-source interface uses `trace_source_to_exit` and
`simulate_tip_wave`. It requires the actual gun coherence model; no phases are
invented for a classical source without coherence data. A later coherent start
plane must be computed by that same physical gun wave chain. This entry point
does not accept an arbitrary downstream source as a physical FEG equivalent.

Run a saved instrument profile with:

```powershell
.\.venv\Scripts\python.exe -m temsim.hybrid_cli --profile <instrument.toml> --output <new-directory> --detector camera --sample-width-nm 40 --sample-pixels 512 --source-particles 1024 --exit-particles 4096
```

The width above is an example numerical input, not a qualified setting. Use
`--intermediate-grids` with a JSON array of `HybridGrid` fields when needed.
`--hybrid-start-z-mm` requests a later checkpoint from the physical wave chain.
Insufficient sampling or resource budgets fail explicitly. Large column spans
may need intermediate planes and substantially larger quadrature budgets.

All new coherent jobs require working CUDA/CuPy. The wave/frame and diffraction
operations execute on CUDA; canonical tangent rays and downstream Lorentz rays
use CPU. CUDA failure is not retried on CPU. The original source population is
checked for at least 1000 unique positive-current particles (default 1024).
Coherent-state quadrature packets, tangent probes and exit samples are separate
populations and never satisfy that source-count requirement.

Results contain the complex incident and exit modes, mode energies and weights,
original source IDs and launch arrays, action/caustic receipts, material branch
histories, raw and transport diffraction, exit particle arrays, downstream
trajectories and receiver/loss accounting. `hybrid_arrays.npz` uses non-object
arrays; `report.json` identifies their units and provenance. An existing output
directory is never overwritten. Receiver times are elapsed times from the
static handoff, not reconstructed absolute emission times. Earlier inserted
receivers can intercept particles before the selected receiver.

## Coherent modes, amplitudes and caustics

Each `WaveMode` represents a unit complex shape and an absolute probability
weight. Amplitudes interfere within one mode. Independent modes, energy
components and conditional material histories contribute intensities.

The executed full complex entrance wave is expanded in a finite Gaussian
coherent-state frame. Its reconstruction error is checked before propagation.
Frame coefficients are complex quadrature amplitudes, not statistical electron
weights. Each packet follows the existing static-potential ray solver and its
`reduced_action_m`. Canonical transverse coordinates are

\[
u=(x,y,P_x/p_0,P_y/p_0),\qquad P=p_{\rm mechanical}-eA,
\]

where the electron charge is `-e` and `p0` is fixed within a mode interval. All
packets share one total-energy reference, including the scalar potential at the
entrance. The solver's action, originally divided by each ray's own entrance
momentum, is converted to this common reference before evaluating its phase:

\[
\ell_{0}=\frac{p_i}{p_0}\ell_i+
  \left(\frac{p_i}{p_0}-1\right)(z-z_0),\qquad
\varphi=\frac{p_0\ell_0}{\hbar}.
\]

This is canonical reduced action, not geometric path length divided by one
fixed wavelength. A common axial phase does not affect same-mode interference.

Central finite differences give the canonical tangent matrix
`M = [[A, B], [C, D]]`. A Herman--Kluk prefactor transports the amplitude:

\[
C_{\rm HK}=\sqrt{\det\left[
\tfrac12(A+D-i\gamma B+iC/\gamma)\right]},\qquad
\gamma=(2k_0\sigma^2)^{-1}.
\]

The square root is lifted continuously using successive matrix eigenphases.
It is not reset to a principal square root at each observation plane; this
retains simultaneous transverse focus crossings and repeated windings. The
scaled symplectic residual and phase increments are checked. Axial refinement
is still required: a coarse path that skips a complete winding cannot be
certified from its endpoints alone.

Output norm changes are retained as signed numerical probability changes.
Factoring a result into unit shape and weight does not restore current. Finite
physical apertures and bores use the existing resolved absorbing masks, with
slab-face events, thickness subdivisions and tilted-boundary step limits.
Hard-edge diffraction, frame spacing, finite-difference size, integration step
and output domain all require convergence checks.

## Material and outgoing particles

The material stage calls `propagate_potential_specimen` with the configured real
specimen. It retains actual atomic potentials, finite slice thickness, selected
frozen-phonon configurations, conditional inelastic histories and external
fields. Existing unsupported-material and specimen-orientation checks remain
active. The default request uses conditional inelastic trajectories;
`--inelastic zero_loss` selects only the unscattered-inelastic branch and records
the removed weight. No empirical ray collision or extra zero-loss factor is
applied after this calculation. No unconfigured material constants are fitted.

A fresh wave-operator error ledger is bound at the material entrance. Its scope
starts at that plane. The complete upstream semiclassical receipt is retained
as a parent; its approximation error is not represented as a rigorous
full-column wave truncation bound.

For the handoff, Gaussian spatial windows satisfy `sum_j |w_j(r)|^2 = 1`.
Each window is Fourier transformed after a straight-path Wilson phase:

\[
\psi_j(r)=w_j(r)\psi(r)
\exp\left[\frac{ie}{\hbar}\int_{q_j}^{r}A\cdot dl\right].
\]

This cancels a static electromagnetic gauge phase and yields a positive local
mechanical-momentum spectrum. Summing every window and momentum bin preserves
the mode probability. The line-integral quadrature and Fourier boundary are
checked. Production sampling streams windows instead of allocating a complete
four-dimensional position/momentum table.

Mode-stratified systematic samples use probability proportional to intensity.
Every sampled particle in mode `m` has weight `W_m / N_m`, not an extra factor of
intensity. `p_xy = h f_xy`; the forward longitudinal momentum follows the local
kinetic energy. Position, direction, energy and mode provenance are retained
through the existing unified particle transport and physical interception.
Complete absorption yields zero outgoing particles and zero detector current.

This handoff is explicitly a finite-resolution approximation. The transported
angular marginal is broadened by the spatial windows. The nominal angular RMS
is `1 / (2 k sigma)` and is stored with the physical window width.
The `--exit-window-nm` control selects this physical width (default 0.5 nm),
independently of material-grid refinement; its actual lattice spacing is saved.
The raw full Fourier diffraction is exported separately; in magnetic fields it is a
canonical spectrum in the captured gauge, not directly mechanical angles.
Downstream particle propagation does not reproduce later coherent interference,
electron holography or an arbitrary phase-contrast image. Those still require
the coherent wave detector path.

## Ideal single-mode development validation

The completed 4096/8192-particle trials and numerical evidence are recorded in
[hybrid_validation_2026-11-09.md](hybrid_validation_2026-11-09.md). They retain
1024 original positive-current source particles separately from exit samples.

The user-authorized validation source is explicitly ideal and independent of
the default FEG configuration:

```powershell
.\.venv\Scripts\python.exe scripts\validate_hybrid_workflow.py --output <new-directory> --pixels 512 --frame-stride 64
```

The script defines a 300 keV Gaussian with 0.2 nm intensity RMS in each transverse
axis, weight 0.4 at a 1 nA reference current, a short field-free coherent interval,
the actual Si CIF fixture at 0.4 nm thickness and a downstream camera. It traces
at least 1024 original positive-current samples of the ideal Gaussian Wigner
source separately from the coherent quadrature, and then executes CUDA wave
propagation and weighted exit transport. The report labels this
`IDEAL_DEVELOPMENT_VALIDATION` and `physical_feg_phase_chain: false`.
It does not qualify the full microscope or the real cold-field source.

`--resume-incident-manifest` restores an explicitly selected checked entrance.
`--resume-exit-manifest`, together with that entrance, can continue a completed
material exit when source, fields, physical inputs and implementation identities
match. The development script will reject a stale material implementation.

After execution, audit the retained arrays and generate a figure with:

```powershell
.\.venv\Scripts\python.exe scripts\analyse_hybrid_validation.py <completed-directory>
```

The focused small-array tests cover independent Fresnel propagation,
simultaneous caustics and full winding, noncanonical-map rejection, gauge
invariance, multiple incoherent mode weights/energies, CUDA admission, original
source identity, resolved plate geometry, material budget handoff, complete
absorption, complex-wave export and no result overwrites. These operator tests
are distinct from finite-source physical trials. Increasing the original
source count, exit samples and numerical resolution remains necessary for any
instrument or observable convergence claim.

## Method references

The physical-source follow-up is recorded in
[Physical FEG coherence and full-column hybrid validation](feg_coherence_and_full_column_validation.md).
It adds an explicit tip-plane coherence length, preserves the installed FEG
and the original-current interception ledger, and records separately completed
classical references and coherent admission results. Its numerical trials do
not turn the ideal short-column validation above into full-instrument
qualification.

- Herman and Kluk, *A semiclassical justification for the use of non-spreading
  wavepackets in dynamics calculations*, Chemical Physics 91 (1984),
  [original paper](https://doi.org/10.1016/0301-0104(84)80039-7).
- Robert, *On the Herman--Kluk semiclassical approximation*,
  [arXiv:0908.0847](https://arxiv.org/abs/0908.0847).
- Aharonov and Bohm, *Significance of Electromagnetic Potentials in the Quantum
  Theory*, [Physical Review 115, 485](https://doi.org/10.1103/PhysRev.115.485).

The gauge-covariant finite-window measurement above is the explicitly defined
numerical handoff used here; it is not claimed to be an exact positive quantum
phase-space representation.
