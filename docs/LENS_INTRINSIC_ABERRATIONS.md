# Intrinsic single-lens aberrations and system residual aberrations

## Workflow

Select a lens in the instrument component list and open **Parameters → Operating → Intrinsic aberrations...**.
Each row provides a coefficient (mm) and an azimuth (deg) for non-axisymmetric terms, with scientific notation supported.
Apply updates only the selected lens; Reset draft resets only this dialog's draft and still requires Apply.
Existing Cs and Cc retain their original single-lens input controls and are not redefined in the new table.

After editing, recalculate Ray Diagram or Electron beam. The new inputs belong to the operating point and are retained in saved inputs,
calculation snapshots, and result archives; they do not change default excitations in the base TOML.
Inputs from the preceding version (schema 78) load with all new terms zero; resaving uses schema 79.
Old executed caches cannot masquerade as new-version propagation results; required parts are recalculated.

In Scanning Image's Probe Aberrations or Illuminating Image's Image Aberrations,
open **System residual → Measure system residual** to measure the optical snapshot from the latest
completed calculation. Live parameter edits must be recalculated before entering this snapshot.
Measurement has a separate Cancel button; opening the page does not run it automatically.

## Physical definitions and ownership

The new terms are C1 (intrinsic defocus), A1, B2, A2, S3, A3, C5, and A5.
They use the lens's local optical X/Y coordinates, with azimuth measured from local +X toward +Y.
All new coefficients default to zero; existing source, excitation, polarity, position, Cs/Cc, and corrector strengths are not adjusted automatically.
This is a finite-order model and does not yet include all possible higher-order aberrations.

The shared wave-aberration basis is

\[
W(\alpha,\phi)=\sum_{n,m}\frac{C_{nm}}{n+1}
\alpha^{n+1}\cos[m(\phi-\phi_{nm})].
\]

The equivalent thin operator at each lens reference plane uses local transverse coordinate \(q\) and that lens's current focal length \(f\):

\[
S(q)=-W(q/f),\qquad
\psi^+=\exp(i2\pi S/\lambda)\psi^-,\qquad
\Delta\theta=\nabla_q S.
\]

Particle directions and coherent phases use the same analytical polynomial and its derivatives. These coefficients are not residual coefficients
measured directly at an arbitrary system image plane; cross-lens comparisons and aggregation require remeasurement at an explicit reference plane, not direct addition.
Tilt uses the existing bounded first-order canonical/Weyl approximation, with remainder and sampling-budget checks; it is not an exact solution at arbitrary large angles.
Optical tilt transforms optical coordinates only and does not rotate the outer mechanical envelope.

Cs is applied once by the existing Cs operator. Effects of offsets/tilts and corrector fields such as HP arise from actual propagation
and must not be copied into intrinsic terms. Lenses using actual mapped magnetic fields do not superpose these native equivalent terms, avoiding double counting
of the same field; edited values remain saved for the native analytical model.
System residual reports are observational only: they do not write back to lens inputs or apply another correction to the full-column wavefunction.

First-order plane classification and finite scan responses include the new terms' effects on local first-order response, especially C1/A1.
System residual diagnostics explicitly use a nominal first-order baseline without the new intrinsic terms, retaining measurable C1/A1 residuals.

## Residual report

- Probe: the pupil is defined at the specimen's exact Z plane using laboratory X/Y and nominal mechanical angles at the specimen plane.
- Image: displacement is measured at the exact Z of the objective image plane, then mapped back through the position-transfer matrix to specimen reference coordinates;
  the display plane and coefficient reference plane are labeled separately.
- All lenses, currently effective deflections, correctors, optical poses, and intrinsic terms pass through the same production propagation plan.
  The report retains C1 and displays training residual, independent validation residual, semiangle, step size, reference planes, and actual computational backend.
- Cc is not measured by this single-energy diagnostic; energy spread, finite source, specimen scattering, aperture acceptance, and step-size convergence
  are outside its validation conclusions. Small numerical values do not mean a real instrument is aberration-free.

The old single equivalent TEM/STEM aberration transfer function cannot fully represent these distributed column effects. Nonzero new terms
must not be silently ignored or incorrectly converted to Cs in that route; use Electron beam for full propagation.

## Computation and validation scope

C1/A1 quadratic phases are stored in the analytical carrier and need not be densely sampled; higher-order phases refine the grid within the existing budget.
Full-column wave operators support actual CuPy GPU execution; CPU/GPU comparisons check complex amplitudes, not just intensity.
Particle propagation and coherent calculations with new intrinsic terms continue to read the same top-level CPU/GPU option. Auto
selects GPU when available, CPU forces CPU, and Require GPU forbids silent fallback. No additional backend button is introduced.
Ordinary intervals of aligned optics use the existing CUDA integrator, calling the shared CuPy operator at aberration planes;
only full double-precision states at these sparse planes are used for interval handoffs, never restored from plotting-precision trajectories.
Native analytical lenses and multipole fields with pose changes use GPU vector integration, retaining the same field, tilt, and aberration definitions.
Actual energy and flight time continue through electric-field changes. Existing Cs and new terms are each applied once at their respective reference planes.
Arbitrary imported magnetic-field maps and residual-gas collisions remain within the existing CPU support scope; this does not establish that every calculation runs on GPU.
Only unavailable hardware or insufficient GPU memory permits Auto to rerun the CPU path from original inputs; physical-input errors are reported directly.

Targeted checks cover independent analytical formulas, local coordinates and tilt, particle/wave centroids and momentum, actual GPU
complex fields and double-precision particle checkpoint agreement, single application of screens and exact checkpoint restoration, archives, cache invalidation, transactional GUI application, and reference-plane
residual fitting. They are not experimental qualification of the whole TEM across all parameter ranges.

### 2026-10-11 particle GPU comparison

Actual GPU tests ran on NVIDIA GeForce RTX 5090, covering all eight new term classes, tilt,
native multipole fields, existing Cs, electrostatic energy changes, flight time, exact checkpoint restoration, and top-level backend selection.
Both coherent complex-field CPU/GPU comparisons and local narrow-wavepacket centroid-response comparisons passed; this does not imply that wave distributions
should be identical to noninterfering particle distributions.

A warmed-up targeted benchmark used 3000 particles, 2000 integration steps, and a 100 mm propagation interval,
with all eight new terms and existing Cs active in one lens. CPU took 0.608 s and GPU 0.0657 s, approximately 9.3 times faster.
Maximum exact-checkpoint differences were 2.72×10⁻²⁰ m in position and 4.34×10⁻¹⁹ rad in direction;
energy and flight time agreed in this example. Inputs came from `test_intrinsic_particle_backend.py`,
using `fixture(_all_terms())` and `_sources(3000)`, with lens peak field 0.002 T and Cs 0.5 mm,
step 0.05 mm, Z from −50 to +50 mm, image-history spacing 0.5 mm, and checkpoints at 0 and 50 mm.
This numerical example isolates the new route; it is not full-GUI/full-instrument timing and excludes first-time GPU compilation;
sparse particles or small batches with many tilted fields are not guaranteed the same speedup.
