# Ronchigram methods and TEM simulator integration

The useful extension is a reproducible Ronchigram diagnostic connected to the
executed gun-to-specimen wave and the actual configured specimen, with
calibrated probe and pupil observables. The real-material specimen path is
primary. Random amorphous generation is an additional specimen construction
method, not a substitute silently applied to an imported sample.
The existing optical model should continue to determine the wave. A fitted
aberration polynomial must not replace the physical source, reapply aberrations
already present in the wave, or change installed lens excitations.

Status: method review and implementation design, 9 November 2026. No numerical
implementation, coherent execution, instrument retuning or new physical
qualification is included in this checkpoint.

## Reference evidence

The supplied [shared conversation](https://chatgpt.com/share/6af20a10-75d4-83ed-85a6-90ad97016929)
analyses [Ronchigram.com](https://ronchigram.com/). Its discussion was read in
full, and the central computational claims were checked against the public
[C++ source](https://github.com/sukhsung/ronchigram/blob/master/emsrc/calculateRonch.cpp).
These are observations of the retrieved source, not a reproduction of the
conversation's WebAssembly or AddressSanitizer experiments:

- `calculateChi0` evaluates a polar aberration expansion; `calculateChi` uses
  `exp(-i chi)`.
- `generateSample` repeats coarse random values into blocks. `calcDiffract`
  applies the displayed circular mask after specimen scattering.
- `probeGeneration` returns amplitude magnitude; quantitative probe intensity
  requires its square.
- `packageOutput` allocates two images but writes three. The interaction
  parameter combines SI rest energy with an unconverted electronvolt value.
- The phase-limit calculation retains a sentinel when no violation exists;
  the Strehl search lacks a verified bracket and monotonicity check.

These implementation details are reasons to reuse the physical idea rather
than transplant that calculation core. The authors' [Strehl study](https://arxiv.org/abs/2007.12281)
motivates a convergence-angle diagnostic; its result does not establish a
universal minimum of every probe-width or current-containment metric.

## Calculation contract

For a declared paraxial angular pupil, the familiar reference calculation is

\[
W(\alpha,\phi)=\sum_{n,m}\frac{C_{nm}}{n+1}
\alpha^{n+1}\cos[m(\phi-\phi_{nm})],\qquad
\chi=\frac{2\pi W}{\lambda},
\]

\[
\psi_{\mathrm{in}}=\mathcal F^{-1}[A_{\mathrm{illum}}e^{-i\chi}],
\quad t=\exp(i\sigma V_{\mathrm{proj}}),
\quad I_R=|\mathcal F[t\psi_{\mathrm{in}}]|^2.
\]

The positive specimen phase follows this repository's positive electrostatic
potential convention. Here `W` and `C_nm` are lengths, `chi` is in radians,
and `V_proj` is a potential integrated along the beam. Existing code uses
millimetres for aberration coefficients and V Angstrom for projected specimen
potentials; conversions belong at the interfaces.

For the physical instrument, replace the constructed `psi_in` in that
reference equation with the **already executed specimen-entrance mode**.
Retain its full complex amplitude, analytical curvature and tilt, physical
coordinate lattice, energy, reference current and mode weight. Do not multiply
it by another equivalent system-aberration phase or another representation of
an upstream physical aperture.

For independent spatial, energy and material modes, form the observable as
`sum(weight_j * abs(detector_wave_j)**2)`. A weighted average of complex waves
would introduce interference between independent modes. There is no unique
phase plate for the resulting intensity mixture.

The Fourier expression is an ideal far-field diagnostic. An instrument camera
image still requires the installed downstream lenses, stops, recording plane
and detector response. In a magnetic field a canonical Fourier spectrum is
not the mechanical scattering-angle distribution; retain the existing gauge
labels or propagate to an appropriate observation plane.

## Existing implementation and extension points

Paths below are relative to the repository root and describe the inspected
working tree, including its uncommitted assembly-study additions.

| Area | Existing implementation | Required extension |
| --- | --- | --- |
| Aberration convention | `src/temsim/optics/aberration_basis.py` and `aberrations.py` share `W`, `chi`, laboratory X/Y and harmonic-angle conversion. | Reuse these definitions in diagnostics. No parallel coefficient convention. |
| Supported terms | C1, A1, B2, A2, C3, S3, A3, C5 and A5; 15 real Cartesian components. | A later independent task may add B4/C41, D4/C43, A4/C45, S5/C52 and R5/C54 with fitting and serialization coverage. Additional terms do not establish physical correctability. |
| Executed source and wave | `physics/tip_wave_pipeline.py` retains tip-origin checkpoints and explicit specimen entrance/exit stages. | Bind the diagnostic to a captured execution and exact plane, including its approximation and qualification status. |
| Specimen interaction | `specimen/source.py` admits vacuum or an imported CIF/MCIF; `specimen/atomistic.py` builds finite projected atomic potentials. `physics/specimen_wave_transport.py`, `potential_specimen_transport.py` and `multislice.py` provide propagation. | Use the configured material as the primary specimen. Add physically constrained random amorphous structures through the same potential interface; retain a separately labelled statistical phase approximation only where appropriate. |
| Phase comparison | `physics/wavefront_comparison.py` and `gui/wavefront_comparison.py` compare matched modes and preserve actual physical coordinates. | Keep real-space phase separate from angular-pupil aberration phase. Wrapped phase differences cannot supply an unwrapped pi/4 limit. |
| Angular readout | `physics/wave_plane_observables.py::canonical_angular_spectrum` includes full phase and carries a gauge warning. | Add an appropriately labelled Ronchigram view and downstream detector readout; do not merely rename this spectrum a measured camera image. |
| Probe size | `physics/wave_observables.py` provides current-weighted centroid and R95 on retained grids. | Add consistently defined D50/D95, peak location and core-current reporting to the same result. D95 is twice R95, not an RMS conversion. |
| Physical controls | `optics/probe_service.py` already separates intrinsic outputs from saved learning gains. | Read outcomes after physical controls are applied; do not introduce direct coefficient knobs as hardware controls. |
| Assembly studies | `assembly_error_study.py` stores fixed-pupil residuals and correction headroom. | Associate selected stages with separately executed coherent observations rather than constructing waves from their fitted coefficients. |

Historical functions in `stem_wave_imaging.py` already show a
pupil-before-specimen FFT chain. However, `physics/source_admission.py`
deliberately blocks that reduced-order source route for new production images.
Using a new diagnostic entry point to bypass this gate would not be a valid
integration. The development tip-origin path is available with explicit
limits; it is not full-image qualification.

## Proposed first implementation

1. **Captured observation input.** Accept an executed specimen checkpoint,
   its instrument identity, physical plane, mode identities and energies,
   source/current reference, material history and numerical limits. Missing
   phase remains unavailable. Live GUI state must not silently alter a stored
   observation.
2. **Configured material first.** Use the imported specimen's atomic species,
   coordinates, orientation, finite shape and thickness to build its slice
   potentials. Extend the source interface for disordered atomic structures
   and reproducible amorphous generation; do not swap in a random phase
   texture. Record composition, density, structural constraints and seed.
   Keep the structure unchanged during a control comparison and use the
   existing relativistic interaction constant at each mode's energy.
3. **Joint readout.** Show incident probe intensity, specimen diffraction and
   an explicitly selected single-mode phase view for the same execution.
   Record angular and spatial axes, raw probability/current data and display
   scaling separately. Illumination aperture and detector acceptance have
   different owners and controls.
4. **Probe-quality metrics.** Report centroid-referenced D50/D95, peak position,
   current in a specified core and represented current loss. Preserve a
   separate optical-axis reference where relevant. Never normalize away
   physical interception to improve a displayed probe.
5. **Assembly comparison.** Allow nominal, perturbed, intrinsic-compensated
   and saved-learning-compensated stages to link to coherent results from
   their exact captured states and common specimen. Keep absent coherent
   results explicitly unavailable. Continue displaying hardware headroom
   and local diagnostic residuals separately.

This scope strengthens the observable link between hardware and image
formation. It does not require a browser/WebAssembly implementation or a
neural network. Existing CuPy acceleration and GUI job coordination remain
the intended execution mechanisms.

## Comparison of wave calculations

The website compresses the upstream optics into an angular aberration phase
and the specimen into one two-dimensional random transmission function. Its
main image therefore requires two two-dimensional transforms and a complex
pointwise multiplication, rather than axial propagation inside the specimen.

This repository has distinct numerical layers:

- The historical equivalent-pupil TEM/STEM functions resemble that Fourier
  construction, but their production source-admission gate remains closed.
- The development tip-origin path propagates captured complex modes through
  the gun and column. The unified field route applies a scalar forward
  electromagnetic operator, with a quadratic carrier and a differential
  residual. It does not reduce every lens to a fitted aberration screen or
  claim an unrestricted Dirac/Maxwell solution.
- The standalone elastic multislice operator uses symmetric Fresnel half
  steps around each slice transmission: `P(dz/2) T_j P(dz/2)`, with
  `P(dz) = F^-1 exp(-i*pi*lambda*dz*(qx^2+qy^2)) F`.
- The full-column specimen coupling instead splits each material phase
  around distributed column transport: `T_j^(1/2) U_column(dz) T_j^(1/2)`.
  This retains admitted lens fields through the specimen interval. Source,
  energy and frozen-phonon modes remain independent weighted branches.

`specimen/atomistic.py` uses the optional abTEM/ASE backend with the neutral
Lobato independent-atom potential and finite slice projection. The model
contains atomic species and positions, but not bonding-charge redistribution.
Its frozen-phonon sampling displaces atoms about an existing structure; it
does not create a physically validated amorphous structure. The elastic
multislice kernel itself has no inelastic channel; the full pipeline has
separate attenuation and conditional-history models with their own limits.

For amorphous material, a structural generator should retain composition,
density, thickness and short-range order, with minimum-separation checks and
appropriate structural validation or relaxation. Merely displacing a crystal
or accepting a random packing is not proof of a realistic material. The
[abTEM atomic-model tutorial](https://abtem.github.io/doc/user_guide/tutorials/advanced_atomic_models.html)
also marks its displaced-diamond carbon construction as an approximation.
Generated atomic structures should feed the same slice-potential calculation
as imported structures.

A faster statistical projected-potential model may additionally specify a
physical correlation spectrum and potential variance calibrated for a stated
material and thickness. It remains a declared approximation. Pixel block
size alone provides neither those material properties nor three-dimensional
structure. Keep structural disorder, thermal configurations and detector
noise as distinct random processes with recorded identities.

A crystalline imported specimen will generally show its own convergent-beam
diffraction structure. It should not be forced to resemble an amorphous
alignment Ronchigram. The observation must follow the selected material.

## Subsequent pupil quality tools

The pi/4 rule requires an **unwrapped angular-pupil aberration phase**, a
declared reference and valid support. The current real-space phase comparison
does not supply this automatically. Preserve intentional defocus and tilt;
any piston convention must be explicit. When the threshold is not reached,
report a range-limited lower bound rather than an enormous recommended angle.

For a qualified pupil representation, define peak Strehl relative to a
phase-flat pupil with the **same amplitude, support and input flux**, at the
same fixed observation plane. Record whether the peak may move. The existing
ideal-column reference retains focus and changes some optical actions, so it
must not automatically be used as this Strehl denominator. A multimode or
physical-reference peak ratio requires its own definition.

An aperture study should sample its declared range, retain all observed
threshold intervals and refine only verified brackets. Do not assume global
monotonicity or equate Strehl 0.8 with minimum D95. A numerical pupil mask is
a hypothetical diagnostic; changing a real upstream aperture requires new
propagation through its actual plane and retains the resulting current loss.

Fitted assembly residuals remain local diagnostics. In particular, coefficients
fitted on a 0.5 mrad mathematical pupil cannot justify a 30 mrad coherent
prediction without independently established coverage and fit accuracy.
Likewise, a fitted residual already encoded by full-field wave propagation
must not be applied again as a corrective phase screen.

## Control and coordinate continuity

- The column frame remains right-handed X/Y with the tip at Z = 0. Detector
  orientation and any display projection are explicit transforms.
- Polar coefficient components use `C*cos(m*phi)` and `C*sin(m*phi)`.
  Mathematical components are not physical service channels. A2 upper/lower
  controls retain their existing meaning; AC/DP12 remains one coil assembly.
- Intrinsic service outputs and explicit saved learning feedforward remain
  separately recorded. A naturally coupled physical response is not an
  additional software learning command. No CNN-based inference is implied
  by the existing corrector-learning terminology.
- Physical assembly azimuth, field-only azimuth error and diameter-axis tilt
  retain distinct meanings. No fitted phase view changes CAD or assembly.

## Validation requirements for implementation

Use bounded analytical/operator tests for vacuum propagation, phase-object
norm conservation, Fourier orientation, SI conversions and pupil sampling.
In vacuum an ideal pupil calculation must return `abs(A)**2` in diffraction
regardless of its aberration phase. With zero aberration, same-pupil Strehl
must be one; an unmet phase threshold must remain range-limited.

Exercise pure C1/A1/A2 cases and mixed modes; verify that changing detector
acceptance does not change the incident probe, and that independent modes
add intensities. Fix sample identity while varying a control. Check physical
pixel size against the FFT lattice, not against the user aperture radius:
for a small-angle grid of total width `2*alpha_max`,
`dx = wavelength/(2*alpha_max)`.

New coherent execution must pass the repository's usable CUDA/CuPy check and
must not fall back to CPU. New finite-source particle comparisons must use
at least 1000 original positive-current source particles, usually 1024, with
emitted, surviving and effective populations retained separately. These
requirements do not make a calculation converged. Spatial, axial, mode,
energy and specimen sampling require their own comparisons before making
quantitative microscope-performance claims.
