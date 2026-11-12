# Particle transport and near-specimen mixed coherence

New calculations use physical particle transport through the gun and column,
then introduce an explicit mixed-coherence model near the specimen. Coherent
propagation from the physical Tip is retired. A historical Tip-wave archive
can still be read and displayed, but its calculation cannot resume.

Keep the applied lens strengths, corrector compensation, source energy
distribution and physical apertures. First calculate the particle beam with
Preview or High accuracy. The coherent stage consumes the accepted particle
population and its captured instrument settings; a change of optics requires
a compatible particle result. New finite-source physical trials require at
least 1000 original positive-current source particles. Increased particle count
and a smaller integration step test convergence without retuning the optics.

The handoff retains particle identities, absolute weights, energies and
interceptions. The local mixed-state lift and phase convention are explicit
model assumptions. Particle positions and flight times do not uniquely recover
a quantum phase originating at the Tip. A local mixed-coherence result must
identify its handoff plane and assumptions rather than claim an executed
Tip-to-specimen phase chain.

The boundary is the specimen upper face. Its coordinates are taken from an
exact float64 particle checkpoint, never a display interpolation. If that
plane is absent, the original upstream particles continue to it with their
actual energies, flight times and physical interception rules. New particles
are not emitted. In Finite-Field Optics the captured vector potential converts
mechanical slopes into canonical frequencies using the electron charge.

The default local model is a covariance-matched minimum-uncertainty Gaussian
mixture with a fitted common angular phase and a 10 pm per-axis source-image
translation RMS. The translation width is a declared model assumption, not a
measured specimen coherence length. Original particle energies are retained;
stationary wave propagation uses their current-weighted mean wavelength and
rejects energy spans greater than 2 eV.

Within one coherent mode, complex amplitudes interfere. Independent source
modes, energies and material histories contribute intensities. The incoherent
mixture has no single phase. Specimen propagation, diffraction and BF/ADF
readout keep the mode weights and physical losses; lost current is not restored
by normalization.

For raster acquisition, choose **Mixed coherent CUDA** on **Scanning Image**.
It consumes the same local-source coherence settings and the selected original
particle population. The independent scan worker can prepare compatible
particles without enabling the retired Tip-wave STEM flag. Its qualified local
material/field split, ideal BF/ADF windows, single-position exit diffraction and
raw export are documented in [mixed coherent STEM](MIXED_STEM_SCAN.md). These
ideal angular windows do not establish downstream detector conjugacy.

For propagation, the retained density factors are expressed in an equivalent
nonorthogonal component basis. This unitary change preserves the complete
mixed density, intensity and current; no component is deleted. The original
density eigenvalues separately define purity and coherent fraction. A phase
image belongs to the selected mathematical propagation component, not to a
unique phase of the complete electron beam.

The default material result is the coherent zero-loss branch, with physical
inelastic removal recorded. A plane inside the specimen truncates only the
executed final slice; it does not move the original specimen faces or lattice.
Inserted downstream recording devices absorb their intercepted current even
when their readout is not selected.

The current command-line interface is described by:

```powershell
.\.venv\Scripts\python.exe -m temsim.mixed_wave_cli --help
```

Use the handoff and numerical controls reported by that interface. A numerical
grid width is a representation budget, not a fitted probe diameter. New
coherent work requires usable CUDA and CuPy and does not retry on CPU after a
CUDA failure. Reading completed results and classical particle calculations
remain available without CUDA.

Reuse a compatible complete particle result without regenerating its source:

```powershell
.\.venv\Scripts\python.exe -m temsim.mixed_wave_cli --result saved.temresult --plane exit --output outputs/mixed-from-result
```

Or calculate a saved input working point before the local wave stage:

```powershell
.\.venv\Scripts\python.exe -m temsim.mixed_wave_cli --profile inputs.temwp --particles 16384 --quality "High accuracy" --plane exit --output outputs/mixed-from-profile
```

`--quality Preview` retains the explicitly requested original population.
The saved particle step is retained unless `--step-mm` is supplied. `--plane
upper` observes the handoff itself; `--z-mm` selects an exact downstream plane.
The specimen insertion, thickness, material and all optical strengths come
from the saved inputs. Neither command inserts Si, refocuses the probe or
removes active fields automatically. Dynamic downstream configurations that
the local stationary model cannot support fail explicitly.

Each new output directory contains `inputs.json`, `report.json` and
`mixed_wave_arrays.npz`. The NPZ stores the retained handoff identities,
weights, positions, slopes, energies and clocks; the original gun identity,
weight and interception ledger; and each mode's complex envelope, coordinate
and carrier metadata, intensity and complete-phase canonical angular spectrum.
Intensities are absolute probabilities relative to the original emitted
current. Per-mode spectra use their own angular coordinates and are not added
by array index when their energy or grid differs. The report records local
coherence assumptions and physical versus numerical losses. No pickle loading
is used, and an existing output path is rejected.

The former `temsim.hybrid_cli`, `scripts/trace_tip_wave.py` propagation commands
and physical Tip-wave constructors reject execution before CUDA checks, cache
reuse or physical preparation. Configuration-only inspection and historical
array readers remain available. Their retained request types and numerical
operators support archive compatibility and mathematical verification; they
do not reopen the retired workflow.
