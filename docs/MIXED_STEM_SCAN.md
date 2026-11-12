# Mixed coherent STEM scanning

The Scanning Image page can acquire BF and ADF images from the executed particle
beam at the specimen entrance. The particle calculation retains the selected
source distribution, lens and corrector excitations, aperture interception and
integration step. A mixed coherent density is constructed at the specimen upper
surface, then propagated through the inserted specimen on CUDA. No coherent
Tip-to-specimen calculation is involved.

## Operation

1. Select **Finite-Field Optics**, insert an atomic specimen and choose its CIF,
   orientation, thickness and position. Set at least 1000 original
   positive-current source particles; the normal selected particle count is
   retained in both Preview and High accuracy.
2. Set the raster dimensions, pixel pitch and scan origin on the Scanning Image
   page. Select the mixed coherent CUDA acquisition and start the calculation.
3. Read BF and ADF as absolute currents referenced to the original emitted beam.
   The exit diffraction panel shows one actual raster position nearest the scan
   origin. Its reported coordinate is the selected pixel coordinate, including
   any half-pixel offset of an even raster.

A usable CUDA device and execution backend are required before source
preparation. An unavailable device, failed CUDA operation, cancellation or
failed numerical qualification produces no replacement image. There is no CPU
coherent fallback. Classical particle transport may use CPU.

The current analytic finite-CIF reference requires a proved rectangular repeat
aligned with the laboratory axes and quarter-period transverse sites, including
this Si[110] case. After the CUDA check, material preflight rejects unsupported
CIF geometry before new classical transport or coherent source construction.
Arbitrary atomic positions need a separately qualified direct or nonuniform
Fourier structure-factor implementation; the workflow does not silently
substitute the former radial-core discretization.

The command-line entry uses the same acquisition workflow:

```powershell
.venv\Scripts\python.exe -m temsim.mixed_stem_cli --profile working-point.temwp --particles 16384 --output outputs\new-mixed-stem
.venv\Scripts\python.exe -m temsim.mixed_stem_cli --result executed-particles.temresult --output outputs\new-mixed-stem-from-result
```

An executed result is reused only when its captured input snapshot matches the
requested state. The exact specimen entrance is reconstructed from the saved
particle restart data. Display-decimated particle arrays are not accepted as an
exact entrance factory. Existing output files are never overwritten.

## Reusing a completed classical boundary

An explicitly exported, immutable particle boundary can be reused to refine
the wave calculation without repeating the already completed classical
transport. The public interchange API is
`temsim.physics.particle_wave_source_io.load_particle_wave_source(directory)`.
It returns a `ValidatedParticleBoundary` with the source, complete captured
inputs, full original gun ledger, emission reference and original producer
metadata. Reading these classical arrays does not require CUDA.

```python
from temsim.physics.particle_wave_source_io import load_particle_wave_source
from temsim.physics.mixed_stem_scan import MixedStemScanRequest, simulate_mixed_stem_scan

boundary = load_particle_wave_source("validated-particle-boundary")
state = boundary.restore_inputs_for_wave()
result = simulate_mixed_stem_scan(
    state, boundary.source, MixedStemScanRequest(), validated_boundary=boundary,
)
```

This is an explicit transfer of completed classical phase space to a later
wave solver. It does not restart old Lorentz checkpoints or recertify the
particles as freshly transported by the new solver implementation. The old
source input digest and generation implementation remain unchanged; the wave
record identifies its own implementation separately. The current state must
match the captured input graph and external inventory. Optical or source-law
changes require new particle transport; a change to wave sampling can reuse
the unchanged classical boundary.

The six-file interchange contains `inputs.json`, `source.json`,
`source-phase.npz`, `original-gun.npz`, `emission-reference.npz` and
`checksums.json`. The loader verifies file and source digests, the complete
input snapshot, producer provenance, original positive-current count, unique
emitted IDs and ancestry checksum, exact handoff membership and weights,
survival, actual clocks, energies and optional canonical-frequency shifts.
It rejects object arrays and bounds both compressed and expanded input sizes.
`save_particle_wave_source(boundary, new_directory)` writes another checked
copy without overwriting an existing directory. Neither API renormalizes
current or restores clipped particles. Every new coherent calculation still
enters through the CUDA policy and its numerical qualification gates.

## Physical meaning and limits

### Finite atomic-potential discretization

Material-grid convergence must include the atomic-potential representation,
not only the propagation step. The original abTEM finite radial-table path
uses a grid-dependent inner radial sample. A nucleus on a pixel and the same
nucleus between pixels can therefore give different represented Fourier
coefficients. The retained failed acquisitions and independent static
translation checks distinguish this effect from source, lens or tolerance
changes. A static potential comparison alone does not quantify a complete
BF/ADF observable error.

The finite Lobato Fourier representation integrates the same continuous
potential, with abTEM's scaled coefficients:

\[
 V(r)=\sum_i a_i\left(\frac{2}{\mu_i r}+1\right)e^{-\mu_i r}.
\]

Use cycles per angstrom for lateral frequency \(q\),
\(\beta_i=\sqrt{\mu_i^2+(2\pi|q|)^2}\), and \(u_i=\beta_i|z|\).
An odd primitive of the lateral Fourier transform is

\[
 S_i(q,z)=\operatorname{sgn}(z)2\pi a_i
 \left[\frac{2(1-e^{-u_i})}{\mu_i\beta_i^2}
 +\frac{\mu_i\{2-(u_i+2)e^{-u_i}\}}{\beta_i^4}\right].
\]

For a real atomic position \((x_j,y_j,z_j)\), a finite slice uses
\(S_i(q,z_{s+1}-z_j)-S_i(q,z_s-z_j)\), multiplied by
\(e^{-2\pi i q\cdot(x_j,y_j)}\). Summing the actual selected atoms and dividing
by the lateral area gives Fourier-series coefficients in volt angstrom.
No infinite-thickness projection is substituted for a finite slice.
`expm1` and stable interval evaluation avoid subtracting nearly equal
exponentials. The infinite-depth limit is
\(8\pi a_i[1/(\mu_i\beta_i^2)+\mu_i/\beta_i^4]\), providing an independent
normalization check against the projected scattering factor.

Finite-reference construction retains physical atomic-site clipping and
screened-potential padding before cropping the observation region. A crystal
repeat accelerates a proven periodic operator; it must not fill missing sites
at a finite specimen edge. Material/source currents, propagation bandwidth,
field errors and the existing combined acceptance budgets remain separate
checks. The finite slice representation and compact periodic storage do not
turn a completed but unqualified scan into an accepted image.

The raster translates the executed entrance density relative to the crystal.
It does not retrace scan-coil drive waveforms. Default BF and ADF are ideal exit
Fourier annuli of 0–15 mrad and 50–150 mrad. They are not currents collected by the
instrument's downstream, millimetre-sized detectors: that requires the
appropriate downstream lens and detector-conjugacy calculation.

The source lift retains the original particle IDs, weights, positions,
mechanical slopes, energies, current and interception losses. Its canonical
frequency includes the magnetic vector-potential correction. The selected
source-image coherence parameter belongs to the local mixed-state model; it is
not a retuning of the optical lenses. Retained coherent mode weights are not
rescaled to restore discarded current.

The material calculation is explicitly a finite periodic-domain model. It uses
an exact forward free-space angular-spectrum operator, static elastic
multislice potential and a qualified symmetric split for the local near-uniform
axial field. It is not the complete general column wave kernel or an isolated
probe boundary solver. Exact Bloch reduction proves equivalence to the same
finite-grid periodic operator, not convergence with respect to physical field
of view.

Detector bandwidth coverage, material sampling, slice refinement, field
splitting and independent resident/operator comparisons are qualified before
images are published. An independently expanded domain is compared at held-out
positions that include the four actual raster corners and selected diffraction
pixel. The domain change contributes to the same combined 5% budget relative
to BF/ADF modulation. The raw selected diffraction must separately meet its
combined 5% intensity-L1 budget on a common reciprocal readout grid; neither
pattern is conditionally normalized to hide current changes. Two-cell boundary
diagnostics report their sampled modes, positions and depths. Held-out checks
are not a strict boundary-error certificate for every raster pixel or a claim
of isolated-probe field-of-view convergence.

The source and general column solvers retain their existing 1e-9 boundary
gates. The finite periodic-domain result does not claim to execute those
general boundary solvers through the specimen. The recorded field-omission
estimate is a sampled/refined bound, not a continuum interval certificate. No
budget is relaxed when a candidate fails. A 512-pixel source representation is
not assumed to resolve a 150 mrad material-exit annulus; the material grid is
independently planned and qualified.

Actual per-particle energies, including the configured energy spread, affect
the particle trajectory and source ancestry. Material propagation currently
uses one current-weighted stationary mean wavelength. It does not integrate
separate energy modes. The report includes the configured FWHM, actual energy
range and free-phase/angular-axis sensitivity estimates; these estimates do
not certify the full material energy-mode approximation. An actual source
energy span above 2 eV is rejected. Selected frozen-phonon, finite absorption or
inelastic, separate high-angle-tail and phase-object alternatives are rejected
when the implemented elastic multislice path cannot represent them. Missing
inelastic material data must not be interpreted as a prediction of no real
inelastic scattering.

BF, ADF, uncollected exit intensity, upstream interception and source/numerical
loss channels use the same original-emission denominator. Absolute currents
are the corresponding fractions times the original reference current. No
post-loss normalization is applied.

## Saved data

The acquisition exporter writes non-object NumPy arrays and JSON metadata:

| File | Contents |
| --- | --- |
| `scan-and-diffraction.npz` | Raster coordinates in nm, detector fractions and currents in A, loss ledgers, one exit diffraction probability array and reciprocal axes in inverse angstroms |
| `entrance-source-arrays.npz` | Original entrance IDs, weights, coordinates, slopes, energies, timing and canonical-frequency data when present |
| `original-gun.npz` | Original gun bundle and interception data |
| `inputs.json` | Captured instrument inputs for this calculation |
| `source-checkpoint.json` | Source construction and coherence/sampling record |
| `report.json` | Numerical qualification, physical scope, energy approximation, detectors, selected diffraction coordinate and provenance |
| `checksums.json` | SHA-256 hashes of the saved acquisition files |

Main array names are `scan_x_nm`, `scan_y_nm`, `fraction_bf`, `fraction_adf`,
`current_a_bf`, `current_a_adf`, `diffraction_probability`, `qx_inv_angstrom` and
`qy_inv_angstrom`. Detector maps follow the two-dimensional raster shape.
Diffraction is probability per reciprocal pixel per original emitted electron.
It is one unscaled exit pattern, not a scan average, normalized display pattern
or a 4D-STEM cube. Logarithmic display scaling changes neither the saved array
nor the reported probability ledger.

## Detached 10 nm Si[110] example

The existing **Physical layout → model editor → sample → Place…** operation can
move the foil along the column axis. Turn **Include children** off, choose the
displayed local or global coordinate frame, enter the target Z, and review the
staged geometry before saving. This places the sample part only. The existing
stage, holder, pole pieces, objective yokes, field references and EDS marker
remain at their fixed absolute coordinates. A historical stage anchor that
followed the sample is retained at its previous fixed origin in the edited
candidate. The foil's mechanical envelope must remain inside the fixed holder
tip, stage and pole gap. Objective symmetry is measured about the fixed
pole-gap centre, not the movable foil.

Changing thickness now automatically preserves the saved upper surface and
extends or retracts only the downstream face. Both the Sample page and generic
parameter editor use the validated transaction
`temsim.specimen.axial_geometry.set_sample_thickness(state, thickness_nm)`.
Normal layout refreshes persist the corresponding sample-part position in a
private input archive, so a later TOML refresh or state restore cannot move it
back. Invalid dimensions or a slab outside the holder/stage/pole gap are
rejected atomically, without clipping thickness or moving optical elements.

The centre moves by `0.5 × (new thickness − old thickness) × 1e-6 mm`, with
thicknesses in nm. Thus 0.4 nm to 10 nm automatically moves the centre by 4.8 nm.
Direct scripted thickness assignment also preserves the reference; the normal
layout must still validate and synchronize the complete geometry before use.
An intentional **Place…** operation chooses a new upper-surface reference.

New captures and operating profiles explicitly store the upper-surface and
initial lattice-thickness references. An old archive without these optional
fields remains readable at its actual saved centre and thickness. Metadata
activation uses that saved upper surface and thickness, without moving it or
reinterpreting it against the current default. In particular, an already
correct 10 nm archived specimen starts with a 10 nm lattice reference, so its
previously translated CIF is not compensated twice. Old particle input/source
digests remain unchanged; explicit later wave handoff permits only this
checked metadata migration, with every other captured field unchanged.

The historical 10 nm example below manually shifted the centre by 4.8 nm and
translated/wrapped a CIF copy to retain the original entrance termination.
It preserves the original CIF and uses local thickness arithmetic rather than
a large laboratory-Z subtraction to set an atomic lattice phase.

The 2026-11-11 example retains the current default optical parameters, 0.3 eV
source FWHM, 0.025 mm particle step, 2 mm history setting and 10 pm source-image
coherence RMS. A detached state increases only the specimen thickness to 10 nm
and its centre by 4.8 nm, keeping the original upper surface fixed. The physical
orientation quaternion remains authoritative for Si[110].

The example helper in
`outputs/mixed-stem-si110-10nm-20261111/prepare_specimen.py` writes a translated,
wrapped expanded-site CIF in the output directory. It retains the unit cell,
all occupancies and the crystal termination at the entrance. The bundled CIF
and default working-point bytes remain unchanged. Its
`prepare_geometry_anchor(state)` function persists the sample position in a
detached input archive through the normal component placement, geometry
experiment and physical-layout validators. A later layout application or input
restore therefore retains the requested upper surface. The saved user default
profile is not automatically modified. Its pure-geometry receipt
compares all 461 atoms in the original 0.4 nm layer against the same entrance
layer in the 10 nm specimen; the maximum nearest-site discrepancy is
4.43e-14 angstrom, below the declared 5e-8 angstrom geometry tolerance. This
geometry check alone is not a wave or source-count convergence calculation.

The fixed-entrance backend has 20 passing static regression cases covering
increased/decreased thickness, repeated layout and save/restore, old 10 nm
activation, atomic rejection/rollback, operating-profile persistence,
ordinary catalog reloads, explicit Place, unchanged non-sample geometry and
unchanged magnetic fields at fixed physical coordinates. These software/geometry
checks do not replace CUDA wave or
particle-count convergence evidence.

`plot_scan.py` in the same output directory renders saved BF/ADF currents and
the selected exit diffraction, checks raw file hashes before and after, and
writes a plot receipt. Its synthetic layout-check fixture is explicitly
labelled as no simulation and is not scientific evidence for the acquisition.
