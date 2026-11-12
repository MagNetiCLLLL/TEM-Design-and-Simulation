# Hybrid development validation — 9 November 2026

> **Retired Tip-origin execution.** The Tip-to-observation calculation described below
> is retained as historical documentation and evidence, not an executable workflow.
> New work uses [particle transport and near-specimen mixed coherence](MIXED_COHERENCE.md).
> Existing inputs, numeric results and archives remain unchanged and readable;
> Tip-origin propagation and continuation are rejected before CUDA or cache reuse.

Status: **executed ideal-source development workflow**, including real Si
material propagation, local diffraction, weighted exit sampling and the
existing particle integrator/physical camera receiver. This is not physical
FEG, full-column, experimental or OEM qualification.

CUDA was rechecked after reboot with an actual CuPy reduction on the NVIDIA
GeForce RTX 5090. Both complete trials used CUDA coherent operations and CPU
classical trajectories. No CPU coherent fallback was used.

## Inputs and continuation

- Explicit ideal single mode, 300 keV, 0.2 nm intensity RMS per transverse axis.
- Mode weight 0.4 at a 1 nA reference; 1024 original positive-current samples
  of the same Gaussian Wigner source, seed 11. All 1024 survived the short
  field-free source-reference trace; effective original count was 1024.
- One nanometre of executed semiclassical propagation before the specimen.
  The saved 512-pixel entrance was restored with checksum and source/field
  identity checks, preserving its reduced-action and caustic receipts.
- Actual `tests/fixtures/cif/Si.cif`, 0.4 nm thickness, two 2-angstrom slices,
  frozen phonons disabled. The material solver enlarged its computational
  domain to 2112 x 2112 pixels over 16.32 nm.
- Gaussian exit-analysis width 0.5 nm, nominal angular RMS broadening
  0.313336119 mrad. Every analysis window and Fourier bin was retained.
- Camera at sample-centre Z + 1 micrometre in the declared field-free
  development layout. Other receivers were withdrawn for this trial.
- Specimen feedback onto the externally imposed fields was omitted.

The earlier specimen-exit cache was rejected by the continuation check because
the global solver implementation identity changed during development. Its
results were retained. The material stage was re-executed under the current
solver identity, using the same executed entrance and physical inputs. The
8192-particle run successfully restored this new specimen exit without repeating
the material calculation.

The solver source identity for these completed runs was:

```text
9daba99bcc428144c071525b24333ae1697bcf7f337b625962ba9c3914ea2c96
```

## Executed results

| Quantity | 4096 exit particles | 8192 exit particles |
| --- | ---: | ---: |
| Original positive-current source particles | 1024 | 1024 |
| Particles received by camera | 4096 | 8192 |
| Specimen exit weight per reference electron | 0.39999522059257436 | 0.39999522059257436 |
| Detector current | 0.3999952205925743 nA | 0.3999952205925743 nA |
| Downstream probability-balance residual | 0 | 0 |
| Maximum diffraction/sampling weight discrepancy | 2.22e-16 | 2.22e-16 |
| Energy discrepancy in the field-free drift | 0 eV | 0 eV |
| Maximum radial cumulative-distribution discrepancy | 0.00392547 | 0.00254426 |

The cumulative-distribution comparison uses the executed **local-window
transport spectrum**, which is the distribution actually sampled. It does not
assert equality with the unwindowed raw diffraction. The predefined audit
threshold was 0.05; weights were checked to 1e-11 absolute probability, drift
positions to 1e-12 m and energy to 1e-6 eV. Actual drift discrepancies were below
1.4e-23 m in this exact field-free case; this is a software comparison, not a
physical accuracy claim.

Among surviving intensity, the fraction above 3 mrad was:

| Distribution | Fraction |
| --- | ---: |
| Incident ideal Gaussian analytical distribution | 0.000653280 |
| Real Si unwindowed diffraction | 0.002518127 |
| Local-window transport spectrum | 0.003664498 |
| 4096 sampled particles | 0.002441406 |
| 8192 sampled particles | 0.003784180 |

This verifies that real specimen scattering changed the diffraction and that
finite-sample tails need more particles than the bright central intensity.
The two particle counts are a bounded exit-sampling comparison, not proof of
original-source, optical-grid or full-instrument convergence.

## Current and process accounting

The executed entrance weight was `0.4000000003638321`, including the separately
reported semiclassical numerical change from the defined `0.4` input. The
initial coherent-frame reconstruction L2 error was `5.868098790313275e-09`.

Material probability accounting retained:

- Numerical bandwidth removal: `4.779771257568122e-06` per reference electron,
  approximately 11.95 ppm of incident current.
- Numerical support removal: zero.
- Physical removal in this configured elastic trial: zero apart from a
  `1.6653345369377348e-16` probability roundoff residual.
- Material balance residual: zero.

The small current reduction is therefore **not specimen absorption**. Lost
numerical bandwidth was not restored by rescaling. No second intensity weight
or zero-loss probability was applied to the sampled particles.

This custom CIF has no configured validated inelastic constants; its existing
material model reports inelastic collisions inactive. The actual trial covers
the elastic zero-loss branch. General conditional inelastic histories remain
connected through the existing material API, with their own source data and
admission requirements; they were not physically qualified by this trial.

## Focused numerical validation

Seventeen focused tests passed, including bounded CUDA mathematical operators.
The main 16-test run and the additional magnetic-field test were executed
separately. Only existing Pydantic deprecation warnings were observed.

Tests cover free propagation against an independent Fresnel FFT, Gaussian
frame reconstruction, continuous simultaneous caustics and full phase winding,
symplectic rejection, gauge invariance, mode weights/energies, CUDA admission,
original source counts, physical-window invariance under lattice refinement,
finite plate geometry, material error-budget handoff, complete absorption,
complex-wave exports, and exact separable windows on a rotated rectangular
lattice.

The additional test compares actual reduced-action transport in a uniform
1 T axial magnetic field over 1 mm at 200 keV with the paraxial solenoid's
analytic circular Gaussian solution. Initial intensity RMS is 30 nm per axis.
The initial coarse quadrature produced about `8.7e-4` maximum cell-amplitude
error, above the unchanged `3e-5` test tolerance. Increasing the represented
grid from 48 to 96 pixels, reducing frame spacing from eight to four pixels
and Gaussian frame width from eight to four pixels resolved the comparison.
The refined test passed both the cell-amplitude check and a `5e-4` complex L2
error limit. The physical source, field and interval were unchanged.

This reinforces the explicit requirement for frame convergence: a small norm
error alone does not certify a propagated phase. No general semiclassical
accuracy or arbitrary-field convergence claim follows from these examples.

## Saved artifacts and reproduction

- `outputs/hybrid-ideal-2026-11-09/resumed2/`: complete 4096-particle result.
- `outputs/hybrid-ideal-2026-11-09/resampled8192/`: complete 8192-particle result.
- Each contains `report.json`, `hybrid_arrays.npz`, `validation.json` and
  `validation.png`. Raw complex incident/exit waves, original launch arrays,
  diffraction, exit directions/weights and receiver histories are preserved.
- The corresponding `*-cache` directories contain committed intermediate
  wave checkpoints with checksum-verified arrays and dependency records.

Run a fresh ideal calculation:

```powershell
.\.venv\Scripts\python.exe scripts\validate_hybrid_workflow.py --output <new-directory>
.\.venv\Scripts\python.exe scripts\analyse_hybrid_validation.py <new-directory>
```

`--resume-incident-manifest` restores a checked ideal entrance.
`--resume-exit-manifest`, together with its original incident manifest, restores
a checked completed material exit. Stage, plane, source, fields, incident-wave
lineage and material input/implementation identities must match. A changed
implementation is not silently admitted for material continuation. The audit
script reads completed results and performs no new source/wave propagation.

For real instrument profiles use the API/CLI described in
[hybrid_particle_wave.md](hybrid_particle_wave.md). The full physical FEG and
instrument controls were not retuned or qualified here. Downstream particle
transport retains mechanical trajectories and current but does not reproduce
subsequent coherent interference.
