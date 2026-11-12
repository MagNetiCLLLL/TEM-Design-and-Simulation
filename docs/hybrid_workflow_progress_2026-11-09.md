# Hybrid workflow progress — physical FEG nanoprobe round

> **Retired Tip-origin execution.** The Tip-to-observation calculation described below
> is retained as historical documentation and evidence, not an executable workflow.
> New work uses [particle transport and near-specimen mixed coherence](MIXED_COHERENCE.md).
> Existing inputs, numeric results and archives remain unchanged and readable;
> Tip-origin propagation and continuation are rejected before CUDA or cache reuse.

Final round savepoint, 2026-11-10 00:02 UTC: fresh physical FEG trials with 2048
and 4096 original positive-current particles completed the incident column.
The final 4096-particle specimen D95 is 9.18497777 nm, current 2.71972656 uA,
and alpha95 7.82919815 mrad. The C2 aperture remains 100 um and all original
current weights/interception are retained. Current mechanical/field axes are
unchanged; the explicitly tuned scalar controls include HPC normal/skew and
Objective focus. The native D95 and RMS waist offsets and finite-source
uncertainty remain separate from geometric topology qualification.

The three result images, input profiles, original source arrays, specimen
arrays and numerical comparisons are saved in
[the Si[110] round archive](development/si110_hybrid_20261109/README.md).
The 20 nm tip coherence length is an explicitly unmeasured Gaussian-Schell
assumption. All 48 CUDA gun modes reached Z = 450 mm with 9.979200252 uA.
The original large wave cache remains under
`outputs/si110-global-hybrid-2026-11-09/global-hybrid4096/wave_cache`.

The full global coherent Si[110] calculation has not yet completed. Spatial
support compaction passed the earlier computational bottleneck, but the
order-4 attempt correctly rejected the kinetic phase bound near C1. The
order-8 attempt then stopped on an envelope sampling/device-budget rejection
near C1. Its last progress message was Z = 504.406018 mm. The bounded refinement
needed 25.6454 GiB of estimated device work against the configured 24 GiB
budget; the initially requested grid needed 35.4375 GiB. The rejected operator
was not applied. The complete gun mixture at Z = 450 mm remains the reusable
checkpoint; partial progress inside the column is not a restart plane.
No task-owned calculation remains running. There is no accepted physical-FEG
specimen/detector result. No
independent local-cell substitute, restored aperture current or relaxed error
threshold is being used.

The user requested completion of this calculation-optimization round, direct
Git publication, saved result pictures and then forced shutdown. The final
attempt record and result pictures are now saved for that direct publication.
No broad test suite is being rerun solely for upload. The
historical pause and ideal-mode sections below do not describe this current
physical-source calculation or its publication status.

## Historical ideal-source continuation

At 21:33 UTC, the ideal single-mode -> real Si -> weighted exit particles ->
camera workflow completed with both 4096 and 8192 exit particles. Seventeen
focused tests passed. See [the ideal-source validation record](hybrid_validation_2026-11-09.md)
for evidence and limits. That result is not a physical FEG/full-column result.

## Historical pause snapshot

Saved at 2026-11-09 21:09 UTC at the user's explicit request to stop calculation
and preserve progress before reboot. **Do not resume automatically.** Wait for
the user to request continuation. The implementation is not yet fully validated
through detector output. No Git commit or push was requested or performed.

Workspace: `F:\tem_simulator_v2`; branch `master`; current HEAD
`9554a3ade6b820437fe4329c5427a7361314f0fb`.

## Requested calculation and agreed validation source

Use the existing `reduced_action_m` to combine particle and wave calculations:
declared coherent modes, amplitude transport, continuous caustic phase,
real-specimen diffraction, weighted outgoing particles and detector tracing.
Ignore specimen feedback onto the externally imposed fields. The user selected
an explicitly labelled ideal single-mode source for development validation.
Do not change the default FEG source to make it coherent implicitly.

Follow `agent.md`: project files in English, user communication in Chinese;
new coherent runs require verified usable CUDA, without CPU fallback. Each
finite-source physical trial must have at least 1000 original positive-current
particles; use 1024. Small mathematical/operator and mocked tests are distinct.
Do not count Gaussian quadrature packets, tangent probes or exit samples as
original source particles. Preserve absolute weights and physical losses.

## Files added in this task

- `src/temsim/physics/hybrid_semiclassical.py`: Gaussian coherent-state frame,
  canonical tangent rays, common-reference reduced action, Herman--Kluk
  amplitude and continuously lifted caustic prefactor.
- `src/temsim/physics/hybrid_exit.py`: gauge-covariant local Fourier windows,
  streamed systematic sampling, independent mode weights and mechanical
  outgoing momentum. Spatial-window width can be specified in physical units.
- `src/temsim/physics/hybrid_pipeline.py`: physical gun API, coherent column,
  existing material propagation, physical downstream particle/receiver path,
  source-count and CUDA admission, geometry and probability records.
- `src/temsim/hybrid_cli.py`: profile-based command-line entry and complete
  complex-wave, diffraction, particle and accounting exports.
- `scripts/validate_hybrid_workflow.py`: independent ideal-source CUDA
  development trial, including explicit checked entrance continuation.
- `tests/test_hybrid_semiclassical.py`, `tests/test_hybrid_exit.py`,
  `tests/test_hybrid_pipeline.py`: focused mathematical and integration-boundary
  regressions.
- `docs/hybrid_particle_wave.md`: equations, usage, scope and references.

All files are saved locally. These are new untracked files. The repository
already contained many unrelated modified and untracked corrector, coordinate,
FEG, assembly-error, GUI, documentation and test files before this task. Do not
reset, overwrite or stage that work indiscriminately.

## Completed verification

Actual CuPy CUDA execution was verified on the RTX 5090. The final focused run
completed successfully before the pause: **15 tests passed** with only the
existing Pydantic deprecation warnings:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_semiclassical.py tests/test_hybrid_exit.py tests/test_hybrid_pipeline.py -q
```

Coverage includes independent free-space Fresnel comparison on CPU and CUDA,
simultaneous transverse caustics and complete winding, noncanonical-map
rejection, coherent-frame reconstruction, gauge invariance, independent mode
weights and energies, CUDA rejection before expensive work, original-particle
identity/count, physical-window invariance under lattice refinement, resolved
plate faces, material error-budget binding, complete absorption, complex-wave
export and protection against overwriting results. These are small operators
or mocked boundaries, not full-source instrument qualification.

Imports, command-line help, syntax and whitespace checks passed for the nine
implementation/test/method files. This progress note was written afterward.
No broad project or coherent regression suite was run.

## Actual development calculation

The benchmark uses a declared ideal 300 keV single Gaussian mode, intensity RMS
0.2 nm per transverse axis, mode weight 0.4 at a 1 nA reference current, and seed
11. It traces 1024 positive-current original samples of the same Gaussian Wigner
source independently of the coherent quadrature. It uses a 1 nm field-free
semiclassical interval before the actual Si CIF fixture, specimen thickness
0.4 nm, no frozen phonons, and the existing zero-loss material calculation.
It is **not** a physical FEG phase chain or a full microscope working point.

The 256-pixel attempt was correctly rejected by the material carrier-boundary
gate. The numerical input was then increased to 512 pixels over 4 nm; no source
size or physical field was retuned and no error threshold was relaxed.

Executed entrance results:

- Plane: `1599.1999998000001 mm`.
- Grid: `512 x 512`, physical width 4 nm.
- Coherent-state packet count: `91955` (not original electrons).
- Initial complex reconstruction L2 error: `5.868098790313275e-09`.
- Input weight: `0.4`; executed incident weight: `0.4000000003638321`.
- The small positive numerical change is recorded, not normalized away.

The real material wave calculation completed and was committed to the private
checkpoint store before the pause:

- Plane: `1599.2000002 mm`.
- Mode ID: `declared-ideal-single-mode/phonon:0`.
- Exit weight: `0.39999522059257436`.
- Grid: `2112 x 2112`, spacing about `0.00772727 nm`, physical width `16.32 nm`.
  This is automatic domain/sampling refinement, not a changed specimen/probe.
- Material branch, numerical-support and current-loss records are in the
  checkpoint manifest. Do not label the entire weight difference as absorption
  without inspecting those records.

**The last run was interrupted during local diffraction sampling.** No complete
exit-particle result, downstream detector result or final `report.json` exists
for this trial. Do not report end-to-end success or detector-current validation.
The original source arrays had not reached final export; their exact generation
law, count, seed and settings are retained in the script and cache dependencies.

## Durable checkpoints

The store manifests and arrays are on disk, not dependent on Python process
memory. Use `ExecutedWaveStore` with the saved dependency and key to verify
checksums when restoring; do not use pickle or bypass source identities.

Completed entrance from run 3:

```text
outputs/hybrid-ideal-2026-11-09/run3-cache/673ce01d86d2a783410ad7d9c5962b24fa3e285cd0b04190b7f13e8168208789-e6bcfe9248df4fa58d2f0f9e8422cbd8/manifest.json
```

Run 4 continued that exact verified entrance. Its local entrance copy is:

```text
outputs/hybrid-ideal-2026-11-09/run4-cache/72f057928513e2fc0e0d5e848e2cb6e7dd67d124bef8daeff19aa818e904f0f3-6b72bff34cab4b8297aa9db2e713120a/manifest.json
```

Most advanced checkpoint, **completed real Si specimen exit**:

```text
outputs/hybrid-ideal-2026-11-09/run4-cache/d42143c220117711de4e0e5c762354b6605b1a23ec434ab800f1ea97d05ee618-28d7aaeef5bc414c9b1089e3376090b5/manifest.json
```

The existing script supports entrance continuation, which avoids repeating the
expensive semiclassical part. After the user asks to continue, an available
command is:

```powershell
.\.venv\Scripts\python.exe scripts\validate_hybrid_workflow.py --output outputs/hybrid-ideal-2026-11-09/resumed1 --resume-incident-manifest "outputs/hybrid-ideal-2026-11-09/run3-cache/673ce01d86d2a783410ad7d9c5962b24fa3e285cd0b04190b7f13e8168208789-e6bcfe9248df4fa58d2f0f9e8422cbd8/manifest.json"
```

Choose a new output directory. This regenerates/traces the same 1024 ideal
source reference samples, reuses the entrance and reruns material transport.
For the fastest continuation, add a strictly checked specimen-exit continuation
option using the completed exit above; that option has **not** been implemented.

## Remaining work after explicit resume

1. Verify CUDA again after reboot. Read this note and current Git state.
2. Restore the completed specimen exit with its identity and material history.
   Finish local diffraction, weighted particle generation and actual detector
   tracing. Inspect absolute probability/current balance and energy/mode lineage.
3. Check the resulting Si diffraction against the incident distribution and
   the particle angular histogram. Keep finite-window broadening explicit.
4. Verify final exports include both complex waves, original source arrays,
   modes, material histories, diffraction, particles and receiver accounting.
5. Record numerical evidence in the method document and report the actual
   validation limits. Do not claim real-FEG/full-column or convergence
   qualification based on the ideal short-column example.

Performance observations for continuation:

- Large material domains make dense Gaussian-window work expensive. The
  2112-pixel output is 16.32 nm wide, so a 0.5 nm window entails many windows.
- Production exit sampling already streams windows and keeps the physical
  width fixed during lattice refinement. It does not allocate the complete
  four-dimensional joint distribution.
- A mathematically exact fast path for an explicitly empty field inventory
  and zero polynomial gauge has just been added to `_line_phase`. It returns
  zero without sampling the empty field. The interrupted process had imported
  the preceding, slower equivalent implementation. The final 15 tests cover
  the saved code.
- Initial coherent-frame reconstruction now uses the exact inverse-FFT window
  synthesis instead of a dense packet-by-pixel sum. The executed entrance was
  computed with the earlier equivalent dense implementation; the saved tests
  compare the updated operator with direct reconstruction/Fresnel propagation.
- A potential next optimization, **not implemented**, is exact separable
  Gaussian evaluation on orthogonal lattices. Gaussian windows factor into X/Y
  vectors, and coherent packet synthesis can be expressed as a matrix product
  of X and Y factors. This could remove billions of redundant exponentials
  without deleting tails or changing weights. Verify equivalence before use.

## Pause state

The task-owned validation Python processes were stopped. The final pytest run
had already exited successfully. Process inspection confirmed no remaining
`validate_hybrid_workflow.py` or hybrid-test process. Partial runs and completed
checkpoints were retained. There is no resumable in-memory Python/GPU session.
