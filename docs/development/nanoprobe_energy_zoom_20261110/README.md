# Nanoprobe energy restoration and condenser zoom

## Zero-FWHM reference

The 300 kV startup working point was rerun with 16,384 original
positive-current particles and a 0.025 mm maximum column integration step.
All 16,384 reached the gun exit. The physical 100 µm C2 aperture admitted
5,116 particles to the specimen entrance at 1599.1999998 mm, with effective
count 5,116 and current 3.122559 µA. The specimen-surface particle D95 was
31.516 pm, the observed D95 waist was 0.125 nm upstream, and the current-weighted
particle α95 was 12.87425 mrad.

The same executed particle input supplied a near-specimen mixed-state CUDA
calculation with 19 coherent modes and 10 pm RMS modeled source-image mixing.
The reconstructed wave D95 was 130.063 pm at the upper surface. The wave then
crossed a 10 nm Si[110] specimen at one midpoint probe position on a
1536 × 1536 grid with 800 slices. The selected vacuum plane was
1599.2000148 mm, 5 nm after the specimen exit. The downstream intensity in the
central 1 nm crop changed by a relative L1 of 4.86 × 10⁻⁸ under padding-grid
refinement. The reported BF and ADF currents at that exposure were
2.727509 µA and 27.81017 nA. See the [scalar evidence](zero-energy-example.json).

![Si[110] exit intensity, diffraction and intensity at the selected downstream plane](si110-selected-plane.png)

## Transverse Beam

After a particle Preview or High accuracy run, enable **Coherent wave** in
**Ray Diagram → Beam analysis → Classical rays**. The view reads the exact
selected Z and launches the independent Tip wave solver on CUDA using the
accepted particle result's instrument settings. A completed wave switches
the display once its intensity readout is ready. Failed or obsolete wave
requests leave the particle display available. The physical Si example
above uses a separate near-specimen reconstruction.

On the startup instrument, the default Tip numerical domain was rejected
by the 0.001 rad near-tip transverse electric phase check. The measured
estimate was 0.397304 rad. With the previously documented 1000 nm endpoint
and 0.4 µm checked radius, all 144 near-tip modes executed. The bounded
full-gun check was stopped before a complete gun-exit checkpoint was produced.
The original phase tolerance remained in effect.

## Restored energy width and rejected zoom

The source was restored to 0.300 eV FWHM while retaining the original
phase-space sampling law and 100 µm C2 aperture. Fresh source traces were
executed at 2,048, 8,192 and 16,384 original positive-current particles.
At 2,048 particles, the restored baseline, 13.88 mrad, and 15.54 mrad
surveys gave modeled specimen-surface coherent D95 values of 194.77,
209.33, and 238.51 pm. The baseline particle α95 was 12.78 mrad.
Chromatic defocus was the largest fitted source of the ray spread.

A refocused 13.30 mrad candidate gave a 202.14 pm modeled specimen-surface
coherent D95 at 16,384 particles and an observed wave waist 0.113 nm
upstream. It failed the full-column crossover condition. Its broad
corrector waist moved from 1307.2700 to 1308.3637 mm, crossing the
1308.125 mm equal-distance plane between TL21 and TL12. The remaining
seven resolved column crossovers retained their nearest lens and side.
See the [rejected crossover comparison](rejected-13p30-topology.json).

The resolved gun crossover moved from 44.8155 to 44.8602 mm, remaining
downstream of the electrostatic Gun Lens. The near-emitter covariance
event in the initial 10 nm remains unresolved as a finite two-dimensional
waist. See the [gun comparison](gun-topology-comparison.json).

## Selected 0.300 eV working point

The final adjustment reduced C3 excitation to 45.5113165468% and refocused
with the Upper Objective at 29.3263075550%. C2 is 10.1936923650%. The wider
of the two passing crossover margins was selected. The specimen-plane
modeled D95 difference between those alternatives was 0.053 pm at 16,384
particles.

| Case | Original particles | Particle α95 | Particle D95 at surface | Modeled wave D95 at surface | Wave waist from surface |
| --- | ---: | ---: | ---: | ---: | ---: |
| Restored width, original optics | 8,192 | 12.8511 mrad | 226.715 pm | 197.651 pm | −0.2500 nm |
| Selected optics | 8,192 | 13.3009 mrad | 233.652 pm | 205.574 pm | −0.1875 nm |
| Selected optics | 16,384 | 13.3047 mrad | 229.356 pm | 202.231 pm | −0.1250 nm |

At 16,384 particles, 5,192 original particles reached the specimen entrance,
with effective count 5,192, current 3.168945 µA, and RMS energy spread
0.127756 eV. The sampled particle RMS waist was at the surface and the
particle D95 minimum was 1.000 nm upstream in both selected trials. The
wave waist bracket at 16,384 was [−0.1875, −0.0625] nm. Relative to the
16,384-particle value, the count refinement changed the modeled surface
wave D95 by 1.65%. Both wave calculations passed the 1152-to-1536 lateral
grid and ±0.25 nm axial placement checks.

The independently traced signed geometric pupil gave an A2 magnitude of
7.31 nm at the selected settings, compared with 56.15 nm in the restored-width
baseline. The CUDA reconstruction retains chromatic displacement from the
energy-dependent particle transport and uses a single current-weighted mean
wavelength. Its phase model extends continuously beyond the fitted pupil
support. The reported D95 and wave focus refer to that model. A sensitivity
run with additional signed A1, C3, C5 and A5 phase gave 202.025 pm and a
waist 0.1875 nm upstream, with the full bracket within the same placement
tolerance.

All eight resolved column crossovers retained their nearest lens and side in
the actual, incident and downstream cohorts through the camera plane. The
critical waist was at 1307.8190 mm, 0.3060 mm on the TL21 side of the
1308.125 mm equal-distance plane. The installed Normal Nanoprobe branch and
direct specimen focus passed the executed topology comparison.
See the [column comparison](column-topology-comparison.json) and
[numerical evidence](restored-energy-focus.json).

The selected [startup working point](../../../profiles/default-nanoprobe.temwp)
was exported from the executed 16,384-particle input snapshot and reloaded
with matching source, field, geometry and optical controls. The
[working-point receipt](working-point-receipt.json) records the exact
identities. The preceding [zero-FWHM working point](../../../profiles/feg-nanoprobe-300kv-zero-fwhm-20261110.temwp)
is available separately.
