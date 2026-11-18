# Stationary Nanoprobe tuning with ideal vacuum coils

The resumed 30 mrad tuning, measured working point and desktop profile are
documented in the [stationary probe result](RESUMED_TUNING.md).

This study compares the parked probe with the historical 300 kV, approximately
30 mrad corrected Si[110] acquisition. The observation plane is the specimen
upper face at 1599.1999975 mm. The physical probe metrics use original particle
weights, mechanical slopes and the current-weighted chief direction. RMS is a
radius; D95 is twice the radius enclosing 95% of transmitted current about
the current centroid. The central 50 pm disk uses the fixed column axis.

## Reference measurements

| Sample entrance | Positive originals | Surviving / effective | Current (µA) | Alpha95 (mrad) | RMS radius (nm) | D95 (nm) | 50 pm disk (µA) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical corrected probe used in the joint Si[110] scan | 2000 | 2000 / 2000 | 10.000 | 29.71914 | 1.57825 | 2.44648 | 2.405 |
| Archived ideal-coil defaults, raster and Descan parked | 3000 | 3000 / 3000 | 10.000 | 0.179402 | 50.23778 | 178.44210 | 0.000 |

The second row is a reanalysis of the separately saved exact entrance
boundary. Its parent section ends at the **lower** face, 5 nm downstream.
The upper-face radial position–slope covariance is +5.029e-12 m rad.
The recorded beam is already expanding at the upper face. From the Mini
Condenser centre to the entrance, independently minimizing the two widths on
the retained axial nodes returned the same sampled position, 1542.00249 mm.
RMS there was 4.626 nm and D95 was 16.412 nm. The adjacent observations at
1540 and 1544.0024 mm do not resolve any separation between the continuous
minima. The original saved trajectories also contain other upstream waists.

The same 5 nm laterally periodic Si[110] stage scan and 20 pm raster pitch
give a pair of dark-field peaks 0.140 nm apart in the historical acquisition.
The crystal projection gives 0.13609 nm. The native line at x = +0.010 nm
has no registered pair in the archived stationary ideal-coil scan. This is
also true of the 1024 and 2048 prefixes of that same 3000-particle transport.
The 50–150 mrad dark-field and 0–15 mrad bright-field windows use ideal
specimen-exit directions in the classical elastic screen model.

[Archived coil metrics](stationary-default.json) ·
[Executed stationary scan](stationary-si110/report.json) ·
[Absolute current maps](stationary-si110/scan.npz) ·
[Native line](stationary-si110/native-profile.csv) ·
[Historical Si scan](../si110_joint_probe_scan_20261101/README.md) ·
[Historical independent RMS and D95 focus observations](../fege_focus_20261110/README.md)

## Electrical and numerical design

All candidates retain the desktop three-condenser, probe-corrected assembly,
source law, 0.3 eV energy FWHM, aperture boundaries and winding geometry.
Electrical groups divide existing complete turns inside the original
winding allocations. The Objective retains its original two 580-turn
windings. The downstream projector lenses keep their original currents and
their full upstream tails contribute to all designs.

The source-free design computes the fields of the fixed winding groups and
varies their signed currents to place the paraxial source images in the
allowed regions. The first real 1024-particle
gun found r95 = 0.670708 µm, far smaller than the 12.024 µm archival
reference used to seed the search. Its measured radius sets the target
angular gain for subsequent designs. The second independent 1024-particle
gun found r95 = 0.670750 µm. Their measured gun-exit radial slope/position
coefficients were 0.902025 and 0.902159 per metre. Every changed field recipe
requires a fresh gun because its field extends upstream.

The nonlinear design uses on-axis derivatives of the actual round-coil fields,
physical HP1, HP2, HPC and HPol normal/skew coefficients, a relativistic
mechanical ray pencil and an independently solved Objective current. The
pencil is a zero-current design operator. Tabulated fields, a finite-order
off-axis expansion and discrete test energies limit its predictions. The first
fixed-conjugate seed below was measured with the full native Lorentz fields
and the original emitted population.

## Executed current adjustments

The first fixed-conjugate seed transmitted all 1024 positive-current originals
(effective count 1024) and 10 µA. At the exact specimen entrance it gave
alpha95 = 1.673649 mrad, RMS radius = 13.172916 nm, D95 = 56.553873 nm and
0.087891 µA in the fixed-axis 50 pm disk. The independently minimized widths
between 2 mm upstream and 2 mm downstream of the entrance were:

| Quantity | Smallest observed | z from the upper surface (nm) | Neighboring observations (nm) |
| --- | ---: | ---: | --- |
| RMS radius | 13.172901 nm | +21.696 | [+21.586, +21.806] |
| D95 | 56.115126 nm | +2239.712 | [+2239.240, +2240.183] |

The signed difference z(RMS) − z(D95) was −2218.015 nm, bounded by
[−2218.597, −2217.434] nm. The same native Si line gave peaks of 40.45 and
41.93 nA, separated by 0.120 nm with a 21.39 nA saddle. This single-count
result falls far below the historical angular target. It illustrates how a
central current component can produce a visible feature with a broad D95.

[Optics and focus brackets](conjugate-seed/optics.json) ·
[Original surface particles](conjugate-seed/entrance.npz) ·
[Electrical settings](conjugate-seed/currents.json) ·
[Fixed-source Si scan](conjugate-seed/si110.json)

## Reproduction

Run with the project environment from the repository root. Source-free
searches are independent of the finite physical source budget. All new
physical runs use at least 1024 original positive-current particles.

- `archive_focus.py ARCHIVE OUTPUT.json` reads checked historical observation
  arrays and the upper-face boundary saved alongside the particle section.
- `design.py OUTPUT` builds the first fixed-conjugate current seed using the
  captured 13-lens axial coefficients and integer-turn geometry.
- `zoom.py OUTPUT --gun GUN.npz --seed RECIPE.json --expand` allows intermediate
  images to move within their physical regions. `--relay-sign` adds an HP2 to
  HP1 imaging constraint. `--hp-radius-um` controls illumination in the first
  primary hexapole.
- `pencil.py RECIPE.json OUTPUT`, `aberrations.py RECIPE.json OPERATOR.npz OUTPUT`
  and `joint.py` fit physical multipole commands and round currents. Joint
  fits can include the measured original gun energy distribution and a
  central-pupil objective. Use `--help` for their numerical options.
- `trial.py OUTPUT --particles 2048 [--recipe RECIPE.json]` runs stationary
  physical transport from the gun in the selected current field. The specimen
  is retracted for vacuum through-focus observations. Each exact original
  particle retains its absolute weight, energy, physical clock and mechanical
  stopping history. Complete 25 mm blocks are checkpointed. To resume a
  completed gun for the same snapshot, add `--resume-gun GUN_DIRECTORY`.
- `scan.py SOURCE_DIRECTORY OUTPUT` uses the checked stationary entrance and
  the historical elastic Si[110] stage raster. `measure.py OUTPUT` separately
  runs the unretuned desktop instrument with its raster and Descan enabled.
- `topology.py TRIAL REPORT.json` checks the finite two-dimensional source images
  against the saved native phase planes for trials that retained them.
- `archive_trial.py TRIAL SCAN OUTPUT [--recipe RECIPE.json]` writes the native
  entrance, registered line, through-focus widths, two focus brackets and the
  current maps with their input identity and file checksums.
- `export_profile.py TRIAL PROFILE.toml` creates a desktop operating profile and
  reloads it through the profile and assembly paths to verify the winding groups,
  multipole currents, round drives, sample and scanning settings.
