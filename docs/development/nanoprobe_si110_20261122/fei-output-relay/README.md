# Nanoprobe Normal output-relay checkpoint

The saved operating point uses a +50 V gun lens, C1 at −34 A, C2 at +9 A, the 100 µm C2 aperture and weak C3 at +6.472 A. The final stored supplies are TL12 at −320.006 A, Mini Condenser at +226.70154583 A and both series objective packs at −1.0000 A. All positions use the physical gun tip as the origin.

| Measurement at the time of the checkpoint | Result |
|---|---:|
| Starting AC scan pivot from signed 1 µrad commands | 1423.043 mm |
| End of lower scan winding | 1420.000 mm |
| Isolated Mini–objective front at the initial objective current | 1479.643 mm |
| Isolated Mini–objective front at the stored objective current | 1480.449 mm |
| Front measured in the entire energized column at the stored currents | 1479.941 mm |
| Physical specimen upper surface | 1599.200 mm |

At the time recorded in the checkpoint the final million-primary through-focus and signed-pivot measurements were still in progress. The completed intermediate −319.1 A TL12 case with the original objective used 524,288 physical tip primaries. It placed the radial waist at 1422.811 mm with a 1422.909 mm pivot and a 74.98 µm specimen D95. [Recorded intermediate trajectories and landings](tl12/result.json)

The [timestamped checkpoint](checkpoint.json) records all winding excitations, material field checksums, completed measurements and replay commands. The [field archive](field-replay/manifest.json) contains the native final material solution, the gun electrostatic potential, and the static and signed scan tracking tables. Run [restore_checkpoint.py](restore_checkpoint.py) with the project Python to reinstall the fields at their recorded paths. The native volume was reconstructed and verified byte for byte.

The preceding [gun and condenser study](../fei-condenser/README.md) supplies source and hardware definitions. The final trial uses the same physical model and source distribution.
