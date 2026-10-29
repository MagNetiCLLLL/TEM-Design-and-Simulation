# Native local-linear column response diagnostic

Within this executed engineering-field model, the large ID306 sample-plane tail is absent from the stable local first-order prediction and appears predominantly in the full-field map minus that prediction. This is a local nonlinear remainder, not a unique spherical-aberration attribution or an additive decomposition of beam widths.

The one calibration batch executed 17 mathematical probes in 30.182299 s on the exact original 23,513-node / 479-checkpoint mesh. All survived without a stop. The static field snapshot, potential/geometry/plan identities, zero induction work, original inputs and saved full-source files passed their guards.

## Calibration and stability

The expansion reference is the actual current-weighted original poly gun-exit mean. Every probe uses K = 300000.3 eV and the mean original exit clock. The probes have no assigned beam current and are not an electron source.

Reference [x, y, tx, ty] = [-5.392479438142326e-10, 4.464022023867627e-10, -3.245360024787658e-10, 2.6339926049250556e-10] in [m, m, rad, rad]; clock = 3.12662048705e-09 s.

The perturbations are 1 nm in x/y and 10 nrad in tx/ty, plus their half sizes. Both four-coordinate central differences share the same reference, requiring 17 trajectories. Predictions use all 769 original source columns and their unchanged absolute weights. They are mathematical arrays stored separately from executed particles.

Across every source point and every requested through-focus plane, h versus h/2 changes the predicted x/y vector by at most 2.02818178549e-05 nm (predeclared limit 0.01 nm), and the slope vector by 6.52411084383e-10 rad (limit 1e-6 rad).

## Native profile comparison

| Quantity | Executed full field | Local first-order prediction (h/2) |
|---|---:|---:|
| Surface D90 (nm) | 1.45143367557 | 2.56226079259 |
| Surface D95 (nm) | 2.44206493829 | 2.8576209483 |
| Surface RMS radius (nm) | 1.34842485356 | 0.80096147239 |
| Surface alpha95 (mrad) | 29.7696327391 | 29.627622307 |
| Surface ID306 centered radius (nm) | 29.1742189259 | 2.22061930147 |
| Smallest observed D95 (nm) | 2.21178210833 | 0.202748886734 |
| Its offset from the sample (nm) | 13.5000000228 | -48.0000001062 |

ID306 raw x/y is [24.521393912993208, -15.871161607943225] nm in the actual result and [1.8287808360523214, -1.2630797085563956] nm in the prediction. The raw vector remainder is [22.692613076940887, -14.60808189938683] nm, norm 26.9879740818 nm. It is the largest source-point remainder at the sample.

The large-ray result and the whole-population width answer different questions: the actual surface D95 is smaller than the first-order D95 even though its largest tail is much larger. The first-order minimum is a mathematical reference, not a demonstrated attainable physical probe; it does not authorize retuning, rescaling, or imposing another focus definition. No D95 quadrature subtraction is used.

## Evidence and limits

- Actual population: the already executed complete constant-K 769-particle comparison, original current 10 uA. No new full-source propagation was performed.
- Only 17 differential probes were newly propagated, with the unchanged complete native field model and physical geometry; no legacy matrix, empirical Cs term or alternate source was introduced.
- The local response is stable to the tested perturbation sizes. This does not certify every field approximation, step discretization, source distribution or experimental microscope.
- No physical beam current, qualification, feasibility claim, or size acceptance threshold is attached to the mathematical prediction.

Raw evidence directory: `F:\tem_simulator_v2\outputs\probe-tail-attribution-20261028\native-local-linear17-h005`

report.json SHA256: `0f2cfdab00c7741eb04d7d7f5d545ccc601bb02f7f5785fbe613b9d67bbde654`
calibration-probes.npz SHA256: `14d02713c8b8e0d5233df61541b3c1032819948ad5afc6449348f9bb254f4a0a`
mathematical-predictions.npz SHA256: `50d661f9565b5cf2c1d6cf76339eff5e92a63978e0ae4834a3bb70e9ba424adb`
