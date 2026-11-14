# Ideal Gaussian magnetic-lens third-order reference

Zero-current independent Gaussian lens variational ODE reference only; no finite source or clipped fit.

| B peak (T) | Sigma (mm) | Cs (mm) | Cc (mm) | First focus z (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | 3 | 92190.3378 | 181.024693 | 148.721853 |
| 0.15 | 6 | 3196.79378 | 90.5964971 | 76.0886324 |
| 0.15 | 12 | 157.241342 | 46.0397556 | 41.0678728 |
| 0.30 | 3 | 1598.39689 | 45.2982485 | 38.0443162 |
| 0.30 | 6 | 78.6206709 | 23.0198778 | 20.5339364 |
| 0.30 | 12 | 12.9873226 | 14.633539 | 11.2962507 |
| 0.60 | 3 | 39.3103354 | 11.5099389 | 10.2669682 |
| 0.60 | 6 | 6.49366132 | 7.31676948 | 5.64812535 |
| 0.60 | 12 | 4.55341426 | 7.83647373 | -0.583720154 |

The field centre is z=0. These are first paraxial image planes, not distances from a hard field edge. The Gaussian field is explicitly continued to all z.

All nine numerical reference checks passed. They test step/tail repeatability and dimensionless scaling, not finite-source sampling, clipped-pupil fitting, higher radial field orders, material geometry, or wave resolution.

Worst relative Cs changes: step 1.89206e-14; input tail 2.35255e-14. Worst equal-kappa dimensionless scaling error: 3.23153e-14.

The same B_peak*sigma has the same dimensionless kappa=|q|*B_peak*sigma/(2p). Cs/sigma, Cc/sigma, focus_z/sigma and the scaled pupil conversion then agree. A larger axial width at fixed peak changes kappa; it is not a pure geometric rescaling.

The tube does not enter the intrinsic zero-current coefficient. Its permitted pupil and transmitted finite source remain separate physical tests. The outward-pupil convention is delta_r=-Cs*theta*|theta|^2+Cc*theta*(delta_E/E0).
