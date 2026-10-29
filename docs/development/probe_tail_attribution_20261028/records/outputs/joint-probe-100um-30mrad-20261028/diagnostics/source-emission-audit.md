# Read-only audit of the actual emitted source

The executed source is **not a uniform or entirely Gaussian joint distribution**. Its initial positions and slopes are truncated Gaussian disks. Its energy distribution is a separately transformed, bounded distribution. The records inspected here use deterministic six-dimensional Halton samples, not random draws, Sobol samples, a product quadrature, or a curved-surface emission law.

This audit reads existing archives and reconstructs emission only. It performs no gun trace, field solve, column propagation, optimization, or coherent calculation. No source parameters, raw arrays, or existing result files were changed.

## Evidence and reproduction

- Machine-readable report: `source-emission-audit.json`.
- Reproduction: `.venv/Scripts/python.exe outputs/joint-probe-100um-30mrad-20261028/diagnostics/source_emission_audit.py > NEW-audit.json`, run from the repository root. Use a new output path.
- The JSON records hashes for the emitter, energy, identity and curvature implementations, the audit script, each original `inputs.json`, `candidate.json`, `particles.npz`, and the relevant archived arrays.
- Original archives: `seed193-h02-v2/candidates/trial-000001`, `seed769-h02/candidates/trial-000001`, and `joint22-trial47-n3073-h02/candidates/trial-000001`, under `outputs/joint-probe-100um-30mrad-20261028`.

The 3073-point archive uses trial 47 optical controls; the 193/769-point archives use the original seed controls. Consequently their downstream radii are **not** a fixed-optics convergence comparison. Their emitter parameters are identical except for count, so emission-only reconstruction does isolate the source rule.

## Actual source branch

All three complete snapshots contain `ColdFieldEmitter` without `_surface_model`, `_tip_curvature_nm_inv`, `_tip_coherence`, `_emission_sampling`, `_emission_quadrature`, `_emission_cdf_quadrature`, or tuning support-probe records. Thus `emitter.py:421-456` executes the legacy flat launch branch. General assembly metadata mentioning a curved tip does not select an absent runtime surface-emission model.

`emitter.py:29-40` uses Halton indices `18 + source_id` and inverse radial CDF sampling. The six bases, in order, are `(2, 3, 5, 7, 11, 13)` for position radius/azimuth, slope radius/azimuth, and the two energy coordinates. There is no sorting or particle-ID permutation in this mapping. Each archived weight is exactly `1/N`.

| Marginal | Implemented rule |
|---|---|
| Initial position | Circular Gaussian conditioned on radius at most `3 sigma`; per-axis untruncated sigma is `5 / 2.354820045 = 2.1233045 nm`; cutoff about `6.3699135 nm`. |
| Initial slopes | Circular Gaussian conditioned on slope norm at most `3 mrad`; per-axis untruncated sigma is `1 mrad`. This is not a statement that the radial RMS is 1 mrad. |
| Initial energy | Exponential term plus a logistic-like quantile term, followed by sample-dependent centering/RMS scaling and a bounded sigmoid map to positive energy. Mean is 0.3 eV and standard deviation is `0.3 / 2.354820045 = 0.1273982700 eV`; support is bounded by 0.01 and 1.3 eV. |

The energy implementation is in `src/temsim/physics/chromatic.py:36-50`, `84-110`, and `139-180`. Despite the phrase "inverse-normal approximation" in the helper's docstring, its expression is a logit, `log(q/(1-q))/1.813799364`, followed by empirical mean/std normalization. The later sigmoid mapping changes that distribution again. The configured 0.3 eV energy width is therefore an RMS-equivalent FWHM, not a demonstrated literal Gaussian FWHM.

The code intends separate position, direction and energy marginals through distinct Halton coordinates. It does not explicitly condition energy on position or angle. Finite deterministic point sets can still have local coincidences and correlations; small pairwise correlations cannot certify complete independence.

## Count-dependent energy nodes

Position and slope prefixes are bitwise identical when increasing the source count. Energy prefixes are not: empirical normalization and the fitted bounded-energy map depend on the entire requested population.

| Change in count | Largest change in an existing particle's launch energy |
|---|---:|
| 193 to 769 | 0.0036185060 eV |
| 769 to 3073 | 0.0070654975 eV |

For ID 306 specifically, launch energy changes from `0.553124768751 eV` at N=769 to `0.546817083793 eV` at N=3073, while its initial position and direction remain exactly identical. A count refinement therefore changes both the sampled population and the existing particles' energy assignments. This is a concrete numerical property to include when interpreting source-count sensitivity; this audit does not establish its contribution to the final probe error.

Measured launch-energy ranges from source reconstruction are:

| N | Minimum (eV) | Maximum (eV) | Mean (eV) | Standard deviation (eV) |
|---:|---:|---:|---:|---:|
| 193 | 0.0148947485 | 0.6509911198 | 0.3000000000 | 0.1273982700 |
| 769 | 0.0151204236 | 0.6625713595 | 0.3000000000 | 0.1273982700 |
| 3073 | 0.0111382479 | 0.7948374133 | 0.3000000000 | 0.1273982700 |

The initial kinetic energies above are reconstructed from the saved emitter parameters and hashed code. They are not claimed to be an independently saved launch-energy array. All reconstructed initial positions, unit directions, normals and IDs match their archived emission arrays exactly. For these actual records, the maximum difference between archived gun-exit kinetic energy minus 300000 eV and reconstructed launch kinetic energy is `8.08e-9`, `1.12e-8`, and `1.07e-8 eV`, respectively. This supports the energy-to-ID correspondence; no permutation or weight change was found.

At N=769, energy's Pearson correlations with initial radius and slope norm are `-0.0036273` and `-0.0061112`; at N=3073 they are `-0.0002429` and `-0.0012822`. There is no strong aggregate linear correlation in these inspected statistics.

## Seed769 outlier ID 306

Its Halton sequence index is 324. Its six coordinates are approximately `(0.134765625, 0.00548696845, 0.9792, 0.38483965015, 0.51389932382, 0.98862084661)`.

| Source ID | Initial radius (nm) | Slope norm (mrad) | Launch energy (eV) | Final centered radius (nm) |
|---:|---:|---:|---:|---:|
| 303 | 2.5148143 | 0.9695190 | 0.2603917 | 0.4338069 |
| 304 | 1.6361783 | 1.3041646 | 0.3259339 | 0.1999228 |
| 305 | 3.5423058 | 1.7158342 | 0.3999900 | 0.6530913 |
| **306** | **1.1356248** | **2.6275978** | **0.5531248** | **28.6893694** |
| 307 | 2.9848752 | 0.2335217 | 0.2456219 | 0.1180623 |
| 308 | 2.0778647 | 0.7134064 | 0.2957296 | 0.1917243 |
| 309 | 4.3296004 | 1.0478240 | 0.3363961 | 0.3640931 |

ID 306 launches at `(1.134949956, 0.039143635) nm`, with slopes `(-1.969238481, 1.739646502) mrad`. Its initial radius is only at the empirical 13.654th percentile, whereas its slope norm and kinetic energy are at the 98.179th and 97.789th percentiles. It is a high-angle/high-energy combination, not an initial-position tail particle. Adjacent IDs are adjacent sequence indices, not necessarily neighboring positions in six-dimensional phase space.

In the saved seed769 transport, ID 306 is already the largest-radius particle at the gun exit (`0.801619027 um`) and at the C2 reference plane (`30.100268325 um`), before its final 28.689 nm centered radius. This localizes a large-radius signature upstream. It does not prove whether the later tail is a correct physical response, a numerical transport error, or a combination. In particular, this audit does not attribute it to spherical aberration or to a manually added aberration term.

## Remaining evidence required

The source and archive correspondence checks passed within their stated scope. They are not transport or physical qualification. The next discriminating evidence is fixed-control, fixed-source step refinement of the same outlier and field path. Source-rule convergence also needs to account for the count-dependent energy transformation. No such additional propagation was performed by this audit.
