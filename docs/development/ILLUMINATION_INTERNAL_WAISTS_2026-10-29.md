# Executed internal-waist and source-association audit

Date: 2026-10-29. This report concerns one frozen **193-particle**, maximum-step
**0.2 mm** classical candidate. It establishes bounded observations of that
population, **not qualification of a default preset or the complete CEOS optical
conjugacies**. No optical controls, physical geometry, specimen position, source
law or absolute particle weights were changed for this capture.

## Execution and scope

The exact input is
`outputs/illumination-control-20261029/column-193-h0.2/baseline/inputs.json`.
The diagnostic executed the native gun followed by the production unified
column, stopping at 1420 mm without moving the physical specimen or replacing
the column geometry. It used `trace_source_to_exit` and `_trace_segment`, with
the production field/action and C2-aperture ownership audits. The final specimen
focus and driven scan pivot are outside this capture.

The run completed in **172.788 s**, within its 180 s budget, with one numerical
job, one Numba thread and one BLAS thread. There were **11,588 integration nodes**
and **7,750 exact observation planes**. Requested spacing was at most 0.25 mm
through the captured column, 0.05 mm between TL22/TL21 and between the AC coil
reference centers, and 0.01 mm over 1407–1410 mm. Inserted observation planes
split the actual integration mesh; this is not a uniform 0.2 mm mesh.

All **193/193 particles**, carrying **10 microampere**, remained active through
1420 mm. No interception was observed. Source IDs, original weights, emission
coordinates/directions, gun-exit phase and source energies matched the frozen
193-particle baseline exactly. The input snapshot was unchanged after execution.

## Definitions and required Normal-mode observations

`finite_region` identifies a finite two-dimensional RMS waist region. It
requires executed sides with larger position covariance in every transverse
direction, negative/positive definite symmetric position-slope covariance on
the respective sides, and one directional converging-to-diverging transition
in each principal direction. It does not require one exact common point focus.
Its representative plane is an actual sampled plane near the radial RMS
minimum, and its full two-dimensional growth bracket can be wider than the
radial sign-change bracket.

Native D95 is separately measured about each plane's actual current-weighted
centroid. Its minimum can occur elsewhere. Neither an RMS region nor a D95
minimum alone proves a strict source-image conjugacy or identifies a scan pivot.

The selected Normal-mode requirements from
[the control contract](../OPTICAL_CONTROL_ROLES.md) give these results:

| Required interval | Requirement | Observed finite 2D regions | Executed radial bracket, mm | Full 2D growth bracket, mm |
| --- | --- | ---: | --- | --- |
| C2–C3, 650–915 mm | At least one 2D waist | 1 | 696.25–696.50 | 696.25–696.75 |
| C3–ADL, 915–1088.2 mm | No converging-to-diverging waist | 0 | None | None |
| TL22–TL21, 1166.35–1249.95 mm | Exactly one 2D waist | 1 | 1211.273132–1211.323102 | 1210.423640–1212.222564 |
| AC pair, 1387.5–1412.5 mm | Exactly one 2D waist | 1 | 1408.51–1408.52 | 1408.49–1408.55 |

All four required intervals resolved under their explicit sampling bounds;
their unresolved lists are empty. This preserves active weak C3 with no
C3-to-ADL crossover. It does not establish that C3 acts as an isolated thin lens.

The central representative plane is 1211.273132 mm, with RMS radius
10.449316 micrometre, D95 49.053825 micrometre and alpha95 3.055606 mrad.
The central native D95 minimum among observed planes is instead at
1210.773431 mm, D95 47.426620 micrometre. These large finite distributions
should not be described as demonstrated sharp source images.

The complete inventory also records one region between C1 and C2
(radial bracket 503.00–503.25 mm), and a broad region near HP1 discussed below.
Gun-exit–C1, ADL–TL22 and 1412.5–1420 mm have no finite region. All recorded
intervals resolved at their stated spacing; all evidence, including directional
envelope maxima, is retained in the raw report.

## HP1: a shallow broad region, not a demonstrated additional source image

The installed **probe HP1 center is 1300.85 mm**. In the TL21-to-AC interval,
the finite-region observer finds a broad region represented at **1298.5 mm**,
with a full growth bracket **1277.160436–1308.75 mm**. The representative RMS
radius is **61.351007 micrometre** and D95 is **229.554618 micrometre**.

The left and right bracket RMS radii are only **1.006062** and **1.001020** times
the representative radius. Their D95 values are **229.104544** and
**226.129885 micrometre**, respectively: the RMS representative is not a D95
minimum. Within this bracket, the sorted principal RMS radii have minima at
1301.375227 mm (42.650408 micrometre) and 1295.000000 mm (44.085982 micrometre).
These are minima of the sorted covariance eigenvalues, not fixed laboratory
axes or separately proven source conjugate planes.

A weighted linear regression on the original emission position explains only
**0.293%** of this plane's position variance; regression on original launch
angle explains **94.253%**. This supports interpreting it as a broad,
predominantly angle-associated envelope region. It does **not** justify counting
it as a demonstrated additional CEOS source image. Conversely, it must remain
in the finite-region inventory rather than being silently discarded.

These regressions use the original finite population and energy mixture, not
independent differential source perturbations. They are association evidence,
not a source-imaging acceptance test.

## AC region: actual concentration, distinct RMS and D95 minima

At the executed RMS representative **1408.51 mm**, the beam has:

- RMS radius **412.665858 nm**, r95 **723.916194 nm**, D95 **1447.832387 nm**;
- alpha95 **2.692977 mrad** and current **10 microampere**;
- sorted principal RMS minima at **1408.51** and **1408.52 mm**, approximately
  **280.763724** and **302.239623 nm**.

The native D95 minimum among the actual observations is **963.107812 nm** at
**1408.60 mm**, with neighboring measured planes **1408.59–1408.61 mm**. It is
not the RMS representative plane. These results replace the earlier free-drift
extrapolation as the evidence for this finite AC waist.

The beam clearly concentrates inside the AC pair. However, linear association
with the original emission position explains only **0.187%** of position
variance at the RMS representative; the original launch-angle polynomial
through third order explains **60.525%**. Together with the finite-beam map
test below, this prevents treating the concentration as proof of a sharp,
strict source image. No common scan pivot was measured by this static capture.

## The local chief map does not image the full finite population

The separate local scan-response analysis proposed a near-axis image plane
around **1408.08086 mm**, with position magnification about 0.08946 and a small
position-per-input-angle block. That is a differential mechanical chief map,
not a measured finite-source image. An offline consistency test applied its
full 4-by-4 transfer to the saved finite phase space, using metres and
mechanical `dx/dz, dy/dz`, subtracting the saved zero-drive reference before
mapping and restoring it at the specimen. No canonical-angle substitution was
made.

For the **original 193-particle column record**, the actual 1420 mm phase maps
to predicted specimen **D95 = 336.376560 nm**, whereas its actual propagated
specimen D95 is **2.209807 nm**. The centered position residual has RMS
**67.665381 nm**. Starting at 1400 or 1410 mm gives the same prediction. Using
the newly captured dense AC data gives 336.381004 nm, only 0.004444 nm different.
Thus neither the internal AC reconstruction nor this upstream mesh refinement
accounts for the discrepancy.

The saved physical input graphs differ only by 193 versus 769 source points;
the scan-private reference additionally inserts `wobble_enabled = False`.
The independently reconstructed native potential contributions have identical
per-source potential digests and the same geometry. Full runtime/field digests
remain different because their input/reference captures differ; they are not
misreported as identical. The scan chief energy is 300000 eV; actual AC energies
span 300000.014895–300000.650991 eV.

The residual's association with the upstream mechanical angles explains:

| Polynomial terms included | Explained position variance | Remaining RMS, nm |
| --- | ---: | ---: |
| Linear | 70.2959% | 36.878627 |
| Through quadratic | 71.6338% | 36.038522 |
| Through cubic | 99.999485% | 0.153499 |

This is strong evidence that high-order angular response matters across this
finite beam. It is consistent with substantial nonlinear cancellation during
downstream propagation. It does not independently assign an aberration
coefficient or prove which optical element supplies each term. In particular,
the local chief conjugacy must not be presented as finite-source imaging
qualification or used to predict the finite probe size by magnification alone.

## CEOS interpretation and remaining limits

The [CEOS reference audit](CEOS_PROBE_CORRECTOR_REFERENCE_2026-10-28.md) and
[Muller et al., Figure 2 and p. 444](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)
distinguish source-image and aperture/field-ray conjugacies. A source image near
DP12 is not automatically the downstream front focal plane. For one fixed
linear transfer, strict point-plane imaging requires `B = 0`; a pure input
tilt producing specimen shift with unchanged mechanical direction requires
`D = 0` and nonzero `B`. Finite waist-to-waist transport need not have `B = 0`.

The required Normal/central/AC finite-region windows pass at this sampling.
That statement does not mean the entire CEOS source-image count, exact optical
conjugacies, full-column focus or driven scanning behavior has passed. The
shallow HP1 region and large central/AC finite widths require the distinctions
above. No source-count convergence, step convergence, final default-preset
qualification, specimen diffraction or coherent scanning image is established
by this audit.

## Provenance and local raw evidence

The raw bundle is currently **local and unarchived**, under
`outputs/illumination-control-20261029/internal-waists-193-h02/`. It is not made
durable merely by committing this report. `report.json`, `receipt.json`,
`particles.npz` and `source.npz` retain the exact planes, full original IDs and
weights, phase/energy history, interception ledger and all unresolved lists.
`source-association-audit.json` records the offline comparisons and 16 input/file
fingerprints. Readback verified all four original receipt file hashes.

The following are 16-hex prefixes of the recorded SHA-256 identities; full
values remain in those JSON records. Array identities include dtype and shape
before the original array bytes, using the existing diagnostic hash routine.

| Identity or file | SHA-256 prefix |
| --- | --- |
| Input snapshot digest | `397424316f54475c` |
| Solver implementation | `799bf51563d3b4cd` |
| Executed runtime | `ae2eb9f0a8083f5e` |
| Executed potential snapshot | `48ea3760bb1ab39b` |
| Physical geometry | `4a763dcae1a043ad` |
| Integration mesh | `cd05e29361ef8a1a` |
| Executed plan signature | `185735f5f3bb1373` |
| `inputs.json` | `114a0fcebe102856` |
| `particles.npz` | `b6edc2f6253011df` |
| `source.npz` | `d3c8a14c336ac810` |
| `receipt.json` | `fecd4b6896e9a3b6` |
| `report.json` | `1b75afb5ecafdcbe` |
| `source-association-audit.json` | `c453878f864a63f3` |
| Original source IDs | `08e94ec878141ceb` |
| Original source weights | `96bf7793256743ba` |
| Emission positions | `9480d15926b2c32a` |
| Emission directions | `3b026569ca7f9e2c` |
| Emission normals | `8ad8d902fca08e59` |
| Source energy offsets | `a0d676cfce1e1be8` |
| Original 193-column particle record | `6ec6a09d91a3e700` |
| Scan native-response JSON | `a44de6315b22c276` |
| Scan static-transfer NPZ | `4c4099ad2899f450` |
| Derived AC conjugacy JSON | `666ca70f589aa2bf` |

The capture driver is `run_internal_waists.py` (prefix
`3158703ada412739`). The offline analysis driver is
`analyze_source_association.py` (prefix `ffd2fcfdae1d863a`). Neither driver
modifies production code or installs a working point.
