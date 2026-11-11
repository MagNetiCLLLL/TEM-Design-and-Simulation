# Microprobe corrector conjugacy and parallel illumination, 2026-11-11

This is a literature-guided, measured operating family for the present combined
finite-field model. The selected profiles provide broad, nearly parallel
illumination at the original Si[110] upper surface. They are separate loadable
candidates; no application-wide preset, Nanoprobe drive or physical component
position was replaced. See [all measured sizes, angles and controls](measured-tables.md),
[the machine-readable results](results.json), and [profile verification](profile-verification.json).
The four loadable controls are discrete executed working points spanning the
requested interval. The continuous C2/C3 proposal solver is retained for further
zoom settings; intermediate proposals require their own finite-source check.

## What was matched

C2 and C3 remain the illumination-zoom controls. C1, source emission, apertures,
Mini magnitude and Objective excitation remain fixed. The Mini polarity is
positive in the model's Microprobe wiring convention, opposite to Nanoprobe.
This sign is not an OEM current calibration. The source emits 10 uA at 300 kV.
The C2 aperture diameter is 100 um. All accepted final trials use 2048 original
positive-current source particles, with the original IDs, absolute weights,
energies and aperture interceptions retained. Source-count and integration-step
comparisons are recorded separately; 2048 particles do not imply a continuous
distribution or experimental accuracy.

The relay solution uses ADL 8.579822332682%, TL22 49.981124115803%,
TL21 61.416161382988% and TL12 18.639375233688%. Mini remains
26.678337967960%, Objective 29.326401112600%, and C1 51.666666666667%.
No lens was moved. The HP2-to-HP1 first-order relay is conjugate and approximately
afocal, with transverse magnification magnitude 0.759938; unit magnification was
not imposed. This is a model design solution, not a published CEOS operating table.
The smallest-area setting is close to the present C3 supply limit; the reported
range should not be extrapolated to smaller areas without checking that limit.
The CEOS paper's double-symmetric core is not reproduced identically: the present
central source image is near 1213.8 mm, rather than the geometric TL22/TL21
midpoint of 1208.15 mm, and the core magnification is non-unit. Here "matched"
means the measured aperture-plane conjugacy, bounded source-image transport and
specified Microprobe illumination criteria. It does not mean a fully calibrated
OEM symmetric aberration-correction condition.

The inherited Nanoprobe multipole drives caused additional radial-only caustics
at the largest Microprobe illumination. Those failed the bounded two-dimensional
waist test. The final Microprobe family reduces the strength and skew-strength
amplitudes of **all four installed hexapoles** to 10% of the inherited values,
while retaining their orientations, physical locations and magnetic relay
settings. HP2 and HP1 are therefore 543559.000930 and 475717.565949 m^-3.
HPC is (-14723.861924, 4575.881859) m^-3 and HPol is
(53835.798746, 31256.220966) m^-3 for normal/skew amplitudes. This is a deliberate
Microprobe excitation change, not a renamed Nanoprobe preset or a software
filter. No quadrupole or surviving particle was artificially removed.

The controlled high-area comparison retains the same source, C2/C3, all magnetic
lenses and clipping: reducing those hexapole amplitudes removes the unresolved
radial-only candidates, lowers mechanical alpha95, and retains four bounded
source-image regions. Its before/after receipts remain in the archive. A
subsequent condenser adjustment places the final high-area endpoint closer to
2.2 um. The reduction is a transport-design choice; it does **not** establish
optimal negative Cs, cancellation of every hexapole aberration, or OEM corrector
performance. Full aberration-coefficient qualification remains separate work.

## How the optical references were interpreted

The [CEOS paper, printed pp. 443-444 and Figure 2](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)
describes a **STEM CESCOR**: the principal hexapole planes are relayed conjugately;
the diagram also contains axial source images and alignment deflectors. It
explicitly allows the condenser Mini to replace TL11 and weighted DC offsets of
the existing scan coils to emulate DP12. Source-image zeros and aperture-ray
zeros are different objects. The paper does not prescribe the present
Microprobe excitation values or a universal Microprobe crossover count.

[Williams and Carter, Chapter 9, especially Figure 9.6, printed p. 147](https://doi.org/10.1007/978-0-387-76501-3_9)
provides the relevant parallel-illumination and double-deflection construction:
source imaging near the objective prefield front focal plane can illuminate the
specimen approximately in parallel; two coils can create a virtual pivot away
from either physical coil. Its lens names cannot be mapped literally to this
project: in some diagrams “C3” denotes an objective prefield function.

The [Titan training material, Module 1 Figures 2 and 5](https://emfacility.science.oregonstate.edu/sites/emfacility.science.oregonstate.edu/files/2025-09/Titan%20training%20manual_0.pdf)
shows coupled C2/C3 zoom and the Mini polarity distinction. Mini Micro/Nano
selection and the TEM/Probe illumination ray path are related controls with
different meanings; a Mini mode label alone does not select a unique set of
source images.

The [author's EPFL ray diagrams](https://www.epfl.ch/labs/lsme/tem-ray-diagrams/)
explicitly illustrate near-parallel TEM illumination via Mini/objective coupling
and distinguish manufacturer-specific implementations. Their thin lenses and
illustrative magnifications are not a dimensional specification for this model.
For a useful counterexample, [Laulainen's Cambridge thesis, Figure 2.11, printed
p. 48](https://doi.org/10.17863/CAM.93209) draws weakly convergent Microprobe SED
with a small probe on the specimen. That is a different illumination objective
from the broad parallel TEM beam requested here. Its additional source images
must not be imposed solely because both modes are named Microprobe.

No publicly inspected source supplies an exact CEOS Microprobe excitation table
for this custom geometry. The architecture below is inferred from those
functional principles and tested in the model. It is not asserted identical to
every TFS or CEOS microscope. Bibliographic details, inspected page numbers and
local research-file hashes are in [research-sources.json](research-sources.json).
Copyrighted book pages and vendor figures are not redistributed in this archive.

## Crossover count and location

The qualification interval is the **executed gun exit at z=450 mm through the
specimen upper surface at z=1599.1999998 mm**. There are four bounded
two-dimensional finite-current RMS source-image regions in each final profile,
for both the local transmitted population and the common specimen-surviving
cohort. The earlier gun/extractor acceleration region is not assigned an exact
crossover count by this downstream survey. Historical display/interpolated gun
histories are insufficient for a precise additional gun-waist claim.

| Component interval | Final Microprobe source crossover | Placement and interpretation |
| --- | --- | --- |
| Gun exit - C1 | None observed | Survey begins at the executed 450 mm exit. |
| C1 - C2 | One | Near z=503.255 mm; all emitted current is still present. |
| C2 - C3 | One | Moves with C2/C3 zoom; see each executed native bracket in the measured table. |
| C3 - ADL | None observed | No inherited Nanoprobe crossover requirement is imposed. |
| ADL - HP2 | None observed | HP2 is an aperture/conjugate reference, not a source waist. |
| HP2 - TL22 | None observed in the final family | The inherited high-area branch had an unresolved radial-only caustic here. |
| TL22 - TL21 | One | Near z=1213.8 mm, between DP22 (1186.35 mm) and DP21 (1229.90 mm). |
| TL21 - HP1 | None observed in the final family | Directional evidence is retained; inherited high-area settings were ambiguous here. |
| HP1 - TL12 | None observed | HP1 remains conjugate to HP2, not another source waist. |
| TL12 - upper AC coil | None observed | Pupil relay and steering transport. |
| Upper - lower AC coils | None observed | A static source crossover inside the pair is not required for this Microprobe family. |
| Lower AC coil - Mini | None observed | Two-coil steering is assessed by the downstream mechanical response. |
| Mini - Upper Objective/specimen | One before the specimen | Source-image region near the objective front-focal function, followed by broad nearly parallel specimen illumination. |

“None observed” refers to the reported native sampling and bounded-waist test;
it is not a claim to resolve every possible microscopic caustic. Actual component
centres are 500, 650 and 915 mm for C1, C2 and C3, and 1522.5 mm for Mini.
The last broad interval is split conceptually at the Upper Objective; the source
image is above it. No Nanoprobe direct-to-specimen focus acceptance rule is used
to reject this intentional parallel-TEM source image.

Each accepted source-image region has converging-to-diverging transitions in
both covariance directions and bounded growth on both sides. A minimum of radial
RMS alone is insufficient. The full audit retains separate native RMS and D95
minimum planes. Some observation brackets are 0.005 mm locally; other zoom
points have 0.5 mm condenser brackets or 0.1 mm final-stage brackets. The measured
table and JSON disclose the actual spacing. No interpolated focus is represented
as an executed plane or as an experimental positional accuracy.
At a coarsely sampled condenser waist, the smallest observed RMS/D95 is an upper
bound on the intervening continuous minimum; it must not be interpreted as the
exact nanometre-scale source-image size. The specimen-plane criteria use the
exact executed physical surface, independently of those axial-waist brackets.

### RMS, D95 and angular statistics

RMS radius is the square root of the current-weighted mean squared distance from
the local current centroid. D95 is a **diameter** enclosing 95% of the current
alive at that plane. It is neither twice RMS radius nor 95% of the emitted gun
current. Alpha95 is the angular cone semi-angle containing 95% of the current,
measured about the actual current-weighted chief direction; angular RMS is a
different second-moment statistic. Angles use mechanical electron directions,
including azimuthal rotation in the immersion field.

At the final nominal 0.971 um point, the saved-phase decomposition gives radial
angular RMS 0.003173 mrad and tangential RMS 0.039032 mrad, versus total mechanical
angular RMS 0.039160 mrad. Most of the remaining angular spread is rotational,
not a strongly converging radial cone. The native canonical-angle surrogate is
0.006558 mrad at 95% current; it is not substituted for the measured mechanical
alpha95 of 0.054894 mrad. The full static-potential postanalysis and its gauge
and small-angle limitations are retained with that trial.

The C1 source image contains the full 10 uA; subsequent waists contain the current
that passed the C2 aperture. Comparing only the final surviving cohort at C1
selects a smaller source subset and can give a different waist size/angle. Both
populations are archived without renormalizing any losses. The specimen is an
illumination plane in Microprobe, not a required small-source-waist plane.

## Deflector groups and special placement

The following are physical positions in the current assembly. Their functions
are literature-guided design interpretations. Only the AC/DP12 group has the
native response and finite-source shift tests described below; the other rows
are not claims of completed per-group calibration.

| Group | Physical z (mm) | Functional placement and constraint |
| --- | --- | --- |
| Gun deflectors | 402 / 418 | Align the accelerating source/gun exit to apertures and the condenser axis; evaluate at the local energy. A source crossover need not coincide with a coil. |
| Condenser deflectors | 800 / 820 | Between C2 and C3 and upstream of their moving source image; centre/tilt the zoom beam and align apertures. A remote virtual pivot follows from the calibrated two-coil ratio. |
| BSh / BTlt group | 1027.5 / 1052.5 | Upstream of ADL; provide independent corrector entrance displacement and angle. The labels are control functions, not a requirement for a source image between the coils. |
| DPH2 / DPH1 | 1144.5 / 1300.85 | At the respective principal hexapoles; centre the beam through nonlinear fields and control alignment/feed-down. These are conjugate aperture planes. |
| DP22 / DP21 | 1186.35 / 1229.90 | Straddle the central source-image region near 1213.8 mm; permit relay alignment in position and angle. |
| DP11 | 1346.30 | Exit alignment upstream of TL12 (1366.30 mm); acts with downstream steering in the exit transport. |
| AC / DP12 | 1387.5 / 1412.5 | One physical two-coil assembly. The DP12 service alias adds common DC bias to the same coils; AC and alignment commands share physical limits. |
| Image / Diffraction / Descan | 1674.4 / 1699.4 | One shared post-specimen two-coil host. Image shift, diffraction shift and descan require different target-plane response constraints, not extra physical elements or an imposed upstream source waist. |

For a mechanical first-order map, a kick at a plane gives
`delta_r = B delta_theta` and `delta_theta_out = D delta_theta`.
An ideal specimen-conjugate pivot has `B=0` for pure tilt, whereas an ideal
front-focal relation has `D=0` for pure shift. For two coils the four-dimensional
matrix `[B_upper B_lower; D_upper D_lower]` permits independent position and
angle commands when it has full rank. Therefore an effective pivot can lie
outside either coil, and a driven pivot need not be a static finite-source waist.
See [Williams and Carter Figure 9.6](https://doi.org/10.1007/978-0-387-76501-3_9)
and [Rodenburg's author-hosted tutorial](https://www.rodenburg.org/RODENBURG.pdf).

## Native conjugate and steering evidence

The archived [native maps](native-maps.json) and [refinements](refined-conjugates.json)
use small nominal-energy zero-current reference pencils. They do not count as
finite-source qualification. Their original input identities and inherited
hexapole drives are retained. At an axial zero reference, changing hexapole
amplitudes does not change the limiting linear hexapole derivative; this is why
the magnetic relay solution is applicable to the reduced-strength family.
Finite-amplitude and aberration claims still require the actual source trials.

HP2 (1144.5 mm) is conjugate to HP1 (1300.85 mm): the B-block norm is
9.49e-8 m/rad at a 0.1 mm maximum step and 9.45e-8 m/rad at 0.05 mm.
Both transverse magnifications have magnitude 0.759938. The canonical
position-to-angle block has diagonal entries approximately 3.9e-5 /m
(Frobenius norm approximately 5.52e-5 /m),
consistent with the intended near-afocal core. This concerns aperture-plane
transport and does not place a source waist at either hexapole centre.

The downstream native canonical front-focal relation is near
z=1584.346231 mm, with canonical D-block norm 1.69e-9. Its mechanical D-block
norm is 1.626 because the specimen is in an immersion field. A canonical zero
alone therefore does not certify pure mechanical shift or the finite-source
semi-angle. A distinct HP1-conjugate pupil image lies near z=1598.382504 mm
(native initial bracket 1598.382-1598.383 mm); this is not the source waist,
not a verified Microprobe coma-free STEM plane, and not a physical scan coil.
These interpolated reference derivatives do not imply nanometre experimental
knowledge of either plane.

The DP12 software reference is at 1384.0 mm; it is a zero-length virtual plane,
not an additional physical coil at that position. In this assembly its
field-free moment construction distributes an alignment command with shares
1.14 and -0.14 over the actual 1387.5/1412.5 mm coils. Those shares must not be
replaced by equal common-mode gains or inferred from a picture. The installed
AC opposite-coil command and these DP12 offsets jointly span the physical
outputs. The native response below includes the actual implementation and
combined coil limits, rather than assuming a field-free construction remains
exact in the immersion optics.

The AC/DP12 four-channel native response has scaled condition number 12.41
(1 um position / 1 mrad angle scaling). Separate 50 nm X and Y shifts were
validated with the full 2048-particle nominal source in the inherited-drive
relay: 480 particles reached the sample, with unchanged current. Displacements
were 49.999963 nm along the requested axis; mean mechanical angle changed by
about 1.05e-8 mrad, and relative D95 change was below 4.5e-9. These are numerical
differential residuals, not real-instrument accuracy. The small-angle pure-tilt
commands were checked with reference pencils only. The archived commands are
local calibration evidence, not a claim of full raster, full field of view or
detector-image qualification. The final reduced-hexapole family additionally
requires its own finite-beam scan validation before image acquisition.

## Validation and limits

All final family profiles retain the original source law: 5 nm virtual-source
FWHM, nominal emission energy 0.3 eV, energy FWHM 0.3 eV and the existing
20 nm coherence-length assumption. That coherence length is an unmeasured
design assumption in a classical Gaussian-Schell representation. These trials
do not run a coherent wave pipeline. They use CPU classical particle transport
with two worker threads and a 0.1 mm maximum integration step. Usable CUDA was
not required or asserted for this classical work.

The current loss is physical C2-aperture interception. Source, survivor and
effective counts are reported separately. No lost current is restored and no
aperture is changed to satisfy alpha95. Original tail-particle IDs, radii,
angles and weights are in each waist audit. Max-radius/R95 and max-angle/alpha95
are reported so an isolated gross outlier cannot be hidden by the 95% statistic.

Inherited-drive numerical controls separate source sampling from integration:
at the nominal point, 1024 particles with step 0.2 to 0.1 mm changed specimen
D95 by -0.000245% and alpha95 by +0.000351%, without changing current. At fixed
0.1 mm step, increasing 1024 to 2048 changed D95 by -0.193%, alpha95 by -1.470%
and current by +0.418%. The 2048 launch is not wholly nested in 1024, and batch
energy normalization also changes the sampled angles. These are measured
sensitivity results, not formal error bars. See [comparisons.json](comparisons.json)
and the separate final-family controls in [final-comparisons.json](final-comparisons.json).

The Si sample was replaced by vacuum **only during incident-beam diagnosis**,
with the physical surface/geometry unchanged. Loadable profiles restore the
original atomic sample, its orientation quaternion, and the original absolute
local CIF path. The quaternion aligns physical [110] with the beam; legacy
zone-axis/in-plane hint fields still read [001]/[100] and are not the authoritative
physical orientation. Profile roundtrip checks verify this explicitly, including
CIF access from a different working directory. No specimen scattering, DF scan,
HRTEM image, complete Cs coefficient, hardware current calibration or space-charge
qualification was performed in this conjugacy study.

## Evidence and reproduction

[Beam envelopes](beam-envelopes.png) and [specimen footprints](illumination-footprints.png)
are original figures plotted from actual observations. The same plots are
available as SVG. Dashed envelope curves are R95, not RMS or D95 diameter.

`column/` retains complete trial receipts, immutable numerical snapshots,
vacuum diagnostic input profiles, waist/particle audits, full envelope arrays,
and selected native phase-space planes for final candidates. Every original
particle column, including intercepted particles, is retained at those selected
planes. Full raw histories remain in the ignored output folder and have SHA256
hashes in [artifact_manifest.json](artifact_manifest.json). Original gun launch
and exit arrays are shared with the unchanged
[earlier Microprobe source archive](../microprobe_20261111/README.md).
The separate `native-gun-exit.npz` files retain the executed 450 mm row. Offline
checks reproduce every source ID and weight, transverse gun-exit phase and clock
bitwise. Kinetic-energy reconstruction differs from nominal-plus-offset by at
most one floating-point ULP (5.82e-11 eV); this rounding is retained and reported,
not corrected by changing physical energies or source weights.

The `drivers/` files preserve the exact research scripts and their original
workspace-path dependencies. They are evidence copies rather than a standalone
installed command. The working driver paths are under
`outputs/microprobe-match-20261111/`, using the verified original gun caches in
`outputs/microprobe-20261111/extractor/new-n1024` or `new-n2048`. Reproduction
requires those full original output caches, the current repository/runtime,
and the numerical snapshots. A fresh checkout's curated phase-space evidence
supports independent metric verification but does not include every multi-GB
raw propagation history.

Preparatory driver errors and rejected exploratory optical points are preserved
separately. An observation-grid preparation exceeded the existing 2 GiB history
budget and was revised to fewer local observation windows, without raising the
budget, reducing source count, changing the physical source or relaxing a
waist test. The rejected inherited high-area settings remain explicitly marked
with unresolved directional evidence rather than being relabeled as qualified.
See [candidate-status.json](candidate-status.json) for explicit selection,
domain checks and rejected-point directional evidence.

The measured four-crossover contract is scoped to this family in
[microprobe-contract.json](microprobe-contract.json). Historical Microprobe
reference counts remain historical; they do not qualify or veto this family.
Nanoprobe source-image, AC-waist and direct specimen-focus rules remain separate.
