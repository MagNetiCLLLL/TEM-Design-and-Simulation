# Probe-corrector initialization and angular transfer

Date: 2026-10-28. This record separates existing historical evidence from the
current complete-source Finite-Field Optics observations. The initial comparison
was prepared from saved inputs, reports and arrays. The follow-up sections below
record a subsequently executed legacy column reference and a unified reference
with identical controls, using the existing validated complete gun source.
No coherent calculation, source replacement or
preset installation was performed.

## Result and scope

Probe-corrector settings can change the specimen convergence semi-angle.
This includes the HP nonlinear fields and the transfer/matching lenses. C2/C3
remain the primary convergence controls; that role does not imply zero angular
coupling from ADL, TL22/TL21, TL12, Mini Condenser or Objective.

The old Normal HP on/off comparison directly measures a small angular change
with a much larger change in probe size. Current independent lens differences
also measure angular coupling. Neither observation identifies the corrector
initialization as the sole reason for the current size error, nor establishes
that any historical control set qualifies the present field model.

Here alpha95 is the current-weighted angular containment semi-angle relative
to the weighted chief direction. D95 is the diameter about the specimen-plane
weighted centroid containing 95% of the current that actually reaches that
plane. A sampled maximum angle or a mechanical pupil edge is a different
observable. There is no imposed transmitted-current floor.

## Which old model and which old branch?

The model names must not be interchanged. The historical `custom` and current
legacy `ideal` paths use finite-length parameterized fields, including the
configured equivalent aberration actions. Historical `custom` propagation
includes canonical paraxial RK4 transport through finite round-lens and
multipole profiles. It is not the separate `ideal_thin` approximation, which
uses thin optical events and drifts. See [simulation modes](../SIMULATION_MODES.md)
and [the historical comparison](FINITE_FIELD_PARTICLE_VALIDATION_2026-10-27.md).

There are also distinct historical operating points:

| Reference | Model / branch | Important distinction |
| --- | --- | --- |
| 14 October Normal | Historical `custom`, weak C3 Normal | C2-C3 crossover required; no C3-ADL crossover. The archived complete 3,000-particle run gives alpha95 approximately 30 mrad. |
| Earlier strong-C3 reference | Historical parameterized corrector, subsequently documented as Large angle | C3, ADL and TL12 strengths differ substantially from Normal. Its central and AC reference points cannot be imported into the Normal comparison. |
| 27 October focused preset | `unified`, Finite-Field Optics Normal | Fitted near alpha95 10 mrad with 49 original particles. It is a newer finite-field working point, not a thin or legacy calculation. |
| `ideal_thin` | Separate thin approximation | The older finite-field and parameterized-field results do not qualify this mode. No comparable complete thin-model crossover chain is established here. |

The archived 14 October Normal fine-step result uses 3,000 emitted particles,
2,661 specimen survivors, 0.025 mm maximum step, alpha95
29.999987656 mrad, RMS radius 0.138880444 nm and D95 0.559281785 nm.
Those are fixed-population observations, not particle-count convergence or
present-unified qualification. The tracked evidence is
[nanoprobe_normal_20261014.json](nanoprobe_normal_20261014.json).

## Initialization: what changed and what did not

The following values compare the archived Normal input with the current
`baseline193/observations.inputs.json`, rather than comparing an optimized
candidate with the initial state. Percent values are excitation magnitudes;
Mini Condenser retains negative Nanoprobe polarity.

| Control | Archived Normal | Current finite-field default |
| --- | ---: | ---: |
| C2, percent | 23.225625142668733 | 13.899040164772698 |
| C3, percent | 9.207984804305022 | 15.68823681182136 |
| ADL, percent | 10.0 | 10.0 |
| TL22, percent | 59.93152503582202 | 59.93152503582202 |
| TL21, percent | 59.95527906841963 | 59.95527906841963 |
| TL12, percent | 36.004558708747176 | 36.004558708747176 |
| Mini Condenser magnitude, percent | 43.88646617222637 | 43.88646617222637 |
| Objective, percent | 68.9801 | 68.97810600833917 |
| HP2 strength, m^-3 | 423294.84130355046 | 4371028.449290636 |
| HP1 strength, m^-3 | 253195.31789553943 | 2616320.416243901 |
| HP1 orientation, rad | -0.04199279823147041 | -0.042012393991127016 |
| HP2 orientation, rad | 0.0 | 0.0 |

Current HP1 and HP2 strengths are respectively 10.3332101 and 10.3262030 times
their archived values. ADL and all three transfer-lens strengths are unchanged.
Thus the default is not merely the old Normal point executed by another solver,
but the near-10-mrad result is also not evidence of an undocumented TL12 change.
The large HP rematch and C2/C3 changes must be interpreted together with the
different nonlinear field model.

The initialization path is explicit. In
[MainWindow](../../src/temsim/gui/main_window.py), standard startup creates and
assembles the state, selects `unified`, and then applies the operating modes.
[Operating-mode loading](../../src/temsim/operating_modes.py) merges the
model-specific device overrides from
[the catalog](../../configs/operating_modes/catalog.toml). Its `unified.nano_probe`
override changes C2, C3, Objective and HP controls; it does not override ADL or
TL22/TL21/TL12. Mini Condenser mode selection preserves its live magnitude and
sets the Nanoprobe sign. A
[model-menu switch](../../src/temsim/simulation_modes.py) alone preserves lens
excitation. Saved inputs and explicit profile restoration are separate paths;
the standard-startup chain is not a guarantee that an already restored session
uses the default. See the
[finite-field default record](FINITE_FIELD_NANOPROBE_DEFAULT_2026-10-27.md).

## Direct old-Normal evidence: HP changes the angle

The archived `nonlinear_comparison` keeps the same source, original weights,
stops and round-lens strengths. All three cases retain 2,661 particles.
At the 0.05 mm step it records:

| Executed historical comparison | alpha95, mrad | RMS radius, nm |
| --- | ---: | ---: |
| Linear reference: added Cs and HP disabled | 30.0051376482691 | 0.1374753155 |
| Added Cs enabled, HP disabled | 30.0047145251404 | 1.7607226695 |
| Added Cs and HP enabled | 29.9999876470261 | 0.1389037714 |

Enabling the HP correction changes alpha95 by -0.004726878 mrad, or about
-4.73 microradians, in this controlled comparison. Its effect on probe size is
much larger. This does not show that the complete corrector has negligible
angular influence: ADL and the transfer lenses were held fixed in this test.
These are the legacy model's added-Cs and HP actions; they must not be described
as a new unified-field counterfactual. Source:
[tracked Normal evidence, `nonlinear_comparison`](nanoprobe_normal_20261014.json).

## Current independent differences

These measurements all start from `high-angle-restart`, whose alpha95 is
29.315612268236276 mrad. They use the complete 193-particle source, original
weights, identical gun ancestry and a 0.05 mm column step. The same 19 original
IDs are intercepted at C2 Aperture, leaving 174 specimen arrivals. No automatic
focus compensation is applied to an independent difference.

| Changed control only | Actual increment | Resulting alpha95, mrad | Change in alpha95, mrad |
| --- | --- | ---: | ---: |
| HP1 and HP2 together | Multiply each strength by 1.01 | 29.31246198579035 | -0.00315028244593 |
| HP1 strength only | Multiply by 1.0001 | 29.315959217690864 | +0.000346949454588 |
| HP1 orientation only | +0.00001 rad | 29.31566642506376 | +0.000054156827484 |
| TL22 magnitude only | +0.001 percentage point | 29.31721491842975 | +0.001602650193476 |
| TL21 magnitude only | +0.001 percentage point | 29.31441329440305 | -0.001198973833226 |
| TL12 magnitude only | +0.001 percentage point | 29.31644621153215 | +0.000833943295875 |

A percentage-point increment is not a relative-percent strength change. These
local finite differences establish nonzero coupling at this operating point;
they do not establish the response to a large change or the corresponding
derivative at the corrected candidate. The HP effects are evaluated on the
same physical population, so these differences are not caused by replacing
current weights or changing which particles reach the specimen.

## Current observed angular chain

The following table is a direct remeasurement of saved arrays, not new
propagation. Within each case it follows the same original particles that
survive to the specimen: 193 for `baseline193`, 174 for `high-angle-restart`.
Those are different physical cohorts between the two cases. In particular,
the high-angle cohort's upstream C1 statistics exclude the 19 particles that
are subsequently intercepted; the actual 193-particle C1 beam is unchanged.

For each named interval the table selects its narrowest stored radial
checkpoint. It does not claim a refined minimum, a common two-dimensional
waist, or a conjugate-plane qualification. Coarse checkpoint radii can change
substantially when the observation plane moves toward a narrow waist.

| Reference | Default Z, mm | Default RMS radius, nm | Default alpha95, mrad | High-angle Z, mm | High-angle RMS radius, nm | High-angle alpha95, mrad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1-C2 interval | 503.486037 | 14.117186 | 0.108581 | 503.486037 | 11.982125 | 0.081901 |
| C2-C3 interval | 720.929273 | 17.584589 | 0.201505 | 671.500000 | 7.806432 | 0.478763 |
| Corrector central interval | 1202.178571 | 2058.174838 | 2.063677 | 1201.803739 | 20910.731574 | 6.168749 |
| AC pair interval | 1400.000000 | 21.682614 | 1.023846 | 1400.500000 | 512.715556 | 3.053275 |
| Extra radial minimum above specimen, searched over 1560-1590 mm | 1585.703521 | 954.553929 | 8.017223 | 1585.703521 | 3390.112556 | 23.944919 |
| Specimen upper surface, an endpoint observation | 1599.1999975 | 0.171771 | 9.819338 | 1599.1999975 | 34.558246 | 29.315612 |

The corrector-central to AC alpha95 ratios are approximately 0.496 and 0.495;
AC to specimen ratios are approximately 9.591 and 9.601. These are ratios of
observed quantiles, not independently measured optical magnifications. They
show why equal angles at all source images must not be assumed. The condenser
Normal requirement passes at the declared sampling, but the full crossover
chain and corrector image qualification remain unresolved in these captures.

## A unit position relay is not a unit angular relay

The old Normal evidence includes a nominal-energy full first-order canonical
map from HP2 at 1144.5 mm to HP1 at 1300.85 mm. Its two position magnifications
are approximately 0.999999999994, and the position-from-angle block has norm
approximately 5.8344e-13 m/rad. However, the position-to-canonical-angle block is
approximately

```text
C = [[ 9.6684574, 0.4083683],
     [-0.4083683, 9.6684574]] m^-1.
```

That nonzero block rules out interpreting the reported position magnification
alone as an afocal full -I map. The coordinates are canonical quantities at
planes within fields, not an automatic measurement of mechanical ray angles.
The relay also stops at HP1; it does not include TL12, Mini Condenser and
Objective downstream to the specimen. An inverse-magnification angle rule
requires additional conjugacy, energy and field assumptions. Source:
[Normal `hp2_hp1_first_order`](nanoprobe_normal_20261014.json); see also
[the map and angle contract](FINITE_FIELD_OPTICAL_ACCEPTANCE_CONTRACT_2026-10-27.md).

## Historical crossover information and missing evidence

For old Normal, the tracked topology check executes a C2-C3 bracket with its
central checkpoint at 672 mm, RMS radius approximately 138.151385 nm. Its
interpolated candidate is near 671.5266646 mm with an inferred radius of
approximately 0.0106868 nm. That inferred radius is not an independently
executed exact waist and must not be compared directly with a differently
sampled current checkpoint. The recorded C3-ADL interval contains no additional
waist under its bounded sampling.

The earlier strong-C3 reference is different: C3=46.0598152184%,
ADL=79.8069433594% and TL12=49.9770148353%. Its documented central radial
minimum is 1208.148342 mm; principal-direction minima are separated by about
6.23 micrometres and their widths are about one micrometre, so the strict
common two-dimensional test fails. Its separately executed TL12-CM waist near
1400.00070 mm has RMS radius 2.15835 nm. Its sample alpha95 is approximately
30.0067 mrad and D95 approximately 0.4953038 nm. These observations belong to
that strong-C3 point, not to the later Normal point. See the
[probe-corrector reference](../probe_corrector_reference.md).

The historical raw output directories named `probe_relay_validation`,
`corrector_dph_20261010` and `ac_scan_pivot_20261010` are not present in this
checkout. The tracked Normal JSON remains available, but a complete archived
Normal table containing every crossover's executed position, size and alpha95,
and an AC-to-specimen differential map, was not recovered. Missing angles must
not be inferred from the 30 mrad specimen value, nor may waist-size ratios be
substituted for a measured transfer magnification.

A new legacy comparison using current geometry, implementation and the current
complete gun source must be labelled as such. The executed follow-up below
supplies such a reference. It supplements the old record rather than restoring
the missing original output directories or reproducing the original software
and 3,000-particle run.

## Local evidence

Current raw inputs, reports and `observations.npz` files are under
`outputs/finite-field-joint-preset-20261028/`:

- `baseline193` and `high-angle-restart`: initialization and angular-chain data.
- `hp-common-plus`, `hp1-relative-plus`, `hp1-angle-plus`, `tl22-plus-v2`,
  `tl21-plus-v2`, and `high-tl12-plus`: independent angular-coupling observations.
- `residual-shape-diagnostic-v1.md` and `residual-shape-diagnostic-v2.json`:
  additional shape diagnostics and content hashes; empirical polynomial
  residuals are not corrected physical probes.

The current complete-source identity is
`4e3bac1af215cfe7b8589e64363541ba2039654fb01be49e8a5abee3017c0815` and solver
identity is `f34f2ff6f73532f29faa3652fd88dd902625bd248d2196e5d94c3f0b08c5cd21`.
All 193 source columns and original weights remain present in the captures.
The high-angle specimen current is 9.015544 microampere from a 10-microampere
source. None of this establishes a qualified 30 mrad, sub-0.7 nm finite-field
preset or a new scanning image. See the separate
[alpha95 search record](FINITE_FIELD_ALPHA95_SEARCH_2026-10-28.md).

## Executed complete-193 legacy Normal reference

The subsequent `legacy-normal-complete193` run completed one current-
implementation `custom` column execution on the already validated complete
193-particle gun source. It uses current geometry and the archived Normal
incident controls, not the old executable or its 3,000-particle sample. Total
reported elapsed time is 13.072690 seconds, including 8.272153 seconds of column
transport. The gun was not recalculated; no column prefix or downstream source
subset was substituted. The result is a reference, not an installed preset.

Independent inspection of the requested and executed State JSON confirms that
the changes from `baseline193` are exactly: `simulation_mode=custom`, the old
C2/C3 and Objective percentages, old HP1/HP2 strengths, and old HP1 orientation.
ADL, TL22/TL21/TL12, CM magnitude/polarity, geometry and source settings remain
the baseline inputs. The source IDs, original weights, source energy offsets,
initial transverse coordinates/slopes and clocks match the actual cached gun
arrays exactly. Reconstructing the source digest gives the identity recorded
above. The saved initial kinetic-energy arrays differ from the unified capture
by at most 5.8207661e-11 eV through floating-point reconstruction; bitwise
identity is not claimed for that derived array.

All 193 source columns remain in the saved arrays. The final 174 survivors and
the 19 C2-aperture-intercepted IDs exactly match `high-angle-restart`. Their
original transmitted weight is 0.9015544041450778, giving 9.015544 microampere
from the 10-microampere source. The recorded interception histories are not
identical inside the aperture plate: at the shared 764.9177896585 mm checkpoint,
the legacy capture still records 193 arrivals and the unified high-angle capture
175; by 765 mm both record the same 174. Reported stop positions differ by up to
0.1 mm. Equality of the final cohort must not be promoted to equality of every
intermediate survival mask. The default unified beam instead retains all 193.

The reference contains 3,994 exact observation planes from 450 mm to the
1599.1999975 mm specimen upper surface. The maximum saved-plane gap is
0.4998695074 mm and the maximum column integration step is 0.05 mm. Additional
local observations cover 1207.15-1209.15 mm at nominal 0.02 mm spacing and
1399.9-1400.1 mm at nominal 0.001 mm spacing. These are observation spacings,
not separate source or integration refinements.

### Executed reference chain and population definitions

The five interior entries below satisfy the report's executed two-dimensional
bracket test at this fixed sampling. They are not claims of exact continuous
minimum positions or converged minimum radii. The specimen is an endpoint
observation, not a two-sided waist. All values were independently recalculated
from `observations-historical_controls_custom.npz` and match the report.

"Actual" includes every original particle physically present at that plane.
"Common" uses the same 174 original IDs that reach the specimen, with their
unaltered original weights. Thus C1-C2 and C2-C3 have 193 actual particles before
the limiting aperture, while both populations have 174 downstream. Common-
cohort statistics are a comparison only; no input population was pruned.

| Reference | Z, mm | Actual N | Actual RMS radius, nm | Actual alpha95, mrad | Common-174 RMS radius, nm | Common-174 alpha95, mrad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1-C2 executed bracket | 503.4860372912 | 193 | 14.117185507 | 0.108580644 | 11.982124856 | 0.081901040 |
| C2-C3 executed bracket | 671.5000000000 | 193 | 9.202415844 | 0.634641509 | 7.808642718 | 0.478763014 |
| Corrector internal executed bracket | 1202.1785714286 | 174 | 2089.117068075 | 6.104659799 | 2089.117068075 | 6.104659799 |
| AC executed bracket | 1400.0010000000 | 174 | 2.305022074 | 3.062205927 | 2.305022074 | 3.062205927 |
| Extra CM-specimen executed bracket | 1585.7035208003 | 174 | 3211.771458176 | 23.981262815 | 3211.771458176 | 23.981262815 |
| Specimen upper surface | 1599.1999975000 | 174 | 0.142413899 | 29.369081661 | 0.142413899 | 29.369081661 |

The C2-C3 requirement passes and no C3-ADL waist is found at the declared
sampling. The old Normal central reference is near 1202.17857 mm in this
execution, not the strong-C3 branch's 1208.148342 mm reference. In the extra
1207.15-1209.15 mm observation window there is no executed two-dimensional
waist: its lowest sampled radius is at the lower window boundary, 1207.15 mm,
with RMS radius 19230.238230 nm. That boundary minimum is not a new waist.

### Comparison with current unified observations

| Actual specimen observation | Actual N | alpha95, mrad | RMS radius, nm | D95, nm |
| --- | ---: | ---: | ---: | ---: |
| Current unified default | 193 | 9.819338272 | 0.171770685 | 0.509215608 |
| Current unified high-angle restart | 174 | 29.315612268 | 34.558245898 | 183.290985600 |
| Historical Normal controls, current custom reference | 174 | 29.369081661 | 0.142413899 | 0.589372246 |

The default row uses its entire actual 193-particle specimen population; it is
not silently restricted to the legacy survivor subset. For a separate comparison
on exactly the same 174 original IDs and exactly shared observation planes,
the arrays give:

| Common plane and fixed 174-ID cohort | Default RMS, nm / alpha95, mrad | High-angle unified RMS, nm / alpha95, mrad | Legacy Normal reference RMS, nm / alpha95, mrad |
| --- | ---: | ---: | ---: |
| Corrector central, 1202.1785714286 mm | 1350.885365 / 1.557474 | 20973.278949 / 6.168749 | 2089.117068 / 6.104660 |
| AC reference, 1400.0000000000 mm | 11.770578 / 0.772352 | 1029.376174 / 3.053275 | 2.458080 / 3.062206 |
| Specimen upper surface | 0.118260 / 7.408900 | 34.558246 / 29.315612 | 0.142414 / 29.369082 |

These fixed-plane rows do not identify each case's own best-focus plane. For
example, the legacy AC sampled waist is at 1400.001 mm, while the high-angle
unified radial minimum is near 1400.5 mm. Comparing each minimum without its
plane and grid would mix image displacement with minimum size. The common-174
default sample angle is also not its full-population alpha95 of 9.819338 mrad.

The reference demonstrates that a similar specimen angular quantile can coexist
with very different central/AC image sizes and specimen aberration. It does not
isolate a unique cause: relative to the high-angle unified restart, the executor,
HP settings, CM and Objective settings differ. Relative to the default, C2/C3
also differ. The large HP initialization ratio alone cannot be assigned all of
the size difference. The historical same-source HP on/off comparison above has
a narrower causal scope and must remain separate.

The full reference crossover-chain status remains `UNRESOLVED`. No particle-
count or half-step convergence was established by this new reference. Its
sample covariance-drift waist offset is approximately +0.497491 nm; this is not
an exact finite-field best-focus measurement or common two-dimensional focus
qualification. Alpha95 is 29.369082 mrad, not the requested 30 mrad. This result
therefore does not qualify or install the requested unified preset.

Raw evidence is in
`outputs/finite-field-joint-preset-20261028/legacy-normal-complete193/`:

- `report.json`: completed reference and actual/common-cohort chains.
- `requested.inputs.json` and `observations-historical_controls_custom.inputs.json`:
  identical requested and executed physical inputs.
- `observations-historical_controls_custom.npz`: full 193-column record, SHA256
  `7c29b06439580eab82dd8e84c795804d66d0720c4599c7e345fb678c0b5bf129`.

The validated source-section SHA256 is
`0d62afe83714e7e3745354ffccdcfa15b923dad0ed6a282a92c4d43f8fe04bcb`.
The executed helper is `replay_legacy_normal_fixed_source.py`, SHA256
`3efe40537591c1dacf98c0259802dc10d816d2c5a8f13f1d6c8c2efe0caf6989`.

## Strict comparison: same Normal controls, different field model

The completed `legacy-controls-unified193` follow-up changes only
`simulation_mode` from `custom` to `unified`. Independent comparison of the
complete executed State JSON confirms equality of every other field, including
the historical HP strengths and all round-lens settings. Both inputs round-trip
through the current State schema without changes. This isolates the selected
transport model at fixed physical inputs; it does not isolate one nonlinear
term or one optical component within that model.

Both records have exactly the same 3,994 observation positions, all 193 original
source IDs, weights and source energy offsets, and identical initial transverse
phase-space coordinates and clocks. The source and solver identities are those
recorded above. The derived initial kinetic energies differ by no more than
5.8207661e-11 eV. Both retain the same final 174 IDs and intercept the same 19
at C2 Aperture. The one-plane difference in plate-interior survival described
above also applies to this pair; intermediate survival masks are not identical.
No source particle was removed from the saved records or reweighted.

| Actual specimen result, identical controls | alpha95, mrad | RMS radius, nm | D95, nm | Current, microampere |
| --- | ---: | ---: | ---: | ---: |
| Current `custom` implementation | 29.369081661 | 0.142413899 | 0.589372246 | 9.015544041 |
| Current `unified` implementation | 29.330967704 | 188.773703128 | 854.265910425 | 9.015544041 |

D95 uses the original weight of the actual 174 specimen arrivals as its
denominator in both rows. Neither row establishes the requested qualified
30 mrad unified preset. Both full crossover chains remain `UNRESOLVED` at the
0.05 mm integration step and declared observation sampling. Total reported
elapsed times are 13.072690 and 148.672205 seconds respectively; the existing gun
source was reused, without a reused column prefix.

The following downstream comparisons use those same 174 original IDs. The
first three rows compare exactly the same physical plane. The final row instead
compares each model's own executed AC bracket at this fixed sampling.

| Observation | Custom Z, mm | Custom RMS, nm / alpha95, mrad | Unified Z, mm | Unified RMS, nm / alpha95, mrad |
| --- | ---: | ---: | ---: | ---: |
| Corrector-central shared checkpoint | 1202.178571429 | 2089.117068 / 6.104660 | 1202.178571429 | 2106.519764 / 6.110223 |
| AC shared geometric plane | 1400.000000000 | 2.458080 / 3.062206 | 1400.000000000 | 686.101566 / 3.057813 |
| Extra CM-specimen shared bracket | 1585.703520800 | 3211.771458 / 23.981263 | 1585.703520800 | 3061.350003 / 23.936242 |
| Each model's executed AC bracket | 1400.001000000 | 2.305022 / 3.062206 | 1399.700000000 | 306.493690 / 3.057813 |

The upstream C1-C2 and C2-C3 bracket positions also agree, at 503.4860372912
and 671.5 mm. Their actual populations contain 193 particles, with custom versus
unified RMS radii of 14.117185507 versus 14.117186165 nm and 9.202415844 versus
9.197967903 nm, respectively. Their common-174 statistics must not replace
these actual upstream beam statistics.

Neither model has an executed two-dimensional waist in the extra
1207.15-1209.15 mm window; both lowest sampled radii occur at its lower boundary.
The unified model also has no such waist inside the fine 1399.9-1400.1 mm window:
its executed AC bracket at 1399.7 mm lies outside that window. Its additional
near-specimen bracket at 1599.1997975 mm is only a fixed-grid observation, not a
refined best-focus solution. The specimen covariance-drift waist offsets are
approximately +0.497491 nm for custom and -9442.929505 nm for unified; neither
offset is an exact finite-field refocus or a two-dimensional qualification.

Direct same-ID transverse-position differences already accumulate in the
corrector transfer region. The weighted RMS norm of the raw unified-minus-custom
position difference is approximately 0.745 nm near HP2 at 1144.318746 mm,
1.831 nm at TL22 center, 89.419 nm at the shared central checkpoint, 202.156 nm
at TL21 center, 214.849 nm near HP1 at 1300.777902 mm, and 745.478 nm at TL12
center. At the shared 1400 mm AC plane it is 688.583 nm. These are same-ray
position differences, not differences between beam radii and not fitted or
subtracted corrections. They measure cumulative transport differences, which
can also grow in drifts; they do not assign each increment to the nearby lens.

Thus the large specimen size discrepancy is already evident before the final
Objective region even with the old HP strengths held exactly equal. The roughly
10.33-fold change in the installed HP initialization cannot explain this strict
pair. The observations support further examination of field-model and matching
differences through the corrector transfer section. They do not establish HP
initialization, a particular round-lens aberration order, or any single
component as the unique cause.

The paired raw evidence is
`outputs/finite-field-joint-preset-20261028/legacy-controls-unified193/`, containing
`report.json`, `requested.inputs.json`,
`observations-historical_controls_unified.inputs.json` and the complete
`observations-historical_controls_unified.npz`. The latter has SHA256
`bd89c16ae2b5ff866cdfa8aff6f7aeb5796a2708e53d167fee3c10f1f58e82d1`.
