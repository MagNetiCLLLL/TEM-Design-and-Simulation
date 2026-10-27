# Weighted coherent segments

Whole-column method optimization was deferred on 2026-10-25 at the user's
request. See [the attempts and handoff record](COHERENT_PROPAGATION_ATTEMPTS_2026-10-25.md)
for verified milestones, retained caches, unresolved method limitations and
future research directions. Neither the nine-energy nor nominal-energy branch
has reached the specimen and main screen.

Finite-Field Optics can explicitly enable **Allow weighted coherent resampling**
under the coherent calculation's advanced numerical settings. Controlled support
projection must also be enabled. Both options remain disabled by default.
These are numerical settings: lens excitations, geometry, electromagnetic fields,
source energy modes, Cs/Cc ownership and physical boundaries are unchanged.

## Exploratory preview

Exploratory labels accompany the phase plate, independent section views,
electron-arrival display and saved-state overlays as well as the calculation
page. A mixed result identifies every preview member and keeps its budget and
loss figures separate. Selecting a non-preview member does not hide the fact
that the surrounding mixed result also contains preview states. In mixed-result
labels, missing member profile metadata is reported as unavailable rather than
being treated as a precision certificate.

The explicit **Exploratory preview** option supplies an initial projection-loss
allowance of `0.01` and complex L2 allowance of `0.5`, relative to each original
gun-exit mode, together with a 2048-pixel post-gun column limit and a suggested
2 mm maximum column step. The gun retains its original numerical grid settings.
These are adjustable numerical settings,
not different lens excitations or a reduced source-mode mixture. Disabling the
GUI preset restores the numerical allowances and axial step saved before
enabling it. The ordinary default column step remains 0.5 mm.

Preview records identify the preset, effective column pixel limit, original
allowances and actual cumulative projection costs. The `1%` probability limit
is relative to the gun-exit reference, not a per-pixel intensity error bound or
a pointwise phase guarantee. Numerical projection loss remains distinct from
aperture, wall and detector absorption. The original physical-domain, sampling,
kinetic and forward-propagation gates remain active. A preview can still fail
if its numerical allowances or valid model domain are insufficient.
Projection budgets do not bound axial discretization or splitting error. The
2 mm suggestion is not a whole-column convergence qualification; interfaces
and field-table nodes can require shorter intervals.

The wider allowances are explicit, optional column-only request overrides,
`column_preview_support_probability` and `column_preview_support_l2`. Their
default `None` values preserve historical request behavior, including earlier
preview runs with `0.001 / 0.1` budgets. They require exploratory preview,
controlled support projection and weighted resampling. Explicit previews may
now request at most `0.5` relative numerical probability loss and `4` cumulative
complex L2 cost. These large ceilings support severely approximate exploration;
they are not recommended accuracy targets or a convergence certificate. The
default preset remains `0.01 / 0.5`. The ordinary request's wave-grid settings
retain their strict validation; only the internally created post-gun grid opts
into the larger exploratory range. Material handoff keeps
its existing backend and pixel settings while carrying the same explicit
budget policy and non-resetting support history. A change of these overrides
changes the column cache identity, not the executed gun source.

The L2 value is a conservative cumulative complex-amplitude projection bound.
It is neither an intensity-loss fraction nor a pointwise phase-error limit.
The independent probability budget limits numerical intensity loss to its
declared fraction of the original gun-exit reference; the default preset uses
one percent. Actual losses, both original allowances and any representation
estimates remain visible in preview results. Optical excitations, source-mode weights, field definitions
and the physical/sampling admission gates are not changed by this preset.

A new preview branch may reuse the executed gun wave before its column
allowances have been issued. It must not increase an already spent strict
ledger or relabel a strict result as a preview. Complete segment caches remain
bound to their own preset, numerical settings, fields and geometry.

## Complex projection and weights

An incoherent source mixture is propagated as separate complex modes. Within
one mode, the operation uses the orthonormal discrete Fourier transform and
retains the original complex coefficients on a smaller frequency rectangle.
The retained frequencies and their phases are not reconstructed from intensity
samples. The physical period, origin, affine orientation, analytical curvature
and tilt remain unchanged. The sampling pitch changes with the lattice size.
The implementation follows the signed-frequency and `norm="ortho"` conventions
in the [SciPy FFT reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html).

For an input mode weight `w` and discarded Fourier coefficients `c`, the initial
projection consumes `w * sum(abs(c)**2)` probability per reference electron and
`sqrt(w * sum(abs(c)**2))` absolute complex L2 allowance. The current lattice
may be unit-normalized as part of the existing mode/weight factorization, but
its absolute weight includes every recorded physical and numerical loss.
Projection never restores that lost weight. Different source modes are never
combined as coherent amplitudes or dropped to meet a memory target.

Spatial projection, boundary taper and Fourier projection share the original
source allocation. Legacy projections and boundary tapers retain their original
one-percent individual allowances. A pressure-targeted Fourier or spatial
projection may use the remaining cumulative allowance, without changing either original total.
Its version-four receipt binds the preceding ledger digest, both remaining
allowances and the typed resource constraint. Every old event is still validated
under its original individual limit. Spatial pressure receipts use schema v6
and additionally bind their selected share of the remaining allocation and
their distinct centered-window candidate family. They do not reinterpret old
spatial receipts. Segment boundaries, cache restore, physical clipping and changed
mode weight cannot renew those allowances. Conditional material branches retain
their existing nonspendable common support history.

The reduced spectrum leaves a guard at 76 percent of the new Nyquist frequency.
The ordinary 80-percent envelope gate, boundary check, full field-domain check,
kinetic phase budget and resource checks still apply. Candidates are provisional
until these checks pass. An inadmissible candidate leaves the original field and
ledger intact. Lossless refinement can still be required afterward. This local
Parseval projection error is not a certificate for subsequent discretized
propagation, axial steps, source convergence or the complete microscope.

## Segments and storage

Segments use the captured field providers and the maximum requested chunk size.
Compact coils contribute actual support faces. Gaussian round lenses contribute
their nominal entries and exits, since Gaussian fields have no finite end. Tilted faces are
marked as nominal axial locations rather than complete transverse boundaries.
All overlapping fields and Gaussian tails remain active on both sides.

The route starts from the executed electron-gun and accelerator output, continues
through every intervening lens and coil, crosses the specimen, and then continues
through the downstream optics to the requested detector. A saved entrance wave
is the preceding segment's complex output, never a newly prescribed specimen
probe. Each checkpoint retains the complex envelope, analytical phase carrier,
coordinates, energy and time references, absolute mode weights and accumulated
error and clipping histories. Saving an intensity image alone cannot restart
coherent propagation. There is no mandatory all-column in-memory history.

Checkpoints permit stage-by-stage execution and release of completed work. They
do not reduce the bandwidth of the current wave or guarantee that a single
high-order optical step fits memory. Any lattice reduction must independently
pass the same controlled complex projection and sampling gates below.

The geometry executor can subdivide these segments further. The weighted
policy's version-three identity specifies `typed_resource_failure_only` and
`remaining_for_admissible_grid` allocation.
Saving a segment, entering a step or completing a canonical transport does not
itself spend a projection allowance. A normal sampling failure first attempts
lossless Fourier refinement. A typed host/device/pixel resource failure can
instead propose a controlled projection of the current unexecuted entrance.
Every candidate must pass the original physical and sampling gates before use;
an unresolved failed output is never filtered into acceptance. Resource retries
keep the first unexecuted pressure entrance and its original ledger fixed. If
the actual operator still exceeds the resource limit, a bounded retry may choose
a smaller candidate from that same entrance, using the measured growth demand.
Discarded proposals spend nothing. Only a successful operator commits one
projection receipt; cancellation or failure cannot commit a proposal. Necessary
entrance boundary taper keeps its original explicit validation and budget rules.
The initial input, a distinct pressure entrance and any live trial are included
in memory admission. Temporary candidate owners are released after the accepted
operator, before preparing the next residual.

When the resource failure specifically follows an unresolved scalar-phase
increment, the executor first proposes the existing controlled spatial window.
This can remove nearly empty peripheral coordinates while retaining the original
pixel pitch, origin and analytical phase carrier. A same-period spectral
projection alone cannot reduce a large phase gradient at those outer coordinates.
Ordinary spatial proposals retain their original one-percent event allowance.
Under typed resource pressure, spatial proposals try increasing shares of the
unchanged remaining allowance: 1, 2, 5, 10, 20, 50 and 100 percent. Each trial
starts from the same unexecuted wave and original ledger. The first candidate
that passes actual entrance admission and the strictly decreasing prior-grid
constraint proceeds to the complete physical operator. This bounded family
does not claim a global minimum-loss window or spend every proposed share.
Only actual successful losses are charged. Its workspace is admitted before
allocation; all ordinary boundary, spectrum and physical-operator checks still apply.
Spatial reduction changes the physical window and the scalar coefficient
range, so it does not inherit the fixed-period spectral growth forecast.
An unavailable or initially rejected window falls back to the spectral route
from the unchanged entrance and ledger. Other resource failures retain their
existing spectral-first order.

Pressure selection checks an even FFT-friendly candidate family with fixed
aspect ratio, choosing the largest resource-admissible grid. Its nested guarded
passbands minimize discarded probability within that family; this is not a claim
of a global two-dimensional optimum. A failed sampling retry contributes its
measured growth scale to the resource forecast. At least a factor of 1.25 plus
the usual five-percent guard is reserved, so one complete finite DFT band can
be embedded inside the ordinary 80-percent Nyquist guard. This does not promise
that a subsequent physical operator will need only one refinement.
The actual operator is checked again after projection. The unchanged entrance
and other caller-owned arrays remain included in host-memory admission.
Optional GPU execution keeps its CPU fallback; Required GPU retains the device
limit. If the least-loss feasible candidate exceeds either remaining allowance,
no projection is committed. Selection uses one forward transform and never
reconstructs amplitudes from an intensity image.

The former low-pressure schedule has a version-one identity, and the first
pressure-only schedule has a version-two identity with a one-percent event cap.
Their saved histories remain intact and cannot be silently reinterpreted.
A new branch can reuse an actual gun-exit wave that has not yet started its
column budget; this preserves the executed gun and its losses without resetting
an already propagated column history.
An explicit version-two to version-three numerical transition must instead
retain the old policy, exact wave, original ledger prefix and source allocations
in a checked transition receipt. It authorizes future pressure allocation only;
none of the old event accounting changes.

When disk segmentation is enabled, each completed mode segment is saved with its
source, field, geometry, operator, exact interval and numerical identities. A
later mode's failure does not erase previously committed segments. All-mode
checkpoints are published only after every mode reaches the same plane. The
source implementation is verified before publication; cancellation and failed
verification cannot publish a partial numerical result.

Complete unified checkpoints also receive a durable observation index. Reopening
the same cache or changing the requested observation plane can reuse the nearest
complete checkpoint at or upstream of that plane, provided the source, captured
fields, geometry, numerical controls and implementation still match. The index
adds references rather than another copy of the complex wave arrays. Reuse checks
payload checksums and every mode's phase, electric and support history. A partial
mode set is never an observation prefix, and a downstream checkpoint is never
silently back-propagated. If preparing a different scan range changes the
captured field, that is a different calculation and cannot reuse the old prefix.
Streamed detector masks retain the original source, phase/support histories and
physical losses. At a detector's exact plane, an incident checkpoint still needs
its mask, whereas an already transmitted checkpoint does not apply it a second
time when continuing downstream.

## Working memory

Spectral resampling uses tiled CPU FFTs with a preflight of 49 bytes per input
cell plus bounded line/tile workspaces and separately counted retained arrays.
It does not allocate a second full coordinate grid or materialize the large
analytical phase carrier. The residual executor releases completed transforms,
derivatives and superseded scalar arrays before later H8 applications.

The Taylor residual's device planning estimate is 448 bytes per cell. The actual
private device allocator still enforces the requested hard limit, including FFT
workspaces; an unusual FFT plan that does not fit is rejected or follows the
previously authorized backend fallback. Required GPU never becomes CPU. The
complete host coefficient/Taylor stage reserves 512 bytes per cell plus 64 MiB
for cold field-provider preparation. Non-shared retained input waves are counted
separately, including retry entrances. Validation and canonical preparation use
their own 256-byte estimates; canonical domain expansion separately checks its
actual output size and retained input arrays before allocation. Failed optional
expansion attempts release their temporary waves before the alternative method.

Chebyshev recurrence planning is backend specific: the original CPU base remains
640 bytes per cell; the measured CUDA base is 384 bytes per cell. Both add the
existing 32-byte recurrence allowance, 16 bytes per group output and bounded
coefficient tables. A CPU fallback replans against the CPU host budget. The
rational coefficient certificate, original substep checks and complex operators
are unchanged. These are stage estimates rather than a claim about total process
or desktop GPU memory.

Each single-mode segment can lazily reuse one private device pool and bounded
FFT-plan cache. Host operations between residuals have no active device context,
so host admission does not accidentally inherit a GPU limit. Every residual
still selects its backend from its actual work size. A shape change or pressure
releases obsolete plans and unused pool blocks without freeing live arrays.
Cancellation closes the owner; nested scopes cannot enlarge its hard allocation
limit. The CPU fallback restarts the failed operation from the unchanged host
input.

## Development evidence

The actual previously accepted default FEG mode at 519.047619 mm has been
continued to 520 mm with the original 24 GiB device and 32 GiB host budgets.
Its 6250-square entrance was projected to 5760 square, and the accepted output
at 519.523810 and 520 mm was 5544 square. The complete original physical and
numerical histories were retained, with an explicit numerical-policy transition
record. This is an actual source-mode continuation, not a fresh nine-mode
source-to-screen qualification.

Independent direct reconstruction of the initial 519 mm projection agrees with
the Parseval receipt; the retained analytical phase carrier is exactly unchanged.
A second actual continuation without Fourier projection also reached 520 mm
on a 6600-square grid after the device-lifetime optimization, within the same
24 GiB device budget. Comparing these two executed outputs on a common physical
lattice gives raw complex relative L2 difference `9.956e-5`, intensity L1
difference `4.058e-7` per reference electron, bright-region raw phase RMS
difference `1.185e-4 rad`, and centroid displacement `1.96e-14 m`. The actual
absolute weight difference is `1.10003e-9` per reference electron, matching the
two new numerical projection receipts. Analytical curvature, tilt, origin and
longitudinal phase reference agree without fitting.
A separate finite round-lens and hexapole H8 comparison uses three decreasing
projection budgets and compares the raw full complex fields on a common physical
lattice. Neither comparison fits away translation, piston, tilt or defocus.

The subsequent fresh public nine-mode run completed a common checkpoint at
493.214285714 mm. Four modes completed the next checkpoint at 497.142857143 mm;
energy-sample mode E4 then reached a host-memory admission limit after an
internal accepted step near 494.196 mm. The incomplete mixture was not published
as a completed plane. This failure is retained as development evidence, and
motivates the separate host-array lifetime and stage-memory work.

An explicitly recorded continuation with those memory improvements admitted
the formerly blocked 7546-square work, but subsequently requested an 8100-square
residual grid beyond the declared host and device capacities. It did not complete
the next nine-mode plane. A separate receipt audit found 103 spectral projections
for its E4 mode before the old failure; the largest projection input
used only about 23 percent of the device planning budget. That mode had consumed
essentially all of its probability-projection allowance. This identifies a
scheduling problem: routine low-pressure compression must not repeatedly spend
the allowance needed for later memory pressure. Existing depleted histories are
not reset; a revised policy must start a new column branch from an earlier actual
checkpoint with valid remaining history.

The version-two fresh column branch from the executed 450 mm gun checkpoint
completed all nine modes at 481.818181818 mm. Mode E4 reached a
7000-square grid; a later retry requested 8400 square beyond the original
8192-pixel and host-memory limits. This branch did not reach the specimen.
A read-only complex128 Fourier audit of that saved 7000-square mode showed that
5120-square projection would lose `4.668374728e-9` probability per reference
electron: 4.22 percent of its remaining total, but 4.22 times its former
individual allocation. This establishes an unnecessary allocation restriction
at that plane, not convergence or successful subsequent propagation. The new
pressure-target allocation addresses that restriction without increasing the
original total error allowances.

The version-three continuation admitted that allocation, but a later refined
7680-square device grid exceeded the same 24 GiB budget. A separate explicit
2048-pixel exploratory preview retained all nine original modes and completed
477.941176471 mm. Its E4 mode later requested 2420 square after a proposed
1540-square projection, although almost all of its relaxed loss allowance
remained. The failed operator and its pending proposals were not published as
a completed plane. This exposes a resource-forecast/retry limitation rather
than exhaustion of the probability allowance; neither attempt establishes
source-to-screen completion.

After the transactional retry repair, the actual preview tried decreasing
same-period spectral candidates from 1540 to 630 square at 480.808823529 mm.
All still required approximately 2420 square for the residual scalar-phase
increment, so the attempt failed without publishing a new complete mixture.
A separate read-only spatial-window test on the nearest complete E4 checkpoint
at 477.941176471 mm reduced 1680 square to 376 square with unchanged pixel pitch,
origin and analytical carrier. Its weighted probability loss was `1.09491e-7`
and L2 cost `3.29903e-4`, within the existing event budgets; boundary and envelope
checks passed. This is evidence for a different candidate selection order, not
proof that the failed internal entrance or subsequent column has propagated.

The subsequent spatial-first implementation did propagate all nine original
modes to 481.764705882353 mm. E4 finished on 376 square; the other modes used
560 to 700 square. The aggregate retained total weight `0.998447360519329`,
cumulative numerical probability loss `1.279599433227646e-7`, and cumulative
projection L2 bound `0.0006750353778554561` per original reference electron.
Every original source allocation and accepted history prefix was independently
checked. This is an exploratory segment result, not whole-column convergence.

That complete wave was committed before a completed-index list/tuple identity
comparison failed. The store now compares canonical JSON identities and freezes
the caller's dependency data, while still rejecting changed values. A separate
recovery receipt verified all nine payload checksums and histories before adding
the missing index; the original failed report, manifests and wave arrays were
not rewritten. Subsequent propagation resumes that actual completed mixture.

The next continuation preserved all nine modes through 493.23529411764713 mm,
then stopped at that complete boundary before changing the future axial-step
ceiling. The retained total weight was `0.9984466284507301`, cumulative numerical
probability loss `8.600285425795337e-7`, and cumulative projection L2 bound
`0.0012532244078992465`. E4 used 1936 square and the other modes 288 to 320
square. This complete checkpoint remains separate from partial progress of an
individual mode.

The first continuation with a 2 mm ceiling failed during E0 propagation at
506.6176470588235 mm, before publishing a new complete nine-mode plane. A pending
1944-to-1540 spectral candidate cost `1.370905446e-7` probability per reference
electron, below its remaining `1.108939966e-4` allowance. The failure exposed a
candidate-routing omission: rejection of the spectral candidate's initial
admission did not try the existing spatial-window alternative. The original
run did not save that local rejection reason, so boundary ringing is not an
established diagnosis. This evidence does not demonstrate propagation to 520 mm
or qualify the larger step over the full column.

Resource recovery now tries both existing candidate types, each from the same
unchanged entrance and cumulative ledger. Scalar-phase pressure tries the
spatial window first; other resource pressure tries spectral projection first.
An initial admission failure releases the rejected candidate before trying the
other type. Both spatial paths have an explicit workspace check. Only a fully
executed successful transaction commits a projection receipt. Rejection notes
retain candidate type, plane, shape and reason as text, including in a final
failure; they never retain failed wave arrays. The spatial event allowance and
all physical sampling gates remain unchanged.

The actual retry with this routing repair reached the same E0 failure at
506.6176470588235 mm. Its saved diagnostics establish that the spectral
candidate was rejected for an occupied envelope boundary, and that the spatial
alternative found no smaller admissible window. The old complete nine-mode
plane at 493.23529411764713 mm remains the last published result. The failed
candidate did not consume its proposed error allowance. This separates the
repaired control-flow defect from the remaining representation problem.

A separate capture preserved that actual unexecuted 1944-square entrance and
1540-square spectral candidate. Applying the existing two-cell boundary taper
to the pending candidate cost another `1.90894e-10` probability and
`1.28095e-5` L2 per reference electron. The sequential probability cost was
`1.372814384e-7` and triangle-inequality L2 bound `0.00038306693`.
An independent raw complex comparison gave L2 `0.00036999880`, within that
bound; the original amplitude checksum and source weight were unchanged.
Subsequent lossless embedding to 1936 square passed the original boundary and
spectral gates. This admission audit did not execute the physical operator.

Recovery may now try this composition only after an unexecuted spectral
candidate fails its initial occupied-boundary check. It retains the same
unfactorized source weight for both receipts and preserves the existing taper
event cap. Both events remain pending until the full physical operator
succeeds. Workspace admission, cancellation, coverage and sampling checks
remain active. A failed propagated output is never tapered or used as a new
projection source. Probability costs add sequentially; L2 bounds add by the
triangle inequality, not in quadrature.

In the next actual run, E0 through E3 completed 506.6176470588235 mm, but E4
encountered scalar-phase sampling pressure at 497.0588235294118 mm. These four
partial mode caches are not a completed mixture. An independent E4 continuation
using the original 0.5 mm ceiling crossed that failure and completed
503.2720588235291 mm, then encountered another 2200-square resource request
beyond the 2048-pixel preview cap. A CPU replay reproduced all four pending
spectral receipts from that saved entrance. Its last 1050-square candidate
passed admission after tapering and lossless embedding to 1344 square, but the
outer transaction rejected it because the preceding proposal was 1188 square.
That last candidate's physical operator was never executed. The raw-to-admitted
grid growth therefore needs to inform candidate selection; this failure alone
does not establish that every admissible representation exceeds the pixel cap.
This supports local step refinement, not a claim that E4
or the mixture reached the requested specimen or screen.

For weighted resource recovery with automatic refinement enabled, a typed
scalar-phase sampling failure that exhausts the spatial resource allowance may
now bisect the whole optical interval. Each attempt starts from the same
accepted checkpoint and rebuilds the quadratic reference, both residual halves,
retarded clock and electric phase. Geometry masks and all budgets are admitted
before committing any history. Retries have a 12-attempt limit and must produce
a strictly smaller representable interval. Their diagnostic history is retained
as `potential_geometry.axial_resource_retries`; old records mean an empty
prefix. This does not subcycle a diagonal phase to hide an unresolved final wave.

The H6/H8 scalar-split action also uses the linear identity
`K R + L U^n = L(R + U^n) + U R`, where `K = L + U`, to retain every
noncommuting term with fewer transforms. H8 now uses 27 instead of 45 complex
two-dimensional FFT operations per action. Floating-point grouping changes;
this is not a bitwise cache optimization or a lower-order physical model.
An actual E4 four-millimetre CPU comparison retained the same coordinates,
carrier, weight and kinetic/electric histories. Raw complex relative L2
difference was `4.485e-16`, intensity relative L1 difference `3.208e-16`, and
mean elapsed time improved by about 1.40 times. That local CPU timing does not
establish a whole-column or GPU speedup.

The actual run with both changes crossed E4's earlier 497 mm scalar failure,
but failed in its canonical map after the accepted internal plane
501.53653492647055 mm. The preferred angular-spectrum chart requested 2162
square, while the Collins alternative reported a 10.9846-times sampling scale.
The combined failure then requested 22400 square, beyond the preview and memory
budgets. The four earlier modes again completed 506.6176470588235 mm; there was
no new complete nine-mode plane. This distinct chart-resource failure remains
separate from the scalar-only axial retry and from the corrected
raw-to-admitted candidate forecast.

Canonical recovery now retains the resource requirements of each alternative
chart separately. An immutable, shape-bound hint identifies the preferred
angular-spectrum domain request and the independent Collins sampling failure.
Only a matching weighted resource transaction may use the preferred request to
select a smaller entrance representation; ordinary propagation keeps its
original failure behavior. Every selected candidate still executes the complete
canonical map with the original boundary, alias, flux and coverage checks.
An actual two-packet regression distinguishes a 288-square domain request from
the alternative's approximately eleven-fold sampling request on a 256-square
entrance. It does not treat the latter as a requirement of both algorithms.

Pending spectral candidates may also try quintic boundary windows with widths
2, 4, 8 and 16 cells. Every trial starts from the same raw spectral candidate,
unchanged source weight and pending ledger. Failed trials spend no allowance.
The original one-percent boundary-taper allowance and cumulative limits remain
active. Non-default widths use ledger schema v5; earlier schemas and their
existing history digests retain their original meaning. Only a complete,
successful physical transaction commits the spectral and window receipts.
An actual 503 mm E4 entrance replay accepted an 810-square spectral proposal
with an eight-cell window and lossless embedding to 1024 square. Its combined
pending probability cost was approximately `4.19056e-5` and L2 bound
`0.00652973` per reference electron. This replay demonstrates admissible input
selection, not successful downstream propagation or a completed nine-mode
plane.

The next actual nine-mode trial entered the previously failing E4 canonical
region, but a residual half-step after 501.53653492647055 mm required 406
Taylor substeps of degree 18: 7308 H8 actions. A read-only sample of the live
worker identified active CUDA FFT execution, rather than a deadlock or a new
memory failure. That trial was deliberately terminated to test the reviewed
spatial-pressure recovery. It is not a physical failure result and published
no new complete mixture. The last verified complete nine-mode checkpoint
remained 493.23529411764713 mm. The local termination receipt preserves that
distinction and the exact runner identity.

With v6 spatial pressure allocation, the next actual run completed all nine
original modes at 506.6176470588235 mm and stopped cooperatively at that complete
checkpoint. Total weight was `0.9984350643820183`; cumulative numerical support
probability loss was `1.2424097245540268e-5` and the mixed support L2 bound was
`0.005203045486576188`. Eight grids were 1540 square and E4 was 1960 square.
The full checkpoint, all original mode identities and every preceding ledger
prefix were independently verified before publication. These are intermediate
preview results, not completed specimen/screen propagation or a convergence
certificate.

Residual exponential selection now gives each method the original declared
half-step error allowance. The achieved Taylor remainder is retained as a
diagnostic and as the unchanged flux-check threshold, not imposed as a new
input tolerance on the Chebyshev planner. The selected method's actual series
certificate is recorded and accumulated by the existing geometry error log.
Historical error records are not recomputed. On the captured expensive plan,
the declared allowance was `3.337808257528363e-14` and the achieved Taylor bound
`8.758865964312184e-15`. The Chebyshev certificate was
`8.817396640941025e-15`, within the original declared allowance, with 1628 rather
than 7308 H8 actions and all 406 original sampling checks retained. Independent
matrix-exponential, geometry-continuation and actual CPU/CUDA regressions cover
this routing and accounting change. It does not bound axial splitting error or
establish a whole-column speedup.

Continuing from that complete checkpoint with the corrected exponential
allowance routing completed all nine original modes at 520 mm. All grids were
1944 square. Total weight was `0.998173638036137`; cumulative numerical support
loss was `0.00027385044310326395` and the mixed support L2 bound was
`0.028170267610324085`. The run stopped cooperatively after the complete plane
was saved. It retained the earlier `0.001 / 0.1` source allocations throughout;
the later `0.01 / 0.5` preset must start a separate branch from the original gun
output. These results cross the original resource bottleneck for the actual
nine-mode mixture, but still do not establish specimen or screen completion.

A separate CPU experiment propagated the same actual 376-square E4 entrance
from 481.764705882353 to 485.764705882353 mm with maximum steps of 0.5, 1 and
2 mm. The runs used 9, 5 and 3 boundary steps and took 28.58, 15.99 and 9.20 s.
Relative to the 0.5 mm run, raw complex relative L2 differences were `5.63935e-4`
and `4.68506e-3`; intensity relative L1 differences were `5.4703e-9` and
`4.55564e-8`. A common analytical carrier was factored exactly for comparison,
retaining every differential phase term rather than fitting away piston, tilt
or defocus. Executed electric phase was already present in the wave; the common
free axial phase was not counted twice. No new projection events were needed.
This short segment supports an explicitly approximate preview setting, not a
bound on cumulative phase error at the specimen or screen.

A small-field regression with injected resource and sampling failures exercises
a 128-square entrance, a proposed
96-square projection, lossless refinement to 120 square, and a rejected
140-square resource request. Retrying from the unchanged original entrance
selects 80 square and commits only that projection. Its raw complex output
matches a direct original-to-80 projection followed by the same real field operator
to within `2e-14`; no phase fitting, recentering or intensity repair is used.
Failure and cancellation cases retain the original immutable ledger. This is
a targeted retry validation rather than a default-FEG whole-column result.

That preview used public AC/Descan calibration for the actual fluorescent
screen from the outset. Its source-reuse receipt proves unchanged original
gun fields, all static contributors, source parameters and actual mechanical
boundaries. Only downstream compact scan fields changed; their support starts
after the gun exit. The resolved downstream field identity is recorded
separately, so this is not a claim that the entire calibrated column field is
identical to the older specimen-only branch.

E4 is an index in the source's Halton sampling order, not the central energy.
It has the lowest sampled emission energy, approximately 0.0666215 eV, and an
initial diffraction RMS angle of approximately 178.079 mrad. The other eight
sampled modes range from approximately 67.47 to 113.23 mrad. E4 already needs a
512-square gun-exit lattice while the other modes use 256 square. This source
distinction must be retained; it does not authorize discarding the costly mode.

Bounded GPU-reuse tests exercised actual CUDA complex128 fields on 128-, 256-
and 350-square meshes with a 256 MiB private hard cap. Reused and fresh GPU
outputs were bitwise equal; CPU comparisons, shape changes, live arrays,
cancellation, Require GPU failures and final release passed. A separate
0.01 mm continuation request was rejected because its distance-scaled nonzero
exponential allowance was below float64 precision; no result was accepted and
the gate was not relaxed.

In this archived instrument configuration the specimen plane is 1599.2 mm and
the main fluorescent-screen plane is 2714.15 mm. The straight-column terminal
bound near 2987.15 mm is not the fluorescent screen. Any screen qualification
must also execute the intervening inserted detectors and retain their physical
loss histories. The saved default specimen is vacuum; propagation through that
plane does not qualify a material-scattering calculation.

The complete default-source nine-mode route must be reported separately. Moving
the original 519 mm resource limit does not by itself establish whole-column
completion or permit weaker physical/sampling gates. For a fixed N-by-N lattice,
the main wave storage is proportional to N squared and a two-dimensional FFT
costs O(N squared log N); distance adds propagation steps. A growing adaptive
lattice, retained intermediate arrays and different source-mode bandwidths can
make actual resource use rise sharply. Segmentation bounds history retention,
but cannot alone bound the information in one physical transverse wave.

## Optional preview preparation at segment entrances

The exploratory preview can explicitly enable `column_preview_segment_resampling`.
This opt-in uses support policy v4, `preview_segment_and_retry`. The earlier v3
weighted policy remains resource-triggered only. At the exact beginning of each
declared column segment, the first residual may try a smaller Fourier grid under
the original one-percent per-event limits and remaining cumulative allowance.
No new allowance is issued at a segment, cache hit, material boundary or resume.

A pending spectral candidate that needs an edge window tries the existing
2/4/8/16-cell family, with every trial starting from the same raw projection and
ledger. Boundary, spectrum, coverage and memory checks still apply. Lossless
admission must retain at least a five-percent cell saving and may not enlarge
either axis beyond the original. The physical residual then runs on the private
candidate. Numerical rejection falls back to the original wave and ledger;
domain errors and cancellation remain explicit. Only a successful physical
residual commits the pending projection. Publication still requires a successful
whole mode segment, and a complete observation plane still requires all original
energy samples at the same native Z.

An already executed v3 branch requires an explicit forward policy-transition
receipt before adopting v4. The original arrays, absolute mode weights, field
and geometry definitions, numerical budgets and previously spent histories are
retained. Projection L2 accounting remains an input-representation bound, not a
pointwise phase tolerance or a certificate for all subsequent propagation.

Repeated boundary and spectrum checks may reuse successful scalar results only
for the identical CPU amplitude backed by immutable bytes and the exact same
threshold. The bounded cache holds weak references, never spectra or wave
arrays. Mutable arrays, GPU data and failed checks are not cached; lattice,
memory, coverage and cancellation checks still execute. Two complete small CPU
propagations retained bitwise-identical complex fields, phase carriers and
records while reducing FFT calls from 85 to 75. This is a local execution result,
not a whole-column speedup claim.

The separate `0.01 / 0.5` original-gun preview branch has now completed all nine
energy samples at 520 mm with total weight `0.9982029732586617`, numerical support
probability loss `0.00024451522058864305` and mixed support L2 bound
`0.024248281319728154`. A process exit without a Python error report interrupted
the first attempt; its cause is unknown. Exact cache replay reused the eight
already completed 520 mm modes and calculated the missing ninth mode, preserving
the original implementation, inputs, cache identity and every ledger prefix.
These remain intermediate results; specimen and screen completion require their
own complete checkpoints.

That exact-cache continuation subsequently saved all nine modes at the native
plane `0.5307142857142857 m` and stopped cooperatively. Total weight was
`0.9972720277028626`, numerical support probability loss
`0.0011754607763717487`, and mixed support L2 bound `0.06216110466975261`.
The next preview starts from those same 45 verified mode-array payloads through
an explicit v3-to-v4 policy transition. The existing `0.01 / 0.5` budgets and
all accepted histories are retained, rather than restarted at this plane.

An independent actual E0 comparison propagated the same 506 mm entrance to
507.75 mm with and without an admissible entrance projection. The completed
candidate used 1568-square instead of 2000-square storage, or 61.4656 percent
as many cells. Its absolute weight difference was `2.663978745764961e-6`,
matching the recorded projection cost to approximately `1.4e-16`. Because the
two physical Fourier periods differ, equal-array-index comparison is invalid.
Independent finite-domain quadrature evaluated both complete complex waves on
common SI coordinates, including their original phase carriers and zero outside
their respective domains. The relative raw complex L2 difference was about
`0.00487951`; intensity L1 difference was about `5.4236e-5` per reference
electron. Grid refinement and shifted quadrature agreed, and direct Fourier
summation independently checked the evaluator. No phase, centroid or intensity
fit was applied. This is a local single-mode preview comparison, not a
whole-column convergence certificate. Its timings were not an isolated
performance benchmark.

The v4 continuation later saved all nine original modes at 558.75 mm and
stopped cooperatively, with total weight `0.9902804452862601`, numerical
support loss `0.008167043192946838`, and mixed support L2 bound
`0.21056641994671413`. This remains an intermediate result, not a specimen or
screen result. Its original one-percent probability allowance is unchanged.

## Exploratory phase representation preparation

The independent `column_preview_phase_reselection` option uses support policy
v5. At a declared segment entrance it may move a resolved quadratic envelope
phase into the stored analytical curvature, retain the compensating complex
envelope, and propose a smaller grid. The preparation runs on CPU with at most
twice the propagation pixel ceiling and the unchanged host-memory ceiling.
The admitted physical propagation still obeys its original pixel limit.
The complete field, tilt, origin and absolute intensity are retained; this is
not subtraction of defocus or replacement by a Gaussian beam.

Finite Fourier interpolation after this representation change is not globally
exact merely because the compensating phase cancels at the original nodes.
The opt-in estimator compares complete complex fields on two denser physical
grids and a half-cell shifted grid. Receipt v2 uses a five-percent relative
quadrature criterion or an absolute tolerance of `1e-7 * sqrt(w * P)`, where
`w` is the actual mode weight and `P` is the input wave probability. The entire
absolute tolerance, maximum observed L2 discrepancy, probe spread and roundoff
floor are charged to an independent, explicitly uncertified estimate history.
Receipt v1 retains its earlier relative-only rule; old records are not
reinterpreted. Neither criterion is a propagated or continuous-field error
certificate.

The original strict projection ledger is unchanged. Every later accepted
projection must also leave room for the accumulated representation estimates
within the same original L2 allocation. Representation, spectral projection
and any boundary taper remain private until the actual first residual succeeds;
failure or cancellation cannot spend either history. Source bindings, complete
history prefixes and the distinction between estimates and bounds survive
checkpoint storage, material handoff and recording stops. This option is not
eligible for high-accuracy qualification.

On the actual E4 input at 541.428571 mm, preparation used a 3300-square
temporary grid and admitted a 1176-square candidate. The comparable candidate
without curvature reselection was 1680 square. An independent complete H8
physical step ended at 1500 square versus 2000 square, with no further support
events. A common-physical-coordinate comparison including both full carriers
gave relative raw complex L2 about `0.00698271`; this includes the two different
input projections, so it does not isolate the representation error. Separate
6144- and 8192-square representation checks observed weighted L2 discrepancies
of roughly `3.72e-8` to `3.92e-8`. The v2 preparation charged `1.05972591519e-7`
as an estimate in addition to its strict projection costs. These are local
preview comparisons, not whole-column convergence or timing guarantees.

Coefficient preparation also reuses the scalar potential already sampled on
the identical plane and avoids constructing known-zero dynamic-coil arrays
after the original domain checks. Independent old-path comparisons retained
bitwise-identical complete CPU waves and energy/clock histories. Controlled
512-square coefficient timings improved by approximately 1.21 to 1.57 times;
this is not a whole-column speed claim. Actual CUDA regression separately
checks the phase-preview complex field and both numerical histories against
CPU execution.

## Nominal emission energy preview

The independent `source_nominal_energy_preview` option executes one energy at
`electron_gun.emitter.emission_energy_ev`. This is the declared initial kinetic
energy before acceleration. It is distinct from
`electron_gun.emitter.energy_spread_fwhm_ev`, which specifies the full width at
half maximum of the configured emission energy distribution. The current cold
field gun happens to configure both values as `0.3 eV`; their equality does not
make them interchangeable. Regression coverage also uses a `2 eV` nominal
energy with an unchanged `0.3 eV` FWHM.

This explicit exploratory option requires one energy sample. It leaves the
physical gun parameters intact and labels the result as not integrating the
configured energy spread. It is not a claim of temporal-coherence convergence.
The ordinary energy-distribution calculation retains its validation against
silently replacing a finite spread by one sample. The new source has its own
cache identity and executes the actual accelerating boundary and gun. No old
energy-quadrature member is relabelled or reweighted to stand in for it.
