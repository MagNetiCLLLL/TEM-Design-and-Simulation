# Prescribed ideal magnetic-lens aberration study plan

Date: 2026-11-13. Scope: a prescribed ideal magnetic field, its natural spherical aberration, and the effect of an absorbing vacuum-tube restriction. This is not a material/pole-piece design, an installed-instrument calibration, a coherent-wave calculation, or a new microscope preset.

The numerical reference has been completed before this finite-source plan. The finite-source gates below are declared before inspecting finite-source study outputs. A failed gate will remain a failure or an unresolved result; source size, pupil, fit order, observation plane, or tolerances will not be changed silently to obtain a pass.

## 1. Field and independent coefficient reference

Use a single centred, axisymmetric field

`B0(z) = B_peak * exp[-z^2 / (2 sigma^2)]`

at 300000 eV. The field centre is z=0. The main grid has `B_peak = 0.15, 0.30, 0.60 T` and `sigma = 3, 6, 12 mm`: nine fields. The axial Gaussian is explicitly evaluated at all z; sigma is a field-width parameter, not a hard boundary. Do not truncate a nonzero fringe field at a nominal support face.

Construct the regular near-axis vacuum-series azimuthal potential. Obtain all magnetic components from the curl of that same potential. Baseline finite-ray transport uses A_phi through radial order 9. Compare orders 7 and 5 on the predeclared representatives. A finite series is divergence-free but retains a vacuum curl-B truncation residual; it is not a complete material magnetic-field solution.

`ideal_lens_reference.py` supplies the independent zero-current first/third-order Lorentz-ODE reference. It uses the first paraxial focus event, a baseline entrance at -8 sigma, the -10 sigma entrance comparison, and maximum steps sigma/32 and sigma/64. Its validity radius is `min(0.5 sigma, 0.004 m)`; this is a declared reference domain, not a measurement of a physical tube. It reports Cs, Cc, the first image plane, and the full signed pupil conversion. It does not fit clipped rays.

The image plane can lie inside the nonzero field; the strongest grid case has its first focus before z=0. This is a valid declared conjugacy, not a thin-lens focal length measured from a hard field edge. Record image z relative to the field centre.

The reference parameter `kappa = abs(q) B_peak sigma / (2p)` is dimensionless. Equal-kappa cases must agree in Cs/sigma, Cc/sigma, image_z/sigma, and image radial slope times sigma. The completed reference's step/tail/scaling checks establish numerical repeatability for this specified ODE, not experimental precision or finite-source qualification.

## 2. Original source and reuse contract

The baseline finite source is a uniform-area disk of radius 0.95 mm, monochromatic at 300000 eV, with total emitted current 1 microampere. Use at least 1024 original positive-current particles, then grow to 2048. Source IDs and samples are deterministic and nested. Any auxiliary zero-current chief probes are additional and do not count toward this population.

The entrance is -8 sigma. Initial mechanical slopes are the fixed matrix returned by `paraxial_focus_reference` for zero Larmor-frame slope at the nominal energy. This is the paraxial zero-Larmor convention. It must not be relabelled as an independently reinitialized exact full-potential canonical condition for every field order.

For one fixed field and source family, tube, radial-order, integration-step and source-growth comparisons preserve the same original positions, mechanical directions, energies, emission times, IDs and declared source boundaries. The added 1024 particles are genuinely transported. Compatible old trajectories, clocks and physical loss flags may be reused only after checking the relevant field, boundary, solver and numerical dependencies.

Across different B_peak/sigma fields, the uniform-disk spatial sampling, scalar energy/current and zero-Larmor convention remain fixed. The physical entrance z and its very small field-dependent mechanical slopes are explicitly different. Do not claim bitwise-identical complete mechanical sources across all nine fields.

Each published N-particle population has the declared 1/N source measure. Earlier raw 1024-particle records remain immutable; publishing the larger 2048-particle population does not rewrite their raw weights or restore intercepted current. Report emitted, surviving and effective particle counts separately. Every surviving or intercepted packet retains its original ID and its proper share of the emitted current.

## 3. Predeclared comparison families

### A. Fixed-input-disk field/tube grid

Use the 0.95 mm disk for all nine fields and tube inner diameters 1.5, 2.0 and 5.76 mm. Within each field, all three tube cases receive the identical complete emitted source. Changing the tube changes mechanical interception only; it does not change the prescribed Gaussian, its intrinsic coefficient reference, or the input source law.

The tube is an absorbing cylindrical boundary along the declared transport interval. Record real path interception, not an output-angle selection or a post hoc current renormalization. Do not assume that checking only the source or final plane proves clearance throughout the path.

### B. Matched 10 mrad reference-pupil comparison

At sigma=6 mm and the three B_peak values, deliberately change the source radius to

`R = 0.010 / abs(image_radial_slope_per_initial_position)`.

Here 10 mrad means the outward **paraxial pupil edge slope** 0.010 rad. It is not a claim that the observed current-weighted mechanical alpha95 equals 10 mrad. Report alpha95, angular RMS, maximum sampled mechanical angle and pupil definition separately. Use the 5.76 mm tube to admit the declared disk; any added clipped comparison must be labelled separately.

These are three explicitly different source-radius experiments. They are not silent refinements of the fixed 0.95 mm source. All other source inputs remain fixed and each family uses its own exact 1024-to-2048 prefix.

### C. Filled-tube comparison

At B_peak=0.30 T and sigma=6 mm, set `R = 0.95 * tube_radius` separately for the three tube IDs. These deliberately different disk radii distinguish pupil filling from changing the field itself. The 2.0 mm tube gives R=0.95 mm and may reuse its exactly compatible family-A result.

The 5.76 mm filled-tube case gives R=2.736 mm and an approximately 130 mrad paraxial edge slope. It extends beyond the existing 100 mrad single-lens finite-pupil benchmark declaration. Keep it as an explicitly larger-pupil exploration, retain any failures, and do not inherit the older benchmark's qualification or claim validated wave C5 coefficients.

## 4. Fixed image, native waist observations and numerical mesh

For each field/source family, the primary Cs reference plane and primary D95 comparison plane are the independent paraxial image plane. Never move this plane to the minimum of a clipped or energy-perturbed population.

Observe, without optical retuning, a predeclared through-focus window

`half_width = max(0.1 sigma, 2 * abs(Cs_reference_m) * alpha_edge_slope^2)`.

The edge slope is derived from the **emitted** family radius, not the transmitted pupil after clipping. Use 129 near-focus planes and 65 whole-transport observation planes, retaining exact entrance and paraxial image coordinates. Within one fixed-source field, all tube, count, step and order comparisons use the same observation coordinates and stop. Source-growth comparisons must also retain the same integration mesh/checkpoint policy, so source count cannot accidentally alter numerical segmentation.

Report native D95(z) and RMS-radius(z) minima separately, with actual neighbouring-plane brackets. Native best-focus observations are secondary; they do not replace the fixed-plane Cs comparison or justify autofocus. The ideal-lens study does not apply an installed Mini Condenser/Objective preset or move a physical specimen to a fitted waist.

## 5. Fitting and held-out validation

Keep the independent variational Cs as the intrinsic third-order reference. At its fixed image plane, use the declared outward paraxial pupil coordinates and the sign convention

`delta_r = -Cs * theta * abs(theta)^2 + Cc * theta * (delta_E/E0)`.

The finite source is monochromatic, so its measured spot does not test chromatic broadening. Cc is an optional independent energy-derivative diagnostic, not a measured monochromatic blur.

Assign training IDs by `radical_inverse(source_id + 1, base=5) < 0.5`; the disk radius/azimuth coordinates use bases 2/3. The held-out IDs are disjoint and never enter that model's fit. Thus 1024 particles supply approximately 512 training and 512 holdout trajectories; growth to 2048 appends both sets to approximately 1024 each. Report actual set counts rather than assuming exact halves.

The declared Cartesian fit is the existing fifteen-column `fit_wave_gradient` basis, including its fifth-order term and omitting C7. Add two axisymmetric geometric fits with angular powers [1,3,5] and [1,3,5,7] to diagnose higher-order contamination. Record all three fitted Cs estimates, conditions and holdout residuals. The Cartesian fit and [1,3,5,7] geometric fit have the coefficient comparison gates below; the lower [1,3,5] geometric fit is an additional contamination diagnostic. Do not claim that a fitted higher-order term is an independently qualified physical wave coefficient. Changing the declared fit basis after seeing a result requires a separately identified comparison.

Keep the image plane, initial rays, field and gauge fixed during each coefficient comparison. Do not remove defocus, recentre coordinates or adjust intensities to hide a discrepancy. A reported fitted linear term is a diagnostic, not a refocused experiment.

If the tube intercepts part of the declared pupil, label its fitted coefficient as a **censored-pupil diagnostic**. It cannot qualify a changed intrinsic Cs. Report the independent field-derived Cs unchanged and keep all original current losses visible.

## 6. Proposed numerical gates

These are proposed computational checks for this study, not manufacturer or experimental error bounds. A full-source run and a diagnostic fit can have different gate eligibility.

| Comparison | Proposed gate | Required interpretation |
| --- | --- | --- |
| Source growth | Exact prefix positions, mechanical directions, energies, times, IDs and boundaries | No recentering, energy rescaling, regenerated prefix, duplicate IDs or fabricated trajectories. Preserve earlier raw records. |
| Step refinement on a fixed source | Maximum paired final-position difference <= 1 nm; relative primary fixed-plane D95 difference <= 0.001 | Compare the same reference plane and retained original IDs; report membership changes instead of hiding them. |
| Spherical coefficient | Relative signed fitted-Cs difference from variational Cs <= 0.001 | Apply independently to both declared fits only when the full declared pupil is admitted. Finite higher-order contamination may fail this gate and must remain visible. |
| 1024 to 2048 source growth | Relative primary fixed-plane D95 difference <= 0.01; relative fitted-Cs difference <= 0.001 | Refit the enlarged training subset, retain the disjoint holdout, and report actual source/tube populations. A censored-pupil Cs comparison remains ineligible for intrinsic-Cs qualification. |
| Field order 7 to 9 | Maximum actual-path or occupied-radius field difference divided by B_peak <= 1e-4 | Evaluate at the same physical coordinates. Also report order-5 comparisons diagnostically. |
| Vacuum-series residual | Maximum sigma * norm(curl B) / B_peak <= 1e-3 | Evaluate on the actually occupied beam domain, not the full unused tube bore. Divergence-free construction alone does not prove this residual small. |

Relative differences name their reference denominator. For Cs, use the positive nonzero independent coefficient of the same field/reference conjugacy. Retain failed, unresolved, ineligible and passed states separately. A small holdout residual alone does not establish field-order, source, integration or coefficient convergence.

For field grids, include the occupied radial envelope of actual baseline trajectories and physically reached points up to interception. Compare the order-7 and order-9 fields on identical points, rather than comparing different trajectories and calling that a field error. Report the grid/actual-path sampling and observed maximum radius so a nominal large but unfilled tube does not create an unrelated failure or an artificial pass.

## 7. Representative refinements and reproducibility

Use the 0.30 T, 6 mm case as the central representative, the weak 0.15 T, 3 mm case as the long-focus/large-Cs representative, and the 0.60 T, 12 mm case as the strongest dimensionless field representative. Include the central-field filled 5.76 mm tube as the largest occupied-pupil representative. Compare radial orders 5/7/9 and a halved axial step with fixed source, boundary, planes and stop. Additional refinements prompted by a failed check are new labelled numerical cases; their original failures remain in the record.

Record every request, source schema/count, particle ID/weight array, field parameters/order/domain/gauge, tube geometry, reference conjugacy, actual integration mesh, observation coordinates, solver implementation, helper bytes, elapsed time and physical loss ledger. Hash source/field inputs and raw output arrays or files. Reuse must be supported by exact compatibility receipts, not a matching display label. Run the study serially with bounded CPU allocation; no coherent calculation is included.

## 8. Interpretation limits

Clipping can reduce transmitted D95 while leaving the prescribed field's intrinsic Cs unchanged. It can also bias a censored finite-pupil fit. Neither effect is evidence that a smaller tube corrected the field.

A larger Cs does not by itself imply a larger observed blur when the entrance radius is fixed: changing field strength or width also changes the pupil angle. Report Cs, actual mechanical angular statistics and finite-ray residuals together; the matched-pupil family provides a separate comparison.

The prescribed ideal magnetic field produces natural nonlinear ray aberrations without requiring pole material or machining errors. The present study measures that model. It does not predict how a real pole bore changes B0, establish a material magnetic-circuit solution, qualify coherent resolution, or prove sampling convergence beyond the explicitly tested original populations.
