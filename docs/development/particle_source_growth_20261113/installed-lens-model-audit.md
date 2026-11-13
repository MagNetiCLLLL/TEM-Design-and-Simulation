# Installed two-condenser round-lens static audit

Scope: frozen focus inputs, implementation b2900cf7e79501458a6c463558ef134002bce351694617833ca8d65a111871e8. Eight installed round-lens owners; the objective has two field lobes. No new particle emission, Lorentz transport, coherent calculation, FEM solve, Cs experiment, or optical retuning was performed.

The 1%-of-peak interval is a display definition. Gaussian all-Z tails are evaluated outside it and outside native seven-sigma support. Neither interval is a hard physical or numerical field cutoff. Peak fields below are the static on-axis current model, not measured or OEM-calibrated fields.

| Lens | Axis centre (mm) | Main bore / clear passage (mm) | Pole body bore (mm) | Declared vacuum passage (mm) | Coil inner diameter (mm) | On-axis absolute peak (T) | 1% interval (mm) | Native 7-sigma support (mm) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| C1 lens | 500.00000000 | 20 | 19.2 | 20 | 96 | 0.931782565838 | 479.115726 to 520.884274 | 447.857143 to 552.142857 |
| C2 lens | 650.00000000 | 20 | 19.2 | 20 | 96 | 0.218397747512 | 613.452521 to 686.547479 | 558.750000 to 741.250000 |
| Mini Condenser lens | 948.50000000 | 20 | 20 | 20 | 34 | 0.0412692008086 | 925.109613 to 971.890387 | 890.100000 to 1006.900000 |
| Objective lens total | 1025.20000000 | 5.76 | 5.76 | 5.76 | 60 | 0.473008011604 | 954.911426 to 1095.488574 | 848.650000 to 1201.750000 |
| Objective lens upper lobe | 1023.85000000 | 5.76 | 5.76 | 5.76 | shared 60 | 0.237600687511 | 953.678840 to 1094.021160 | 848.650000 to 1199.050000 |
| Objective lens lower lobe | 1026.55000000 | 5.76 | 5.76 | 5.76 | shared 60 | 0.237600687511 | 956.378840 to 1096.721160 | 851.350000 to 1201.750000 |
| Diffraction lens | 1322.90000000 | 20 | 21.5 | 20.0 | 60 | 0.0934940821652 | 1287.814420 to 1357.985580 | 1235.300000 to 1410.500000 |
| Intermediate lens | 1492.90000000 | 20 | 21.5 | 20.0 | 60 | 0.0648786840215 | 1449.043025 to 1536.756975 | 1383.400000 to 1602.400000 |
| Projector 1 lens | 1672.90000000 | 20 | 21.5 | 20.0 | 60 | 0.123693589877 | 1620.271630 to 1725.528370 | 1541.500000 to 1804.300000 |
| Projector 2 lens | 1875.40000000 | 20 | 21.5 | 20.0 | 60 | 0.0324437803101 | 1816.924033 to 1933.875967 | 1729.400000 to 2021.400000 |

C1/C2 additionally declare 5.76 mm vacuum passage on the shared cartridge and adjacent pole records. Their 19.2 mm `mechanical_bore_diameter_mm` is the configured pole-body hole, distinct from the main 20 mm lens bore and the declared vacuum envelope. Those fields must not be treated as interchangeable measurements. Post-objective pole-body bores are 21.5 mm while the configured clear passage is 20 mm. Objective upper/lower physical pole centres are 991.5/1058.9 mm, distinct from their Gaussian field centres 1023.85/1026.55 mm.

The JSON inventory retains Gaussian width parameters, amplitudes, offsets, standard deviations, FWHM, excitation/polarity, geometry intervals, and every named bore field.

## Actual force model

`unified_field_runtime.py:163-193` captures the installed analytical providers and wraps each frozen round field in `HighOrderRoundLensPotential(radial_order=3, axial_domain="gaussian_all_z")`. The aligned-plane/native sampling paths retain this order; the initial first-order adapter metadata does not describe the executed unified field.

For local cylindrical radius r and current on-axis Gaussian B0(z), the executed round potential is A_phi = r B0 / 2 - r^3 B0_second / 16. Its curl gives B_z = B0 - r^2 B0_second / 4 and B_r = -r B0_first / 2 + r^3 B0_third / 16. Divergence of B is identically zero. The remaining vacuum curl residual is (curl B)_phi = r^3 B0_fourth / 16; a truncated near-axis series is not a complete current-free vacuum solution or a reconstruction of pole geometry.

Relevant code: `src/temsim/physics/unified_field_runtime.py:181`, `high_order_round_lens.py:197-225,260-270`, `unified_potential_sampler.py:63-89`, `potential_native_sampler.py:79-86,361`, and `unified_field_particle.py:402-427`. The actual particle derivative is relativistic Lorentz force including current E/B, actual kinetic energy, flight time, and declared mechanical interception; it is not an empirical Cs kick.

All current `lens_field_map_descriptors` are empty. The round provider is `GeometryAwareAnalyticFieldProvider`, status `provisional_geometry_bound_analytic_not_fem`. Gaussian amplitudes/widths are prescribed engineering parameters. Field normalization is only enabled for C1. No iron geometry, coil NI, nonlinear B-H curve, saturation, or hysteresis is solved in these current B0 profiles. The gun electrostatic source solver is a separate retained axisymmetric electrode/Laplace path.

## Detached diameter comparison

Existing geometry validation accepted C2 main bore 20 to 25 mm and coil inner diameter 96 to 120 mm at unchanged lens drives, polarity, and absolute positions. Across twenty identical interior XYZ points, both A and B were bitwise equal to the original arrays. Geometry identity and the initial geometry-bound capture identity changed. The production high-order numerical potential identity changed for the main-bore edit because its admitted radial domain changed; the coil edit left that numerical identity unchanged.

The corresponding minus-25% candidates were rejected by existing geometry validation: the effective C2 aperture would not fit the main bore, and the smaller coil would overlap the shared cartridge. No invalid candidate was forced to run. These static observations establish model coupling and cache semantics only; they do not measure a change or absence of change in physical Cs.

Receipts: `static-lens-diameter-comparison/report.json`, `static-lens-diameter-comparison/field-samples.npz`, and `installed-lens-static-inventory.json`. Original frozen inputs and live project definitions were not modified.

Actual pole-body trials were also checked through the ordinary geometry validator. Single C2 pole-bore edits violated the paired shared-interface constraints. Single Objective pole edits violated its paired-pole constraints, and coupled C1/C2 pole-only edits still violated cartridge/tube constraints. These rejected inputs produced no field or Cs result. Their receipts are in the separate `static-lens-diameter-comparison-actual-pole`, `static-lens-diameter-comparison-actual-objective-pole`, and `static-lens-diameter-comparison-shared-interface-pole` directories. A valid pole-bore study needs a consistent linked geometry, not a bypass of these checks.

## Relation to a real magnetic lens

The pole bore, magnetic gap and pole shape determine a real lens field together with coil ampere-turns and material magnetization. The main housing bore, coil inner diameter, pole-body hole, vacuum liner and aperture plate opening are different quantities. In particular, the 70 um C2 aperture is not a 20 mm lens bore.

The current axial Gaussian parameters are engineering assumptions. Some C1/C2 dimensions and field widths were rescaled while preserving the isolated integral of B0 squared. That preserves an isolated first-order focusing quantity; it does not determine the field derivatives, spherical aberration or an OEM working point. The present field model does not solve magnetic saturation, hysteresis or the mutual nonlinear magnetization of a shared yoke. Radial series convergence is distinct from particle-count and RK4-step convergence. No measured-field or independent FEM comparison currently establishes a percentage error for this working point.

The gun electric field is separate: this accepted planar cold-FEG uses the closed axisymmetric electrode/grounded-liner Laplace/P1 scalar potential path. Its existence does not make the lens Gaussian magnetic profiles a material FEM solution.

## Available alternatives and a controlled bore study

| Method | Suitable use | Inputs or limitations |
|---|---|---|
| Existing project axisymmetric nonlinear A_phi FEM | First controlled pole-bore/Cs study, including saturation | Actual C2 currently has eight missing recipes: assign coil NI, materials/B-H, mesh and exterior domain; solve paired Objective and shared circuits consistently. No magnetic hysteresis model. |
| Independent 2D axisymmetric FEM, for example FEMM | Independent magnetostatic comparison of the same geometry | Nonlinear materials supported; mesh and outer-boundary convergence remain necessary. |
| 3D nonlinear FEM, for example COMSOL or Elmer | Non-axisymmetric pole geometry, alignment errors, slots and multipoles | Higher setup/memory cost; requires explicit 3D geometry and material data. |
| 3D integral magnetostatics, for example ESRF Radia | Alternative open-region treatment of current coils and nonlinear magnetic bodies | Segmentation and interaction-matrix convergence required; not an electron-optical calibration by itself. |
| Measured or independently solved field/potential | Calibrate the existing instrument instead of predicting unknown geometry | A map does not automatically adapt to a changed bore. The present unified pipeline needs retained magnetic potential rather than an arbitrary B-only table. |

For each valid bore geometry, report both fixed-NI behavior and a refitted-NI comparison at matched energy, conjugate planes, focus and convergence angle. Measure Cs with a declared pupil/reference family and held-out ray tests; report higher-order fit residuals. Then compare the complete 0.3 eV finite-source probe at the unchanged specimen entrance with at least the qualified source population, retaining absolute current and physical interception. A reduction of D95 caused by clipping is not evidence of a lower field-derived Cs. Preserve crossover ordering and apply the existing focus tolerance rather than moving the specimen to the fitted focus.

Primary references consulted for method selection:

- [JEOL objective-lens description](https://www.jeol.com/words/emterms/20121023.060558.php): pole gap and bore, field shape, and resolution tradeoffs.
- [FEMM manual](https://www.femm.info/Archives/doc/manual.pdf): axisymmetric linear/nonlinear magnetostatics and boundary treatment.
- [COMSOL magnetic-lens example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.particle.magnetic_lens/magnetic_lens.html): magnetic field and charged-particle tracing.
- [ESRF Radia introduction](https://www.esrf.fr/home/Accelerators/instrumentation--equipment/Software/Radia/Documentation/Introduction.html): three-dimensional nonlinear integral magnetostatics.
- [EOD electron-optical design](https://www.lencova.com/index.php/eod): independent FEM and aberration-computation workflow.

True pole-body edits were also attempted through the existing editor. C2 upper-pole bore 19.2 to 24.0 or 14.4 mm was rejected because the shared C1/C2 interface poles must match. Objective upper-only and lower-only bore 5.76 to 7.2 mm were rejected because the symmetric pole pieces must match. A linked C1-lower/C2-upper bore change to 24.0 or 14.4 mm was rejected by the fixed continuous vacuum-tube/shared-cartridge constraint. No invalid pole geometry was used to sample a candidate field. Thus the direct bitwise field comparison applies to the accepted main-bore and coil-inner-diameter edits; the pole-body uncoupled-field conclusion is a code/model deduction, not a completed legal pole-body experiment. Separate rejection receipts are retained in `static-lens-diameter-comparison-actual-pole`, `static-lens-diameter-comparison-actual-objective-pole`, and `static-lens-diameter-comparison-shared-interface-pole`.
