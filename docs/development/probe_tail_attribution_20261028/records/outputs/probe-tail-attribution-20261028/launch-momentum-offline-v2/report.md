# Launch momentum and gun-exit phase: offline diagnostic

The saved gun-exit phase is substantially better described by initial position plus kinetic transverse momentum than by initial position plus launch slope alone. This is explanatory evidence for the importance of the prescribed launch distribution, not proof of a gun chromatic-aberration coefficient or of numerical convergence.

## Definitions and source contract

- `angular_rms_mrad = 1` is the each-axis local slope Gaussian scale at emission, before the radial 3 mrad cutoff. It is not the divergence at the fully accelerated gun exit (emitter.py:428-432; parameter_registry.py:136).
- `virtual_source_fwhm_nm = 5` is the projected physical emission footprint; it is not an independently measured or positioned virtual source (emitter.py:424-427; parameter_registry.py:135).
- Launch kinetic energy is 0.3 eV plus the prescribed per-particle energy offset. Gun tracing explicitly initializes mechanical momentum from this energy and the emission direction (tracing.py:187-194,222-223).
- Relativistic mechanical momentum magnitude is p(K) = sqrt((K e)(K e + 2 m c^2))/c. The fit uses px/p_ref and py/p_ref, where p_ref is fixed at 300000.3 eV for both cases. No vector potential was evaluated and no canonical-momentum assertion is made.
- The finite sample has launch-energy RMS 0.127398270 eV, 42.4661% of its 0.3 eV mean. The configured 0.3 eV FWHM field is RMS-equivalent, not a measured spectral FWHM (chromatic.py:66-75; parameter_registry.py:138-139).

## Matched particle 306

Its launch energy changes from 0.553124768751 eV to 0.3 eV with exactly the same initial position and direction. The exact relativistic launch-momentum ratio is 1.35784744571 (sqrt-energy ratio 1.35784727756). The observed gun-exit radius ratio is 1.36111099945; the slope-norm ratio is 1.37570679491.

The close ratios are consistent with strong launch-momentum scaling. They do not prove an ideal uniform accelerator: finite source position, focusing fields, nonlinear response, and numerical error remain in the actual trajectories.

## Deterministic held-out comparison

Source IDs with id % 5 == 1 were held out before fitting. Both partners of each poly/mono pair stay in the same split: 615 training IDs (1230 rows) and 154 held-out IDs (308 rows). ID306 is held out. Both models have exactly five coefficients per output: intercept and four inputs. Only training rows determine standardization and least-squares coefficients. No energy label or higher-order term was added.

| Gun-exit target | Angle-model held-out RMS error | Momentum-model held-out RMS error | Momentum-model R² |
|---|---:|---:|---:|
| gun_exit_x_m | 57.2631458 nm | 0.328231709 nm | 0.999999103808 |
| gun_exit_y_m | 55.119749 nm | 0.314485322 nm | 0.999999160159 |
| gun_exit_tx_rad | 0.0310725965 urad | 0.00103315974 urad | 0.999968668544 |
| gun_exit_ty_rad | 0.0299072109 urad | 0.000994145918 urad | 0.999970249348 |

ID306-specific errors (this particle was held out):

| Model | Case | Position-vector prediction error (nm) | Maximum position-component error (nm) | Slope-vector prediction error (urad) |
|---|---|---:|---:|---:|
| angle | poly | 221.424231 | 165.192408 | 0.122383333 |
| angle | mono | 8.81982465 | 6.64031599 | 0.0042196794 |
| momentum | poly | 1.87190248 | 1.40436602 | 0.00585477551 |
| momentum | mono | 0.326485175 | 0.243901232 | 0.000843421284 |

Training design condition numbers are approximately 1.03. Per-case errors, coefficients, training-only scales and all predictions are retained in report.json / arrays.npz.

## Tail shape versus RMS area

At sub-eV energies, p² is approximately 2mK. If direction and energy are independent and mean K is fixed, E[p_perp²] is approximately unchanged by energy spread alone. Its distribution can nevertheless become a scale mixture with heavier tails. It would be incorrect to infer increased RMS emittance merely from a larger energy spread.

| Axis | Mono kinetic normalized RMS area (m) | Poly kinetic normalized RMS area (m) | Poly/mono change | Mono p kurtosis | Poly p kurtosis |
|---|---:|---:|---:|---:|---:|
| x | 2.17704966e-15 | 2.17587831e-15 | -0.0538043% | 2.78564336 | 3.17082076 |
| y | 2.16476192e-15 | 2.15981786e-15 | -0.228388% | 2.76791811 | 3.1518646 |

These are mechanical momentum covariance areas at emission, not canonical invariants or a source-brightness calibration. The small RMS differences include the actual finite quadrature correlations. The ideal longitudinal-acceleration calculation in the JSON conserves transverse mechanical momentum only as a stated kinematic reference; it is not a replacement propagation.

## Existing native linearization support (read-only inspection)

- `unified_field_particle.py:447` provides `trace_unified_response`, accepting explicit phase, kinetic energy and flight time and executing the shared complete fields. With no coil increments it changes no physical drive. Its result contains actual A, mechanics and field identities. No variational ODE solver was found in the inspected unified path.
- `first_order.py:37,319` already builds a nine-ray central-difference response. The unified dispatch in `core.py:611-615` retains the full captured field model, but the public first-order helper uses a zero reference and nominal energy, so it does not automatically represent the actual saved off-axis source.
- A future isolated test could derive a local Jacobian at a declared mathematical reference with fixed K and h/h2 central differences, then compare its offline first-order predictions with the already executed constant-K source. Reusing the exact donor plan avoids a possible m/mm round-trip mesh change. Clipping, original weights and the nonlinear full-source results must remain authoritative.
- This would measure a local-linear remainder, not uniquely identify spherical aberration. No fitted surrogate, reference probe or linearized model was propagated in this analysis.

## Limits and reproducibility

No field was constructed or sampled; no gun or column propagation was run. Source reconstruction matches the saved emission positions, directions, normals, IDs and absolute weights exactly. The source law is prescribed and does not constitute a self-consistent microscopic emission or space-charge calculation. Numerical gun refinement is a separate experiment. A good linear fit cannot separate all energy-dependent focusing from nonlinear fields or discretization errors.

The JSON records SHA256 hashes of both input snapshots, raw particles, reports and governing source files. arrays.npz preserves launch energies, kinetic momenta, actual exit phase, model features, predictions and the held-out mask. The accompanying analysis-source.py is the executable audit. No raw source arrays, optics or results were modified.
