# Final exact-Objective diffraction verification

The final Objective excitation was 29.223518226455024%. D = 13.589999609390503%, I = 8.532044964017839%, P1 = 32.1132572718703%, and P2 = 9.826874422815507%.

A fresh production unified Lorentz response used nine zero-current differential pencils at a 0.05 mm step bound. The source was the actual specimen upper surface, Z = 1025.1999975 mm; the virtual main-screen diffraction reference was Z = 2140.15 mm. Spatial increments were 1e-10 m and mechanical-slope increments 1e-7 rad. The canonical input conversion used the actual captured vector-potential gradient. The source emitter was guarded against invocation.

| Quantity | Final exact working point |
| --- | ---: |
| Canonical position-response norm | 8.392709698605317e-7 |
| Effective camera length | 49.999999174983316 mm |
| Relative error against 50 mm | 1.6500333765989694e-8 |
| Pencils reaching the reference | 9 / 9 |
| Emitter invocations | 0 |

The final exact point satisfies the differential criteria (position-response norm below 1e-3 and camera-length relative error below 3%). The independent 0.1/0.05 mm mesh checks and half-amplitude check in proposal.json belong to the nearby Objective value 29.2235172%, not to this exact final point. Their numerical residuals are preserved without relabeling them as measurements at the new excitation.

final-exact-objective.json contains the complete matrices, runtime/field/geometry receipt, exact controls, input identities and artifact hashes. final-exact-inputs.json.gz is the complete prepared snapshot, including archived input bytes; final-exact-input.toml is the matching explicit profile. final-override.json is the reusable C2-specific projector override with these final measurements and the qualification limits.

This is a zero-current on-axis differential reference. It does not provide a finite-source current, RMS, D95, finite-angle collection, specimen diffraction signal, scanning image, coherent resolution or experimental qualification. The complete finite-source incident-focus checks remain separate. No project source, global catalog or old projector evidence file was changed by this final observer.
