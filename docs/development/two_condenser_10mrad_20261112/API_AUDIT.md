# C2 convergence API and GUI audit for the >=10 mrad / 70 um study

This audit made no source-particle calculation. The new physical incident study is owned by the separate >=1000-original-particle workflow.

## Aperture handling

The ordinary C2 stored-preset application reads the C2-specific aperture table. AssemblyPanel displays that table's diameter rather than a fixed 100 um value. ContinuousApertureComponent stores a continuous radius and exposes diameter_mm/diameter_um; 70 um is representable. The generic current/aperture gate accepts 20-250 um, so it does not reject 70 um.

A genuine entry-point defect was found in seed_illumination: its historical 120 um exploration pupil applied to every Nanoprobe, including physical C1+C2 hardware. The narrow fix now applies that historical override only to installed three-condenser hardware. True C2 keeps its own preset diameter; the calibrator's later explicit c2_aperture_mm parameter still takes precedence. C3 and an installed-but-off C3 retain the former 120 um seed behavior. The five new configuration-only cases guard the emitter against invocation; the complete test_assembly_illumination.py module passes all 41 cases.

## Existing convergence-control limits

The generic Direct Alignment condenser device list is C2+C3. For physical C1+C2 hardware its corresponding GUI control is deliberately hidden, rather than inventing an absent C3. The production Direct Alignment API also explicitly rejects unqualified unified-field calibration. A C2-specific live convergence control has not been qualified by this study; this does not block application of the independent stored C2 preset or low-level C2/Objective editing.

The separate generic calibrate_illumination API still reads the historical 30 mrad Nanoprobe target and defaults to 193 particles (its old lower bound is 32). It must not be used through those defaults for the present >=10 mrad / >=1000-original-particle physical trials. The dedicated physical study driver enforces its own original-particle budget. These unrelated API-policy defaults were intentionally left unchanged in this narrow source fix.

The old catalog's reference_c2_aperture_diameter_um=100 is a reference for the C2+C3 historical path, not a 100 um hardware clamp. The warm-seed observer reads the actual current aperture. The 100 entries in the alignment-constraint table are percent excitation bounds, and preferred_current_pa=100 is a current setpoint; neither is an aperture-diameter default.

## Source freeze and validation

The one source modification is the C2-versus-three-condenser condition in src/temsim/optics/assembly_illumination.py. Solver source identity after the change is 8862d211bd0ee8fcd6c07c3d744fb20b900667c1f91f84eda5c23cc0bea56dc5. No other source or global configuration change was made by this audit. Source remains frozen after this fix.

Validation: tests/test_assembly_illumination.py, 41 passed, no failures or skips. Report: seed-regressions.xml.

## Deferred projector rematch

The isolated projector workflow is prepared in projector/propose_from_final_controls.py. It requires final upstream controls in upstream-controls.json and a unique case name. It will use a guarded zero-current on-axis proposal plus production unified differential checks; it will not qualify finite-source alpha95, finite-angle detector acceptance or wave performance. No new projector calculation has started before the final incident working point is available.
