# Projector matching for the new C2 >=10 mrad study

A separate weak-C2 provisional calculation is recorded under weak-c2-provisional. Its upstream Objective is explicitly provisional; a final exact-point check remains separate. The previous study directory remains unchanged.

Copy upstream-controls.template.json to upstream-controls.json only after the final incident working point is known. Fill all four final pre-specimen strengths, set ready=true, and use a unique case_name. The observer requires the physical C2 aperture diameter to be 0.07 mm. It does not determine or qualify the finite-source alpha95 or focus.

propose_from_final_controls.py retains the earlier workflow: captured-potential on-axis Larmor proposal, D/I solve with P1/P2 held, then nine zero-current production unified pencils at 0.1 and 0.05 mm and a half-amplitude check. Source emission is guarded against invocation. Full prepared inputs, explicit profile, receipts, controls and hashes are recorded under the unique case directory. Existing result directories are not overwritten.

This remains a differential diffraction reference at the given Objective and specimen upper surface. The actual >=10 mrad finite-source distribution, detector acceptance, scattering images and coherent-wave performance require their own validation. No src or global configuration file is changed by this workflow.
