# Cold field emission gun apertures and electrical boundaries

Research checkpoint: 2026-11-05. Simulator baseline:
`04e1d5682dab6a5392b542f139f4efeeaccc33c5`.

The extractor already intercepts electrons outside its own opening. The separate
`feg_dpa_aperture` also intercepts electrons, but has no assigned electrical
potential in the current gun field model. These are different components. The
identity, electrical connection and potential of the separate gun aperture remain
unresolved; its name and location do not establish that it is an anode.

This checkpoint preserves the literature findings, simulator inspection and
completed diagnostics. It does not change the gun geometry, electrical model,
source, operating point or controls.

## Component identity and the corrected decision

| Component | Meaning in this investigation | Simulator component |
| --- | --- | --- |
| Extraction anode or extractor | The electrode immediately downstream of the emitter that provides the extraction field; its own bore also limits transmitted electrons | `feg_extractor` |
| Electrostatic gun lens | The focusing electrode downstream of the extractor | `feg_electrostatic_lens` |
| Separate gun aperture | The independently represented aperture after the gun lens in the current profile | `feg_dpa_aperture` |
| Intermediate or terminal anode | An electrode with a role and electrical connection specified by a particular gun design | No identity with `feg_dpa_aperture` has been established |

The earlier proposal to reinterpret the existing gun aperture directly as an
intermediate anode is withdrawn. A published FEI design contains an anode at a
similar relative location, but that is insufficient to identify the simulator's
strip as that electrode. Do not rename it, move it, ground it, or assign the FEI
example voltage solely on that analogy. Likewise, do not confuse the opening in
the extraction anode with the separate downstream gun aperture.

The current `Extractor Aperture` alias on `feg_dpa_aperture` is ambiguous and must
not be used as evidence of electrical or physical identity with `feg_extractor`.
It is retained unchanged in this documentation-only checkpoint.

## Book evidence

### Volume 1

In *Principles of Electron Optics*, second edition, Volume 1, Chapter 10,
printed pages 155-156, equation 10.45 and figure 10.12 describe a conducting
aperture with a defined potential and fringing fields. This supports including
the conductor in the electrical boundary; it does not specify a TEM gun
aperture's wiring. Figures 10.19-10.20, printed page 166, illustrate a cathode
field calculation; the caption specifies a 10 mm total width and a 2 mm
anode-cathode distance for that example. Those dimensions are not universal
commercial cold-FEG dimensions. [Volume 1 Chapter 10](https://doi.org/10.1016/B978-0-08-102256-6.00010-9)

### Volume 2

Volume 2, printed pages 1058-1059, figure 43.4 and section 43.3, distinguishes
the emitter, extraction electrode, electrostatic lens and anode. It does not
identify a separate post-lens strip matching the simulator. The book uses
"wehnelt" for the extraction/current-control electrode even in its cold-FEG
discussion; that terminology does not imply a negative thermionic Wehnelt bias.
[Volume 2 Chapter 43](https://doi.org/10.1016/B978-0-12-813369-9.00043-9)

Printed pages 1187-1189, section 50.1 and figure 50.1, separate extraction from
acceleration and focusing as an approximation. The aperture shown in both
subsystems belongs to the same extraction electrode, not an additional physical
plate. The approximation requires a sufficiently thin diaphragm, small aperture
and weak trajectory bending in the fringe regions. The subsequent lens system
depends on the particular gun design.
[Volume 2 Chapter 50](https://doi.org/10.1016/B978-0-12-813369-9.00050-6)

## Manufacturer designs and experimental sources

### FEI cold field emission gun design

[US20180323036A1](https://patents.google.com/patent/US20180323036A1/en),
figure 2 and paragraphs 0084-0090, shows CFEG source 4, extractor 9,
electrostatic lens 8b, downstream anode 11 and accelerating electrodes 13.
The illustrative positive potentials are:

| Element | Stated example |
| --- | --- |
| Extractor 9 | Approximately 4.5 kV |
| Electrostatic lens 8b | Adjustable, approximately 0.05-2 kV |
| Anode 11 | Approximately 4 kV |
| Accelerating electrodes 13 | Progressively approximately 7.5-60 kV |

The different potentials imply distinct electrical nodes for the lens and anode.
They do not establish separate power supplies. The cited passage does not state
the absolute reference to earth or the full supply/divider wiring. Figure 2 is
explicitly approximately to scale; it is not a manufacturing drawing. This
design also includes a permanent magnetic source lens. It establishes a possible
architecture, not the identity of the existing strip or a verified construction
of a named commercial X-CFEG/E-CFEG instrument.

### JEOL fixed gun aperture

[US20160013012A1](https://patents.google.com/patent/US20160013012A1/en),
figure 1 and paragraphs 0053-0056, places fixed gun aperture 117 downstream of
acceleration tube 114 and gun alignment coils 115 and 116. Its example opening
diameter is about 0.5 mm, with electron-limiting and gas-conductance functions.
The description permits several source types, including cold field emission.
It does not give aperture 117's electrical connection or potential. Neither its
location nor its diameter should be transferred to a different upstream plate.

### Hitachi cold cathode circuit

[EP2096661A1](https://patents.google.com/patent/EP2096661A1/en), figure 1,
shows the cathode, two extractor electrodes and the accelerating section.
Its circuit distinguishes the extraction supplies and a divider toward ground.
It is another complete electrical architecture, not evidence for an additional
independent strip between its second extractor and accelerator.

### Actual instruments and experiments

The [Thermo Fisher X-CFEG application note](https://documents.thermofisher.com/TFS-Assets/MSD/Application-Notes/xcfeg-application-note-an0171.pdf),
page 3, describes gun-lens settings from 1 to 8 with finer 0.1 steps. It does not
provide internal aperture wiring or a conversion of those settings to kV.
Do not interpret a gun-lens control number as an electrode voltage.

A [2025 experimental study using a modified JEOL cold field emitter](https://epub.uni-regensburg.de/77543/1/1-s2.0-S0304399125000579-main.pdf),
pages 2-3, identifies beam-defining apertures in the extractor and at the source
exit. It does not establish the position or voltage of a separate post-lens
strip matching this model. Published patents, commercial control documentation
and experimental apparatus descriptions have different evidential scopes.

## Current simulator implementation

Coordinates are measured from the tip, with positive Z downstream. The following
are model values from [FEG.toml](../../configs/instruments/gun/FEG.toml), not
measured OEM dimensions.

| Component | Axial geometry | Opening and electrical treatment |
| --- | --- | --- |
| Extractor | Z = 6-10 mm, centre 8 mm | 4 mm bore diameter; 4 kV relative to the tip; included in the electrode field boundary and bore interception |
| Electrostatic gun lens | Z = 14-22 mm, centre 18 mm | 8 mm bore; setting 1.2 kV relative to the extractor, hence 5.2 kV relative to the tip |
| Separate gun aperture | Mechanism Z = 23-25 mm; plate centre 24 mm; plate thickness 0.2 mm | Working diameter 4 mm; carrier bore 6 mm; electron interception, but no assigned electrical potential |
| Accelerator | Z = 30-370 mm | Ten configured electrode stages; 300 kV total acceleration setting |

The gun-lens voltage reference is explicitly a historical additive convention,
not an OEM calibration. In the current field gauge, tip potential rise is zero
and ground is the high-tension voltage above it. At 300 kV, a grounded conductor
must therefore receive an internal potential rise of 300000 V, not zero.

The separate aperture's placement records a user-defined relative order. Its
strip and screw/rod topology derives from an objective-aperture photo reference;
the metadata does not establish a cold-FEG construction, material grade or
electrical connection. The working opening, carrier bore and mechanism diameter
are different quantities.

Relevant implementation locations:

- [ExtractorElectrode and voltage conversion](../../src/temsim/optics/electron_gun/electrostatic.py):
  `effective_aperture_radius_mm` and `potential_rise_from_tip_v`.
- [Gun component membership](../../src/temsim/optics/electron_gun/field_emission.py):
  `bore_components` includes the extractor; the separate aperture is its own component.
- [Particle interception](../../src/temsim/optics/electron_gun/tracing.py):
  `_clip_body_bores` removes particles outside the bore within the component's
  axial extent and records the component key. `_resolve_aperture_crossing`
  separately handles the gun aperture. The bore helper checks accepted step
  positions; the small checks below do not qualify continuous wall-contact accuracy.
- [Closed gun field](../../src/temsim/physics/closed_gun_field.py):
  the request explicitly records `upstream_apertures` as
  `not_assigned_electrical_potentials`.

## Preserved numerical evidence

The [evidence manifest](../development/cold-feg-aperture-research-2026-11-05/records/manifest.json)
lists hashes and scope for the records below. Historical outputs are preserved
without changing their values or replacing them with a new operating point.

### Earlier default gun transport

The [saved summary](../development/cold-feg-aperture-research-2026-11-05/records/default-gun-summary.json),
[plane arrays](../development/cold-feg-aperture-research-2026-11-05/records/default-gun-planes.npz)
and [input snapshot](../development/cold-feg-aperture-research-2026-11-05/records/default-input-snapshot.json.gz)
retain a completed classical run at the baseline commit. It used 3000 original
positive-current source particles, a 10 microampere source, a 5 nm FWHM flat
source, 1 mrad angular sigma and 0.3 eV launch energy.

At the separate gun aperture, Z = 24 mm, the centred D95 was approximately
1.346 micrometres against a 4 mm opening. All 3000 particles survived to the
gun exit, with effective count approximately 3000 and unchanged 10 microampere
current. These are results for that model and input, not cold-FEG calibration
or evidence that a real aperture should pass the same fraction. No new transport
was run when saving this checkpoint. The original snapshot retains its original
instrument-root path; replay elsewhere requires an explicit relocation.

### Omitted conductor in the field request

The [conducting-aperture audit](../development/cold-feg-aperture-research-2026-11-05/records/conducting-aperture-audit.json)
changed aperture radius, thickness, axial position, X offset and enabled state
individually. All five changes left the field request unchanged. Its request
SHA-256 was `e26f42fd100dc49d4a856edea4423ae79cdce04847a97fdd866d306d6fe7ff4a`.

Sampling an existing matching field cache at R = 2.5 mm and Z = 23.9/24.1 mm
gave potential rises of 8523.7546/8778.7281 V, a difference of approximately
254.9735 V across the nominal plate thickness. These values diagnose the
omitted conductor; they are neither its correct assigned potential nor a
mesh-converged prediction for a conducting plate.

A synthetic annular boundary-array fixture confirmed that the existing boundary
builder can fix finite-thickness metal while leaving the bore open. Its 6 mm
outer radius and 5200 V value were test inputs, not proposed physical dimensions
or a recommended electrical connection. No field was solved for that fixture.

### Extractor clipping check

The [extractor check](../development/cold-feg-aperture-research-2026-11-05/records/extractor-clipping-check.json)
uses coordinate arrays and the production clipping helper, without source
emission or field propagation. At Z = 8 mm, R = 1.9 mm passes and R = 2.1 mm is
blocked with key `feg_extractor`. R = 2.1 mm at Z = 5 or 11 mm is outside the
extractor extent and is not removed by that component. This establishes active
geometric filtering, not a measured energy-filtering response.

The reproduction scripts are
[check_extractor_clipping.py](../development/cold-feg-aperture-research-2026-11-05/reproduce/check_extractor_clipping.py)
and [audit_conducting_aperture.py](../development/cold-feg-aperture-research-2026-11-05/reproduce/audit_conducting_aperture.py).
From the repository root on Windows, run:

```powershell
.\.venv\Scripts\python.exe docs/development/cold-feg-aperture-research-2026-11-05/reproduce/check_extractor_clipping.py --output outputs/extractor-clipping-check.json
.\.venv\Scripts\python.exe docs/development/cold-feg-aperture-research-2026-11-05/reproduce/audit_conducting_aperture.py --output outputs/conducting-aperture-audit.json
```

Both scripts refuse to overwrite an existing output. The second requires a
matching local field cache and fails without it; it never silently computes a
new field. The cache is not distributed with this checkpoint.

## Next implementation decisions

1. Identify the separate aperture's physical support, insulation and electrical
   connection in a specific reference architecture. Until then, preserve its
   identity and flag the missing electrical boundary; do not supply a guessed
   default potential or silently substitute an anode.
2. Represent conductor connection and voltage reference explicitly. An attached
   conductor inherits its electrical node. Grounding is a boundary condition;
   a physically floating conductor would require an appropriate charge constraint.
3. Make CAD, particle interception and field boundaries consume the same conductor
   geometry, including plate thickness and bore walls. Keep the open bore as
   vacuum. A two-dimensional axisymmetric approximation cannot exactly represent
   an offset strip or its asymmetric support.
4. Include conductor geometry, connection and potential in field-request/cache
   identity for each supported field provider. Apply placement constraints for
   the selected architecture rather than assuming every gun aperture occupies
   the same interval.
5. Expose electrical node, reference, resolved potential, dimensions and provenance
   in the controls. Preserve existing profiles and do not reinterpret historical
   gun-lens values or retune the source to make a beam fill an aperture.
6. Before accepting a changed model, check conductor equipotentials, voltage-gauge
   invariance, mesh convergence, shared geometry and conserved particle weights.
   New finite-source trials require at least 1000 original positive-current
   particles, normally 1024 or more, plus increased sampling for convergence.
   Any new coherent calculation remains subject to the repository CUDA policy.

Third-party books, articles and figure copies remain local reading material under
the [reference-copy policy](README.md). This checkpoint stores authored findings,
source links and simulator-generated evidence.
