# CAD detail levels and evidence rules

Established 2026-10-11; extended 2026-10-18 with Probe Corrector deflector laminates and rigid-carrier rules for ordinary deflectors. Applies to STEP and native software 3D models of existing parts.

The instrument uses a **functional material-block** level of detail: supported pole pieces/pins, bulk winding packs, yokes, electrode plates, and insulation sections can be displayed separately. No part models individual wires, threads, surface lettering, or unconfigured cooling tubes, fasteners, or seals. STEP retains analytical curved solids; within the same software view, all parts use the same angular tessellation setting, lighting, and material-category colors.

A consistent geometric detail level does not mean all parts have equally accurate real dimensions. Existing configurations, patent embodiments, and qualitative structural diagrams must be recorded separately; decorative solids must not make poorly supported components appear “equally precise.”

## Admission criteria for internal solids

1. Retain the identity and assembly relationships of material subparts, actual electrode surfaces, working openings, and physical coil centers explicitly listed in the original TOML.
2. If manufacturer information/original papers directly support a structural family matching the current component, simplified internals at the same detail level may be built. Dimensions that remain configuration values or allocations within an envelope must be explicitly labeled as reconstruction.
3. An optical channel name, harmonic order, or alternative patent scheme alone cannot justify adding poles, coils, brackets, yokes, or housings.
4. Real components with unresolved internal structures retain their original outer envelopes, labeled `configured_envelope`. The volume of this closed placeholder geometry is not actual material volume and must not be used for mass, thermal, or field calculations.
5. Pure control channels, virtual optical planes, and zero-thickness detector active surfaces do not receive invented finite-thickness solids.

## Treatment by category in this round

| Category | Retained evidence | Details not added |
| --- | --- | --- |
| Round lenses/objective/mini condenser | Original material subparts and sections; lens patents support the family of pole pieces, bulk coils, and return yokes | No invented winding counts, cooling grooves, or unknown fasteners; virtual upper/lower objectives do not become extra hardware |
| Main probe corrector HP1/HP2 | Müller 2006 text and Fig.3 directly describe six pole pins, coils, and a yoke | Do not transplant its 152 mm OD, 30 mm pole length, or 4 mm bore radius; do not add unconfirmed housings or insulating supports |
| DPH/DP | Originally configured coaxial, dual-direction saddle winding packs and axial/radial relationships; the CEOS coil patent supports flexible carriers, radial insulation covers, and interlayer bonding, represented as `bonded_flexible_pcb_equivalent` | Do not duplicate drive channels as hardware or infer rigid formers, screws, clamps, or host fasteners; do not claim the patent is the winding drawing of a specific model |
| Stigmators with 8 declared coils | `coil_count` and `coil_topology` both supplied in the original configuration | 8 coils do not imply 8 iron cores or pole pieces; undeclared topologies remain envelopes |
| Ordinary deflector/scan pair | Declared physical coil centers, lengths, and bulk X/Y winding packs; four hosts explicitly selecting `rigid_insulating_carriers_pinned` in C3 + Probe Corrector gain sourced rigid insulating carriers, spacer sleeves, insulating housings, and locating pins | Do not infer OEM structures from optical names or automatically extend them to other configurations; no unknown magnetic cores, external mounting parts, or duplicate Descan hardware |
| Accelerator | Electrodes with defined thicknesses and patent-supported interstage insulation sections | No entrance/exit support rings; thermionic modules without electrode thicknesses retain only envelopes |
| Electrostatic blanker/quadrupole electrodes | Explicit electrode functions, gaps, and lengths | No unconfirmed insulating carriers; do not replace them with magnetic coils |
| Wien and filter multipole carriers | Original envelopes/branch positions | A 12-pole patent or optical multipole channel is insufficient to assign a physical pole count to this instrument |
| Apertures/slits | Original thicknesses, supplied working openings, and positions | Unknown working openings are not replaced with vacuum bores; a two-blade carrier without a working slit width does not receive arbitrary blades |

## Comparison with the original TOML

The comparison baseline is the saved assembly parameters before CAD integration in this round, including shared Tip and subassembly dependencies; earlier Git versions are not substituted for previously completed optical adjustments. Original text and SHA are saved for comparison, and the same parsed document is passed through both the original geometry path and the new CAD path.

Check part identities, parent/child relationships, order, axial centers, shapes, openings, and overall installation transforms. List added internal solids and removed unknown details separately; “unchanged outline” must not become “identical internal geometry.” Energy filters separately validate bending coordinates and full-column installation transforms. Comparison figures use the same camera, scale, and section.

This check does not change particle/wave inputs or substitute CAD graphics for electromagnetic-field calculations.

## Passive laminates in Probe Corrector deflectors

The auxiliary-structure update of 2026-10-18 applies only to DPH1/DPH2 and DP11/DP21/DP22. It retains the existing four copper winding packs and adds two substrate carriers, two radial cover layers, and one interlayer adhesive region per component. See [Probe Corrector internal 3D structure](cad-probe-corrector.md) for evidence, dimensions, and limitations. Ordinary TEM deflector/scan pairs do not automatically inherit this structure; their references and hosts require separate review.

The additions retain TOML parameters, `SolidSpec`, unified material categories, native 3D, and analytical STEP formats. This is a laminate equivalent at functional material-block detail, not a turn-by-turn PCB manufacturing file. Cover rings represent radial cover surfaces only and do not establish complete insulation of conductor sidewalls, lead exits, or seams. Polymer and adhesive types, insulation performance, and bond strength are unspecified; graphical material categories are not performance calibrations.

Interlayer bonding can represent a literature-supported method of relative fixation; attachment of the insert to the actual lens, hexapole, or vacuum tube remains unconfirmed. New adhesive layers must not be used to infer OEM host mounting, nor may shoulders, clamps, or fasteners be added for decorative completeness. New geometry requires separate checks of solid contacts, absence of volumetric interpenetration, actual material bores, and host clearances; pre-addition solid counts and validation results cannot be reused.

## Rigid carriers for ordinary two-stage deflectors

The four Condenser, BSh/BTlt, AC Scan, and Image/Diffraction hosts in C3 + Probe Corrector each select the same evidence-supported rigid-carrier structural family. See the [ordinary TEM deflector structure record](cad-multipoles.md#ordinary-tem-deflectors-rigid-carrier-update-2026-10-18) for specific evidence, dimension allocations, and comparisons. This is an explicitly labeled engineering equivalent; it does not confirm identical OEM internals for all four groups or force different structures merely because their functions differ.

Each axial station retains four copper winding packs and gains two insulating carriers with copper-shaped recesses and continuous radial backing, an insulating spacer sleeve, an insulating housing, and two insulating locating pins with mating round holes. The housing uses the insulating scheme explicitly described in the source; ferrite cores or flexible PCB laminates from other schemes are not appended. The locating pins fix the carriers relative to each other, not the complete assembly to its microscope host. Radial backing, pin-hole azimuths, and all millimeter dimensions are engineering allocations within the envelope, not manufacturer measurements.

The original configuration still defines both axial field centers, existing copper solids, and effective field parameters; the additions do not extend axial allocations or add field sources. Descan continues to share the Image/Diffraction solids; pure control records must not create a fifth material group. More detailed fixation does not mean terminals, turn-by-turn insulation, full-column axial mounting, material grades, or magnetic-field calibration are resolved; these omissions must remain explicit. New solids must pass the same native-mesh, analytical-STEP, save/reload, and assembly-clearance checks; old-model regression results are not substitutes.

## 2026-10-21 downstream Probe component refinements

HP1 reuses HP2's declared generic multipole winding-former/bonding structural family but is generated from HP1's own window and core dimensions; no OEM former is inferred from CEOS photographs. TL21/TL12 use generic insulating end-plate structures specific to their interfaces. DP22, DP21, and DP11 receive edge sealing only at the end not touching a pole piece; DPH1 can be sealed at both ends. End sealing is labeled structural-family inference, and millimeter parameters are labeled engineering reconstruction. Three existing copper-end contacts and gaps in lens magnetic-return connections remain limitations; new detail must not create the impression of a complete manufacturing drawing. See the [inspection record](probe-lower-detail-review-2026-10-21.md).


## 2026-10-21 repair of three DP copper-end contacts

The subsequent version uses explicitly shorter copper lengths with fixed centers and insulation at both ends for DP22/DP21/DP11, leaving positive clearances to the pole pieces; this supersedes the previous section's single-end sealing and three unresolved contacts. Only these three winding materials receive this publicly labeled engineering closure; pole-piece magnetic circuits and optical parameters are unchanged. Real coil fields depend on conductor layout; current propagation uses separately configured effective fields, so identical software results before and after do not prove equal fields in a real assembly. See the [end-insulation and clearance repair record](probe-contact-isolation-2026-10-21.md) for dimensions and evidence.
