# Deflector current-to-field audit

Date: 2026-11-15. Scope: source-code inspection and vacuum magnetostatic
calculations only. No source emission, particle propagation, coherent wave,
optical retuning, or experimental coil qualification was performed.

## What the current program means

The column deflectors do not currently consume amperes. The single-station
components expose `kick_x_mrad` and `kick_y_mrad`
(`src/temsim/optics/single_plane_deflector.py:12`, `:101`); two-station components
expose upper/lower versions of those angles. Logical correction/scan drives
are added on their physical host, with limits checked after addition
(`src/temsim/physics/instrument_magnetic.py:93`). A shared scan channel is not
an additional physical winding.

`column_dipole_fields` (`src/temsim/physics/instrument_magnetic.py:143`) turns
the reference deflection into a finite, uniform transverse field:

\[
 B_x={p_0\over q}{\theta_y\over L_{eff}},\qquad
 B_y=-{p_0\over q}{\theta_x\over L_{eff}}.
\]

Here angles are in radians, length is in metres, and `q=-e`. The relativistic
reference momentum is taken from the beam voltage
(`src/temsim/physics/core.py:217`):

\[
 p_0={\sqrt{T^2+2T m_ec^2}\over c}.
\]

Thus a positive X kick of an electron travelling in +Z corresponds to positive
integrated By; a positive Y kick corresponds to negative integrated Bx. This
follows directly from `q(v cross B)`. The field is frozen from reference inputs,
not recomputed from the energy of each test electron. Different particle
energies respond differently to the same frozen field. Retaining an angle
command while changing the reference beam voltage changes the inferred field;
this is the present angle-command convention, not a physical fixed-current
law.

The finite field is a top-hat axial model. It retains the signed field integral
and effective length, but is not a Biot--Savart solution of the displayed
conductors. The posed-field equivalent uses local
`Az=-By*x+Bx*y`, with the same finite axial window
(`src/temsim/physics/posed_column_fields.py:262`).

The Gun Deflector is a separate interface: its existing inputs are Bx/By in
mT at two stations, multiplied by a soft axial window
(`src/temsim/optics/electron_gun/alignment.py:17`, `:79`). Its blanking setting
is also a field amplitude. It is not an ampere input or a coil-current rating.

## What the CAD supplies, and what it does not

DPH/DP inserts contain four saddle-loop conductor envelopes, arranged as two
concentric, orthogonal opposed pairs
(`src/temsim/cad_multipoles.py:57`, `:81`). The explicit CAD dimensions are
radial layer thickness, full angular span, conductor-band width and axial
copper length (`src/temsim/corrector_model_3d.py:24`, `:40`). Copper length and
effective field length are intentionally distinct. For example DPH1/DPH2
have 14.5 mm copper allocation but 4 mm native effective field length.

The solids are bulk allocations. They do not define individual trace paths,
turn count, electrical connectivity, coil polarity, lead routing or current.
The four displayed loops therefore must not be counted as four optical poles
of one quadrupole. A dipole channel consists of an opposed pair; the other
pair supplies the other steering direction. CAD `x`/`y` names identify
mechanical pair placement and do not establish the sign or orientation of
an electrical-to-optical response matrix.

The CEOS patent supports rolled flexible-PCB coils, opposed dipole windings,
and different multipoles through their electrical connection. It does not
establish the dimensions, turns, traces or calibration of this application's
DPH/DP components. Material, resistance, magnetic-core and thermal models are
not part of the ideal vacuum-current field.

## A physically defined conversion

For prescribed currents in vacuum, compute each complete closed-circuit
unit-current field by the Biot--Savart integral:

\[
 \mathbf b_j(\mathbf r)={\mu_0\over4\pi}
 \sum_{n\in j}\oint_{C_{jn}}
 {d\boldsymbol\ell'\times(\mathbf r-\mathbf r')
 \over|\mathbf r-\mathbf r'|^3},\quad
 \mathbf B(\mathbf r,t)=\sum_j I_j(t)\mathbf b_j(\mathbf r).
\]

`b_j` has units T/A. With unknown individual conductor size, a declared
uniform pack-current distribution can replace the discrete turns; its
normalization must still preserve the specified physical N and I. A solid
annular or saddle CAD volume by itself is not a current distribution. The
direction of current and its closed return path must be specified.

Field-integral calibration at the unchanged reference energy can retain
native commands without pretending to preserve the entire old field shape:

\[
 \begin{bmatrix}\theta_x\\\theta_y\end{bmatrix}
 = {q\over p_0}
 \begin{bmatrix}
 -\int b_{1y}dz & -\int b_{2y}dz\\
 \int b_{1x}dz & \int b_{2x}dz
 \end{bmatrix}
 \begin{bmatrix}I_1\\I_2\end{bmatrix}.
\]

The integral is along the reference +Z axis, over the full finite-coil field
including its tails. This is a small-angle, straight-reference calibration;
full propagation must subsequently use the actual vector field and actual
particle momentum. It is not a claim that finite saddle fields reproduce the
native top-hat shape, off-axis aberrations or downstream focus. Misalignment
and X/Y coupling require the full 2x2 matrix, not two unsigned scalar gains.
For shared drives, add all current contributions on the host once.

In this ideal linear model the basis can be calculated once and reused when
only current changes. Rebuild the basis after a change to N, coil geometry,
orientation or circuit topology. Include these dependencies and the field
implementation version in cache identity. Fast scanning may use a
quasi-static current envelope, but does not thereby model inductance, power
electronics, eddy currents, propagation delay or dynamic electric fields.

## Bounded analytic scale check, not an installed calibration

For two opposed, thin-filament saddles on radius `a`, each with N turns,
straight axial length `l` and full span `2 phi`, with their current senses
chosen to reinforce the dipole, direct integration gives

\[
 \left|\int B_\perp dz\right|
 = {2\mu_0 N I l\sin\phi\over\pi a}.
\]

This formula is derived here from Biot--Savart, not supplied as an OEM coil
formula. It includes the full axial integral of the straight portions; the
end-arc transverse contributions have zero full-axis integral. End arcs are
nevertheless essential to the local field profile and circuit closure.

For a representative centreline of the present bulk bands, take
`l=L_copper-band`, `a=mean layer radius`,
`phi=span/2-band/(2a)`. This is an explicitly approximate centroid model,
not a uniform-pack integration or a generated individual-turn winding.

| Physical host | Copper length (mm) | Inner-pair integral per A-turn (T m) | Outer-pair integral per A-turn (T m) |
| --- | ---: | ---: | ---: |
| DPH2 | 14.50 | 7.95313519e-7 | 7.63836060e-7 |
| DPH1 | 14.50 | 7.95313519e-7 | 7.63836060e-7 |
| DP22 | 14.22 | 7.78620282e-7 | 7.47803519e-7 |
| DP21 | 14.32 | 7.83924600e-7 | 7.52905759e-7 |
| DP11 | 14.32 | 7.83924600e-7 | 7.52905759e-7 |

Each value already includes the two opposed saddles; N is the count in each
saddle, not the sum over the pair. Inner/outer radii differ, so equal N and I
do not imply equal X/Y gains. The executed authoritative default assembly
(FEG / C3 + Probe Corrector / No Energy Filter, Nanoprobe / Diffraction)
uses **300 keV**, with positive magnetic rigidity **0.0021006577809314134
T m**. Its native DPH2 1 mrad X command, examined on a detached state, gives
**By=0.5251644452328534 mT** over **4 mm**, or **2.1006577809314154e-6
T m**. Bx is zero, and the original state is exactly unchanged.

At that same reference energy, the DPH centroid model needs approximately
**2.6412952 A-turn** (inner pair) or **2.7501422 A-turn** (outer pair) for a
1 mrad response. This does not supply missing physical N or determine an
ampere value for the program's current default model. It matches the
integrated field condition only; its finite-coil field is not the native
4 mm top-hat.

An independent scalar integration of the complete DPH inner-pair on-axis
Biot--Savart expression over `z=(-infinity,+infinity)` was executed using
SciPy adaptive quadrature. It gave 7.953135192287502e-7 T m per A-turn versus
the analytic 7.953135192237451e-7, a relative difference 6.30e-12. The
representative centre field was 5.287755563169186e-5 T per A-turn. This is
only a formula/normalization check for the declared centroid filament; it
does not validate the application field against a real deflector.

The closed saddle circuit was additionally integrated directly in three
dimensions using successively doubled Gauss--Legendre orders 4, 8, 16 and
32 per straight/arc segment. Against order 64 at four beam-region points,
maximum component error normalized by peak reference field was 6.7663e-5,
2.9625e-9, 6.4075e-16 and 7.6890e-16 respectively. The last two values are
at binary64 round-off. The centre field agreed with the separate analytic
axis expression. This demonstrates quadrature convergence for this bounded
filament example, not convergence of an unknown real winding geometry.

Reproduce with `.venv/Scripts/python.exe
docs/development/corrector_windings_20261115/check_deflector_fields.py`.
The script patches source emission/transport entry points to fail, reads
defaults without writing them, and writes an English JSON receipt to
`outputs/corrector-windings-20261115/deflector-field-check.json`.

## Primary sources

- [CEOS, US20090079531A1, Multipole coils](https://patents.google.com/patent/US20090079531A1/en): especially Figs. 5--8 and the dipole/connection discussion. Supports construction family and circuit topology, not installed dimensions or turn counts.
- [Oregon State University, The Biot--Savart Law](https://books.physics.oregonstate.edu/GSF/bs.html): the vacuum current-source law and its superposition.
- [CERN, Analytical design, beam rigidity](https://e-publishing.cern.ch/index.php/CYRSP/article/download/1619/1331/7986): the relativistic `p/q` scale. The search index supplied the stated equation; the complete 68 MB document was not downloaded.
- [CERN/JUAS, Thomas Zickler, Normal-conducting accelerator magnets, Lecture 2](https://indico.cern.ch/event/569714/contributions/2303819/attachments/1336494/2168024/JUAS_lecture_2017_Lecture_2_Slides.pdf): reference beam-rigidity equation and distinction between beam requirement and coil design.

## Recommended implementation boundary

Keep the existing mrad/mT channels and saved field strengths unchanged in
this task. Add a separately identified vacuum Biot--Savart coil source only
after circuit paths, per-coil counts and signed current channels are defined.
Do not relabel existing mrad controls as amperes. An optional current
calibration can preserve chosen signed field integrals; it must report the
field-shape change, retain unknown electrical data explicitly, and be tested
for zero, sign reversal, linearity, reference-energy response, symmetry,
geometry/cache invalidation and no duplication of shared hosts. It cannot
claim physical current conversion merely because an editable CAD winding or
a native Gaussian/top-hat field exists.
