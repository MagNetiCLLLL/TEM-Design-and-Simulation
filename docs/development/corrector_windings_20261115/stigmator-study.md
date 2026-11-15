# Stigmator current-to-field audit

Date: 2026-11-15. Scope: read-only model/source research. No particle emission,
coherent propagation, installed lens adjustment, installed excitation change or
tolerance change was performed. The numerical examples below are field/unit
calculations, not a new
finite-source physical trial.

## Present implementation

The column stigmators currently accept two dimensionless percentage commands,
not amperes. `optics/model.py:177-195` stores `max_strength_m2`,
`strength_x_percent`, `strength_y_percent`, and channel angles. In
`optics/stigmator_field.py:29-42`, with coordinates and longitudinal derivatives
in metres,

\[
\begin{aligned}
f(z)&=\exp[-(z-z_0)^2/(2\sigma^2)],\qquad \sigma=L_{\rm FWHM}/2.355,\\
n(z)&=f(z)k_{\max}(u_x\cos2\alpha_x+u_y\cos2\alpha_y),\\
s(z)&=f(z)k_{\max}(u_x\sin2\alpha_x+u_y\sin2\alpha_y),\\
K(z)&=\begin{pmatrix}n&s\\s&-n\end{pmatrix},\qquad
\mathbf r''=-K\mathbf r,
\end{aligned}
\]

where `u_x` and `u_y` are percentages divided by 100, and `n,s,k_max` have
units m^-2. The default basis angles 0 and 45 degrees span normal and skew
quadrupoles. They are not two ordinary dipole axes. Equal nonzero commands
therefore do not mean a 45-degree rotation of the optical tensor.

`physics/posed_column_fields.py:216-241` constructs the same existing field from
the local vector potential

\[
A_z={p\over q}\left[-\tfrac12n_0x^2+\tfrac12n_0y^2-s_0xy\right]f(z),
\quad A_x=A_y=0,
\]

giving

\[
B_x={p\over q}(n_0y-s_0x)f(z),\qquad
B_y={p\over q}(n_0x+s_0y)f(z),\qquad B_z=0.
\]

Here `q=-e` is signed; `physics/core.py:217-222` calculates relativistic momentum
from the selected acceleration voltage. Thus the physical normal/skew gradient
coefficients are `(p/q)*(n,s)` in T/m. Their magnitudes obey
`|G|=(p/e)|k|`; keeping the sign convention is necessary when a physical coil is
connected to the existing control. The present Gaussian field is an effective
optical model, not a field calculated from the reconstructed winding. A
divergence-free vector potential construction alone does not make this
longitudinally modulated transverse polynomial a current-free, exact finite
vacuum-coil fringe solution.

The Gun Stigmator is a different interface:
`optics/electron_gun/alignment.py:153-209` takes `gradient_t_per_m` and a rotation,
with a finite soft window. It already uses a magnetic gradient in T/m, but still
has no physical ampere/turn calibration.

## Material geometry and missing electrical information

`cad_multipoles.py:265-307` represents a material stigmator as eight rectangular
bulk radial loops in two interleaved four-coil groups, with group angular bases
0 and 45 degrees. Their radial, tangential and axial allocations are engineering
fractions of the configured body. The code explicitly says those allocations
do not rescale the field. `cad_stigmator_support.py` adds nonmagnetic supports,
insulation and adhesive around the same bulk allocation.

The default condenser stigmator record in
`configs/instruments/column/C3_ProbeCorrector.toml:821-848` declares eight coils,
ID 20 mm, OD 56 mm, 30 mm body length and 6 mm effective field FWHM. None of those
values supplies winding turns, individual turn centre lines, winding handedness,
per-channel current, or an ampere-to-gradient calibration. A geometrical coil
count of eight is not a turn count of eight.

The corrector QPH2, QPC, QPH1 and QPol records are explicitly
`layout_role="control_channel"`, with `physical_host_status="unverified"`
(same column file at lines 285-300, 472-487, 573-588 and 655-670). Their optical
coordinates do not establish four extra material winding assemblies. They must
not receive invented independent physical turns from their display envelopes.
The current probe A1 fine control uses QPol outputs; this does not independently
identify an OEM stigmator cartridge (`optics/corrector_channels.py:104-130`).

## Ideal vacuum conversion that can be implemented honestly

For a declared closed-loop winding in vacuum, solve Biot-Savart and superpose
the field of every group:

\[
\mathbf B(\mathbf r)={\mu_0\over4\pi}\sum_j I_j
\sum_{t=1}^{N_j}\oint_{C_{jt}}
{d\boldsymbol\ell\times(\mathbf r-\mathbf r')\over
|\mathbf r-\mathbf r'|^3}.
\]

For an unresolved homogeneous pack, average unit-turn loops over its fixed
allocated section and multiply by `NI`; this preserves the field source's
spatial range. Metal area remains a CAD/packing input. It is not a free magnetic
gain. Exclude magnetic cores, pole pieces, saturation, resistance and heating
in the requested ideal-vacuum model. Include the closed-loop return sections;
using only the long axial sides loses the finite fringe field.

At each longitudinal plane, extract

\[
G_n={1\over2}(\partial_xB_y+\partial_yB_x),\qquad
G_s={1\over2}(\partial_yB_y-\partial_xB_x).
\]

The extraction is a projection of the full computed field near the axis; retain
and report residual dipole and higher-order components. Compute each electrical
channel at +1 A, including the explicitly saved winding directions and turns,
to obtain a response matrix

\[
\begin{pmatrix}G_n(z)\\G_s(z)\end{pmatrix}
=C(z)\begin{pmatrix}I_a\\I_b\end{pmatrix},
\quad [C]={\rm T\,m^{-1}\,A^{-1}},\qquad
\begin{pmatrix}n(z)\\s(z)\end{pmatrix}
={q\over p}C(z)\begin{pmatrix}I_a\\I_b\end{pmatrix}.
\]

Do not assume that a CAD `basis_angle_deg` alone fixes the electrical polarity
or optical tensor phase. Verify these from the actual generated field. A
well-conditioned 2x2 matrix can be inverted for a chosen peak-gradient or
integrated-gradient target. A single current fit cannot generally reproduce the
entire old Gaussian axial shape and fringe field. Report the chosen matching
condition and the remaining field-shape error separately; do not silently
replace the optical provider or refocus the probe.

For repeated field queries, precompute two unit-current vector potentials or
fields and scale their linear combination. The cache identity must include all
turn counts, envelope/path geometry, winding directions, group assignment,
pose, integration resolution, and source-model version. Current can remain a
separate linear coefficient when the cached data are explicitly normalized to
1 A. Conductor repacking changes a turn-resolved field and must invalidate that
geometry-dependent cache.

## Executed bounded field study

Run with the project environment:

```powershell
.venv/Scripts/python.exe docs/development/corrector_windings_20261115/study_stigmator_current.py
```

The script fails if source emission or Gun source transport is requested. It
constructs the installed default Nanoprobe/Diffraction state without tracing a
particle, verifies the whole optical state is unchanged, and records the
selected voltage and original native stigmator commands. The installed default
is 300 kV, with all three column stigmator percentage pairs zero. The nonzero
QPol probe-corrector correction remains separate and is not reassigned to these
three material bodies.

Using the installed SciPy constants, the executed state gives magnetic rigidity
`p/e=0.0021006577809314134 T m`; the signed `p/q` is negative. A separate native
100% channel reference at 300 m^-2 corresponds to gradient magnitude
`0.630197334279424 T/m`, not 300 T/m and not 300 A. The script projects the
original frozen field at each reference plane and checks its signed gradient
against the tensor conversion. It also restores six independent detached
instrument snapshots, sets only one copied channel to 100% at a time, captures
that copy's native frozen field and verifies its nonzero signed normal/skew
gradient against the expected tensor-to-field conversion. Thus the sign/unit
checks include six nonzero cases in addition to the three zero default cases.
The original components and complete installed state remain unchanged. The zero
installed controls are not replaced by the nonzero reference examples.

To demonstrate a possible physical current source, the script additionally
constructs a closed rectangular representative centreline halfway between the
inner and outer edges of each of the eight saved bulk coil frames, at its
radial middle plane. This is explicitly a representative-path engineering
assumption, not the full pack average, not the unknown real turn routing, and
not an installed runtime source. Each four-coil group is assigned alternating
`+1,-1,+1,-1` current signs; the first loop normal points outward radially.
Straight wire segments use their exact finite-segment Biot-Savart integral.
An independent closed-square analytic result checks normalization and direction.
The study uses installed `scipy.constants.mu_0`, as the annular-coil solver does;
this differs negligibly from the legacy `4*pi*1e-7` approximation in the separate
deflector study. No project constant or existing tolerance was changed.

For that assumed source the central response per ampere-turn per coil is:

| Material body | `C_n,a` | `C_n,b` | `C_s,a` | `C_s,b` | Units |
| --- | ---: | ---: | ---: | ---: | --- |
| Condenser Stigmator | approximately 0 | +0.00126180459349 | -0.00126180459397 | approximately 0 | T/m per A-turn |
| Objective Stigmator | approximately 0 | +0.00126180459349 | -0.00126180459397 | approximately 0 | T/m per A-turn |
| Diffraction Stigmator | approximately 0 | +0.00008916912908 | -0.00008916912910 | approximately 0 | T/m per A-turn |

The off-diagonal phase is a useful warning: a radial-loop winding basis is not
automatically the same phase as the optical focusing basis. Signs and phase
must be established from the generated magnetic field, as done here. The
resulting matrices are well conditioned. Central finite-difference derivative
steps 10, 5, 2.5, 1.25 and 0.625 micrometres agree at the last two steps to
relative 2.23e-9 or better, below the predeclared 1e-7 bound. Central dipole
residuals are below 1e-16 T/A.

The assumed loop source has an approximately 21.1 mm gradient FWHM for the
Condenser/Objective Stigmator and 29.1 mm for the Diffraction Stigmator, sampled
at 0.05 mm spacing. Their original optical Gaussian FWHMs are 6, 6 and 8 mm.
The respective integral/peak widths over +/-100 mm are about 21.639 and
30.806 mm, while the original Gaussian full integral/peak widths are 6.386 and
8.515 mm. Matching just the central gradient would consequently leave a large
integrated-gradient mismatch. These are evidence against silently replacing
the existing Gaussian provider, rather than a validated replacement field.

The per-coil turn counts for these three bodies remain unknown. For any chosen
fixed N, `I=(required ampere-turns)/N`; conductor metal area is not used to
invent a magnetic gain. Complete matrices, assumed paths, 100% reference
current solutions and original installed native control values are stored in
[`validation/stigmator-current-study.json`](validation/stigmator-current-study.json).
No new current calibration was installed.

## Primary references consulted

- [JEOL stigmator explanation](https://www.jeol.com/words/semterms/20201020.111014.php):
  alternating winding directions in each four-coil set, a second set offset by
  45 degrees, and two current controls. It supports the topology and correction
  principle, not the numerical dimensions of this project's reconstruction.
- [CERN Accelerator Engineering and Technology, chapter 8](https://cds.cern.ch/record/2743158/files/Bordry2020_Chapter_AcceleratorEngineeringAndTechn.pdf):
  magnetic rigidity and normalized integrated quadrupole strength. The search
  index exposed the relevant equations; opening the full PDF encountered the
  CERN anti-bot page during this audit. The repository formula/sign derivation
  above was checked directly from its own vector potential and Lorentz law.
- [OpenStax University Physics, Biot-Savart law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law):
  the vacuum line-current integral and vector superposition. Its textbook
  statement that permeability is exactly `4*pi*1e-7` predates the current SI;
  retain the project's constant for regression rather than modifying it here.
- [NIST magnetic units](https://www.nist.gov/publications/units-magnetic-quantities):
  after the 2019 SI revision vacuum permeability is experimentally determined.
  This does not justify changing an existing numerical tolerance or constant
  in a winding-input task.

No source above supplies this project's winding current, insulation packing,
measured field map, or experimental qualification.
