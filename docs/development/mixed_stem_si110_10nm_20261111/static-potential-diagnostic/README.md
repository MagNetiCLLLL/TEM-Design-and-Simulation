# Static sampling diagnostic of the finite 10 nm Si[110] potential

This is a CPU atomic-potential diagnostic, not a source, wave propagation,
detector or scan calculation. No production or pinned source file was edited.
The script is `../diagnose_static_potential.py`; runtime was 109.76 seconds.

All eight baseline combinations use the same 208 actual Si sites in the
periodic rectangular primitive, the same translated input CIF, the same 100 A
finite slab and 400 A axial vacuum padding on each side. The original Lobato
finite-projection integrator and float32 slice storage are retained. Potential
slices already contain their Z integral in V A, so the diagnostic sums them
once in float64 without another multiplication by dz. No full-FOV tiling was
allocated. Arrays, metadata and hashes are in `report.json`, the twelve NPZ
files and `checksums.json`.

The baseline atom-array SHA256 is
`0b8ea8b0a8b84ec456adb0567a82eb5fc3601c39146207f04836b5e487700060`.
The CIF SHA256 is
`e55aea3dcd96d74e295e529195365216bc1c22716c87bf763c254bbde8148a98`.
All baseline configurations have the identical atom-array hash. The frozen
potential implementation hash is also recorded; no current solver identity is
substituted for the pinned implementation used by this diagnostic.

## Results

For every fixed lateral grid, doubling the slice count from 800 to 1600
changes the projected DC by only 1.281e-6 to 1.287e-6 relative. The relative
complex Fourier L2 change in q = 2.5 to 7.6 inverse angstroms is 2.8e-8 to
3.2e-8. This rules out a large slice-count dependence of the summed projected
potential in these cases. It does not certify a multislice wave integrator.

Fourier coefficients are compared at identical physical reciprocal coordinates
of the same rectangular primitive. The following values use 280 by 224 pixels
and 1600 slices as a comparison reference, not a claimed continuum solution.

| Full square grid | Primitive X by Y | Quarter-cell X index | Relative complex L2, q=2.5 to 7.6 / A |
| --- | --- | --- | ---: |
| 1920 | 240 by 192 | 60, integer | 1.621841% |
| 2080 | 260 by 208 | 65, integer | 0.726201% |
| 2160 | 270 by 216 | 67.5, half pixel | 1.726964% |
| 2240 | 280 by 224 | 70, integer | 0, comparison reference |

The 270-pixel X case is worse than both neighboring grids, and even worse than
the coarser 240-pixel case, on this common-q comparison. This is a measured
nonmonotonic sampling effect. A low-q fit does not identify a global half-pixel
shift of the physical crystal: its inferred translation is approximately
1e-6 A, many orders below a pixel. Individual weak coefficients have larger
phase errors and must not be mistaken for a global coordinate translation.

Four additional controls physically translate every atom by half an X pixel,
rebuild the static potential at 800 slices, and remove only the analytically
known Fourier translation phase. Relative complex Fourier L2 discrepancies in
q=2.5 to 7.6 / A are 3.442765%, 2.973622%, 2.733649% and 2.593666% for primitive
X counts 240, 260, 270 and 280 respectively. These violate exact translation
covariance at the reported discretization errors, while using the same
continuous atomic model. The 270 case mostly changes coefficient phase/sign:
its amplitude-only L2 discrepancy is 1.38e-6, whereas its complex discrepancy
is 2.733649%. Comparing only magnitudes would conceal that effect.

## Mechanism and limits

The installed abTEM finite real-space integrator chooses an inner radial
limit from the lateral sampling, `min(sampling) / 2`, and uses a constant first
radial sample inside that limit. The measured inner limits range from
0.0141763 A to 0.0121511 A. Thus changing the grid simultaneously changes the
core regularization and the sampling of atomic positions. In the 270-pixel
case, quarter-cell columns fall halfway between X pixels. The nonmonotonic
common-q coefficients and the half-pixel translation controls support this
mechanism independently of any wave calculation.

These results do not prove that this mechanism alone explains the failed BF
and ADF modulation errors. The q interval is a potential Fourier diagnostic
near the nominal ADF frequency range; it is not detector intensity. Static
projected coefficients do not replace finite-slice wave convergence,
domain/field qualification, or the unchanged 5% image and diffraction gates.
Choosing only an integer-aligned grid sequence is not by itself a proof of
translation-invariant convergence. A corrected finite-slab potential operator
must pass common-q and half-pixel controls before a new matched propagation
can test its effect on the scan.
