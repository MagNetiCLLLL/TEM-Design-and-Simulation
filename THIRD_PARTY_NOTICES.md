# Third-party notices

TEM Design & Simulation uses software and data from other projects. Their
authors retain their copyrights and license terms. The application's
GPL-3.0-only license does not relicense those dependencies, their data, fonts,
native libraries or documentation.

This notice covers the required CPU runtime dependencies installed in the
Windows development environment reviewed on 2026-10-22. It records 17 direct
dependencies and a total of 74 distributions, including transitive dependencies.
The exact versions below describe that environment, not a promise about future
installations or other platforms. The application wheel does not embed these
dependency wheels or their DLLs; normal installation obtains them separately.

Original texts, copyright notices and provenance are in
[`third_party_licenses/`](third_party_licenses/README.md). The
[`inventory.json`](third_party_licenses/inventory.json) records all 74 package
versions, license metadata, dependency relationships, notice-file paths and
SHA-256 hashes. Upstream supplements have versioned source URLs and hashes in
[`supplements.json`](third_party_licenses/supplements.json). This is a notice
inventory, not certification that every possible binary distribution satisfies
all license obligations.

## Required dependencies in the reviewed environment

The license column summarizes the main package license. The original texts
and additional component notices apply where a package contains more than one
license.

| Dependency | Reviewed version | Principal license / additional notices |
| --- | --- | --- |
| abTEM | 1.0.10 | GPL-3.0-or-later |
| ASE | 3.29.0 | LGPL-2.1-or-later |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| ImageIO | 2.37.4 | BSD-2-Clause |
| manifold3d | 3.5.3 | Apache-2.0 |
| Matplotlib | 3.11.1 | Matplotlib's PSF-based license; bundled font notices |
| Numba | 0.66.0 | BSD-2-Clause; third-party notices |
| NumPy | 2.4.6 | BSD-3-Clause, 0BSD, MIT, Zlib, CC0-1.0; native-library notices |
| Pillow | 12.3.0 | MIT-CMU; bundled library notices |
| PyOpenGL | 3.1.10 | BSD-style PyOpenGL license; separate freeglut and GLE terms |
| pyqtgraph | 0.14.0 | MIT; CET color-map data CC-BY-4.0 and other color maps CC0-1.0 |
| PySide6 | 6.8.3 | LGPL-3.0-only / GPL-2.0-only / GPL-3.0-only alternatives; module-specific terms |
| SciPy | 1.18.0 | BSD-3-Clause; bundled native-library notices and exceptions |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| tifffile | 2026.7.31 | BSD-3-Clause |
| tomli_w | 1.2.0 | MIT |
| xraylib | 4.3.0 | BSD-3-Clause; upstream author notices |

## Components requiring particular attention

**abTEM and ASE.** Their GPL/LGPL terms continue to apply. abTEM's
GPL-3.0-or-later grant permits using version 3 with this GPL-3.0-only application.
Distributing a combined work or modified dependency requires following the
applicable source-disclosure and notice requirements; merely listing a package
name here does not meet those requirements.

**pyFFTW and FFTW.** The reviewed CPU dependency closure includes pyFFTW 0.15.1.
The Python wrapper has a BSD-3-Clause license, while its bundled Windows FFTW
libraries are GPL-2.0-or-later. The installed DLL version strings identify FFTW
3.3.5, and this archive includes its upstream `COPYING` text. Both licenses
must be retained. The wrapper's permissive metadata alone does not describe
the complete native dependency. Before shipping those binaries, obtain and
provide the matching corresponding source and build information as required.

**Qt / PySide6 / Shiboken.** The reviewed PySide6, PySide6_Essentials,
PySide6_Addons and shiboken6 versions are all 6.8.3. This project uses the
open-source licensing route. Qt modules have different license options, and
some modules are GPLv3-only under their open-source offering. The archive
preserves the versioned PySide OSS license texts, including the Qt GPL
exception, without asserting that every file has the same license. A
redistributed Qt build must also retain its module-specific copyrights and
third-party notices and satisfy the selected GPL/LGPL source and installation
requirements. Qt provides [module licensing information](https://doc.qt.io/qt-6.8/licensing.html)
and [third-party code notices](https://doc.qt.io/qt-6.8/licenses-used-in-qt.html).

**PyOpenGL's optional GLE component.** The installed distribution includes GLE
notices with IBM source-license terms and Artistic terms for documentation and
examples, separate from PyOpenGL's BSD-style license. These terms must not be
treated as a blanket GPL-compatible grant. The review found no explicit
`OpenGL.GLE` or GLE extrusion API usage in this application's source or in the
installed pyqtgraph source. That static check does not certify every possible
runtime load. The current application wheel does not include GLE binaries.
An installer that bundles PyOpenGL must separately assess the actual GLE
payload and either omit unused GLE components or establish that their inclusion
and distribution meet the relevant terms. Original GLE notices and author
credits are preserved in the archive; their inclusion is not a compatibility
endorsement.

**Color-map attribution.** The CET continuous color maps supplied through
pyqtgraph originate with **Peter Kovesi, Centre for Exploration Targeting**, and
are distributed through the [Colorcet project](https://colorcet.holoviz.org/)
under **CC BY 4.0**. See the [original color-map project](https://peterkovesi.com/projects/colourmaps/)
and [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
This application does not modify those installed CET tables. pyqtgraph's
MIT software license does not replace this data license. Its separate CC0
notice for viridis, magma, plasma, inferno and cividis is also preserved.

**Other native code, fonts and data.** NumPy and SciPy wheels include their
own bundled-library notices, including OpenBLAS/LAPACK and, where present,
GCC runtime components under the GCC Runtime Library Exception. llvmlite 0.48.0
includes Apache-2.0 terms with the LLVM exception as well as BSD-2-Clause terms.
Matplotlib includes font notices; Pillow and numcodecs include additional
native-library terms. tqdm 4.70.0 uses MPL-2.0 and MIT terms. Preserve these
original notices and exceptions when redistributing the corresponding files;
do not collapse them into the main Python package's license.

## Scope and distribution responsibilities

The archived texts were collected from installed distribution metadata and
files, supplemented from official upstream sources where needed. This method
does not prove that every native component was identified or that every
upstream package supplied complete notices. A frozen executable, offline
installer, container or bundled environment needs an inventory of the exact
artifacts actually included. Supply applicable corresponding sources,
installation/relinking information, retained notices and modification notices
as required by their licenses. A license URL or this notice alone is not a
substitute for those obligations.

Optional GPU/CUDA packages and the separate CAD environment are outside this
CPU inventory. Their Python package license does not automatically cover
bundled CUDA libraries, drivers, CAD kernels or other native components; audit
the exact optional artifacts before redistributing them. Developer tools are
also outside the runtime inventory. AutoScript is outside this review's scope.

Manufacturer reference PDFs, screenshots and downloaded originals are kept
locally and excluded from new repository distributions. They are not licensed
under this project's GPLv3 grant; prior Git history is not rewritten. Other
third-party datasets and documents retain their own provenance and terms.
Upstream names and trademarks identify dependencies and references only and
do not imply endorsement.
