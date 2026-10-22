"""Collect notices from the installed required CPU dependency closure.

Run from the project virtual environment. This reads distribution metadata;
it never imports numerical libraries, installs packages or contacts a server.
Version-pinned upstream supplements are retained separately in supplements.json.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party_licenses"

# These summaries describe the principal package license, not every bundled
# native library, dataset or font. The original notices remain authoritative.
SUMMARIES = {
    "matplotlib": "Matplotlib license (PSF-based); bundled font licenses",
    "numba": "BSD-2-Clause; see bundled third-party notices",
    "pyopengl": "BSD-style PyOpenGL license; bundled freeglut/GLE notices",
    "pyqtgraph": "MIT; CET color-map data CC-BY-4.0; other color-map data CC0-1.0",
    "pyfftw": "BSD-3-Clause wrapper; bundled FFTW GPL-2.0-or-later",
    "manifold3d": "Apache-2.0",
    "xraylib": "BSD-3-Clause; upstream notices included",
    "pandas": "BSD-3-Clause; see bundled notices",
    "contourpy": "BSD-3-Clause",
    "cycler": "BSD-3-Clause",
    "kiwisolver": "BSD-3-Clause",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "colorama": "BSD-3-Clause",
    "partd": "BSD-3-Clause",
    "jinja2": "BSD-3-Clause",
    "zict": "BSD-3-Clause",
    "google-crc32c": "Apache-2.0",
    "comm": "BSD-3-Clause",
    "traitlets": "BSD-3-Clause",
    "jupyterlab-widgets": "BSD-3-Clause",
    "ipython-pygments-lexers": "BSD-3-Clause",
    "prompt-toolkit": "BSD-3-Clause",
    "ipympl": "BSD-3-Clause; bundled JavaScript notices",
    "scipy": "BSD-3-Clause; bundled native-library notices and exceptions",
    "tomli-w": "MIT",
}

COMPONENT_NOTES = {
    "pyopengl": [
        "The optional bundled GLE library has IBM source terms and Artistic documentation/example terms; these are not a blanket GPL-compatible grant.",
        "No explicit OpenGL.GLE usage was found in this application or installed pyqtgraph during this review. A redistributed dependency bundle needs a separate GLE inclusion/license decision.",
    ],
    "pyqtgraph": [
        "CET continuous color maps: Peter Kovesi, Centre for Exploration Targeting, distributed through Colorcet; CC-BY-4.0. Source: https://peterkovesi.com/projects/colourmaps/ and https://colorcet.holoviz.org/ .",
        "The application does not modify the installed CET tables. The supplied package also includes a separate CC0 notice for viridis, magma, plasma, inferno and cividis data.",
    ],
    "pyfftw": [
        "The Windows FFTW DLLs in this installed pyFFTW distribution identify FFTW 3.3.5. The wrapper and native library have different licenses; retain both.",
    ],
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect() -> dict:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["dependencies"]
    markers = default_environment()
    markers["extra"] = ""
    roots = {
        canonicalize_name(req.name)
        for value in requirements
        if (req := Requirement(value)).marker is None or req.marker.evaluate(markers)
    }
    queue = list(sorted(roots))
    packages = {}
    supplements = json.loads((DEST / "supplements.json").read_text(encoding="utf-8"))
    while queue:
        key = canonicalize_name(queue.pop(0))
        if key in packages:
            continue
        dist = metadata.distribution(key)
        meta = dist.metadata
        applicable = []
        for value in dist.requires or []:
            req = Requirement(value)
            if req.marker is None or req.marker.evaluate(markers):
                applicable.append(canonicalize_name(req.name))
                queue.append(req.name)
        files = []
        for file in sorted(dist.files or [], key=str):
            source = str(file).replace("\\", "/")
            filename = Path(source).name
            if not any(word in filename.lower() for word in
                       ("license", "licence", "copying", "notice", "copyright", "legal", "authors", "attribution")):
                continue
            # A commercial-license pointer is not an open-source grant. The
            # versioned upstream Qt OSS license texts are supplied below.
            if filename == "LicenseRef-Qt-Commercial.txt":
                continue
            original = dist.locate_file(file)
            if not original.is_file():
                continue
            data = original.read_bytes()
            # Keep paths short enough for Windows wheel/sdist staging. The
            # original filename remains in the inventory's source_path.
            relative = Path("packages") / key / (sha256(source.encode())[:12] + ".txt")
            if any(item["path"] == relative.as_posix() for item in files):
                raise ValueError(f"Notice path hash collision for {key}: {source}")
            output = DEST / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            files.append({"path": relative.as_posix(), "source": "installed-distribution", "source_path": source,
                          "sha256": sha256(data), "bytes": len(data)})
        for item in supplements:
            if key not in item["packages"]:
                continue
            expected = item.get("package_versions", {}).get(key)
            if expected and expected != dist.version:
                raise ValueError(f"Update upstream supplements for {key} {dist.version}; expected {expected}")
            data = (DEST / item["path"]).read_bytes()
            if sha256(data) != item["sha256"]:
                raise ValueError(f"Supplement checksum mismatch: {item['path']}")
            files.append({k: item[k] for k in ("path", "source", "url", "sha256", "bytes")})
        raw = meta.get("License-Expression") or meta.get("License") or ""
        summary = SUMMARIES.get(key, raw if len(raw) < 180 else "See original license text")
        if not summary:
            summary = "; ".join(c.removeprefix("License :: ") for c in meta.get_all("Classifier", []) if c.startswith("License :: ")) or "See original license text"
        packages[key] = {
            "name": meta["Name"], "version": dist.version, "direct": key in roots,
            "license_summary": summary, "metadata_license_expression": meta.get("License-Expression"),
            "metadata_license": meta.get("License"),
            "license_classifiers": [c for c in meta.get_all("Classifier", []) if c.startswith("License :: ")],
            "project_urls": meta.get_all("Project-URL", []),
            "component_notes": COMPONENT_NOTES.get(key, []),
            "requires_in_this_scope": sorted(set(applicable)), "notice_files": files,
            "notice_status": "texts_collected" if files else "no_text_found_review_before_redistribution",
        }
    result = {
        "schema_version": 1,
        "scope": f"Installed {markers['platform_system']} CPU runtime dependency closure; no optional extras, developer tools, CAD environment or proprietary SDK",
        "method": "importlib.metadata + packaging requirement markers; no numerical-library imports",
        "requirements": requirements,
        "marker_environment": markers,
        "direct_dependency_count": len(roots), "distribution_count": len(packages),
        "limitations": ["Notice collection is not a complete binary or copyright-provenance audit.",
                       "A future environment, package update or redistributed native binary requires a fresh inventory.",
                       "License texts do not replace corresponding-source or installation-information obligations."],
        "packages": [packages[key] for key in sorted(packages)],
    }
    (DEST / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = collect()
    print(f"Collected {result['direct_dependency_count']} direct dependencies and {result['distribution_count']} installed distributions.")
