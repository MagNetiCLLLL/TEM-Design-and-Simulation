"""Record primary references without redistributing third-party books or figures."""
from pathlib import Path
from hashlib import sha256
import json

HERE = Path(__file__).resolve().parent
rows = [
    dict(id="ceos2006", title="Muller et al., Advancing the Hexapole Cs-Corrector for the STEM (2006)",
         url="https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf",
         local="ceos-mueller-2006.pdf", pages="Printed 443-444; PDF pages 2-3; Figure 2 visually inspected",
         role="STEM CESCOR topology, HP-plane conjugacy, multipole/deflector order, Mini/TL11 and DP12 scan-coil alias",
         limit="Not a published Microprobe excitation table or a requirement for identical Microprobe source images"),
    dict(id="titan", title="Titan Training Material, Module 1, public university facility copy",
         url="https://emfacility.science.oregonstate.edu/sites/emfacility.science.oregonstate.edu/files/2025-09/Titan%20training%20manual_0.pdf",
         local="titan-training.pdf", pages="PDF pages 4 and 6; Module 1 pages 2 and 4; Figures 2 and 5 visually inspected",
         role="C2-C3 zoom; Mini polarity and Micro/Nano selection; distinguish TEM/Probe condenser ray paths",
         limit="Schematic operating principles, not calibrated current values for this model"),
    dict(id="williams2009", title="Williams and Carter, Transmission Electron Microscopy (2009), Chapter 9",
         url="https://doi.org/10.1007/978-0-387-76501-3_9",
         public_copy="https://cur.ac.rw/mis/main/library/documents/book_file/digital-66630b222e2283.91709642.pdf",
         local="williams-carter-tem.pdf", pages="Printed 142-147; PDF pages 194-199; Figures 9.5 and 9.6 visually inspected",
         role="Parallel illumination, objective front focal plane, two-coil shift/tilt and remote virtual pivot",
         limit="Its C3 can denote the objective prefield; use functional planes rather than equating component names"),
    dict(id="alexander2025", title="Duncan T. L. Alexander, TEM Ray Diagrams, EPFL (2025)",
         url="https://www.epfl.ch/labs/lsme/tem-ray-diagrams/",
         pages="Author notes and TEM/STEM diagram descriptions", role="Mini/objective coupling, parallel TEM versus focused STEM, manufacturer differences",
         limit="Ideal thin lenses; illustrative magnifications; not a CEOS Microprobe design drawing"),
    dict(id="laulainen2022", title="Joonatan Laulainen, Diffraction Between the Spots: Scanning Electron Diffraction of Beam-sensitive Disordered Materials (2022), Cambridge thesis",
         url="https://www.repository.cam.ac.uk/items/ed8d5eb1-789f-4d3e-94f7-c8bec6b7f246",
         doi="https://doi.org/10.17863/CAM.93209", local="zed-thesis.pdf",
         pages="Printed 48; PDF page 65; Figure 2.11 visually inspected",
         role="A distinct weakly convergent Microprobe SED ray path with a small specimen probe",
         limit="This SED example is not the present broad, nearly parallel TEM illumination prescription"),
    dict(id="rodenburg2004", title="John Rodenburg, Double deflection (2004)",
         url="https://www.rodenburg.org/RODENBURG.pdf", pages="Three-page author-hosted tutorial",
         role="Remote pivot from combined deflection", limit="Conceptual coil construction, not present-model calibration"),
    dict(id="zeltmann2026", title="Steve Zeltmann, Probe Corrector Wisdom, v2.1 (2026-10-11)",
         url="https://sezeltmann.com/assets/pdf/probecorr-manual.pdf", pages="Section 3.1",
         role="Independent Microprobe/TEM alignment and virtual DP12 practical context",
         limit="Author laboratory notes, not OEM certification; uncertain details are not used as hard constraints"),
]
for row in rows:
    if "local" in row:
        path = HERE / "references" / row.pop("local")
        row["inspected_local_file"] = str(path.relative_to(HERE)).replace("\\", "/")
        row["sha256"] = sha256(path.read_bytes()).hexdigest()
        row["bytes"] = path.stat().st_size
(HERE / "research-sources.json").write_text(json.dumps(dict(
    accessed="2026-11-11", scope="Primary-source research and visual inspection; third-party PDFs remain local, uncommitted",
    sources=rows), indent=2) + "\n", encoding="utf8")
print("Recorded seven primary references; no third-party PDF copied into the archive")
