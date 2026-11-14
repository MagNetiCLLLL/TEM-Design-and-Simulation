"""Capture real C1+C2 vacuum metadata without creating or transporting a beam."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path/"agent.md").is_file())
sys.path.insert(0, str(ROOT/"src"))
from temsim.column.layout import LayoutConfiguration, CorrectorAssembly, C3Hardware
from temsim.column.module_assembly import resolve_module_assembly

configuration = LayoutConfiguration(corrector=CorrectorAssembly.NO_CORRECTOR,
                                    c3_hardware=C3Hardware.TWO_CONDENSER, c3_excited=False)
assembly = resolve_module_assembly(configuration)
keys = ("condenser_lens_1", "condenser_lens_2", "mini_condenser", "objective_lens",
        "objective_upper_pole", "objective_lower_pole")
parts = []
for key in keys:
    part = assembly.part(key)
    parts.append(dict(key=key, source_file=part.source_file,
        global_interval_mm=[part.start_z_mm, part.end_z_mm], global_center_mm=part.center_z_mm,
        dimensions={key: value for key, value in part.data.items() if key in {
            "bore_diameter_mm", "mechanical_bore_diameter_mm", "mechanical_clear_bore_diameter_mm", "vacuum_inner_diameter_mm"}}))
path = ROOT/"configs/instruments/column/C2.toml"
result = dict(schema="ideal-lens-real-vacuum-context-v1", scope="Assembly metadata only; no field solution, source emission or transport",
    assembly_selection="FEG / physical C1+C2 / No Energy Filter",
    authoritative_column_path=path.relative_to(ROOT).as_posix(),
    authoritative_column_sha256=sha256(path.read_bytes()).hexdigest(), parts=parts,
    liners=[asdict(row) for row in assembly.vacuum_liner_segments if row.start_z_mm < 1090 and row.end_z_mm > 440],
    effective_bore_segments=[asdict(row) for row in assembly.vacuum_bore_segments if row.start_z_mm < 1090 and row.end_z_mm > 440],
    interpretation={
        "continuous_tube": "Global Z 450 to 960.5 mm, ID 5.76 mm and radius 2.88 mm; this caps the larger 20 mm main lens bores for C1 lens, C2 lens and Mini Condenser lens.",
        "narrower_hardware": "The C2 aperture carrier and beam deflector at Z 755 to 815 mm, and the specimen/goniometer gap at 1022.5 to 1027.9 mm, have 5 mm mechanical passages. A separately inserted 70 um C2 optical stop is not the lens bore or a uniform tube.",
        "ideal_study": "The ideal tube IDs 1.5, 2 and 5.76 mm are deliberately declared passive cylinders. Only 5.76 mm matches the reconstructed continuous-tube ID; they do not reproduce the full actual column or an OEM lens bore. The prescribed Gaussian does not respond to tube diameter.",
        "fixed_source": "Radius 0.95 mm means entrance filling 126.67%, 95% and 32.99% of those tube radii respectively; physical clipping must remain in current and finite-source statistics.",
        "matched_filling": "Source radii 0.7125, 0.95 and 2.736 mm are each 95% of tube radii 0.75, 1 and 2.88 mm. They are different prepared source boundaries, not a fixed-source tube experiment.",
        "authority": "Engineering reconstruction from the repository, not OEM measurements or a magnetic material boundary solution."})
target = HERE/"vacuum-tube-inventory.json"
target.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print(target)
