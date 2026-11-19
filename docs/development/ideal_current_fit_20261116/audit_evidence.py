"""Verify frozen fit identities and electrical counts without computing a beam."""
from hashlib import sha256
import json
from pathlib import Path

from temsim.ideal_coil_segments import normalize_recipe, bind_segments


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    reference_path = HERE / "round-lens-field-match/captured-field-coefficients.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    conclusion = json.loads((HERE / "fixed-envelope-segmented-current/round-transfer/final-conclusion.json").read_text(encoding="utf-8"))
    assert conclusion["status"] == "FAILED_TO_MATCH_REFERENCE_DO_NOT_INSTALL"
    assert sha256(reference_path.read_bytes()).hexdigest() == conclusion["reference_sha256"]
    identities = conclusion["original_geometry_identities"]
    geometry_digest = sha256(json.dumps(identities, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert geometry_digest == conclusion["geometry_identity_sha256"]
    groups = 0
    turns = 0
    for owner, original in identities.items():
        captured = reference["ideal_current_coils"][owner]["metadata"]
        assert original["original_current_definition"] == captured["provenance"]["winding_current_definition"]
        assert original["original_winding_records"] == [row["metadata"] for row in captured["coils"]]
        recipe = normalize_recipe(conclusion["recipes"][owner])
        windings = original["original_current_definition"]["windings"]
        assert set(recipe["windings"]) == {winding["key"] for winding in windings}
        for winding in windings:
            cells = bind_segments(recipe, winding["key"], winding["interval_turn_counts"])
            groups += len(cells)
            turns += sum(cell.turn_count for cell in cells)
    assert len(identities) == 13 and groups == 30
    output = dict(scope="Frozen reference identity and recipe syntax only; no source, field solve or wave",
        status="PASSED_IDENTITY_CHECKS_CANDIDATE_STILL_FAILED", owner_count=len(identities),
        closed_current_groups=groups, physical_turn_count=turns,
        reference_sha256=conclusion["reference_sha256"], geometry_identity_sha256=geometry_digest)
    validation = HERE / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "evidence-identity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
