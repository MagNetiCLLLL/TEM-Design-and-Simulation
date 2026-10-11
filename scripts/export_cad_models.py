"""Export a plain JSON SolidSpec catalog using the isolated CAD interpreter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="JSON catalog, with module-local millimetre SolidSpecs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", action="append", default=[], help="Export only this module key (repeatable)")
    args = parser.parse_args()
    # Do not import temsim.__init__ or scientific/Qt modules in the CAD process.
    path = Path(__file__).resolve().parents[1]/"src"/"temsim"/"cad_step.py"
    definition = importlib.util.spec_from_file_location("temsim_cad_step", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8-sig"))
    manifest_name = "step-manifest.json"
    if args.module:
        selected = set(args.module)
        unknown = selected-{row["key"] for row in catalog["modules"]}
        if unknown:
            parser.error("Unknown module keys: "+", ".join(sorted(unknown)))
        catalog["modules"] = [row for row in catalog["modules"] if row["key"] in selected]
        catalog["export_selection"] = sorted(selected)
        # A partial rerun must not overwrite a previous complete-run receipt.
        manifest_name = "step-manifest--"+"__".join(module._filename(key) for key in sorted(selected))+".json"
    report = module.export_catalog(catalog, args.output, progress=lambda name: print("Exporting "+name, flush=True),
                                   manifest_name=manifest_name)
    print(json.dumps(dict(status=report["status"], files=len(report["exports"]), seconds=report["elapsed_s"],
                         manifest=str(args.output/manifest_name))), flush=True)


if __name__ == "__main__":
    main()
