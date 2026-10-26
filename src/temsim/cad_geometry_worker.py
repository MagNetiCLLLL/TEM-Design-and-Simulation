"""Isolated CAD process: request JSON in, atomically completed receipt out.

Execute this file with the separate CAD interpreter. Loading sibling modules
by path keeps the simulator package and its numerical/GUI dependencies unloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_file", nargs="?", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)
    request_path = args.request or args.request_file
    if request_path is None or (args.request and args.request_file and args.request != args.request_file):
        parser.error("Provide exactly one request JSON path")
    if args.response is None and args.output is None:
        parser.error("Provide --output directory or --response JSON path")
    response = args.response or args.output / "result.json"
    root = Path(__file__).resolve().parent
    source = root / "cad_kernel.py"
    spec = importlib.util.spec_from_file_location("isolated_cad_kernel", source)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    started = perf_counter()
    receipt = dict(schema="temsim-cad-worker-result-v1", status="failed", request_sha256=None,
                   request_id=None, draft_digest=None)
    files = [source, source.with_name("cad_step.py"), source.with_name("cad_assembly_validation.py"), Path(__file__).resolve()]
    receipt["implementation_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in files}
    code = 1
    try:
        request = json.loads(request_path.read_text(encoding="utf-8-sig"))
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        receipt.update(request_sha256=kernel.digest(request), request_id=request.get("request_id"),
                       draft_digest=request.get("draft_digest"))
        receipt["result"] = kernel.dispatch(request)
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != receipt["implementation_sha256"][str(p.relative_to(root))]
               for p in files):
            raise RuntimeError("CAD implementation changed during computation; rerun the request")
        receipt["status"] = "complete"
        code = 0
    except Exception as exc:
        receipt.pop("result", None)
        receipt["error"] = dict(type=type(exc).__name__, message=str(exc))
    receipt["elapsed_s"] = perf_counter() - started
    response.parent.mkdir(parents=True, exist_ok=True)
    pending = response.with_name(response.name + "." + uuid.uuid4().hex + ".pending")
    pending.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(response)
    print(json.dumps(dict(status=receipt["status"], response=str(response), elapsed_s=receipt["elapsed_s"])), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
