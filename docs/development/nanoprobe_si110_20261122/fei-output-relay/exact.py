"""Replay a supply setting from the FEI emitter through the measured specimen surface."""

import argparse
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from relay import HERE, OBJ, layered, load_bias, relaunch, sha, write
from column_trace import trace
from focus import run as focus
from gun_bundle import recover
from source_image import measure
from supplies import Supplies


def run(t, o, n, folder, detail=True, step=0.06):
    path = Supplies().get(t, o)
    with ThreadPoolExecutor(max_workers=2) as pool:
        gun = pool.submit(relaunch, path, n)
        table = layered(path)
        packet = gun.result()
    with np.load(table) as d:
        checks = json.loads(str(d["layer_receipts"]))
    if any(x["solution_sha256"] != sha(path) or x["bias_A"] != load_bias(path) for x in checks):
        raise ValueError("Unmatched volume interpolation sources")
    folder.mkdir(exist_ok=True)
    r = trace(path, table, packet, folder, step, detail=True)
    r.update(
        tl12_A=t,
        objective_A=o,
        emitted_particles=n,
        common_field_sha256=sha(path),
        interpolation_checks=checks,
    )
    write(folder / "result.json", r)
    focus(path, folder)
    if detail:
        recover(folder)
        measure(folder)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tl", type=float)
    p.add_argument("obj")
    p.add_argument("name")
    p.add_argument("--count", type=int, default=524288)
    p.add_argument("--step", type=float, default=0.06)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run(a.tl, OBJ if a.obj == "ref" else float(a.obj), a.count, HERE / a.name, not a.quick, a.step)
