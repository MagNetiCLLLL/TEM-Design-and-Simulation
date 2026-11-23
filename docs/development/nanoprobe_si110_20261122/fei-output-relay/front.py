"""Locate the isolated Mini/incident-objective canonical front referred to the specimen."""

import argparse
import numpy as np
from types import SimpleNamespace
from relay import (
    BASE,
    HERE,
    WORK,
    Circuit,
    Global,
    digest,
    load_bias,
    optics_mesh,
    relaunch,
    source_packet,
    write,
)
from material_trial import bore
from material_relay_column import Column, Field
from material_relay_tracking import Trial


def run(path, folder):
    full = load_bias(path)
    bias = {
        k: v if k.startswith(("mini_condenser/", "objective_lens/")) else 0.0 for k, v in full.items()
    }
    pair = WORK / ("global-" + digest(bias)[:16] + ".npz")
    if not pair.exists():
        seeds = []
        for candidate in WORK.glob("global-????????????????.npz"):
            b = load_bias(candidate)
            if all(not v or k.startswith(("mini_condenser/", "objective_lens/")) for k, v in b.items()):
                seeds.append((sum(abs(b[k] - v) for k, v in bias.items()), candidate))
        Circuit(Global(9)).solve(bias, min(seeds)[1], pair)
    if load_bias(pair) != bias:
        raise ValueError("Incorrect isolated pair")
    bore.process(pair, ["dc"], 0.25)
    field = Field(pair, step=0.25)
    m = Trial.__new__(Trial)
    m.m, _, _ = optics_mesh(0.06)
    m.source = source_packet(relaunch(BASE, 131072))
    m.pair = Column.axis_field(m, field)
    isolated = m.fronts()
    from supplies import field as total_axis

    z, b = total_axis(path)
    coeff = np.zeros((len(z), 1, 3))
    coeff[:, 0, 2] = b
    m.pair = Column.axis_field(m, SimpleNamespace(z=z, c=coeff))
    result = dict(
        pair_field=pair.name,
        full_field=path.name,
        excitation=bias,
        all_fronts=isolated,
        installed_field_fronts=m.fronts(),
        definition="Canonical outgoing angular response D=0 at the specimen upper surface. Incident propagation includes only the jointly solved Mini and series objective excitation.",
    )
    folder.mkdir(exist_ok=True)
    write(folder / "front.json", result)
    print("Fronts", result["all_fronts"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("field")
    p.add_argument("folder")
    a = p.parse_args()
    run(WORK / (a.field + ".npz"), HERE / a.folder)
