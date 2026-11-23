"""Absolute projected current along the entire column from every original particle."""

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from column_current import ApertureTracker
from condenser_pair_column import column_mesh, round_field
from condenser_pair_gun import MaterialGunMagnet, identity, selected_field
from condenser_pair_study import EV, OUT, Pair
from condenser_source_fields import FEM
from relay_transfer import advance, electric_reference
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
from temsim.physics.instrument_electric import (
    configure_instrument_electric_domain,
    instrument_electric_end_mm,
)
from temsim.vacuum import bind_gun_environment


EDGES = np.linspace(-6e-6, 6e-6, 501)
COUNT = 32768


def histogram(x, y, weight):
    rows = [np.histogram(v, bins=EDGES, weights=weight)[0] for v in (x, y)]
    np.testing.assert_allclose(np.sum(rows, axis=1), weight.sum(), atol=2e-13, rtol=1e-11)
    return rows


def packet():
    pair = Pair()
    q, d = selected_field(pair)
    with np.load(OUT / "tip-32768.npz") as source:
        g = dict(source)
    np.testing.assert_array_equal(g["current"], q)
    if str(g["signature"]) != identity(pair, d, q, COUNT):
        raise ValueError("Selected magnetic field differs from the original source launch")
    return pair.f, q, d, g


def gun_density():
    f, q, d, g = packet()
    gun = f.state.electron_gun
    bind_gun_environment(f.state)
    configure_instrument_electric_domain(gun, instrument_electric_end_mm(f.state))
    gun._instrument_magnetic_field = MaterialGunMagnet(f, q, d)
    gun._instrument_magnetic_identity = None
    gun.emitter.ray_count = COUNT
    names = ("x_m", "y_m", "tx_rad", "ty_rad")
    launch = EmissionBundle(
        **dict(zip(names, g["launch"].T, strict=True)),
        energy_offset_ev=np.r_[f.emission.energy_offset_ev, np.zeros(17)],
        ray_id=g["ids"],
        weight=g["weight"],
    )
    print("Recovering every emitted trajectory through the accelerating gun", flush=True)
    result = _trace_feg_to_exit(gun, emitted_bundle=launch)
    actual = np.column_stack([getattr(result.exit_bundle, key) for key in (*names, "flight_time_s")])
    np.testing.assert_array_equal(actual, g["phase"])
    np.testing.assert_array_equal(result.z_mm, g["gun_z_mm"])
    current = 1e6 * result.emitted_current_a * g["weight"][:COUNT]
    rows = [histogram(x[:COUNT], y[:COUNT], current) for x, y in zip(result.x_m, result.y_m, strict=True)]
    np.savez_compressed(
        OUT / "density-gun.npz", z=result.z_mm, current_uA=rows, edges_m=EDGES, signature=g["signature"]
    )
    print("Every gun-exit coordinate and flight time exactly reproduced", flush=True)


def column_density():
    f, q, d, g = packet()
    m, all_stops = column_mesh(f, 0.05)
    # These stations are already on the executed RK mesh. The terminal phase
    # space is compared component-by-component to the first full transport.
    z = m.z[all_stops] * 1000
    stops = all_stops[np.isclose(z * 2, np.round(z * 2), rtol=0, atol=1e-9)]
    stops = np.unique(np.r_[stops, all_stops[-1]])
    with np.load(FEM) as table:
        t = ApertureTracker(
            m,
            dict(table),
            "cubic",
            source=(g["phase"], g["weight"], g["ids"], g["energy_offset_ev"], float(g["emitted_A"])),
        )
    field = round_field(f, m.nodes, q, d, 7)
    p = g["phase"]
    lost = np.full(len(p), -1, np.int32)
    phi = electric_reference(t)
    en = t.de + 300000.0
    currents = 1e6 * t.current * g["weight"][:COUNT]
    rows = []
    first = 0
    for group in np.array_split(stops, (len(stops) + 15) // 16):
        for p, lost in advance(t, field, p, en, phi, lost, first, group):
            live = lost[:COUNT] == -1
            rows.append(histogram(p[:COUNT, 0][live], p[:COUNT, 1][live], currents[live]))
        first = group[-1]
    with np.load(OUT / "column-32768-0.05-7.npz") as saved:
        np.testing.assert_array_equal(saved["surface_phase"], p)
        np.testing.assert_array_equal(saved["loss"], lost)
    np.savez_compressed(
        OUT / "density-column.npz",
        z=m.z[stops] * 1000,
        current_uA=rows,
        edges_m=EDGES,
        signature=g["signature"],
    )
    print("Every specimen position, direction, flight time and current weight reproduced", flush=True)


def render():
    with np.load(OUT / "density-gun.npz") as gun, np.load(OUT / "density-column.npz") as col:
        assert str(gun["signature"]) == str(col["signature"])
        np.testing.assert_array_equal(gun["edges_m"], col["edges_m"])
        np.testing.assert_allclose(gun["current_uA"][-1], col["current_uA"][0], atol=1e-13)
        zs = np.r_[gun["z"][:-1], col["z"]]
        rows = np.concatenate([gun["current_uA"][:-1], col["current_uA"]])
    assert np.all(np.diff(zs) > 0)
    np.testing.assert_allclose(rows.sum(axis=2), 15.0, rtol=1e-12)
    np.savez_compressed(
        EV / "all-ray-density.npz",
        z_mm=zs,
        edges_um=EDGES * 1e6,
        bin_current_uA=rows,
        axes=np.array(["x", "y"]),
        source_count=COUNT,
    )
    edges = EDGES * 1e6
    pitch = edges[1] - edges[0]
    cur = rows / pitch
    zedge = np.r_[zs[0], (zs[1:] + zs[:-1]) / 2, zs[-1]]
    ii = np.flatnonzero(rows.any(axis=(0, 1)))
    limit = max(abs(edges[ii[0]]), abs(edges[ii[-1] + 1])) + 0.12
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(
        "All 32,768 trajectories  |  projected current from source to specimen", weight="bold", fontsize=15
    )
    cm = plt.cm.inferno.copy()
    cm.set_bad("#f6f7f8")
    norm = LogNorm(vmin=15 / COUNT / pitch, vmax=float(cur.max()))
    for j, ax in enumerate(axes):
        im = ax.pcolormesh(
            zedge,
            edges,
            np.ma.masked_less_equal(cur[:, j].T, 0),
            cmap=cm,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        ax.set(ylabel=("x" if j == 0 else "y") + " (µm)", xlim=(zs[0], zs[-1]), ylim=(-limit, limit))
        ax.spines[["right", "top"]].set_visible(False)
    annotations = [
        (0, "Source"),
        (500, "C1 lens"),
        (650, "C2 lens"),
        (915, "C3 lens"),
        (1088, "ADL"),
        (1248, "Corrector"),
        (1400, "Scan"),
        (1522.5, "Mini"),
        (zs[-1], "Surface"),
    ]
    for z, label in annotations:
        axes[0].text(
            z,
            0.98,
            label,
            ha="left" if z == 0 else "right" if z == zs[-1] else "center",
            va="top",
            transform=axes[0].get_xaxis_transform(),
            fontsize=8,
            color="#344858",
        )
    axes[-1].set_xlabel("Distance from the effective source  z (mm)")
    fig.colorbar(im, ax=axes, label="Projected current density (µA/µm)", shrink=0.89)
    fig.savefig(EV / "all-ray-density.png", dpi=170)
    print("Absolute current closes at", len(zs), "planes in both projections", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["gun", "column", "render"])
    args = ap.parse_args()
    {"gun": gun_density, "column": column_density, "render": render}[args.stage]()
