"""Draw registered sections of the installed CESCOR solids in their native millimetres."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np

from temsim.cad_catalog import part_solids
from temsim.cad_mesh import _manifold
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.probe_corrector_fem import HP, TL


COLOURS = {
    "Ni–Fe pole pin": "#167f79",
    "Iron pole / return": "#7f8e9a",
    "Winding": "#bc643a",
    "Insulation / support": "#f2c995",
    "Vacuum liner": "#684f78",
}


def draw_section(axis, section, transform, colour):
    contours = []
    for polygon in section.to_polygons():
        if len(polygon) >= 3:
            vertices = transform(np.asarray(polygon))
            codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(polygon) - 1) + [PlotPath.CLOSEPOLY]
            contours.append(PlotPath(np.vstack((vertices, vertices[0])), codes))
    if contours:
        axis.add_patch(
            PathPatch(
                PlotPath.make_compound_path(*contours), facecolor=colour, edgecolor="#38434a", linewidth=0.3
            )
        )


def main():
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {p["key"]: p for p in doc["parts"]}
    hp2 = parts["probe_hp2_hexapole"]
    center = hp2["local_center_z_mm"]
    geometry = doc["geometry"]
    (tube,) = tube_solids(
        {
            **geometry,
            "key": CONTINUOUS_TUBE_KEY,
            "mechanical_inner_diameter_mm": geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"],
        }
    )
    shapes = [(tube.shape, "Vacuum liner", True)]
    for key, row in parts.items():
        if key not in (*HP, "probe_dph1_coils", "probe_dph2_coils") and row.get("parent_key") not in TL:
            continue
        for solid in part_solids(row, parts):
            if solid.material == "soft_magnetic":
                kind = (
                    "Ni–Fe pole pin"
                    if key in HP and solid.region.startswith("pole_pin_")
                    else "Iron pole / return"
                )
            elif solid.material == "copper_winding":
                kind = "Winding"
            elif key in HP or key in ("probe_dph1_coils", "probe_dph2_coils"):
                kind = "Insulation / support"
            else:
                continue
            shapes.append((solid.shape, kind, key in ("probe_hp2_hexapole", "probe_dph2_coils")))
    fig, (side, end) = plt.subplots(1, 2, figsize=(11.6, 5.1), gridspec_kw={"width_ratios": [1.5, 1.0]})
    origin = 450.0

    def longitudinal(p):
        return np.column_stack((origin - p[:, 1], p[:, 0]))

    for shape, kind, at_hp2 in shapes:
        body = _manifold(shape, 112)
        section = body.rotate((0.0, 0.0, -30.0)).rotate((90.0, 0.0, 0.0)).slice(0.0)
        draw_section(side, section, longitudinal, COLOURS[kind])
        if at_hp2:
            draw_section(end, body.slice(center), lambda p: p, COLOURS[kind])
    for key, label in (
        ("probe_hp2_hexapole", "HP2"),
        ("probe_tl22_lens", "TL22"),
        ("probe_tl21_lens", "TL21"),
        ("probe_hp1_hexapole", "HP1"),
    ):
        z = parts[key]["local_center_z_mm"] + origin
        side.plot([z, z], [87, 90], color="#485660", lw=0.6)
        side.text(z, 92, label, ha="center", fontsize=9)
    side.axhline(0, color="#949494", lw=0.45, ls="--", zorder=0)
    side.set(
        xlim=(1120, 1327),
        ylim=(-91, 99),
        xlabel="Column z (mm)",
        ylabel="Signed transverse position (mm)",
        title="Longitudinal section through an HP2 pole pair",
    )
    end.annotate(
        "", xy=(76, 82), xytext=(-76, 82), arrowprops={"arrowstyle": "<->", "color": "#485660", "lw": 0.85}
    )
    end.text(0, 84, "152 mm yoke OD", ha="center", fontsize=9)
    end.annotate(
        "8 mm pole bore",
        xy=(2.5, -3.1),
        xytext=(10, -42),
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        arrowprops={"arrowstyle": "-", "color": "#485660", "lw": 0.7},
    )
    end.set(
        xlim=(-81, 81), ylim=(-81, 92), xlabel="x (mm)", ylabel="y (mm)", title="HP2 transverse section"
    )
    for axis in (side, end):
        axis.set_aspect("equal")
        axis.spines[["right", "top"]].set_visible(False)
        axis.tick_params(labelsize=8)
    fig.legend(
        [Patch(facecolor=c, edgecolor="none") for c in COLOURS.values()],
        list(COLOURS),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.17, wspace=0.19)
    fig.savefig(Path(__file__).with_name("corrector-sections.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
