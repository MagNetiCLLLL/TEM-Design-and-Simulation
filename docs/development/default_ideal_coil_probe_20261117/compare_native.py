"""Compare two transports only on identical physical source IDs and absolute weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from fit_stationary import metrics, write


def compare_guns(native, table):
    records = [json.loads((path / "gun.json").read_text()) for path in (native, table)]
    if records[0]["input_digest"] != records[1]["input_digest"]:
        raise ValueError("Physical source and column inputs differ")
    with np.load(native / "gun.npz") as a, np.load(table / "gun.npz") as b:
        common = (
            "ray_id",
            "weight",
            "alive",
            "blocked_z_mm",
            "blocked_key",
            "emission_position_m",
            "emission_direction",
            "emission_normal",
            "emission_ray_id",
            "emitted_current_a",
        )
        for key in common:
            same = (
                np.array_equal(a[key], b[key], equal_nan=True)
                if a[key].dtype.kind == "f"
                else np.array_equal(a[key], b[key])
            )
            if not same:
                raise ValueError(f"Gun records differ in {key}")
        errors = {
            key: float(np.max(abs(a[key] - b[key])))
            for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "flight_time_s")
        }
        return dict(
            originals=len(a["ray_id"]),
            identical_fields=list(common),
            max_absolute_difference=errors,
            input_digest=records[0]["input_digest"],
            native_sha256=records[0]["sha256"],
            table_sha256=records[1]["sha256"],
            field_validation=records[1]["magnetic_table"],
        )


def compare_columns(native, observation):
    record = json.loads((native / "metrics.json").read_text())
    expanded = json.loads((observation / "optics.json").read_text())
    if record.get("gun_sha256", expanded["gun_sha256"]) != expanded["gun_sha256"]:
        raise ValueError("The column comparison requires identical gun-exit data")
    with np.load(native / "surface.npz") as a, np.load(observation / "surface.npz") as b:
        if (
            str(a["input_digest"]) != str(b["input_digest"])
            or str(a["input_digest"]) != expanded["input_digest"]
            or float(a["z_mm"]) != float(b["z_mm"])
            or float(a["emitted_current_a"]) != float(b["emitted_current_a"])
        ):
            raise ValueError("Native and expanded columns differ in their physical setting")
        ids = a["ray_id"][a["alive"]]
        order = np.searchsorted(b["ray_id"], ids)
        if not np.array_equal(ids, b["ray_id"][order]) or not np.array_equal(
            a["weight"][a["alive"]], b["weight"][order]
        ):
            raise ValueError("Native and expanded columns differ in source membership or current")
        keys = ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")
        phase = np.column_stack([a[key][a["alive"]] for key in keys])
        check = metrics(b["phase"][order], b["weight"][order], float(b["emitted_current_a"]))
        errors = dict(zip(keys, np.max(abs(phase - b["phase"][order]), axis=0).tolist()))
        errors["kinetic_energy_ev"] = float(
            np.max(abs(a["kinetic_energy_ev"][a["alive"]] - b["energy_ev"][order]))
        )
    return dict(
        input_digest=expanded["input_digest"],
        matched_survivors=len(ids),
        full_gun_originals=expanded["original_emitted"]["emitted_count"],
        native_surface=record["surface"],
        expansion_surface=check,
        max_absolute_difference=errors,
        native_D95_focus=record["local_D95"],
        field_comparison=expanded.get("field_witness"),
    )


def compare_gun_effect_at_surface(native, table):
    a, b = [json.loads((d / "optics.json").read_text()) for d in (native, table)]
    if (
        a["input_digest"] != b["input_digest"]
        or not a["native_gun_field"]
        or b["native_gun_field"]
        or a["step_mm"] != b["step_mm"]
        or a["focus_spacing_nm"] != b["focus_spacing_nm"]
    ):
        raise ValueError("Require matching column settings and a direct versus tabulated gun")
    with np.load(native / "surface.npz") as ra, np.load(table / "surface.npz") as rb:
        for key in ("ray_id", "weight", "z_mm", "emitted_current_a"):
            if not np.array_equal(ra[key], rb[key]):
                raise ValueError(f"Specimen states have different {key}")
        errors = dict(
            zip(
                ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"),
                np.max(abs(ra["phase"] - rb["phase"]), axis=0).tolist(),
            )
        )
        errors["kinetic_energy_ev"] = float(np.max(abs(ra["energy_ev"] - rb["energy_ev"])))
    return dict(
        scope="Effect of gun magnetic interpolation at the specimen after identical high-order column propagation",
        input_digest=a["input_digest"],
        matched_originals=a["original_emitted"]["positive_current_count"],
        native_gun_sha256=a["gun_sha256"],
        tabulated_gun_sha256=b["gun_sha256"],
        direct_gun_surface=a["full"]["surface"],
        tabulated_gun_surface=b["full"]["surface"],
        direct_gun_focus=a["full"]["D95"],
        tabulated_gun_focus=b["full"]["D95"],
        max_absolute_difference=errors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operators = {
        "gun": compare_guns,
        "column": compare_columns,
        "gun-at-surface": compare_gun_effect_at_surface,
    }
    parser.add_argument("operator", choices=operators)
    for name in ("native", "other", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write(args.output, operators[args.operator](args.native, args.other))
