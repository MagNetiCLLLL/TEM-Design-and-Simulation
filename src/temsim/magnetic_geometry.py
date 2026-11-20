"""Shared physical material intervals, independent of display envelopes."""

import math


CORRECTOR_SPLIT_RETURNS = frozenset({"pole_sleeves", "radial_return_plates"})


def corrector_return_extent(part):
    """Derive the installed envelope from the two ordered material intervals."""
    if (part.get("mechanical_part_role") not in CORRECTOR_SPLIT_RETURNS
            or part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"):
        return {}
    message = f"{part['key']}: expected two separate, ordered upper and lower return intervals"
    intervals = part.get("material_intervals_mm")
    if not isinstance(intervals, (tuple, list)) or len(intervals) != 2:
        raise ValueError(message)
    for interval in intervals:
        if (not isinstance(interval, (tuple, list)) or len(interval) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in interval)):
            raise ValueError(message)
    (start, upper_end), (lower_start, end) = intervals
    try:
        if not (all(math.isfinite(v) for pair in intervals for v in pair)
                and start < upper_end < lower_start < end):
            raise ValueError(message)
    except OverflowError as exc:
        raise ValueError(message) from exc
    return dict(local_start_z_mm=start, local_center_z_mm=(start+end)/2,
                local_end_z_mm=end, length_mm=end-start)


def validate_corrector_return_extent(part):
    extent = corrector_return_extent(part)
    if any(not math.isclose(part[field], value, abs_tol=1e-8, rel_tol=0)
           for field, value in extent.items()):
        raise ValueError(f"{part['key']}: return envelope must follow its upper and lower material intervals")


def objective_layer_intervals_mm(parent_data, parent_start_mm, profile):
    fields = ("local_start_z_mm", "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
              "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm")
    if any(key not in parent_data for key in fields):
        return ()
    origin = float(parent_start_mm)-float(parent_data["local_start_z_mm"])
    inset = float(parent_data.get("mechanical_coil_axial_inset_mm", 0)) if profile == "magnetic_excitation_coil" else 0
    intervals = tuple((origin+float(parent_data[f"{side}_yoke_start_local_z_mm"])+inset,
                       origin+float(parent_data[f"{side}_yoke_end_local_z_mm"])-inset)
                      for side in ("upper", "lower"))
    if any(end <= start for start, end in intervals):
        raise ValueError("Objective layer intervals must have positive length")
    return intervals
