"""Supplemental lens markers using the shared current-weighted waist diagnostic."""

from temsim.physics.beam_waist import branch_waist_candidates


def first_crossover_after_lens(branch, lens_z_mm, name, stop_z_mm=None,
                               minimum_rays=5, start_margin_mm=0.25):
    """Return the first current-weighted plot-node waist downstream of a lens.

    Use the same positive-current population across the three-node bracket as
    the main waist markers. The historical ``verified`` key denotes a sampled
    RMS minimum with converging-to-diverging correlation, not precise focus
    qualification. The optional stop remains an exclusive search limit.
    """
    candidates = branch_waist_candidates(
        branch,
        z_min_mm=float(lens_z_mm) + float(start_margin_mm),
        z_max_mm=float("inf") if stop_z_mm is None else float(stop_z_mm),
        min_rays=int(minimum_rays),
    )
    first = next(candidates, None)
    if first is None:
        return None
    return {
        "name": name,
        "z_mm": first["z_mm"],
        "rms_radius_mm": first["rms_radius_mm"],
        "verified": True,
        "source_lens_z_mm": float(lens_z_mm),
    }
