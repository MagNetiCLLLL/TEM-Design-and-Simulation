"""Bounded face resolution for annular conductors, in metres.

This only chooses numerical nodes. Electrode faces, bores, voltages and the
vacuum Laplace boundary problem retain their original physical definitions.
"""
from __future__ import annotations

import math


MAX_VERTICES = 1_500_000


def electrode_axial_bands(start, stop, inner, cells_per_bore, coarse_step):
    """Keep four bore radii of fringe resolution at each electrode face.

    Short electrodes retain the previous uniform band exactly. In a long
    narrow conducting channel the two face layers are separated: interior
    target cell widths grow by at most 20 percent towards the middle, capped
    by the larger of the coarse scale and thickness / (2*cells_per_bore).
    Thus shrinking the hole does not demand a number
    of axial cells proportional to thickness / radius throughout the metal.
    The channel remains vacuum in the solve; no field is set to a constant.
    """
    step = inner / cells_per_bore
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Unresolvable annular-electrode mesh scale")
    if stop-start <= 8*inner:
        return [(start-4*inner, stop+4*inner, step)]
    left, right = start+4*inner, stop-4*inner
    bands = [(start-4*inner, left, step), (right, stop+4*inner, step)]
    midpoint = left + (right-left)/2
    interior_step = max(coarse_step, (stop-start)/(2*cells_per_bore))
    position, width = left, step
    while position < midpoint:
        following = min(midpoint, position+width)
        if following <= position or len(bands) >= MAX_VERTICES//3:
            raise ValueError("Unresolvable annular-electrode graded mesh")
        mirrored_left, mirrored_right = right-(following-left), right-(position-left)
        if mirrored_right <= mirrored_left:
            raise ValueError("Unresolvable annular-electrode graded mesh")
        # Use the represented interval widths, so ceil(width/step) cannot
        # spuriously split one cell into two after coordinate subtraction.
        bands.extend(((position, following, following-position),
                      (mirrored_left, mirrored_right, mirrored_right-mirrored_left)))
        position = following
        width = min(interior_step, width*1.2)
    return bands
