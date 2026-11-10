"""Right-handed filter branch mounted in the fixed microscope column frame.

Column +Z is downstream from the tip. Branch +X maps to column +Z,
branch +Y to column +Y, and branch -Z to column +X. The solver may use
branch coordinates internally; entrance positions, slopes and alignment
controls always use column X/Y. No reflection is part of the mounting.
"""

import numpy as np


FILTER_TO_COLUMN_ROTATION = ((0., 0., -1.), (0., 1., 0.), (1., 0., 0.))
ENTRANCE_COORDINATE_CONVENTION = "column_xy_v1"


def column_to_filter_vectors(vectors):
    """Rotate column vectors into the branch, without changing their units."""
    return np.asarray(vectors, dtype=float) @ np.asarray(FILTER_TO_COLUMN_ROTATION)


def filter_to_column_vectors(vectors):
    """Rotate branch vectors into the column, without an origin translation."""
    return np.asarray(vectors, dtype=float) @ np.asarray(FILTER_TO_COLUMN_ROTATION).T


def entrance_phase_space_vectors(x_m, y_m, tx_rad, ty_rad):
    """Map column entrance offsets/slopes to branch position and direction.

    Slopes are px/pz and py/pz in the column, as used by the upstream ray
    transport. Returned directions need not be normalized. A column +X
    offset or slope therefore has a negative branch Z component.
    """
    x, y, tx, ty = np.broadcast_arrays(x_m, y_m, tx_rad, ty_rad)
    position = np.stack((x, y, np.zeros_like(x)), axis=-1)
    direction = np.stack((tx, ty, np.ones_like(tx)), axis=-1)
    return column_to_filter_vectors(position), column_to_filter_vectors(direction)


def restore_entrance_alignment(values, convention):
    """Normalize saved controls; untagged legacy records used reflected X.

    Only saved alignment is migrated. New live/API controls are column X/Y;
    incoming source positions/slopes must never be reflected to preserve a
    previous bug. Return a copy so reading cannot rewrite the source record.
    """
    if convention not in (None, ENTRANCE_COORDINATE_CONVENTION):
        raise ValueError(f"Unsupported Energy Filter entrance coordinates: {convention!r}")
    result = dict(values)
    if convention is None and "alignment_x_mrad" in result:
        value = result["alignment_x_mrad"]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError("Saved Energy Filter X alignment must be a finite number")
        result["alignment_x_mrad"] = -float(value)
    return result
