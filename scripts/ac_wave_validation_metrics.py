"""Native-grid intensity diagnostics for the bounded AC wave validation.

This module only reads executed checkpoints. It neither recentres the field,
rescales its probability nor remaps it onto a common display grid. Temporary
sorted radius/CDF runs make mixed-mode encircled probabilities independent of
the number and size of simultaneously resident modes.
"""
from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.physics.beam_comparison import wave_plane_moments
from temsim.physics.column_wave import _previous_support_error
from temsim.physics.wave_mode_support import incoherent_mode_approximation


_MAXIMUM_CHUNK_CELLS = 262_144


def _acceptance_stages(record):
    """Keep aperture-conditioned empirical estimates distinct from L2 bounds."""
    from collections.abc import Mapping
    found = {}

    def visit(value):
        if isinstance(value, Mapping):
            if value.get('schema') == 'aperture-conditioned-cartesian-window-v1' and 'wall_estimate' in value:
                key = value.get('span_identity'), value.get('upstream_digest')
                if key not in found:
                    guards = [check for mode in value.get('modes', ()) for check in mode.get('guard_checks', ())]
                    found[key] = dict(
                        start_z_mm=value['start_z_mm'], stop_z_mm=value['stop_z_mm'],
                        numerics=thaw_json(value['numerics']), wall_estimate=thaw_json(value['wall_estimate']),
                        maximum_guard_complex_change=max((g['relative_complex_change'] for g in guards), default=0.),
                        maximum_guard_current_change=max((g['relative_current_change'] for g in guards), default=0.),
                        scope=value['scope'])
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(record)
    return list(found.values())


def _quantile(runs, fraction, total, maximum):
    """Return a represented-cell radius, not an interpolated display radius."""
    target = fraction * total

    def enclosed(radius):
        value = 0.0
        for run in runs:
            index = int(np.searchsorted(run[0], radius, side="right"))
            if index:
                value += float(run[1, index - 1])
        return value

    if enclosed(0.0) >= target:
        return 0.0
    lower, upper = 0.0, maximum
    # Keep F(lower) < target <= F(upper). At the end select the smallest
    # actual sample above lower; this also handles gaps and tied radii.
    for _ in range(96):
        middle = lower + (upper - lower) * 0.5
        if middle == lower or middle == upper:
            break
        if enclosed(middle) >= target:
            upper = middle
        else:
            lower = middle
    candidates = []
    for run in runs:
        index = int(np.searchsorted(run[0], lower, side="right"))
        if index < run.shape[1]:
            candidates.append(float(run[0, index]))
    result = min(candidates) if candidates else upper
    if enclosed(result) < target:
        raise ArithmeticError("Native-grid radius quantile failed its cumulative probability check")
    return result


def checkpoint_metrics(checkpoint):
    """JSON-ready intensity moments and radii on every mode's physical grid.

    ``PlaneWave.amplitude`` contains cell probability amplitudes: mode weights
    multiply abs(amplitude)**2, with no extra area factor. Radii are measured
    about the full mixed-intensity centroid, including offsets between modes.
    Edge fractions use the outer ceil(1% * size) cells on each lattice edge;
    they diagnose a finite computational window, not physical absorption.
    """
    result = thaw_json(wave_plane_moments(checkpoint))
    total = float(result["source_fraction"])
    centre = result["centroid_xy_m"]
    result.update(
        schema="ac-wave-native-intensity-metrics-v1",
        total_probability=total,
        mode_count=int(result["population_count"]),
        centroid_xy_nm=None if centre is None else [float(v) * 1e9 for v in centre],
        rms_radius_nm=None if result["radial_rms_m"] is None else result["radial_rms_m"] * 1e9,
        modes=[],
        radius_definition="Weighted represented-cell radius about the mixed laboratory-XY intensity centroid",
        edge_definition="Outer ceil(1% of each lattice dimension), at least one cell; corners counted once",
    )
    ledgers = []
    for mode in checkpoint.beam.modes:
        ledger = _previous_support_error(checkpoint.record, mode.mode_id)
        if ledger is not None:
            ledgers.append(dict(mode_id=mode.mode_id, **ledger))
    result['support_approximation'] = dict(
        modes=ledgers,
        probability_loss_per_tip_electron=sum(row['probability_loss_per_tip_electron'] for row in ledgers),
        maximum_accumulated_relative_l2=max((row['accumulated_relative_l2_bound'] for row in ledgers), default=0.),
        intensity_l1_bound_per_tip_electron=sum(row['reference_mode_weight']*(
            2*row['accumulated_relative_l2_bound']+row['accumulated_relative_l2_bound']**2) for row in ledgers),
        scope='Latest ledger for each current mode, counted once. Numerical support approximation only, not physical absorption or total propagation error.')
    result['incoherent_mode_approximation'] = incoherent_mode_approximation(checkpoint.record)
    result['aperture_conditioned_approximation'] = _acceptance_stages(checkpoint.record)
    for fraction in (50, 80, 95):
        result[f"r{fraction}_m"] = result[f"r{fraction}_nm"] = None
    maximum = weighted_total = edge_total = 0.0
    runs = []
    try:
        with tempfile.TemporaryDirectory(prefix="temsim-ac-wave-metrics-") as directory, ExitStack() as mappings:
            paths = []
            for mode_index, mode in enumerate(checkpoint.beam.modes):
                plane = mode.plane
                ny, nx = plane.amplitude.shape
                rows = max(1, min(64, _MAXIMUM_CHUNK_CELLS // nx))
                x = np.arange(nx, dtype=float) - nx // 2
                ex, ey = max(1, math.ceil(nx * 0.01)), max(1, math.ceil(ny * 0.01))
                norm = edge = 0.0
                weight = float(mode.weight_per_reference_electron)
                if weight > 0:
                    for first in range(0, ny, rows):
                        last = min(ny, first + rows)
                        mass = np.abs(plane.amplitude[first:last]) ** 2 * weight
                        norm += float(mass.sum())
                        row_numbers = np.arange(first, last)
                        outer_rows = (row_numbers < ey) | (row_numbers >= ny - ey)
                        edge += float(mass[outer_rows].sum())
                        middle_mass = mass[~outer_rows]
                        if 2 * ex >= nx:
                            edge += float(middle_mass.sum())
                        else:
                            edge += float(middle_mass[:, :ex].sum() + middle_mass[:, -ex:].sum())
                        if centre is None:
                            continue
                        y = row_numbers - ny // 2
                        b, origin = plane.basis_m, plane.origin_m
                        dx = origin[0] - centre[0] + b[0, 0] * x[None, :] + b[0, 1] * y[:, None]
                        dy = origin[1] - centre[1] + b[1, 0] * x[None, :] + b[1, 1] * y[:, None]
                        radius = np.hypot(dx, dy).ravel()
                        mass = mass.ravel()
                        positive = mass > 0
                        if not np.any(positive):
                            continue
                        radius, mass = radius[positive], mass[positive]
                        order = np.argsort(radius)
                        radius, cumulative = radius[order], np.cumsum(mass[order])
                        maximum = max(maximum, float(radius[-1]))
                        path = Path(directory) / f"m{mode_index}-r{first}.npy"
                        # Rows are contiguous so repeated searchsorted calls
                        # do not copy a strided, potentially large radius array.
                        np.save(path, np.stack((radius, cumulative)), allow_pickle=False)
                        paths.append(path)
                weighted_total += norm
                edge_total += edge
                result["modes"].append(dict(
                    mode_id=str(mode.mode_id), energy_kev=float(mode.energy_kev),
                    grid_shape=[int(ny), int(nx)], probability=norm,
                    declared_weight=weight, origin_m=plane.origin_m.tolist(),
                    basis_m=plane.basis_m.tolist(), edge_1pct_probability=edge,
                    edge_1pct_fraction=None if norm == 0 else edge / norm,
                ))
                del mode, plane
            if not math.isclose(weighted_total, total, rel_tol=1e-10, abs_tol=1e-14):
                raise ArithmeticError("Native-grid radius and moment probabilities disagree")
            if total > 0:
                for path in paths:
                    run = np.load(path, mmap_mode="r", allow_pickle=False)
                    mappings.callback(run._mmap.close)
                    runs.append(run)
                for fraction in (50, 80, 95):
                    radius = _quantile(runs, fraction / 100.0, total, maximum)
                    result[f"r{fraction}_m"] = radius
                    result[f"r{fraction}_nm"] = radius * 1e9
            result["edge_1pct_probability"] = edge_total
            result["edge_1pct_fraction"] = None if total == 0 else edge_total / total
            # Windows forbids deleting an open mapped file. Close mappings
            # explicitly before TemporaryDirectory removes only its own data.
            for run in runs:
                run._mmap.close()
            runs.clear()
    finally:
        for run in runs:
            run._mmap.close()
    return result
