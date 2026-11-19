"""Detached field-only current calibration for fixed physical round windings.

Only an isolated native Gaussian Bz at one stated axial plane is matched.
The winding envelope and integer turns are inputs, never fit parameters. This
module does not propagate particles, qualify focus, or edit the supplied state.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import math
from numbers import Real

import numpy as np
from scipy.optimize import minimize_scalar

from temsim import input_io
from temsim.coil_reference_metadata import ROUND_LENS_PROFILES, round_coil_operating_point
from temsim.coil_winding import OWNER_CURRENT_FIELD
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FrozenAnalyticField, GeometryAwareAnalyticFieldProvider,
    resolve_runtime_lens_field_provider,
)
from temsim.physics.potential_capture import capture_round_lens_potential


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def current_for_field_match(target_bz_t, unit_current_bz_t, *, operating_percent=100.,
                            polarity=1, enabled=True):
    """Solve Bz = Bz_per_signed_ampere * I100 * percent/100 * polarity.

``unit_current_bz_t`` is the field from +1 A actual conductor current at the
same plane and fixed winding definition. The returned reference is nonnegative;
the actual current is signed. Zero/disabled excitation cannot identify I100.
There is no guessed current-rating cap or minimum field-magnitude tolerance.
"""
    target = _finite(target_bz_t, "Target Bz")
    unit = _finite(unit_current_bz_t, "Unit-current Bz")
    percent = _finite(operating_percent, "Operating excitation")
    if not 0 < percent <= 100:
        raise ValueError("Current calibration requires operating excitation in (0, 100]%")
    if not isinstance(enabled, (bool, np.bool_)) or not enabled:
        raise ValueError("Disabled excitation cannot identify a reference current")
    if isinstance(polarity, (bool, np.bool_)) or polarity not in (-1, 1):
        raise ValueError("Current calibration requires polarity +/-1")
    if unit == 0:
        raise ValueError("Zero unit-current field cannot identify a current")
    signed = target / unit
    scale = percent * .01 * int(polarity)
    if scale == 0:
        raise ValueError("Excitation scaling underflows the finite numerical range")
    reference = signed / scale
    if not (math.isfinite(signed) and math.isfinite(reference)):
        raise ValueError("Field-matched current exceeds the finite numerical range")
    if target != 0 and (signed == 0 or reference == 0):
        raise ValueError("Field-matched current underflows the finite numerical range")
    if reference < 0:
        raise ValueError("Target field sign is incompatible with the fixed winding polarity")
    return dict(current_at_100_percent_a=reference, current_a=signed,
                target_bz_t=target, unit_current_bz_t=unit,
                operating_percent=percent, polarity=int(polarity))


def _detached_state(state):
    detached = copy(state)
    detached.lenses = deepcopy(state.lenses)
    # Every provider cache/diagnostic written by the field resolver belongs to
    # this copy. No getter or mutable cache is touched on the supplied state.
    for name in ("_runtime_lens_field_provider_cache", "_runtime_copper_coil_provider_cache",
                 "_runtime_posed_nonlinear_provider_cache", "_field_provider_diagnostics"):
        setattr(detached, name, {})
    detached.simulation_mode = 'unified'
    detached.round_lens_field_model = 'gaussian'
    return detached


def _native(state, key):
    lens = next((item for item in state.lenses if item.key == key), None)
    if lens is None:
        raise ValueError(f"{key}: no native round lens is installed")
    return lens, state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens


def _gaussian_provider(state, key):
    _, native = _native(state, key)
    provider = resolve_runtime_lens_field_provider(state, key, native)
    if type(provider) is not GeometryAwareAnalyticFieldProvider:
        raise ValueError(f"{key}: current calibration requires a native Gaussian field")
    if provider.registration != CoordinateRegistration():
        raise ValueError(f"{key}: current calibration currently requires an aligned, unshifted source axis")
    return provider


def _gaussian_peak_plane(provider):
    frozen = FrozenAnalyticField.from_provider(provider)
    # Include every term centre and densely bracket each term's full native
    # support so separated lobes are searched independently. No source field
    # is truncated by this peak-location search.
    nodes = np.unique(np.concatenate([
        (centre + sigma * np.linspace(-10., 10., 321)) * 1e3
        for _, centre, sigma in frozen.terms_t_m]))
    values = np.abs(provider.magnetic_field_t(nodes))
    if not np.isfinite(values).all() or np.max(values) == 0:
        raise ValueError("The native Gaussian has no nonzero finite calibration field")
    candidates = [(float(values[index]), float(nodes[index])) for index in (0, len(nodes)-1)]
    for index in range(1, len(nodes)-1):
        if values[index] > 0 and values[index] >= values[index-1] and values[index] >= values[index+1]:
            fit = minimize_scalar(lambda z: -abs(float(provider.magnetic_field_t(z))),
                                  bounds=(nodes[index-1], nodes[index+1]), method='bounded',
                                  options={'xatol': 1e-10})
            if not fit.success:
                raise ValueError("Native Gaussian peak-plane search did not converge")
            candidates.append((-float(fit.fun), float(fit.x)))
            candidates.append((float(values[index]), float(nodes[index])))
    magnitude, plane = max(candidates, key=lambda row: (row[0], -row[1]))
    if magnitude == 0:
        raise ValueError("The native Gaussian has no nonzero calibration field")
    return plane, [float(nodes[0]), float(nodes[-1])]


@input_io.using_state_inputs
def calibrate_round_lens_current(state, lens_key, *, matching_plane_z_mm=None):
    """Return a current-only recipe without changing any input or live cache.

All owned windings must already declare valid N. An explicit plane matches
the original signed Gaussian Bz there; otherwise the original Gaussian's
absolute-peak plane is found. Reference excitation is explicitly normalized
to enabled 100%, including when the working-point channel is off or zero.
The original percentage/polarity/enabled state is never changed. Mapped/FEM
recipes are rejected before any solve, load or Gaussian substitution.
"""
    key = str(lens_key)
    assembly = getattr(state, '_resolved_assembly', None)
    if assembly is None:
        raise ValueError("Current calibration requires resolved winding geometry")
    owner = next((p for p in assembly.parts if p.key == key), None)
    if owner is None or owner.data.get('mechanical_profile') not in ROUND_LENS_PROFILES:
        raise ValueError(f"{key}: select an installed round-lens owner")
    descriptor = getattr(state, 'lens_field_map_descriptors', {}).get(key, {})
    if (descriptor and descriptor.get('solver') not in {'analytic', 'analytical'}
            or key in (getattr(state, '_lens_field_map_bindings', {}) or {})):
        raise ValueError(f"{key}: mapped or FEM recipes require their own explicit calibration")
    windings = [dict(key=p.key, data=p.data) for p in assembly.parts
                if p.data.get('mechanical_profile') == 'magnetic_excitation_coil'
                and p.data.get('field_source_key') == key]
    operating = round_coil_operating_point({**owner.data, OWNER_CURRENT_FIELD: 1.}, windings)
    original = _detached_state(state)
    original_lens, _ = _native(original, key)
    percent = _finite(original_lens.percent, 'Original excitation')
    polarity = original_lens.polarity
    if not 0 <= percent <= 100 or isinstance(polarity, (bool, np.bool_)) or polarity not in (-1, 1):
        raise ValueError(f"{key}: original excitation/polarity is invalid")
    enabled = bool(original_lens.enabled)
    original_provider = _gaussian_provider(original, key)
    reference = _detached_state(state)
    reference_lens, _ = _native(reference, key)
    reference_lens.percent = 100.; reference_lens.enabled = True
    gaussian = _gaussian_provider(reference, key)
    if matching_plane_z_mm is None:
        plane, search = _gaussian_peak_plane(gaussian)
        condition = 'native_gaussian_absolute_peak_plane'
    else:
        plane = _finite(matching_plane_z_mm, 'Matching plane z in mm')
        search = None; condition = 'explicit_axial_plane'
    target = float(gaussian.magnetic_field_t(plane))
    if not math.isfinite(target) or target == 0:
        raise ValueError(f"{key}: the selected plane has no nonzero finite Gaussian reference field")
    # Check the existing native law's linear excitation assumption instead of
    # extrapolating a nonlinear or overridden field method silently.
    reference_lens.percent = 50.
    half = float(gaussian.magnetic_field_t(plane))
    reference_lens.percent = 100.
    if not math.isclose(half, target*.5, rel_tol=2e-14, abs_tol=0):
        raise ValueError(f"{key}: native excitation is not linear at the reference plane")
    working_target = float(original_provider.magnetic_field_t(plane))
    expected_working = target*(percent*.01) if enabled else 0.
    if not math.isclose(working_target, expected_working, rel_tol=2e-14, abs_tol=0):
        raise ValueError(f"{key}: native excitation does not follow the declared linear Gaussian reference")
    unit = _detached_state(reference); unit.round_lens_field_model = 'copper_coil'
    unit_lens, _ = _native(unit, key)
    unit_lens.percent = 100.; unit_lens.enabled = True; unit_lens.polarity = 1
    unit._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, OWNER_CURRENT_FIELD: 1.}) if p.key == key else p for p in assembly.parts))
    potential = capture_round_lens_potential(unit, key).potential
    unit_field = float(potential.sample([[0., 0., plane*.001]])['B_t'][0, 2])
    solved = current_for_field_match(target, unit_field, polarity=polarity)
    current = solved['current_at_100_percent_a']*(percent*.01)*int(polarity) if enabled else 0.
    if not math.isfinite(current):
        raise ValueError(f"{key}: operating current exceeds the finite numerical range")
    return dict(schema='round-lens-single-plane-current-calibration-v1', lens_key=key,
                matching_condition=condition, matching_plane_z_mm=plane,
                peak_search_interval_mm=search,
                current_at_100_percent_a=solved['current_at_100_percent_a'], current_a=current,
                reference_target_bz_t=target, reference_excitation_percent=100., reference_enabled=True,
                reference_polarity=int(polarity), original_excitation_percent=percent,
                original_enabled=enabled, original_bz_t=working_target,
                calibrated_operating_bz_t=unit_field*current, unit_current_bz_t=unit_field,
                zero_operating_excitation=(not enabled or percent == 0),
                total_turn_count=operating['total_turn_count'],
                winding_turn_counts={p['key']: int(p['data']['winding_turn_count']) for p in windings},
                quadrature_orders=[coil.quadrature_order for coil in potential.coils],
                model='ideal_prescribed_current_coils_in_vacuum', pole_piece_magnetism=False,
                scope='Isolated field-only scalar match; no focus or all-column equivalence',
                inputs_unchanged=True)
