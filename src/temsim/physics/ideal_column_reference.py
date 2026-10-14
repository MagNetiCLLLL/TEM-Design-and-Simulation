"""An explicit first-order reference of the captured post-gun column.

This counterfactual keeps the same source, energy-dependent linear optics,
defocus, drives and material boundaries. It removes declared lens defects
and nonlinear corrector actions; it is not an achromatic replacement source.
"""
from dataclasses import is_dataclass, replace
from types import SimpleNamespace

import numpy as np

from temsim.immutable_json import json_digest


COLUMN_MODELS = ("physical", "first_order_reference")
REFERENCE_MODEL_VERSION = "captured-column-first-order-c1-v1"


def validate_column_model(model):
    if model not in COLUMN_MODELS:
        raise ValueError("Column model must be physical or first_order_reference")
    return model


def column_model_record(model):
    """JSON-compatible definition, also included in the transformed identity."""
    validate_column_model(model)
    if model == "physical":
        return {"column_model": model}
    return {
        "column_model": model,
        "model_version": REFERENCE_MODEL_VERSION,
        "preserved": ["executed gun-exit modes", "per-mode energy dependence",
            "main-lens focusing and rotation", "intrinsic C1 defocus",
            "quadrupoles and stigmators", "deflections and dynamic drives",
            "apertures and mechanical bores", "physical axial grid"],
        "removed": ["lens Cs", "posed spherical kicks", "intrinsic A1 and higher terms",
            "hexapole nonlinear actions"],
        "limitations": ["single-energy first-order lens reference; not achromatic across modes",
            "native aligned analytical column only; mapped magnetic fields are unsupported"],
    }


def column_plan_record(plan):
    model = ("first_order_reference" if REFERENCE_MODEL_VERSION in
             getattr(plan, "solver_signature", "") else "physical")
    return column_model_record(model)


def column_plan_for_model(plan, model):
    """Return an immutable reference plan on the exact physical plan grid.

    No instrument setting is changed. In particular, removing a screen does
    not rebuild the axial grid or move an aperture. Posed/mapped magnetic
    fields may have nonlinear residuals outside the explicit kick arrays and
    cannot be certified as first order by this transformation.
    """
    validate_column_model(model)
    if model == "physical":
        return plan
    if getattr(plan, "mapped_fields", ()):
        raise ValueError("First-order reference does not support mapped or posed magnetic fields; "
                         "their higher-order residuals cannot be removed by the reference model")
    # Wrappers and streamed segments may receive an already transformed plan.
    if REFERENCE_MODEL_VERSION in getattr(plan, "solver_signature", ""):
        return plan
    intrinsic = []
    for kick in getattr(plan, "intrinsic_aberration_kicks", ()):
        c1 = dict(kick.coefficients).get("c1_mm", 0.)
        if c1:
            intrinsic.append(replace(kick, coefficients=(("c1_mm", c1),)))
    zeros = {}
    for name in ("hex_normal_m3", "hex_skew_m3", "midpoint_hex_normal_m3",
                 "midpoint_hex_skew_m3", "cs_kick_m3"):
        if not hasattr(plan, name):
            continue  # A streamed slice carries only its executed coefficients.
        array = np.zeros_like(getattr(plan, name))
        array.flags.writeable = False
        zeros[name] = array
    policy = column_model_record(model)
    values = dict(**zeros, posed_spherical_kicks=(),
        intrinsic_aberration_kicks=tuple(intrinsic), hexapole_exit_nodes=(),
        solver_signature=repr((getattr(plan, "solver_signature", ""), REFERENCE_MODEL_VERSION)),
        signature=json_digest((plan.signature, policy)))
    return replace(plan, **values) if is_dataclass(plan) else SimpleNamespace(**{**vars(plan), **values})
