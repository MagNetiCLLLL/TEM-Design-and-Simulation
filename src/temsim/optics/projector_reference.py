"""Receiver-aware first-order Image focus with the incident optics held fixed.

The stored Image currents are a Camera reference. Moving the observation to
the upstream screen changes the transfer. Only D/I are refocused here; the
Objective, illumination, receiver insertion, scan calibration and P1/P2 stay
at their captured values. This is not a nonlinear aberration correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.optical_tuning import check_tuning_cancelled
from temsim.physics.recording_stop import tem_projection_reference_plane
from temsim.physics.scan_geometry import IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD


@dataclass(frozen=True)
class ImageProjection:
    receiver_key: str
    receiver_name: str
    receiver_z_mm: float
    receiver_inserted: bool
    magnification: float
    target_magnification: float
    image_residual_m_per_rad: float
    validation_step_mm: float

    @property
    def summary(self) -> str:
        target = self.receiver_name + ("" if self.receiver_inserted else " optical reference (retracted)")
        return (f"First-order sample image at {target}: {self.magnification:.6g}×; "
                f"image residual {self.image_residual_m_per_rad:.3g} m/rad. "
                "Objective and illumination retained; higher-order aberrations are not corrected.")


def focus_image_projector(state) -> ImageProjection:
    """Adapt already-applied Image currents to the selected recording plane.

    Preserve the sample magnification of the stored Camera reference at this
    same working point. Search on a private graph and commit D/I only after an
    independent finer-step sample-canonical transfer passes. Outer preset
    transactions also roll back the initially applied stored currents on error.
    """
    from temsim.optics.direct_alignment import _LiveFirstOrderModel, diffraction_transfer

    check_tuning_cancelled(state)
    work = capture_instrument_snapshot(state).restore()
    # These tiny Jacobian bundles are a CPU solve even when particle transport
    # is configured for CUDA. The caller's execution preference is untouched.
    work.acceleration_enabled = True
    work.acceleration_backend = "Numba CPU"
    work.step_mm = 0.05
    target = tem_projection_reference_plane(work)
    reference = diffraction_transfer(work, float(work.camera.z_mm))
    magnification = math.sqrt(abs(float(np.linalg.det(reference.j_img))))
    if not math.isfinite(magnification) or magnification <= 1.0e-10:
        raise ValueError("The stored Camera Image reference has no usable sample magnification")
    keys = ("diffraction_lens", "intermediate_lens")
    lenses = {str(lens.key): lens for lens in work.lenses}
    seed = np.asarray([lenses[key].percent for key in keys], dtype=float)
    use_equivalent = bool(getattr(work, "equivalent_image_lenses_enabled", False))
    if use_equivalent:
        # Respect an explicitly selected equivalent-lens model. Do not silently
        # replace it with distributed fields during a projector-only switch.
        model = None
        upper = np.asarray([lenses[key].max_percent for key in keys], dtype=float)
    else:
        model = _LiveFirstOrderModel(work, float(work.sample.z_mm), float(target.z_mm),
                                     keys, step_mm=0.05)
        upper = model.upper

    def blocks(vector):
        check_tuning_cancelled(state)
        if model is not None:
            return model.canonical_position_blocks(vector)
        for key, value in zip(keys, vector):
            lenses[key].percent = float(value)
        transfer = diffraction_transfer(work, float(target.z_mm))
        return transfer.j_img, transfer.j_diff_m_per_rad

    def residual(vector):
        a, b = blocks(vector)
        scale = max(math.sqrt(abs(float(np.linalg.det(a)))), 1.0e-30)
        return np.r_[math.log(scale / magnification), np.asarray(b).ravel() / 1.0e-3]

    solution = least_squares(residual, seed, bounds=(np.zeros(2), upper),
                             max_nfev=100, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11)
    for key, value in zip(keys, solution.x):
        lenses[key].percent = float(value)
    check_tuning_cancelled(state)
    work.step_mm = 0.025
    validation = diffraction_transfer(work, float(target.z_mm))
    actual = math.sqrt(abs(float(np.linalg.det(validation.j_img))))
    error = float(np.linalg.norm(validation.j_diff_m_per_rad, ord=2))
    if (not solution.success or not math.isfinite(actual) or not math.isfinite(error)
            or abs(actual / magnification - 1.0) > 1.0e-3
            or error > IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD):
        raise ValueError(
            f"Image focus at {target.name} could not preserve the incident working point: "
            f"sample-image residual {error:.6g} m/rad, magnification {actual:.6g} "
            f"(target {magnification:.6g}); previous settings retained")
    result = ImageProjection(str(target.key), str(target.name), float(target.z_mm),
                             bool(getattr(target, "inserted", False)), actual,
                             magnification, error, float(work.step_mm))
    check_tuning_cancelled(state)
    live = {str(lens.key): lens for lens in state.lenses}
    for key, value in zip(keys, solution.x):
        live[key].percent = float(value)
    return result
