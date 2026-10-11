import math
from dataclasses import replace

import numpy as np
import pytest

from temsim.operating_modes import direct_alignment_by_key
from temsim.optics.direct_alignment_precalibration import (
    interpolated_precalculated_seed,
    interpolated_nanoprobe_seed,
    precalculated_alignment_points,
    precalculated_alignment_ratios,
)


def test_image_precalculation_keeps_lm_and_normal_branches_separate():
    definition = direct_alignment_by_key("image_magnification")
    points = precalculated_alignment_points(definition)

    assert [point.branch for point in points] == [
        "lm", "lm", "lm", "normal", "normal", "normal"
    ]
    ratios = precalculated_alignment_ratios(definition)
    assert len(ratios) == 4
    assert not any(
        row.lower_target == 1000.0 and row.upper_target == 10000.0
        for row in ratios
    )


def test_log_midpoint_seed_is_the_vector_midpoint():
    definition = direct_alignment_by_key("diffraction_camera_length")
    points = precalculated_alignment_points(definition)
    left = points[1]
    right = points[2]
    target = math.sqrt(left.target * right.target)
    lower = np.zeros(4)
    upper = np.full(4, 100.0)

    seed = interpolated_precalculated_seed(
        definition, target, lower, upper
    )

    assert seed is not None
    assert seed == pytest.approx(
        0.5 * (
            np.asarray(left.strengths) + np.asarray(right.strengths)
        )
    )


def test_image_precalculation_does_not_extrapolate_across_branch_gap():
    definition = direct_alignment_by_key("image_magnification")
    seed = interpolated_precalculated_seed(
        definition,
        2_000.0,
        np.zeros(5),
        np.full(5, 100.0),
    )

    assert seed is None


def test_ratio_table_reports_target_ratio_and_per_decade_delta():
    definition = direct_alignment_by_key("diffraction_camera_length")
    first = precalculated_alignment_ratios(definition)[0]
    points = precalculated_alignment_points(definition)

    assert first.target_ratio == pytest.approx(10.0)
    assert first.log10_span == pytest.approx(1.0)
    assert first.strength_delta_per_decade == pytest.approx(
        np.asarray(points[1].strengths) - np.asarray(points[0].strengths)
    )


def test_nanoprobe_seed_requires_the_calibrated_aperture_path():
    # Exercise a declared table independently of the current instrument's
    # calibration. The revised relay deliberately has no stored seed table.
    definition = replace(direct_alignment_by_key("nanoprobe_convergence"), targets={
        "precalculation_aperture_diameters_um": [120., 144.],
        "precalculation_convergence_mrad": [30., 36.],
        "precalculation_vectors": [[20., 30.], [24., 34.]],
    })
    lower = np.zeros(2)
    upper = np.full(2, 100.0)

    matched = interpolated_nanoprobe_seed(
        definition, 33.0, 132.0, lower, upper
    )
    mismatched = interpolated_nanoprobe_seed(
        definition, 33.0, 100.0, lower, upper
    )

    assert matched is not None
    assert matched == pytest.approx([22., 32.])
    assert mismatched is None


def test_revised_nanoprobe_does_not_reuse_the_old_corrector_seed_table():
    definition = direct_alignment_by_key("nanoprobe_convergence")
    assert not any(key.startswith(("precalculation_", "aperture_scaling_"))
                   for key in definition.targets)
    # These requests matched the former 0.25 mrad/um table. They must now
    # fall through to the live candidate search, not its former focus branch.
    for target_mrad, diameter_um in ((3., 12.), (33., 132.), (60., 240.)):
        assert interpolated_nanoprobe_seed(
            definition, target_mrad, diameter_um, np.zeros(2), np.full(2, 100.)
        ) is None
