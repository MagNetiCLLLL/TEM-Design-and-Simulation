"""The desktop starts from solved inputs even when a previous Preview exists."""

from temsim.gui.joint_fem_page import JointFEMPage
from temsim.gui.main_window import MainWindow


def test_joint_fem_opens_without_restoring_preview(qtbot, monkeypatch):
    from temsim.gui.result_files import ResultFiles

    def unexpected_restore(*args, **kwargs):
        raise AssertionError(
            "The saved-result browser was initialized during joint FEM startup"
        )

    started = []
    monkeypatch.setattr(ResultFiles, "__init__", unexpected_restore)
    monkeypatch.setattr(JointFEMPage, "start", lambda page: started.append(page))
    window = MainWindow(joint_fem_startup=True)
    qtbot.addWidget(window)
    assert started == [window.fem_page]
    assert window.centralWidget().currentWidget() is window.fem_page
    assert not hasattr(window, "preview_timer")
    window.close()


def test_rotated_source_image_uses_both_transverse_directions():
    import numpy as np
    from temsim.physics.condenser_relay import source_images

    z = np.linspace(0.54, 0.56, 31)
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    fan = np.zeros((8, len(z), 2))
    for axis in range(2):
        direction = rotation[:, axis]
        fan[2 * axis] = 1e-7 * 0.3 * direction
        fan[2 * axis + 1] = -fan[2 * axis]
        fan[4 + 2 * axis] = 1e-6 * (z[:, None] - 0.55325) * direction
        fan[5 + 2 * axis] = -fan[4 + 2 * axis]
    (image,) = source_images(z, fan)
    np.testing.assert_allclose(image["z_mm"], 553.25)
    np.testing.assert_allclose(image["principal_magnifications"], [0.3, 0.3])

    fan[6] += 10e-9 * rotation[:, 1]
    fan[7] = -fan[6]
    assert source_images(z, fan) == []


def test_corrector_selection_launches_the_displayed_working_point(qtbot, monkeypatch):
    from pathlib import Path

    from PySide6.QtCore import QProcess
    from temsim.joint_fem import load_profile

    calls = []
    monkeypatch.setattr(QProcess, "start", lambda process, program, args: calls.append(args))
    page = JointFEMPage()
    qtbot.addWidget(page)
    assert page.working_point.currentData() is None
    assert "HPol A2" in page.working_point.currentText()
    index = page.working_point.findText("Nanoprobe Normal  |  Corrector relay")
    assert index > 0
    page.working_point.setCurrentIndex(index)
    assert "50,000" in page.retrace.text()
    (args,) = calls
    path = Path(args[args.index("--profile") + 1])
    assert page.profile == load_profile(path)
    assert page.profile["measured_relay"]["field_sha256"] == page.profile["solution_sha256"]
    assert not page.working_point.isEnabled()


def test_hpol_startup_extends_original_source_and_only_changes_two_drives():
    import numpy as np
    from temsim.joint_fem import input_file, load_profile
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT

    current = load_profile()
    relay = load_profile(OPERATING_MODE_CONFIG_ROOT / "fei_50v_corrector_relay/profile.json")
    assert current["source_count"] == relay["source_count"] == 50_000
    assert set(current["excitation_A"]) == set(relay["excitation_A"])
    changed = {k for k, v in current["excitation_A"].items() if relay["excitation_A"][k] != v}
    assert changed == {"probe_hpol_hexapole_x", "probe_hpol_hexapole_y"}
    assert current["assets"]["tracking"] == relay["assets"]["tracking"]
    with (
        np.load(input_file(current, "tracking")) as mesh,
        np.load(OPERATING_MODE_CONFIG_ROOT / "fei_50v/tracking-inputs.npz") as previous,
    ):
        np.testing.assert_array_equal(mesh["phase"][:10_000], previous["phase"])
        np.testing.assert_allclose(mesh["weight_uA"][:10_000] * 5, previous["weight_uA"], rtol=3e-16)
        assert np.unique(mesh["phase"], axis=0).shape == (50_000, 6)
        assert np.all(mesh["weight_uA"] > 0)
        np.testing.assert_allclose(mesh["weight_uA"].sum(), current["source_current_uA"])


def test_conjugate_markers_require_the_measured_operating_field(tmp_path):
    import pytest
    from temsim.joint_fem import load_profile, run
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT

    profile = load_profile(OPERATING_MODE_CONFIG_ROOT / "fei_50v_corrector_relay" / "profile.json")
    profile["measured_relay"]["field_sha256"] = load_profile()["solution_sha256"]
    with pytest.raises(ValueError, match="different magnetic field"):
        run(tmp_path, profile=profile)


def test_aperture_current_drop_does_not_make_a_waist():
    import numpy as np
    from temsim.physics.condenser_relay import beam_widths

    z = np.linspace(0, 0.05, 6)
    launch = np.array([-1.0, 1.0, -10.0, 10.0])
    x = launch[:, None] * (1 + 10 * z)
    xy = np.stack([x, x / 2], axis=2)
    xy[2:, 3:] = np.nan
    widths, roots = beam_widths(z, xy, np.ones(4))
    assert widths[3, 2] < widths[2, 2]
    assert roots == {"x": [], "y": [], "r": []}


def test_corrector_relay_replays_both_intermediate_waists(tmp_path):
    import json

    from temsim.joint_fem import load_profile, run
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT

    profile = load_profile(OPERATING_MODE_CONFIG_ROOT / "fei_50v_corrector_relay" / "profile.json")
    run(tmp_path, profile=profile)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["field_sha256"] == profile["measured_relay"]["field_sha256"]
    assert result["matched_field_receipts"] == 10
    assert result["emitted"] == 50_000 and result["c2_aperture_um"] == 100
    for positions in (
        [image["z_mm"] for image in result["full_column_source_images"]],
        result["column_live_waists_mm"]["r"],
        result["specimen_cohort_waists_mm"]["r"],
    ):
        first, second = [z for z in positions if 1000.0 < z < 1534.5]
        assert 1179.1 < first < 1237.2
        assert 1420.0 < second < 1512.5


def test_default_source_image_is_between_c1_and_c2(tmp_path):
    import json
    from temsim.joint_fem import run

    run(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["emitted"] == 50_000
    assert result["gun_lens_V"] == 50
    assert result["c2_aperture_um"] == 100
    relay = result["condenser_relay"]
    assert 24 < relay["gun_object_z_mm"] < 26
    first, second = relay["source_images"]
    assert 545 < first["z_mm"] < 560 and max(first["principal_magnifications"]) < 1
    assert 750 < second["z_mm"] < 830
    assert max(first["residual_nm_per_urad"]) < 0.01
    for roots in relay["C2_bundle_waists_mm"].values():
        c1, c2 = roots
        assert abs(c1 - first["z_mm"]) < 0.5
        assert abs(c2 - second["z_mm"]) < 0.5
    assert 1 < relay["C2_transmitted"] <= result["gun_exit"]
    assert (tmp_path / "condenser-relay.png").is_file()
