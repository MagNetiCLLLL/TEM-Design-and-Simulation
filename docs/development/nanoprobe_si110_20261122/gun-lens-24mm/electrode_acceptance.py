"""Read the joint gun field and audit primary-electron losses at two physical biases."""

import argparse
from collections import Counter
import json

import numpy as np

from gun_focus import HERE, INPUTS, _trace_feg_to_exit, gun_step_workers, prepare, set_voltage, write_json


def field_samples(field):
    z_mm = np.array([0.5, 3., 8., 12., 14., 16., 18., 20., 22., 24., 28., 32., 42.])
    axis = np.column_stack((np.zeros_like(z_mm), np.zeros_like(z_mm), z_mm * 1e-3))
    off_axis = axis.copy()
    off_axis[:, 0] = 1e-4
    potential = field.potential_rise_v_at_global_positions(axis)
    ez = field.field_at_global_positions_v_per_m(axis)[:, 2]
    er = field.field_at_global_positions_v_per_m(off_axis)[:, 0]
    return dict(z_mm=z_mm.tolist(), axis_potential_V=potential.tolist(),
                axis_Ez_V_per_m=ez.tolist(), Er_at_r_0p1mm_V_per_m=er.tolist())


def trial(gun, emission):
    with gun_step_workers():
        result = _trace_feg_to_exit(gun, emitted_bundle=emission)
    out = result.exit_bundle
    np.testing.assert_array_equal(out.ray_id, emission.ray_id)
    np.testing.assert_array_equal(out.weight, emission.weight)
    weights = np.asarray(emission.weight)
    labels = np.asarray(result.blocked_key, dtype=str)
    np.testing.assert_array_equal(labels == "", out.alive)
    lens = gun.electrostatic_lens
    limits = lens.mechanical_center_from_tip_mm + np.array([-.5, .5]) * lens.mechanical_length_mm
    history = result.equal_time_history
    radii_um = np.hypot(history.x_m, history.y_m) * 1e6
    inside = history.alive & (history.z_mm >= limits[0]) & (history.z_mm <= limits[1])
    aperture = next(p for p in result.plane_arrivals if p.key == gun.dpa_aperture.key)
    scale_uA = gun.emitted_current_a * 1e6
    return dict(original_count=len(weights), positive_current_count=int(np.count_nonzero(weights > 0)),
                exiting_count=int(np.count_nonzero(out.alive)),
                effective_exiting_count=float(weights[out.alive].sum()**2 / (weights[out.alive]**2).sum()),
                emitted_uA=float(scale_uA * weights.sum()),
                exit_uA=float(scale_uA * weights[out.alive].sum()),
                gun_lens_intercepted_uA=float(scale_uA * weights[labels == lens.key].sum()),
                gun_aperture_intercepted_uA=float(scale_uA * weights[labels == aperture.key].sum()),
                first_loss_counts=dict(Counter(labels[~out.alive])),
                maximum_sampled_radius_in_lens_um=float(np.max(radii_um[inside])),
                maximum_radius_at_aperture_midplane_um=float(np.max(
                    np.hypot(aperture.x_m[aperture.reached], aperture.y_m[aperture.reached]) * 1e6)),
                maximum_sampled_radius_to_exit_um=float(np.nanmax(radii_um)),
                maximum_exit_energy_error_eV=result.electrostatic_model_report["maximum_exit_energy_error_ev"])


def component_response(gun):
    """Separate the field of the changed lens electrode from the linked divider."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.unique(np.r_[np.linspace(1.5, 48., 1001), 8., 14., 16., 18., 22., 24., 32., 42.])
    axis = np.column_stack((np.zeros_like(z), np.zeros_like(z), z * 1e-3))
    lens_voltages = (1.2, 1.8)
    fields = []
    for voltage in lens_voltages:
        set_voltage(gun, voltage)
        fields.append(gun.base_electric_field)
    baseline, linked = fields
    stages = gun.accelerator.stages
    reference = [s.voltage_fraction for s in stages]
    # Keep the original absolute ring voltages while changing only the lens electrode.
    tension = gun.accelerator.high_tension_kv
    try:
        for stage, fraction in zip(stages, reference):
            old_ring_kV = lens_voltages[0] + (tension - lens_voltages[0]) * fraction
            stage.voltage_fraction = (old_ring_kV - lens_voltages[1]) / (tension - lens_voltages[1])
        set_voltage(gun, lens_voltages[1])
        isolated = gun.base_electric_field
    finally:
        for stage, fraction in zip(stages, reference):
            stage.voltage_fraction = fraction

    for old, new in zip(baseline.request["rings"], isolated.request["rings"], strict=True):
        assert old["key"] == new["key"]
        change = new["potential_rise_v"] - old["potential_rise_v"]
        assert np.isclose(change, 600. if old["key"] == "electrostatic_lens" else 0., atol=1e-8)
    potentials = [f.potential_rise_v_at_global_positions(axis) for f in fields]
    ez = [f.field_at_global_positions_v_per_m(axis)[:, 2] / 1000 for f in (*fields, isolated)]
    nominal_delta = ez[1] - ez[0]
    isolated_delta = ez[2] - ez[0]
    checks = []
    for location in (8, 14, 16, 18, 22, 24, 32, 42):
        j = np.flatnonzero(z == location).item()
        checks.append(dict(z_mm=location, linked_delta_Ez_kV_per_m=float(nominal_delta[j]),
                           lens_only_delta_Ez_kV_per_m=float(isolated_delta[j])))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, top=.82, bottom=.1, hspace=.33)
    fig.suptitle("Where the gun-lens voltage changes the field", x=.12, y=.97,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.12, .90, "Nanoprobe Normal   |   Tip at 0 mm   |   Extractor +4.0 kV   |   Final anode 300 kV",
             fontsize=10, color="#404e56")
    colors = ("#007771", "#a44f1c")
    for curve, voltage, c in zip(potentials, lens_voltages, colors):
        axes[0].plot(z, curve / 1000, color=c, lw=1.9, label=f"Gun electrode +{voltage:.1f} kV")
    axes[0].set(ylim=(0, 40), ylabel="Axis potential\nrelative to tip (kV)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].axhline(0, color="#37464f", lw=.85)
    axes[1].plot(z, isolated_delta, color="#52539a", lw=2, label="Only the gun electrode changes")
    axes[1].plot(z, nominal_delta, color="#007771", lw=1.8, ls="--", label="Normal coupled divider")
    axes[1].set(ylabel="Change in axial field\n(kV/m)", xlabel="Axial distance from the source (mm)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].set_title("Response to a +600 V change at the gun electrode", loc="left", fontsize=11)
    for ax in axes:
        ax.axvspan(14, 22, color="#d7e1e6", alpha=.65, zorder=-3)
        ax.axvline(24, color="#865058", lw=1, ls=":")
        ax.axvspan(41.5, 42.5, color="#eadcc1", alpha=.8, zorder=-3)
        ax.set_xlim(z[0], z[-1])
        ax.grid(axis="y", color="#d2d8dc", lw=.65, alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
    for at, label in ((18., "Gun electrode"), (24., "Gun aperture"), (42., "1st accelerator ring")):
        axes[0].text(at, 1.025, label, transform=axes[0].get_xaxis_transform(),
                     fontsize=8, ha="left" if at == 24. else "center", color="#354753")
    fig.savefig(HERE / "gun-electrode-field.png", dpi=170, facecolor="white")
    plt.close(fig)
    controls = json.loads((HERE / "operating-point.json").read_text())["controls"]
    working_points = []
    for voltage in (controls["gun_lens_before_kV"], 5.3, controls["gun_lens_after_kV"]):
        identity = set_voltage(gun, voltage)
        if voltage == controls["gun_lens_after_kV"]:
            assert identity == controls["tuned_electric_field_sha256"]
        working_points.append(dict(lens_kV=voltage, field_sha256=identity,
                                   field=field_samples(gun.base_electric_field)))
    write_json(HERE / "electrode-field-control.json", dict(
        applied_change_V=600, initial_lens_kV=1.2, field_at_zero_radius=True,
        baseline_sha256=baseline.report["request_sha256"], nominal_sha256=linked.report["request_sha256"],
        isolated_sha256=isolated.report["request_sha256"], samples=checks,
        working_points=working_points))


def main(field_only):
    gun, record = prepare()
    if field_only:
        component_response(gun)
        return
    count = 2048
    emission = gun.emit(count)
    launch = np.column_stack((emission.x_m, emission.y_m, emission.tx_rad, emission.ty_rad))
    with np.load(INPUTS / record["particle_source"]["file"]) as parent:
        np.testing.assert_array_equal(launch, parent["launch"][:count])
        np.testing.assert_array_equal(emission.ray_id, parent["ids"][:count])
        assert float(parent["emitted_A"]) == gun.emitted_current_a
    assert len(np.unique(emission.ray_id)) == count and np.all(emission.weight > 0)
    np.testing.assert_allclose(emission.weight.sum(), 1.)
    with np.load(INPUTS / record["bore_field"]["file"]) as magnetic:
        valid_radius_um = json.loads(str(magnetic["receipt"]))["local_radius_m"] * 1e6
    results = dict(voltages_relative_to="tip", extractor_kV=gun.extractor.voltage_kv,
                   source_sha256=record["particle_source"]["sha256"],
                   magnetic_sha256=record["bore_field"]["sha256"],
                   magnetic_fit_radius_um=valid_radius_um, conditions=[])
    for kv in [1.2, 1.8]:
        identity = set_voltage(gun, kv)
        field = gun.base_electric_field
        data = trial(gun, emission)
        assert data["maximum_sampled_radius_to_exit_um"] < valid_radius_um
        assert data["maximum_exit_energy_error_eV"] < 1e-3
        results["conditions"].append(dict(lens_kV=kv, field_sha256=identity,
                                          field=field_samples(field), transport=data))
        print(f"{kv:g} kV: {data}", flush=True)
    write_json(HERE / "electrode-acceptance.json", results)
    component_response(gun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-only", action="store_true")
    main(parser.parse_args().field_only)
