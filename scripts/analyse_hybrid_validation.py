"""Audit saved ideal hybrid outputs; no new source or coherent propagation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import h, hbar

from temsim.physics.hybrid_semiclassical import momentum


def analyse(directory, *, make_plot=True):
    directory = Path(directory)
    target = directory/"validation.json"
    if target.exists() or (make_plot and (directory/"validation.png").exists()):
        raise ValueError("Validation artifacts already exist; do not overwrite reviewed results")
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    if report["status"] != "IDEAL_DEVELOPMENT_VALIDATION" or report["physical_feg_phase_chain"]:
        raise ValueError("This audit is specific to the declared ideal field-free development trial")
    # Tolerances are fixed before reading the observed residuals. The CDF
    # criterion is an engineering sampling check, not an iid KS p-value.
    limits = {"weight_absolute": 1e-11, "drift_absolute_m": 1e-12,
              "energy_absolute_ev": 1e-6, "radial_cdf_maximum": .05}
    with np.load(directory/"hybrid_arrays.npz", allow_pickle=False) as data:
        source_ids, source_weights = data["original_ray_id"], data["original_weight"]
        original_count = int(np.count_nonzero(source_weights > 0))
        if original_count < 1000 or len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Invalid original positive-current source population")
        weight, energy = data["exit_weight"], data["exit_energy_ev"]
        direction, position = data["exit_direction"], data["exit_position_m"]
        final_x, final_y = data["downstream_x_m"][-1], data["downstream_y_m"][-1]
        dz = data["downstream_z_mm"][-1]*1e-3-position[:, 2]
        expected = position[:, :2]+dz[:, None]*direction[:, :2]/direction[:, 2, None]
        drift_error = float(np.max(abs(np.column_stack((final_x, final_y))-expected)))
        energy_error = float(np.max(abs(data["downstream_energy_ev"]-energy)))
        source_effective = float(source_weights.sum()**2/np.dot(source_weights, source_weights))
        mode = report["wave_checkpoints"]["exit"]["modes"][0]
        if len(report["wave_checkpoints"]["exit"]["modes"]) != 1:
            raise ValueError("This comparison requires the declared single-mode elastic trial")
        raw = data["diffraction_0_raw_canonical_diffraction"]
        transport = data["diffraction_0_transport_diffraction"]
        frequency = data["diffraction_0_frequency_xy_m1"]
        theta = h/momentum(mode["energy_kev"]*1000)*frequency
        radial = np.linalg.norm(theta, axis=-1)
        sampled_radial = np.linalg.norm(direction[:, :2], axis=-1)
        edges = np.r_[np.linspace(0, .02, 81), .04, .08, np.inf]
        predicted_hist = np.histogram(radial, bins=edges, weights=transport)[0]
        sampled_hist = np.histogram(sampled_radial, bins=edges, weights=weight)[0]
        predicted_cdf = np.cumsum(predicted_hist)/transport.sum()
        sampled_cdf = np.cumsum(sampled_hist)/weight.sum()
        cdf_error = float(np.max(abs(predicted_cdf-sampled_cdf)))
        weight_error = max(abs(float(raw.sum())-report["specimen_exit_weight"]),
            abs(float(transport.sum())-report["specimen_exit_weight"]),
            abs(float(weight.sum())-report["specimen_exit_weight"]),
            abs(float(report["downstream_balance_residual"])))
        ledgers = [row["probability_balance"] for history in report["specimen_exit"]["material_histories"]
                   for row in history["modes"]]
        band_loss = sum(row["numerical_band_loss"] for row in ledgers)
        support_loss = sum(row["numerical_support_loss"] for row in ledgers)
        physical_removal = sum(row["column_absorbed"]+sum(row["first_events"].values()) for row in ledgers)
        mode_probability_error = max(abs(row["residual"]) for row in ledgers)
        sigma = report["specimen_incident"]["upstream"]["sigma_m"]
        ideal_angular_sigma = hbar/(2*sigma*momentum(mode["energy_kev"]*1000))
        high = radial > .003
        result = {"schema": "ideal-hybrid-output-audit-v1", "thresholds": limits,
            "original_positive_count": original_count, "effective_original_count": source_effective,
            "original_source_weight": float(source_weights.sum()), "exit_particle_count": len(weight),
            "camera_particle_count": int(np.count_nonzero(data["downstream_blocked_key"] == "camera")),
            "weight_absolute_error": weight_error, "material_balance_residual": mode_probability_error,
            "field_free_drift_maximum_error_m": drift_error, "energy_maximum_error_ev": energy_error,
            "radial_cdf_maximum_error": cdf_error, "window_sigma_nm": report["diffraction"][0]["window_sigma_m"]*1e9,
            "nominal_window_angular_rms_mrad": report["diffraction"][0]["window_nominal_angular_rms_rad"]*1000,
            "detector_current_a": report["detector_current_a"],
            "material_numerical_band_loss": band_loss, "material_numerical_support_loss": support_loss,
            "material_physical_removal_including_roundoff": physical_removal,
            "conditional_fraction_above_3_mrad": {
                "incident_ideal_gaussian": float(np.exp(-.003**2/(2*ideal_angular_sigma**2))),
                "real_Si_raw_diffraction": float(raw[high].sum()/raw.sum()),
                "local_window_transport": float(transport[high].sum()/transport.sum()),
                "sampled_particles": float(weight[sampled_radial > .003].sum()/weight.sum())},
            "limitations": ["Ideal short field-free column; no physical FEG or full-instrument qualification",
                "This custom Si CIF has no configured validated inelastic constants; only its elastic zero-loss branch was executed",
                "Finite-window particles do not reproduce downstream coherent interference",
                "One source count does not establish physical-source convergence"]}
        result["passed"] = (weight_error <= limits["weight_absolute"]
            and mode_probability_error <= limits["weight_absolute"] and drift_error <= limits["drift_absolute_m"]
            and energy_error <= limits["energy_absolute_ev"] and cdf_error <= limits["radial_cdf_maximum"])
        if make_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
            ny, nx = raw.shape
            x_axis, y_axis = theta[ny//2, :, 0]*1000, theta[:, nx//2, 1]*1000
            ix, iy = np.flatnonzero(abs(x_axis) <= 12), np.flatnonzero(abs(y_axis) <= 12)
            extent = [x_axis[ix[0]], x_axis[ix[-1]], y_axis[iy[0]], y_axis[iy[-1]]]
            norm = LogNorm(vmin=max(raw.max(), transport.max())*1e-7, vmax=max(raw.max(), transport.max()))
            for ax, intensity, title in zip(axes[:2], (raw, transport), ("Si raw diffraction", "Local-window angular distribution")):
                im = ax.imshow(intensity[np.ix_(iy, ix)], origin="lower", extent=extent, norm=norm, cmap="magma")
                ax.set(title=title, xlabel="X direction (mrad)", ylabel="Y direction (mrad)")
            fig.colorbar(im, ax=axes[:2], shrink=.8, label="Probability per reference electron per bin")
            hist = axes[2].hist2d(final_x*1e9, final_y*1e9, bins=80, range=((-12, 12), (-12, 12)),
                                  weights=weight, cmap="magma")
            axes[2].set(title=f"Weighted camera hits ({len(weight)})", xlabel="X (nm)", ylabel="Y (nm)", aspect="equal")
            fig.colorbar(hist[3], ax=axes[2], shrink=.8, label="Probability per reference electron per pixel")
            fig.suptitle("Ideal single-mode development trial | 300 keV | Si 0.4 nm | field-free camera drift")
            fig.savefig(directory/"validation.png", dpi=180)
            plt.close(fig)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not result["passed"]:
        raise ValueError(f"Hybrid validation failed; inspect {target}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyse(args.directory, make_plot=not args.no_plot), indent=2))
