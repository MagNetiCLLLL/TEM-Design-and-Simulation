"""Export a parked trial to the desktop TOML profile and verify its optical controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from temsim.assembly_catalog import AssemblyCatalog

from measure import desktop_state
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
from temsim.operating_modes import condenser_mode_key
from temsim.physics.incremental_particles import capture_emission_phase
from temsim.physics.multipole_coil_field import capture_multipole_coil_fields
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def controls(state):
    multipoles = {
        field.component_key: list(map(float, field.potential.currents_a))
        for field in capture_multipole_coil_fields(state)
    }
    return dict(
        round_model=state.round_lens_field_model,
        condenser_mode=condenser_mode_key(state),
        step_mm=state.step_mm,
        energy_ev=state.electron_gun.nominal_exit_energy_ev,
        source_count=state.electron_gun.emitter.ray_count,
        emitted_current_a=state.electron_gun.emitted_current_a,
        energy_fwhm_ev=state.electron_gun.emitter.energy_spread_fwhm_ev,
        gun_controls=capture_gun_profile_controls(state.electron_gun),
        apertures={
            a.key: (a.enabled, a.z_mm, a.radius_mm, a.offset_x_mm, a.offset_y_mm) for a in state.apertures
        },
        field_digest=get_unified_field_runtime(state).snapshot.numerical_digest,
        entrance_z_mm=state.sample.upper_surface_z_mm,
        round_groups=state.lens_field_map_descriptors,
        lens_drives={lens.key: (lens.percent, lens.polarity) for lens in state.lenses if lens.enabled},
        multipole_electrical_a=multipoles,
        raster=state.ac_deflector.scan_enabled,
        descan=state.descan_deflector.scan_enabled,
        specimen_inserted=state.sample.inserted,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--receipt", type=Path, help="Save desktop round-trip evidence")
    args = parser.parse_args()
    snapshot = InstrumentSnapshot.from_dict(json.loads((args.trial / "inputs.json").read_text()))
    state = snapshot.restore()
    selection, _ = desktop_state(state.electron_gun.emitter.ray_count)
    save_profile(args.profile, state, selection)
    selection, values = read_profile(args.profile)
    restored = default_state()
    AssemblyCatalog().apply(restored, selection)
    apply_profile_values(restored, values)
    apply_physical_layout_to_state(restored, preserve_operating_parameters=True)
    before, after = controls(state), controls(restored)
    for key in before:
        if before[key] != after[key]:
            raise ValueError(f"Desktop profile did not preserve {key}")
    source = [capture_emission_phase(s.electron_gun, s.electron_gun.emit()) for s in (state, restored)]
    if source[0].keys() != source[1].keys() or any(
        not np.array_equal(source[0][key], source[1][key], equal_nan=True) for key in source[0]
    ):
        raise ValueError("Desktop profile did not reproduce the original emission phase")
    print("Verified desktop profile", args.profile, "against", snapshot.digest)
    print("Physical multipole electrical inputs (A)", after["multipole_electrical_a"])
    if args.receipt:
        from hashlib import sha256
        from fit_stationary import write

        write(
            args.receipt,
            dict(
                profile=args.profile.name,
                input_digest=snapshot.digest,
                profile_utf8_lf_sha256=sha256(args.profile.read_text(encoding="utf-8").encode()).hexdigest(),
                matched_fields=sorted(before),
                matched_emission_arrays=sorted(source[0]),
                numerical_field_digest=after["field_digest"],
                condenser_mode=after["condenser_mode"],
                source_count=after["source_count"],
                current_a=after["emitted_current_a"],
                FWHM_eV=after["energy_fwhm_ev"],
                apertures=after["apertures"],
                multipole_electrical_a=after["multipole_electrical_a"],
            ),
        )


if __name__ == "__main__":
    main()
