"""Separate the archived gun-exit handoff from downstream field changes without refocusing."""
import json
from pathlib import Path
from types import SimpleNamespace
import cupy as cp
import numpy as np

from angle_origin import measure
from material_trial import HERE,PREVIOUS,Column,Field,source_and_electrodes
from column_current import ApertureTracker
from condenser_pair_column import specimen
from relay_transfer import advance,electric_reference
from stationary_pupil import half_nodes


def trace_stations(model, entrance, energy_offsets, weights, locations):
    model.t.input = entrance
    stops = model.indices(np.array(list(locations.values())) / 1000)
    states = advance(model.t, model.full, entrance, energy_offsets + 300_000,
                     electric_reference(model.t), model.loss, 0, stops)
    measurements = {"exit": measure(entrance, weights)}
    for name, (phase, loss) in zip(locations, states, strict=True):
        if np.any(loss < -1):
            raise ValueError("A diagnostic particle left the validated field")
        live = loss == -1
        measurements[name] = measure(phase[live], weights[live])
        measurements[name].update(surviving=int(live.sum()), transmitted_uA=15 * weights[live].sum())
    return measurements, phase, loss


def vacuum_focus(model, field, phase, loss):
    """Measure the cone at the geometrical focus with specimen scattering disabled."""
    offsets_mm = np.linspace(0, 1.8, 181)
    z = model.m.z[-1] + offsets_mm / 1000
    if z[-1] > field.z[-1]:
        raise ValueError("The continuation exceeds the solved magnetic domain")
    nodes = half_nodes(z)
    electric = model.f.electric.base_field
    ix = np.clip(np.searchsorted(electric.z, nodes, side="right")-1, 0, len(electric.z)-2)
    dz = np.diff(electric.z)[ix]
    mesh = SimpleNamespace(z=z, nodes=nodes, radius=np.full(len(z)-1, np.inf),
                           owner=np.full(len(z)-1, -1), keys=[],
                           electric=(electric.r, electric.voltage, ix, (nodes-electric.z[ix])/dz, dz))
    table = dict(z_local_m=field.localz, xy_m=field.xy, B_zyx_T=field.table(),
                 valid_radius_m=field.receipt["local_radius_m"])
    tracer = ApertureTracker(mesh, table, "cubic", source=model.source)
    no_extra_fields = cp.zeros((len(nodes), 20))
    live = loss == -1
    rows = []
    def record(i, p):
        s = specimen(p[live], tracer.weights[live], tracer.current)
        rows.append(dict(offset_mm=offsets_mm[i], D95_nm=s["D95_um"]*1000, alpha95_mrad=s["alpha95_mrad"]))
    record(0, phase)
    previous = 0
    for group in np.array_split(np.arange(1, len(z)), 12):
        states = advance(tracer, no_extra_fields, phase, model.en, model.phi0, loss, previous, group)
        for stop, (phase, loss) in zip(group, states, strict=True):
            np.testing.assert_array_equal(loss == -1, live)
            record(stop, phase)
        previous = group[-1]
    best = int(np.argmin([r["D95_nm"] for r in rows]))
    return dict(plane="Vacuum continuation beyond the physical specimen entrance", best=rows[best],
                bracketed=0 < best < len(rows)-1, sampling_um=10., profile=rows)


def audit():
    folder=HERE/"relay-edge"
    r=json.loads((folder/"tracking.json").read_text())
    n=r["count"]
    gpath=Path(r["gun_source"])
    with np.load(gpath) as g:
        ids=g["ids"];actual=g["phase"];actual_de=g["energy_offset_ev"]
    with np.load(PREVIOUS/"evidence/illuminated-normal-particles.npz") as old:
        np.testing.assert_array_equal(ids,old["source_id"][:n])
        old_de=old["energy_offset_eV"][:n]
        w=old["weight"][:n]
        archived=old["gun_exit_phase"][:n]
        old_lives=old["specimen_loss"][:n]==-1
        if np.any(old["front_loss"][:n] != -1):
            raise ValueError("The archived beam had upstream losses before its C2 entrance observation")
        former=dict(exit=measure(archived,w),foil=measure(old["front_phase"][:n],w),
                    specimen=measure(old["specimen_phase"][:n][old_lives],w[old_lives]),
                    surviving=int(old_lives.sum()),transmitted_uA=15*w[old_lives].sum())
        z_mm=old["track_z_m"]*1000
        moment=old["moments"]
        rms_archive=np.hypot(moment[:,1],moment[:,2])*1e6
        archived_mini_rms_um=float(np.interp(1522.5,z_mm,rms_archive))
    field=Field(Path(r["solution"]),step=r["bore_step_mm"])
    source = source_and_electrodes()
    locations = {"foil": 764.9, "mini": 1522.5, "specimen": r["surface_z_mm"]}
    def model_at(step):
        return Column(source, field, gpath, n, step, observation_mm=list(locations.values()))
    model = model_at(r["step_mm"])
    trials={}
    for name,entrance,de in (("archived_exit_in_current_column",archived,old_de),
                             ("matched_current_exit_in_current_column",actual,actual_de)):
        trials[name], phase, loss = trace_stations(model, entrance, de, w, locations)
    fine_step = r["step_mm"]/2
    fine, refined_phase, refined_loss = trace_stations(model_at(fine_step), actual, actual_de, w, locations)
    np.testing.assert_array_equal(loss, refined_loss)
    nested = {}
    for size in (n//2, n):
        live = loss[:size] == -1
        nested[size] = measure(phase[:size][live], w[:size][live])["alpha95_mrad"]
    validation = dict(standard_step_mm=r["step_mm"], half_step_mm=fine_step, refined=fine["specimen"],
                      nested_source_alpha95_mrad=nested,
                      max_survivor_displacement_nm=1e9*np.linalg.norm(
                          refined_phase[loss==-1,:2]-phase[loss==-1,:2],axis=1).max())
    result=dict(original_comparison_ids=n,conditional_incident_uA=15*w.sum(),
                source_de_max_difference_eV=float(np.max(np.abs(old_de-actual_de))),
                weight_convention="Use each archived original absolute weight in all three paired comparisons; the matching prefix carries 7.5 uA.",
                numerical_validation=validation,
                vacuum_focus=vacuum_focus(model, field, phase, loss),
                historical_mini_rms_radius_um=archived_mini_rms_um,
                historical_mini_cohort="All 30396 eventual survivors in the original 32768-particle run",
                field_unchanged_for_exit_swap=field.receipt["solution_sha256"],archived_field_with_archived_exit=former,
                **trials)
    (HERE/"exit-control.json").write_text(json.dumps(result,indent=2)+"\n")
    print([(label,data["specimen"]["alpha95_mrad"]) for label,data in [("archived",former),*trials.items()]],flush=True)


if __name__=="__main__":
    audit()
