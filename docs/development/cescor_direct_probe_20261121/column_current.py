"""Current-only experiments with the archived CESCOR field and a 15 microampere gun."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cupy as cp
import numpy as np
from scipy.interpolate import make_interp_spline

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "probe_corrector_cescor_20261121"
sys.path.insert(0, str(STUDY))
from compare import CircuitTracker, Mesh, POINT, PREVIOUS, START, SURFACE, WORK, metrics, shared_inputs
from column import BASIS, FACTOR
from measure import desktop_state
from temsim.optics.electron_gun.emitter import EmissionSampling
from temsim.physics.incremental_particles import capture_emission_phase
from temsim.physics.closed_gun_field import closed_field_request

OUT = HERE.parents[2] / "outputs/cescor-direct-20261121"
OWNER = np.array(json.loads((POINT / "currents.json").read_text())["owner_per_group"])
ACTIVE = np.r_[0:15, 21:27]
RECIPE = json.loads((STUDY / "evidence/comparison.json").read_text())["probes"]["material"]
OBJECTIVE_NI = -36000
C2_FOIL = (0.7649, 0.7651)


class ApertureTracker(CircuitTracker):
    """Bracket interception on the physical bore without accepting exterior trajectories."""

    def __init__(self, mesh, table, transverse_interpolation="linear"):
        super().__init__(mesh, table)
        xy = table["xy_m"]
        limit = float(table["valid_radius_m"])
        helper = (STUDY / "field_lookup.cu").read_text()
        if transverse_interpolation == "cubic":
            # The admitted transverse disk leaves one full grid cell on every
            # side for the four-node Lagrange stencil, including clipped stages.
            if limit >= xy[-2] or -limit <= xy[1]:
                raise ValueError("Cubic stencil exceeds registered field samples")
            helper = helper[:helper.index("    const double wx[]")] + (HERE / "transverse_cubic.cu").read_text()
        elif transverse_interpolation != "linear":
            raise ValueError(f"Unknown transverse interpolation: {transverse_interpolation}")
        anchor = "    const double u ="
        # Continue the boundary value for virtual RK stages on a rejected step.
        # Every accepted chord and its intermediate stages must remain in-domain.
        exterior = " ||\n        x * x + y * y > radius * radius"
        if helper.count(anchor) != 1 or helper.count(exterior) != 1:
            raise ValueError("The registered corrector field boundary changed")
        helper = helper.replace(
            anchor,
            """
    const double distance = hypot(x, y);
    if (distance > radius) {
        const double scale = radius / distance;
        x *= scale;
        y *= scale;
    }
"""
            + anchor,
        ).replace(exterior, "")
        source = (PREVIOUS / "lorentz.cu").read_text()
        field_hook = "int lo=0,hi=nr-1;"
        fixed = (
            f"add_corrector(node,x,y,{len(xy)},{xy[0]:.17g},{1/(xy[1]-xy[0]):.17g},{limit:.17g},bx,by,bz);"
        )
        stage_guard = " || swept>=r2[nr-1]"
        contact = "if(endpoints>radius2[i]) {loss[j]=i;break;}"
        for token in (field_hook, stage_guard, contact):
            if source.count(token) != 1:
                raise ValueError("The native field or mechanical-contact contract changed")
        source = source.replace(field_hook, fixed + field_hook).replace(stage_guard, "")
        source = source.replace(
            contact,
            f"""
        if(endpoints>radius2[i] && radius2[i]<={limit**2:.17g}*(1+1e-12)) {{loss[j]=i;break;}}
        if(swept>{limit**2:.17g} || swept>=r2[nr-1]) {{loss[j]=-2;break;}}
        {contact}""",
        )
        self.module = cp.RawModule(code=helper + source, options=("--std=c++11", "--fmad=false"))
        self.kernel = self.module.get_function("trace")
        for name, array in zip(("cescor_b", "cescor_iz", "cescor_fz"), self.tables):
            ptr = self.module.get_global(name)
            cp.ndarray((1,), dtype=cp.uint64, memptr=ptr).set(np.array([array.data.ptr], dtype=np.uint64))


def baseline():
    return np.concatenate(
        [RECIPE["background_signed_groups_A"][key] for key in dict.fromkeys(OWNER[ACTIVE])]
    )


def source_rate():
    """Verify the rate-only change at the original emission boundary."""
    nodes = json.loads(gzip.decompress((POINT / "inputs.json.gz").read_bytes()))["graph"]["nodes"]
    (source,) = [n["attributes"] for n in nodes if n.get("type", "").endswith(":ColdFieldEmitter")]
    gun = desktop_state(32768)[1].electron_gun
    gun.emitter.sampling = EmissionSampling(**nodes[source["_emission_sampling"]["ref"]]["attributes"])
    for name, value in source.items():
        if not name.startswith("_") and getattr(gun.emitter, name) != value:
            raise ValueError(f"Source field differs from reference: {name}")
    initial = capture_emission_phase(gun, gun.emit())
    old_rate = gun.emitted_current_a
    original_electrodes = closed_field_request(gun)
    gun_opening = gun.dpa_aperture.diameter_um
    gun.c1_aperture.diameter_um = 1000.0
    gun.emitter.emission_current_na = 15000.0
    updated = capture_emission_phase(gun, gun.emit())
    if closed_field_request(gun) != original_electrodes or gun.dpa_aperture.diameter_um != gun_opening:
        raise ValueError("Gun electrodes or fixed gun aperture changed")
    receipt = json.loads((POINT / "gun.json").read_text())
    envelope = receipt["magnetic_table"]["maximum_executed_radius_m"] * 1000
    opening = gun.c1_aperture
    if (
        not opening.enabled
        or getattr(opening, "_slit_mode", False)
        or np.hypot(opening.offset_x_mm, opening.offset_y_mm) + envelope >= opening.radius_mm
    ):
        raise ValueError("The 1 mm C1 aperture is not clear of every original gun trajectory")
    if not np.isclose(gun.emitted_current_a, 15e-6, rtol=1e-14, atol=0):
        raise ValueError("The selected emission model did not accept the new rate")
    gun_file = POINT / "gun.npz"
    if hashlib.sha256(gun_file.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("The gun-exit checkpoint differs from its source receipt")
    with np.load(gun_file) as g:
        for key in initial:
            if not np.array_equal(initial[key], updated[key], equal_nan=True):
                raise ValueError(f"The source rate changed {key}")
        for key in ("ray_id", "position_m", "direction", "normal"):
            np.testing.assert_array_equal(updated[key], g["emission_" + key])
        np.testing.assert_array_equal(updated["weight"], g["weight"])
        assert g["alive"].all() and np.all(updated["weight"] > 0) and old_rate == g["emitted_current_a"]
    OUT.mkdir(exist_ok=True, parents=True)
    metadata = dict(
        source_A=gun.emitted_current_a,
        gun_exit_A=gun.emitted_current_a,
        gun_exit_z_mm=gun.exit_plane_z_mm,
        gun_aperture_fixed_diameter_um=gun_opening,
        C1_diameter_um=opening.diameter_um,
        C1_z_mm=opening.z_mm,
        maximum_executed_gun_radius_um=envelope * 1000,
        gun_aperture_loss_A=0.0,
        C1_loss_A=0.0,
        gun_input_count=receipt["population"]["positive_current_count"],
        electrostatic_request_unchanged=True,
        original_gun_sha256=receipt["sha256"],
        reference="Prescribed gun-exit phase space; fixed-field, independent-electron transport.",
    )
    with NamedTemporaryFile(mode="w", dir=OUT, suffix=".json", delete=False) as fp:
        json.dump(metadata, fp, indent=2)
        temp_file = Path(fp.name)
    temp_file.replace(OUT / "gun-boundary.json")
    return gun.emitted_current_a


def fields(step_mm, observations, objective_ni=OBJECTIVE_NI, table_file=None):
    path = table_file or WORK / "fine-field.npz"
    with np.load(path) as f:
        table = dict(f)
    cache = WORK / "column"
    inputs = [
        "columns.npz",
        "mechanical.npz",
        "electric.npz",
        f"armco-{abs(objective_ni):.12g}-gap-fine-p8.npz",
    ]
    if not all((cache / name).is_file() for name in inputs):
        shared_inputs(objective_ni)
    mesh = Mesh(step_mm, observations, cache)
    with np.load(cache / "mechanical.npz") as hardware:
        foil = (hardware["names"] == "condenser_aperture_2") & np.isclose(hardware["r"], 50e-6)
        if foil.sum() != 1 or not np.allclose(
            [hardware["lo"][foil][0], hardware["hi"][foil][0]], C2_FOIL, rtol=0, atol=1e-12
        ):
            raise ValueError("The selected 100 micrometre C2 aperture or its registration changed")
    # All round lenses before the specimen are supplied explicitly below.
    mesh.group[0] -= mesh.transfers.sum(axis=0)
    mesh.hex.fill(0)
    background = mesh.base("armco", objective_ni, "gap-fine", 8)
    basis_path = BASIS if BASIS.is_file() else cache / "replay-basis.npz"
    if not basis_path.is_file():
        from stationary_pupil import capture

        capture(json.loads((POINT / "currents.json").read_text()), basis_path, spacing_mm=0.125)
    with np.load(basis_path) as b:
        np.testing.assert_array_equal(b["owners"], OWNER)
        coils = make_interp_spline(b["z"], b["round"][ACTIVE], k=5, axis=1)(mesh.nodes)
    return mesh, table, background, coils


def field_array(mesh, background, coils, current_a):
    total = background + (np.asarray(current_a) @ coils.reshape(len(coils), -1)).reshape(background.shape)
    return np.ascontiguousarray(np.column_stack((total, mesh.hex)))


def pencil_fields(mesh, table, background, coils):
    center = len(table["xy_m"]) // 2
    axis = np.interp(mesh.nodes[1::2] - START, table["z_local_m"], table["B_zyx_T"][:, center, center, 2])
    return np.ascontiguousarray(np.column_stack((background[1::2, 0] + axis, coils[:, 1::2, 0].T)))


def current_ledger(mesh, t, loss):
    stop = loss >= 0
    keys = mesh.keys[mesh.owner[loss[stop]]]
    lost = {key: float(t.current * t.weights[stop][keys == key].sum()) for key in np.unique(keys)}
    front, rear = np.searchsorted(mesh.z, C2_FOIL)
    fraction = lambda k: float(t.current * t.weights[(loss == -1) | (loss >= k)].sum())
    result = dict(
        gun_exit_A=t.current * t.weights.sum(),
        C2_entrance_A=fraction(front),
        C2_exit_A=fraction(rear),
        specimen_A=fraction(len(mesh.z)),
        intercepted_A=lost,
    )
    assert np.isclose(result["specimen_A"] + sum(lost.values()), result["gun_exit_A"], rtol=1e-12, atol=0)
    return result


def source_digest(t):
    return hashlib.sha256(b"".join(a.tobytes() for a in (t.input, t.weights, t.ids, t.de))).hexdigest()
