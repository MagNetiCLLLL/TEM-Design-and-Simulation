"""Reproduce the two-condenser material working point, with all other drives frozen."""

import argparse
from functools import cache, cached_property
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
from condenser_source_fields import Fields, HERE, sha, write
from condenser_source_search import axial_nodes, electrostatic, transfer
from temsim.physics import condenser_pair_field


DEFAULT_OUT = HERE.parents[2] / "outputs/condenser-pair-fem-20261122"
OUT = Path(os.environ.get("TEMSIM_CONDENSER_PAIR_OUT", DEFAULT_OUT))
OUT.mkdir(parents=True, exist_ok=True)
EV = HERE / "evidence/condenser-pair"
EV.mkdir(parents=True, exist_ok=True)
SOURCE = Path(condenser_pair_field.__file__)
PUBLISHED_FEM_DEPENDENCIES = "3bbfb030f6f60625a63041e4639a84f46f4943eee10275d94ebb7222cb59cfae"


@cache
def fem_dependencies():
    digest = hashlib.sha256()
    for name in (
        "magnetic_materials.py",
        "part_materials.py",
        "magnetic_circuits.py",
        "pole_geometry.py",
        "physics/nonlinear_magnetostatics.py",
        "physics/axisymmetric_magnetostatics.py",
        "physics/lens_field_provider.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update((SOURCE.parents[1] / name).read_bytes())
    return digest.hexdigest()


class Pair:
    def __init__(self, fields=None):
        self.f = Fields() if fields is None else fields
        self.parts, self.curves, self.grades = condenser_pair_field.installed_pair(self.f.state)
        self.geometry_id = hashlib.sha256(
            json.dumps(
                dict(parts=self.parts, curves=self.curves, grades=self.grades), sort_keys=True
            ).encode()
        ).hexdigest()

    def field(self, amp, pitch=2.5, domain=550):
        q = self.f.current(*amp)
        metadata = dict(
            q=q[:12].tolist(),
            coils=[w.to_metadata() for w in self.f.units[:12]],
            mesh=[pitch, domain],
            geometry=self.geometry_id,
            solver=sha(SOURCE),
        )
        # The published November 22 keys used the exact shared implementations
        # named above. A change also invalidates every associated gun handoff.
        if fem_dependencies() != PUBLISHED_FEM_DEPENDENCIES:
            metadata["fem_dependency_sha256"] = fem_dependencies()
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        path = OUT / f"field-{key[:20]}.npz"
        if not path.is_file():
            print("Solving coupled field", amp, pitch, domain, flush=True)
            data = condenser_pair_field.solve_pair(
                self.f.state, self.f.units[:12], q[:12], pitch_mm=pitch, domain_mm=domain
            )
            assert str(data["geometry_id"]) == self.geometry_id
            data.update(key=key)
            with NamedTemporaryFile(dir=OUT, suffix=".npz", delete=False) as temp:
                temporary_path = Path(temp.name)
            try:
                np.savez_compressed(temporary_path, **data)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
            print("Field solved", amp, data["nodes"], data["peak_material_T"], flush=True)
        with np.load(path) as d:
            data = dict(d)
        if str(data["key"]) != key:
            raise ValueError("Invalid cached field identity")
        return path, data

    @cached_property
    def optics(self):
        z = axial_nodes(0.00030)
        mid = (z[1:] + z[:-1]) / 2
        ep, ef = electrostatic(z, 0.3)
        u = self.f.round_axis(mid)
        fixed = self.f.current(0, 0) @ u + self.f.objective(mid)[:, 0]
        near = mid < 0.45
        fixed[near] += self.f.far_corrector(mid[near])[:, 0, 2]
        fixed[~near] += np.interp(mid[~near], self.f.zjoint, self.f.bjoint[:, 2])
        return z, mid, ep, ef, u[:4].sum(0), u[4:8].sum(0), fixed

    def prediction(self, amp, data):
        z, mid, ep, ef, u1, u2, fix = self.optics
        result = transfer(
            z, ep, ef, fix + amp[0] * u1 + amp[1] * u2 + np.interp(mid, data["z"], data["induced_T"]), True
        )
        return result[:4]

    def tune(self):
        orig = (-41.0, 19.5)
        self.field(orig)
        # Linearization proposes currents only. Every reported material solution
        # is solved at its actual simultaneous currents and retested.
        anchors = [(-20.0, 0.0), (0.0, 12.0), (0.0, 0.0)]
        fields = [self.field(a)[1] for a in anchors]
        z, mid, ep, ef, u1, u2, fix = self.optics
        y = [np.interp(mid, d["z"], d["induced_T"]) for d in fields]
        base = fix + y[2]
        a1 = u1 + (y[0] - y[2]) / -20
        a2 = u2 + (y[1] - y[2]) / 12
        target = np.array([0.522, 0.795])

        def branch(a, b, field):
            counts, at, mag, gap, _ = transfer(z, ep, ef, field, False)
            if tuple(counts) == (1, 0, 1, 0, 0) and 0.45 < at < 0.55 and mag < 1:
                return float(np.linalg.norm(([at, gap] - target) / [0.022, 0.018]))
            return np.inf

        options = []
        for a in np.arange(-55.0, -4.0, 1.0):
            for b in np.arange(3.0, 39.0, 0.5):
                score = branch(a, b, base + a * a1 + b * a2)
                if np.isfinite(score):
                    options.append((score, a, b))
        if not options:
            raise ValueError("Could not bracket the single-image branch")
        amps = np.array(min(options)[1:])
        for pitch in (2.5, 1.5):
            for trial in range(6):
                path, d = self.field(amps, pitch)
                counts, at, mag, gap = self.prediction(amps, d)
                print("Actual branch", amps, counts, at, mag, gap, flush=True)
                if (
                    tuple(counts) == (1, 0, 1, 0, 0)
                    and 0.45 < at < 0.55
                    and mag < 1
                    and abs(gap - 0.795) < 0.009
                ):
                    break
                base = (
                    fix + np.interp(mid, d["z"], d["induced_T"]) + amps[0] * (u1 - a1) + amps[1] * (u2 - a2)
                )
                candidates = []
                for a in np.arange(amps[0] - 6, amps[0] + 6.1, 0.5):
                    for b in np.arange(amps[1] - 4, amps[1] + 4.1, 0.25):
                        score = branch(a, b, base + a * a1 + b * a2)
                        if np.isfinite(score):
                            candidates.append((score, a, b))
                if not candidates:
                    raise ValueError("Nonlinear current neighborhood lost target branch")
                amps = np.array(min(candidates)[1:])
            else:
                raise ValueError("No accepted nonlinear branch")
        point = dict(
            series_A=amps.tolist(),
            material_file=path.name,
            material_sha256=sha(path),
            predicted_counts=counts.tolist(),
            image_mm=at * 1000,
            magnification=mag,
            gap_mm=gap * 1000,
            mesh_mm=pitch,
            domain_mm=550,
        )
        write(OUT / "point.json", point)
        print("Accepted", point, flush=True)

    def prepare_controls(self):
        work = json.loads((OUT / "point.json").read_text())
        cases = [
            ("matched-current", [-41.0, 19.5], 1.5, 550),
            ("mesh-check", work["series_A"], 1.0, 800.0),
        ]
        for name, drives, pitch, domain in cases:
            path, data = self.field(drives, pitch, domain)
            counts, first, mag, gap = self.prediction(drives, data)
            folder = OUT / name
            folder.mkdir(exist_ok=True)
            shutil.copyfile(path, folder / path.name)
            write(
                folder / "point.json",
                dict(
                    series_A=drives,
                    mesh_mm=pitch,
                    domain_mm=domain,
                    material_file=path.name,
                    material_sha256=sha(path),
                    predicted_counts=counts.tolist(),
                    image_mm=first * 1000,
                    magnification=mag,
                    gap_mm=gap * 1000 if counts[2] else None,
                ),
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["solve", "tune", "prepare-controls"])
    ap.add_argument("--amps", type=float, nargs=2, default=[-41.0, 19.5])
    ap.add_argument("--pitch", type=float, default=2.5)
    ap.add_argument("--domain", type=float, default=550.0)
    args = ap.parse_args()
    p = Pair()
    if args.action == "tune":
        p.tune()
    elif args.action == "prepare-controls":
        p.prepare_controls()
    else:
        print(p.field(args.amps, args.pitch, args.domain)[0], flush=True)
