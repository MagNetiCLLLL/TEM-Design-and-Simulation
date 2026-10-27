"""Retained native gun alignment potentials with their original soft windows.

The declared model is transverse dipole/quadrupole B derived from A_z. It is
not a solved coil/yoke field or a new fringe-field reconstruction. Static
blanking of the existing gun deflector retains its replacement-drive behavior;
it is not the separate time-dependent electrostatic column blanker.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
from temsim.optics.electron_gun.electrostatic import _soft_window_with_derivatives


SCHEMA = "native-gun-magnetic-potential-v1"
_NATIVE = {cls: cls.field_at_global_positions_t for cls in (GunDeflector, GunStigmator)}


@dataclass(frozen=True, slots=True)
class GunMagneticPotential:
    source_id: str
    # Each term: centre m, half-length m, soft edge m, c10,c01,c20,c11,c02.
    terms: tuple
    radial_support_m: float
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if (type(self.source_id) is not str or not self.source_id or self.source_id.strip() != self.source_id
                or any(ord(character) < 32 for character in self.source_id)):
            raise ValueError("Gun potential source_id must be a nonempty identifier")
        if (isinstance(self.radial_support_m, (bool, np.bool_)) or not np.isscalar(self.radial_support_m)
                or not np.isfinite(self.radial_support_m) or self.radial_support_m <= 0):
            raise ValueError("Gun potential radial support must be positive and finite")
        try:
            if any(isinstance(value, (bool, np.bool_, str)) for row in self.terms for value in row):
                raise ValueError("Gun coefficients must be numeric scalars")
            terms = tuple(tuple(float(value) for value in row) for row in self.terms)
        except (TypeError, ValueError) as error:
            raise ValueError("Gun potential terms require finite scalar coefficients") from error
        if not terms or any(len(row) != 8 or not np.isfinite(row).all() or min(row[1:3]) <= 0 for row in terms):
            raise ValueError("Gun potential terms require a centre, positive lengths and five polynomial coefficients")
        if any(not math.isfinite(abs(row[0])+row[1]+row[2]) for row in terms):
            raise ValueError("Gun potential support exceeds the finite coordinate range")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Gun potential provenance must be a mapping")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "radial_support_m", float(self.radial_support_m))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))

    @classmethod
    def from_provider(cls, provider):
        if (type(provider) not in _NATIVE or getattr(provider.field_at_global_positions_t, "__func__", None)
                is not _NATIVE[type(provider)]):
            raise ValueError("Gun potential capture requires an unchanged native alignment provider")
        provider.validate()
        if type(provider.enabled) is not bool:
            raise ValueError("Gun alignment enabled state must be boolean")
        for name in ("mechanical_center_from_tip_mm", "mechanical_length_mm", "mechanical_outer_diameter_mm",
                     "mechanical_clear_bore_diameter_mm", "field_center_offset_mm", "soft_edge_mm"):
            if not math.isfinite(float(getattr(provider, name))):
                raise ValueError("Gun alignment geometry must be finite")
        radius = float(provider.mechanical_clear_bore_diameter_mm)*.0005
        edge = max(float(provider.soft_edge_mm), 1e-6)*.001
        terms = []
        if type(provider) is GunDeflector:
            drives = ((provider.upper_center_from_tip_mm, provider.upper_field_x_mt, provider.upper_field_y_mt),
                      (provider.lower_center_from_tip_mm, provider.lower_field_x_mt, provider.lower_field_y_mt))
            if provider.beam_blanked:
                drives = ((provider.upper_center_from_tip_mm, 0., provider.blanking_field_y_mt),)
            factor = 1e-3 if provider.enabled or provider.beam_blanked else 0.
            for center, bx, by in drives:
                terms.append(((center+provider.field_center_offset_mm)*.001, provider.coil_length_mm*.0005,
                              edge, -by*factor, bx*factor, 0., 0., 0.))
            status = {"blanking_replaces_alignment": provider.beam_blanked}
        else:
            angle = math.radians(provider.rotation_deg)*2
            gradient = float(provider.gradient_t_per_m) if provider.enabled else 0.
            normal, skew = gradient*math.cos(angle), gradient*math.sin(angle)
            terms.append((provider.optical_reference_from_tip_mm*.001, provider.effective_length_mm*.0005,
                          edge, 0., 0., -.5*normal, -skew, .5*normal))
            status = {"rotation_deg": provider.rotation_deg}
        return cls(provider.key, terms, radius, {"native_type": type(provider).__name__,
                   "enabled": provider.enabled, **status})

    @property
    def digest(self):
        return json_digest(self.to_metadata())

    def to_metadata(self):
        return {"schema": SCHEMA, "source_id": self.source_id, "terms": [list(row) for row in self.terms],
                "radial_support_m": self.radial_support_m, "provenance": thaw_json(self.provenance),
                "units": {"position": "m", "A": "T m", "B": "T"}, "static": True,
                "gauge": "global A=(0,0,P(x,y)*native_soft_window(z))",
                "domain": "strict bore in each active field slab; declared native zero outside all slabs",
                "model_limit": "effective transverse alignment field; no coil/yoke Maxwell reconstruction",
                "wave_admission": False}

    def to_dict(self):
        return {"metadata": self.to_metadata(), "digest": self.digest}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"metadata", "digest"} or not isinstance(record["metadata"], Mapping):
            raise ValueError("Invalid gun magnetic potential record")
        meta = record["metadata"]
        try:
            result = cls(meta["source_id"], meta["terms"], meta["radial_support_m"], meta["provenance"])
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete gun magnetic potential record") from error
        if result.digest != record["digest"] or json_digest(meta) != json_digest(result.to_metadata()):
            raise ValueError("Gun magnetic potential identity mismatch")
        return result

    def sample_jet(self, positions_m):
        raw = np.asarray(positions_m)
        if raw.dtype.kind not in "iuf" or raw.shape[-1:] != (3,) or not np.isfinite(raw).all():
            raise ValueError("Gun potential positions must be finite real XYZ metres")
        points = np.asarray(raw, dtype=float)
        x, y, z = np.moveaxis(points, -1, 0)
        shape = points.shape[:-1]
        value = np.zeros((*shape, 3))
        jacobian = np.zeros((*shape, 3, 3))
        hessian = np.zeros((*shape, 3, 3, 3))
        for center, half, edge, c10, c01, c20, c11, c02 in self.terms:
            active = (z >= center-half-edge) & (z <= center+half+edge)
            if np.any(active & (np.hypot(x, y) > self.radial_support_m)):
                raise ValueError("Position is outside the gun magnetic potential bore domain")
            # Do not evaluate an inactive polynomial beyond its declared bore.
            xx, yy = np.where(active, x, 0.), np.where(active, y, 0.)
            f, df_mm, ddf_mm, _ = _soft_window_with_derivatives(z*1e3, (center-half)*1e3, (center+half)*1e3, edge*1e3)
            df, ddf = df_mm*1e3, ddf_mm*1e6
            p = c10*xx+c01*yy+c20*xx*xx+c11*xx*yy+c02*yy*yy
            px, py = c10+2*c20*xx+c11*yy, c01+c11*xx+2*c02*yy
            value[..., 2] += p*f
            jacobian[..., 2, 0] += px*f
            jacobian[..., 2, 1] += py*f
            jacobian[..., 2, 2] += p*df
            hessian[..., 2, 0, 0] += 2*c20*f
            hessian[..., 2, 1, 1] += 2*c02*f
            hessian[..., 2, 0, 1] += c11*f
            hessian[..., 2, 1, 0] += c11*f
            hessian[..., 2, 0, 2] += px*df
            hessian[..., 2, 2, 0] += px*df
            hessian[..., 2, 1, 2] += py*df
            hessian[..., 2, 2, 1] += py*df
            hessian[..., 2, 2, 2] += p*ddf
        if not all(np.isfinite(a).all() for a in (value, jacobian, hessian)):
            raise ValueError("Gun potential evaluation exceeds finite numerical range")
        return {"A_t_m": value, "jac_A_t": jacobian, "hess_A_t_per_m": hessian}

    def sample(self, positions_m):
        jet = self.sample_jet(positions_m)
        gradient = jet["jac_A_t"]
        return {"A_t_m": jet["A_t_m"], "B_t": np.stack((gradient[..., 2, 1],
                    -gradient[..., 2, 0], np.zeros(gradient.shape[:-2])), axis=-1)}
