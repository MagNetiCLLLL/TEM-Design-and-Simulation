"""Immutable executed wave state at a declared physical reference plane."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import math
from types import MappingProxyType

import numpy as np

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.wave_flux import BeamState


# Keep the original emitted-electron reference for existing stored records.
# This labels the current denominator; it does not imply a Tip wave solve.
EMITTED_ELECTRON_REFERENCE = "physical-feg-tip-emitted-electron"


@dataclass(frozen=True)
class WaveCheckpoint:
    beam: BeamState
    plane_z_mm: float
    reference_current_a: float
    record: object
    auxiliary_arrays: object = field(default_factory=dict)

    def __post_init__(self):
        if not self.beam.reference_plane:
            raise ValueError("A wave checkpoint requires a declared electron reference")
        if not math.isfinite(self.plane_z_mm) or self.plane_z_mm <= 0:
            raise ValueError("A wave checkpoint requires a positive finite physical plane")
        if not math.isfinite(self.reference_current_a) or self.reference_current_a < 0:
            raise ValueError("Invalid wave reference current")
        if self.beam.total_weight > 1+1e-10:
            raise ValueError("Wave propagation created electron probability")
        object.__setattr__(self, "record", freeze_json(self.record))
        if hasattr(self.beam, "content_identity"):
            return
        auxiliary = {}
        for name, value in self.auxiliary_arrays.items():
            if (not isinstance(name, str) or not name or len(name) > 80
                    or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)):
                raise ValueError("Checkpoint auxiliary state needs a plain array name")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("Checkpoint auxiliary state cannot contain Python objects")
            auxiliary[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self, "auxiliary_arrays", MappingProxyType(auxiliary))

    @property
    def digest(self):
        # Preserve the existing storage digest definition for old archives.
        if hasattr(self.beam, "content_identity"):
            return json_digest({"record": self.record, "stored_payload": self.beam.content_identity,
                                "plane_z_mm": self.plane_z_mm, "current_a": self.reference_current_a})
        digest = sha256()
        for name, array in sorted(self.auxiliary_arrays.items()):
            digest.update(name.encode())
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        for mode in self.beam.modes:
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, name)
                digest.update(b"none" if value is None else np.ascontiguousarray(value).tobytes())
            digest.update(json_digest((mode.mode_id, mode.energy_kev, mode.weight_per_reference_electron,
                                       None if mode.axial_reference is None else asdict(mode.axial_reference),
                                       mode.scattering_history)).encode())
        return json_digest({"record": self.record, "payload": digest.hexdigest(),
                            "plane_z_mm": self.plane_z_mm, "current_a": self.reference_current_a})

    @property
    def transmitted_current_a(self):
        return self.reference_current_a*self.beam.total_weight
