"""Cold field-emitter geometry and deterministic emission phase space."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np

from temsim.component_keys import FEG_TIP
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.physics.chromatic import cold_feg_energy_offsets
from temsim.optics.electron_gun.tip_coherence import TipCoherence


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(indices.size, dtype=float)
    fraction = 1.0 / float(base)
    work = indices.astype(np.int64, copy=True)
    while np.any(work):
        values += (work % base) * fraction
        work //= base
        fraction /= float(base)
    return values


def _halton_dimensions(count: int, bases: Iterable[int]) -> list[np.ndarray]:
    indices = np.arange(18, 18 + count, dtype=np.int64)
    return [_radical_inverse(indices, base) for base in bases]


def _truncated_gaussian_disk(radial_u, azimuth_u, sigma, cutoff):
    if sigma <= 0.0 or cutoff <= 0.0:
        return np.zeros_like(radial_u), np.zeros_like(radial_u)
    retained = 1.0 - np.exp(-(cutoff * cutoff) / (2.0 * sigma * sigma))
    radius = sigma * np.sqrt(
        -2.0 * np.log(np.maximum(1.0 - radial_u * retained, 1e-15))
    )
    azimuth = 2.0 * np.pi * azimuth_u
    return radius * np.cos(azimuth), radius * np.sin(azimuth)


@dataclass(frozen=True)
class EmissionSampling:
    """Explicit nested source-level quadrature; absent records stay legacy."""
    seed: int = 0
    schema: str = "tip-sobol-antithetic-v1"

    def validate(self):
        if self.schema != "tip-sobol-antithetic-v1" or type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Tip sampling requires the versioned Sobol rule and an unsigned 32-bit seed")
        return self

    def validate_count(self, count):
        if type(count) is not int or not 16 <= count <= 1048576 or count & (count-1):
            raise ValueError("Sobol tip sampling requires a complete power-of-two ray count from 16 to 1048576")
        return count


@dataclass(frozen=True)
class EmissionCDFQuadrature:
    """Captured CDF nodes and original weights, including zero-current probes.

    Signs act on local x/y/slopes before physical tip curvature. No emitted or
    transmitted population is recentered or renormalized by this record.
    """
    coordinates: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]
    signs: tuple[int, ...]
    ray_ids: tuple[int, ...]
    schema: str = "tip-explicit-cdf-v1"
    role: str = "quadrature"
    construction: str = ""

    @property
    def total(self):
        return len(self.coordinates)

    def validate(self):
        if self.schema != "tip-explicit-cdf-v1" or self.role not in {"quadrature", "boundary-probes"}:
            raise ValueError("Explicit tip CDF nodes require a supported schema and role")
        if not isinstance(self.construction, str) or len(self.construction) > 16384:
            raise ValueError("CDF construction description must be a bounded text record")
        arrays = (self.coordinates, self.weights, self.signs, self.ray_ids)
        if any(type(a) is not tuple for a in arrays) or not 9 <= self.total <= 1048576:
            raise ValueError("Explicit tip CDF records require immutable tuples with 9 to 1048576 nodes")
        if any(len(a) != self.total for a in arrays):
            raise ValueError("Explicit tip CDF fields must have the same complete node count")
        if any(type(row) is not tuple or len(row) != 6 or any(
                type(v) is not float or not math.isfinite(v) or not 0 <= v <= 1 for v in row)
               for row in self.coordinates):
            raise ValueError("Explicit tip coordinates must contain six finite float CDF values in [0, 1]")
        if any(type(w) is not float or not math.isfinite(w) or not 0 <= w <= 1 for w in self.weights):
            raise ValueError("Explicit tip CDF weights must be finite non-negative probabilities")
        if any(type(s) is not int or s not in {-1, 1} for s in self.signs):
            raise ValueError("Explicit tip CDF signs must be integer -1 or +1")
        if any(type(i) is not int or not 0 <= i <= np.iinfo(np.int64).max for i in self.ray_ids) or len(set(self.ray_ids)) != self.total:
            raise ValueError("Explicit tip CDF ray IDs must be unique non-negative int64 integers")
        total = math.fsum(self.weights)
        if self.role == "boundary-probes":
            if total != 0:
                raise ValueError("Boundary probes must carry exactly zero physical current")
        elif not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("Explicit tip quadrature weights must sum to one; they are never renormalized")
        return self

    @classmethod
    def from_arrays(cls, coordinates, weights, *, signs=None, ray_ids=None,
                    role="quadrature", construction=""):
        u, w = np.asarray(coordinates), np.asarray(weights)
        if u.ndim != 2 or u.shape[1] != 6 or w.shape != (u.shape[0],) or u.dtype.kind not in "fiu" or w.dtype.kind not in "fiu":
            raise ValueError("CDF coordinates and weights must be numeric N-by-6 and N arrays")
        n = len(u)
        signs = np.ones(n, dtype=np.int64) if signs is None else np.asarray(signs)
        ray_ids = np.arange(n, dtype=np.int64) if ray_ids is None else np.asarray(ray_ids)
        if signs.shape != (n,) or ray_ids.shape != (n,) or signs.dtype.kind not in "iu" or ray_ids.dtype.kind not in "iu":
            raise ValueError("CDF signs and IDs must be complete integer arrays")
        return cls(tuple(tuple(float(v) for v in row) for row in u), tuple(float(v) for v in w),
                   tuple(int(v) for v in signs), tuple(int(v) for v in ray_ids),
                   role=role, construction=construction).validate()

    @classmethod
    def from_dict(cls, row):
        expected = {"coordinates", "weights", "signs", "ray_ids", "schema", "role", "construction"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("Explicit tip CDF record must include every versioned field")
        if row["schema"] != "tip-explicit-cdf-v1":
            raise ValueError("Unsupported explicit tip CDF schema")
        return cls.from_arrays(row["coordinates"], row["weights"], signs=row["signs"],
            ray_ids=row["ray_ids"], role=row["role"], construction=row["construction"])

    @property
    def digest(self):
        from dataclasses import asdict
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmissionQuadrature:
    """Opt-in product quadrature at the tip; never a new physical source."""
    spatial: int
    directions: int
    energies: int
    schema: str = "tip-product-cdf-v1"

    @property
    def total(self):
        return self.spatial * self.directions * self.energies

    def validate(self):
        if self.schema != "tip-product-cdf-v1" or any(type(n) is not int or not 3 <= n <= 65536
                for n in (self.spatial, self.directions, self.energies)) or self.total > 1048576:
            raise ValueError("Tip product quadrature requires integer factors >= 3 and at most 1048576 rays")
        return self


@dataclass
class ColdFieldEmitter:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    name: str = "Cold Field Emission Tip"
    key: str = FEG_TIP
    tip_radius_nm: float = 100.0
    tip_cone_half_angle_deg: float = 5.0
    emitter_material: str = "W(310)"
    work_function_ev: float = 4.4
    vacuum_pa: float = 1.0e-8
    emission_current_na: float = 10000.0
    emission_energy_ev: float = 0.3
    minimum_kinetic_energy_ev: float = 0.01
    virtual_source_fwhm_nm: float = 5.0
    angular_rms_mrad: float = 1.0
    angular_cutoff_mrad: float = 3.0
    energy_spread_fwhm_ev: float = 0.3
    young_decay_width_ev: float = 0.20
    boersch_sigma_ev: float = 0.10
    energy_half_range_ev: float = 1.0
    ray_count: int = 3000

    @property
    def cdf_quadrature(self):
        return self.__dict__.get("_emission_cdf_quadrature")

    @cdf_quadrature.setter
    def cdf_quadrature(self, value):
        if value is None:
            self.__dict__.pop("_emission_cdf_quadrature", None)
        elif isinstance(value, EmissionCDFQuadrature):
            self.__dict__["_emission_cdf_quadrature"] = value.validate()
        else:
            raise ValueError("Expected a captured explicit tip CDF quadrature")

    @property
    def sampling(self):
        return self.__dict__.get("_emission_sampling")

    @sampling.setter
    def sampling(self, value):
        if value is None:
            self.__dict__.pop("_emission_sampling", None)
        elif isinstance(value, EmissionSampling):
            self.__dict__["_emission_sampling"] = value.validate()
        else:
            raise ValueError("Expected an explicit versioned tip emission sampling rule")

    @property
    def quadrature(self):
        return self.__dict__.get("_emission_quadrature")

    @quadrature.setter
    def quadrature(self, value):
        if value is None:
            self.__dict__.pop("_emission_quadrature", None)
        elif isinstance(value, EmissionQuadrature):
            self.__dict__["_emission_quadrature"] = value.validate()
        else:
            raise ValueError("Expected an explicit tip emission quadrature")

    @property
    def curvature_nm_inv(self):
        """Operating curvature; absent historical flat records mean zero."""
        return self.__dict__.get("_tip_curvature_nm_inv", 0.0)

    @curvature_nm_inv.setter
    def curvature_nm_inv(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Tip curvature must be finite and non-negative")
        if value == 0:
            self.__dict__.pop("_tip_curvature_nm_inv", None)
            self.__dict__.pop("_tip_curvature_model", None)
        else:
            model = self.curvature_model
            self.__dict__["_tip_curvature_nm_inv"] = value
            self.__dict__["_tip_curvature_model"] = model

    @property
    def curvature_model(self):
        from temsim.optics.electron_gun.tip_curvature import MODEL
        return self.__dict__.get("_tip_curvature_model", MODEL)

    @curvature_model.setter
    def curvature_model(self, value):
        from temsim.optics.electron_gun.tip_curvature import MODEL
        if value != MODEL:
            raise ValueError("Unsupported continuous tip geometry model")
        if self.curvature_nm_inv:
            self.__dict__["_tip_curvature_model"] = value
        else:
            self.__dict__.pop("_tip_curvature_model", None)

    @property
    def surface_model(self):
        """Explicit v1 surface model; absent on historical parameter records."""
        return self.__dict__.get("_surface_model")

    @surface_model.setter
    def surface_model(self, value):
        if value is None:
            self.__dict__.pop("_surface_model", None)
        else:
            from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
            if not isinstance(value, TipSurfaceModel):
                raise ValueError("Expected a versioned tip surface model")
            value.validate()
            self.__dict__["_surface_model"] = value

    @property
    def coherence(self) -> TipCoherence | None:
        # Keep the pre-existing dataclass field schema and absent-parameter
        # graph byte-for-byte compatible with archived classical gun data.
        # A selected tip model is an explicit captured instance attribute.
        return self.__dict__.get("_tip_coherence")

    @coherence.setter
    def coherence(self, value):
        if value is None:
            self.__dict__.pop("_tip_coherence", None)
        else:
            self.__dict__["_tip_coherence"] = value

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_outer_diameter_mm

    @property
    def emitted_current_a(self):
        if self.surface_model is not None:
            return self.surface_model.current_na * 1e-9
        return float(self.emission_current_na) * 1.0e-9

    @property
    def optical_reference_from_tip_mm(self):
        return 0.0

    @property
    def kind(self):
        return "cold_field_emitter"

    @property
    def shape_profile(self):
        return "feg_tip"

    def validate(self):
        if self.cdf_quadrature is not None:
            self.cdf_quadrature.validate()
            if self.ray_count != self.cdf_quadrature.total:
                raise ValueError("Tip ray count must match the complete explicit CDF table")
            if any(model is not None for model in (self.sampling, self.quadrature, self.surface_model, self.coherence)):
                raise ValueError("Explicit tip CDF quadrature cannot be combined with another sampling or source model")
            if not np.isfinite(self.energy_half_range_ev) or self.energy_half_range_ev <= 0:
                raise ValueError("Explicit tip CDF quadrature requires a finite positive energy half range")
            if getattr(self, "_tuning_boundary_probes", 0) or getattr(self, "_tuning_surface_probes", 0):
                raise ValueError("Explicit tip CDF tables cannot be repartitioned for tuning probes")
        if self.sampling is not None:
            self.sampling.validate().validate_count(self.ray_count)
            if not np.isfinite(self.energy_half_range_ev) or self.energy_half_range_ev <= 0:
                raise ValueError("Sobol tip sampling requires a finite positive energy half range")
            if self.quadrature is not None or self.surface_model is not None or self.coherence is not None:
                raise ValueError("Sobol tip sampling cannot be combined with product, surface, or coherent source models")
            if getattr(self, "_tuning_boundary_probes", 0) or getattr(self, "_tuning_surface_probes", 0):
                raise ValueError("Sobol tip sampling cannot be repartitioned for tuning probes; use High accuracy")
        if self.quadrature is not None:
            self.quadrature.validate()
            if self.coherence is not None or (self.surface_model is not None
                    and self.surface_model.coherence is not None and not self.surface_model.shared_boundary):
                raise ValueError("Product quadrature is classical only for this source model; use a classical law or the shared cap geometric-ray representation")
        from temsim.optics.electron_gun.tip_curvature import validate_curvature
        validate_curvature(self)
        if self.surface_model is not None:
            self.surface_model.validate()
            if self.coherence is not None:
                raise ValueError("Surface emission and historical Gaussian-Schell emission cannot both be selected")
            if int(self.ray_count) < 9:
                raise ValueError("Surface emission requires at least 9 rays")
            return self
        nonnegative = (
            "tip_radius_nm",
            "work_function_ev",
            "vacuum_pa",
            "emission_current_na",
            "emission_energy_ev",
            "minimum_kinetic_energy_ev",
            "virtual_source_fwhm_nm",
            "angular_rms_mrad",
            "angular_cutoff_mrad",
            "energy_spread_fwhm_ev",
            "young_decay_width_ev",
            "boersch_sigma_ev",
            "energy_half_range_ev",
        )
        for attribute in nonnegative:
            value = float(getattr(self, attribute))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{self.name} {attribute} must be finite and non-negative.")
        if self.coherence is not None:
            if not isinstance(self.coherence, TipCoherence):
                raise ValueError("FEG tip coherence must be a TipCoherence parameter record")
            self.coherence.validate()
            if self.virtual_source_fwhm_nm <= 0.0:
                raise ValueError("A coherent tip needs positive spatial FWHM")
        if int(self.ray_count) < 9:
            raise ValueError("Cold FEG ray count must be at least 9.")
        if self.emission_energy_ev <= 0.0:
            raise ValueError("Cold FEG emission energy must be positive.")
        if self.minimum_kinetic_energy_ev >= self.emission_energy_ev:
            raise ValueError(
                "Cold FEG minimum kinetic energy must be below its mean "
                "emission energy."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Cold FEG visible tip length must be positive.")
        return self

    def emit(self, count: int | None = None) -> EmissionBundle:
        self.validate()
        if self.cdf_quadrature is not None:
            requested = self.ray_count if count is None else count
            if type(requested) is not int or requested != self.cdf_quadrature.total:
                raise ValueError("Requested ray budget must match the complete explicit CDF table")
            from temsim.optics.electron_gun.tip_joint_sampling import explicit_cdf_bundle
            return explicit_cdf_bundle(self)
        if self.sampling is not None:
            self.sampling.validate_count(self.ray_count if count is None else count)
        n = int(self.ray_count if count is None else count)
        if n < 9:
            raise ValueError("Cold FEG emission requires at least 9 rays.")
        if self.sampling is not None:
            from temsim.optics.electron_gun.tip_joint_sampling import joint_bundle
            return joint_bundle(self, n)
        if self.quadrature is not None:
            if n != self.quadrature.total:
                raise ValueError("Requested ray budget must equal the complete tip product quadrature; change its factors explicitly")
            if getattr(self, "_tuning_boundary_probes", 0) or getattr(self, "_tuning_surface_probes", 0):
                raise ValueError("Product quadrature cannot be silently repartitioned for tuning probes")
            if self.surface_model is not None:
                from temsim.optics.electron_gun.tip_surface import surface_bundle
                return surface_bundle(self.surface_model, n, quadrature=self.quadrature)
            from temsim.optics.electron_gun.tip_sampling import flat_product_bundle
            return flat_product_bundle(self, self.quadrature)
        if self.surface_model is not None:
            from temsim.optics.electron_gun.tip_surface import surface_bundle
            return surface_bundle(self.surface_model, n,
                support_probes=int(getattr(self, "_tuning_surface_probes", 0)))
        if self.coherence is not None:
            from temsim.optics.electron_gun.tip_coherence import tip_particle_samples
            bundle = tip_particle_samples(self, n)
            if getattr(self, "_tuning_boundary_probes", 0):
                from temsim.physics.optical_tuning import add_source_support_probes
                bundle = add_source_support_probes(bundle, self)
            return bundle
        u_r, u_phi, u_a, u_theta, u_young, u_boersch = _halton_dimensions(
            n, (2, 3, 5, 7, 11, 13)
        )
        spatial_sigma_m = self.virtual_source_fwhm_nm / 2.354820045 * 1e-9
        x, y = _truncated_gaussian_disk(
            u_r, u_phi, spatial_sigma_m, 3.0 * spatial_sigma_m
        )
        angular_sigma = self.angular_rms_mrad * 1e-3
        angular_cutoff = self.angular_cutoff_mrad * 1e-3
        tx, ty = _truncated_gaussian_disk(
            u_a, u_theta, angular_sigma, angular_cutoff
        )
        energy = cold_feg_energy_offsets(
            n,
            self.energy_spread_fwhm_ev,
            self.energy_half_range_ev,
            self.young_decay_width_ev,
            self.boersch_sigma_ev,
            quantiles=(u_young, u_boersch),
            mean_kinetic_energy_ev=self.emission_energy_ev,
            minimum_kinetic_energy_ev=self.minimum_kinetic_energy_ev,
        )
        bundle = EmissionBundle(
            x_m=x,
            y_m=y,
            tx_rad=tx,
            ty_rad=ty,
            energy_offset_ev=energy,
            weight=np.full(n, 1.0 / n, dtype=float),
            ray_id=np.arange(n, dtype=np.int64),
        )
        if getattr(self, "_tuning_boundary_probes", 0):
            from temsim.physics.optical_tuning import add_source_support_probes
            bundle = add_source_support_probes(bundle, self)
        from temsim.optics.electron_gun.tip_curvature import curve_bundle
        return curve_bundle(bundle, self)

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": self.mechanical_outer_diameter_mm,
            "shape_profile": self.shape_profile,
        }
