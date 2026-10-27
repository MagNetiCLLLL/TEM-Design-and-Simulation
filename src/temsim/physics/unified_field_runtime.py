"""One immutable field and mechanical capture for production column consumers.

The physical gun owns emission and acceleration up to its executed exit. This
bundle owns all selected straight-column fields afterwards. Saved empirical
aberrations and equivalent image lenses remain inputs, but are not actions of
this executor. Static E is retained, including small gun tails; a magnetic-only
wave consumer must explicitly reject unsupported acceleration.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
import weakref

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.potential_geometry import PotentialGeometrySnapshot, capture_potential_geometry
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot


_SCHEMA = "unified-field-runtime-v1"
_CACHE = {}
_INTERN = weakref.WeakValueDictionary()
_LOCK = RLock()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class UnifiedFieldRuntime:
    snapshot: ElectromagneticPotentialSnapshot | DynamicPotentialSnapshot
    geometry: PotentialGeometrySnapshot
    source_identity: str
    metadata: Mapping = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self):
        if type(self.snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot) or type(self.geometry) is not PotentialGeometrySnapshot:
            raise ValueError("Unified execution requires frozen field and geometry snapshots")
        if self.geometry.source_scene_geometry_digest != self.snapshot.scene.geometry_digest:
            raise ValueError("Unified field and mechanical captures describe different geometry")
        if not isinstance(self.source_identity, str) or len(self.source_identity) != 64:
            raise ValueError("Unified runtime requires a complete instrument input identity")
        if "emission_time_s" in self.metadata:
            from temsim.physics.emission_timing import validate_emission_time_s
            validate_emission_time_s(self.metadata["emission_time_s"])
        object.__setattr__(self, "metadata", freeze_json(thaw_json(self.metadata)))
        object.__setattr__(self, "digest", json_digest(self._payload()))

    @property
    def start_z_m(self):
        return self.geometry.start_z_m

    @property
    def stop_z_m(self):
        return self.geometry.stop_z_m

    def _payload(self):
        return {"schema": _SCHEMA, "snapshot": self.snapshot.to_dict(),
                "geometry": self.geometry.to_dict(), "source_identity": self.source_identity,
                "metadata": thaw_json(self.metadata)}

    def to_dict(self):
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"schema", "snapshot", "geometry", "source_identity", "metadata", "digest"} or record["schema"] != _SCHEMA:
            raise ValueError("Invalid unified runtime record")
        from temsim.physics.dynamic_potential import SCHEMA as DYNAMIC_SCHEMA
        loader = DynamicPotentialSnapshot if record["snapshot"].get("schema") == DYNAMIC_SCHEMA else ElectromagneticPotentialSnapshot
        result = cls(loader.from_dict(record["snapshot"]),
                     PotentialGeometrySnapshot.from_dict(record["geometry"]),
                     record["source_identity"], record["metadata"])
        if result.digest != record["digest"]:
            raise ValueError("Unified runtime digest mismatch")
        return result

    def require_state(self, state):
        if capture_instrument_snapshot(state).digest != self.source_identity:
            raise ValueError("Instrument inputs or implementation changed after unified field capture")


def unified_source_selection(state):
    """Select physical field owners, including the shared Descan drive once."""
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    from temsim.optics.image_corrector import ImageCorrectorSadPlaneComponent
    lenses = tuple(lens.key for lens in state.lenses if lens.enabled)
    components = {item.key: item for collection in (state.stigmators, state.corrector_elements, state.deflectors)
                  for item in collection}
    auxiliaries = tuple(key for key, item in components.items() if item.enabled and key not in lenses
                        and key not in SHARED_DEFLECTOR_HOSTS and key != "probe_dp12_scan_deflector"
                        and type(item) is not ImageCorrectorSadPlaneComponent)
    return lenses, auxiliaries


def time_dependent_drive(item):
    """Prove only exact constant native waveforms; never test sampled values."""
    if not getattr(item,"enabled",False):
        return False
    scan,wobble=bool(getattr(item,"scan_enabled",False)),bool(getattr(item,"wobble_enabled",False))
    if not scan and not wobble:
        return False
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    if type(item) not in (AcDeflectorComponent,DescanDeflectorComponent):
        return True
    if scan:
        matrix=np.asarray(item.scan_command_matrix_mrad,dtype=float)
        return bool(np.any(matrix[:,0]!=0.) or (item.scan_lines>1 and np.any(matrix[:,1]!=0.)))
    return bool(wobble and (item.wobble_amplitude_x_mrad!=0. or item.wobble_amplitude_y_mrad!=0.))


def unified_runtime_preflight(state):
    """Cheap explicit support checks; never solve fields or mutate settings."""
    issues = []
    if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
        issues.append("Residual-gas collisions are not implemented by the unified field executor")
    if bool(getattr(getattr(state, "nanopulser", None), "installed", False)):
        issues.append("Installed nanopulser requires a retained electric/time-dependent potential")
    lenses, auxiliary = unified_source_selection(state)
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    from temsim.physics.emission_timing import resolve_emission_time_s
    try:
        resolve_emission_time_s(state)
    except ValueError as exc:
        issues.append(str(exc))
    for item in (*state.stigmators, *state.corrector_elements, *state.deflectors):
        if time_dependent_drive(item) and type(item) not in (AcDeflectorComponent, DescanDeflectorComponent):
            issues.append(f"{item.key}: no retained native time-dependent potential law")
    for key in lenses:
        recipe = getattr(state, "lens_field_map_descriptors", {}).get(key, {})
        if recipe and recipe.get("solver") not in {"axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"}:
            issues.append(f"{key}: B-only map has no retained magnetic potential")
    return freeze_json({"schema": "unified-field-preflight-v1", "supported": not issues,
                        "issues": issues, "round_lens_keys": lenses, "auxiliary_keys": auxiliary,
                        "source": "executed physical gun exit; no synthetic source substitution",
                        "electric_policy": "retained native phi and E plus -partial_t(A); no tolerance-based tail deletion",
                        "time_policy": "actual emission epoch plus flight time; native raster flyback requires explicit temporal admission"})


def capture_unified_field_runtime(state, start_z_m=None, stop_z_m=None):
    from temsim.physics.particle_sections import section_limits
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    from temsim.physics.gaussian_remote_domain import POLICY as REMOTE_POLICY
    from temsim.physics.dynamic_potential import capture_native_dynamic_coil
    from temsim.physics.emission_timing import resolve_emission_time_s
    report = unified_runtime_preflight(state)
    if report["issues"]:
        raise ValueError("Finite-Field Optics: "+"; ".join(report["issues"]))
    lower, upper = section_limits(state)
    start = lower*.001 if start_z_m is None else float(start_z_m)
    stop = upper*.001 if stop_z_m is None else float(stop_z_m)
    if not np.isfinite((start, stop)).all() or start < lower*.001 or stop > upper*.001 or stop <= start:
        raise ValueError("Unified runtime span must lie between the physical gun exit and straight-column stop")
    gun = state.electron_gun
    epoch = resolve_emission_time_s(state)
    native = capture_selected_column_potentials(state,
        round_lens_keys=tuple(report["round_lens_keys"]), auxiliary_keys=tuple(report["auxiliary_keys"]),
        include_electric=True, include_gun_alignment=all(getattr(gun, key, None) is not None for key in ("deflector", "stigmator")),
        time_s=epoch)
    rows, dynamic_coils = [], []
    drive_components={item.key:item for item in (*state.stigmators,*state.corrector_elements,*state.deflectors)}
    for row in native.contributions:
        potential = row.potential
        if type(potential) is AnalyticMagneticPotential:
            field=potential.field
            if getattr(field,"dynamic",False):
                dynamic = capture_native_dynamic_coil(row, drive_components)
                if dynamic is not None:
                    dynamic_coils.append(dynamic)
                    continue
                potential=AnalyticMagneticPotential(replace(field,dynamic=False),{
                    **thaw_json(potential.provenance),"dynamic_drive_frozen":False,
                    "static_drive_proof":"native physical coil has identically zero varying coefficients"})
            potential = (HighOrderRoundLensPotential(potential.field, radial_order=3, axial_domain="gaussian_all_z",remote_domain=REMOTE_POLICY)
                         if type(potential.field) is FrozenAnalyticField else AnalyticPotentialDomain(potential))
        rows.append(PotentialContribution(row.source_id, row.owner_ids, potential, row.kind))
    snapshot = ElectromagneticPotentialSnapshot(native.scene, tuple(rows), native.gauge)
    if dynamic_coils:
        snapshot = DynamicPotentialSnapshot(snapshot, tuple(dynamic_coils))
    geometry = capture_potential_geometry(state, start, stop)
    identity = capture_instrument_snapshot(state).digest
    return UnifiedFieldRuntime(snapshot, geometry, identity, {
        "source_contract": report["source"], "electric_policy": report["electric_policy"],
        "emission_time_s": epoch, "time_policy": report["time_policy"],
        "dynamic_snapshot_digest": snapshot.numerical_digest if dynamic_coils else None,
        "round_radial_order": 3, "empirical_aberrations_applied": False,
        "round_remote_domain_policy": REMOTE_POLICY,
        "equivalent_image_maps_applied": False, "geometry_owner": "unified captured finite boundaries",
        "saved_legacy_terms": "preserved in instrument input identity; inactive in this field-owned interval",
        "qualification": "native engineering fields with explicit domains; not OEM field calibration"})


def get_unified_field_runtime(state):
    """Reuse exactly matching immutable captures; edits invalidate both solvers."""
    identity = capture_instrument_snapshot(state).digest
    with _LOCK:
        previous = _CACHE.get(id(state))
        if previous is not None and previous[0]() is state and previous[1].source_identity == identity:
            return previous[1]
        shared = _INTERN.get(identity)
        if shared is not None:
            _CACHE[id(state)] = (weakref.ref(state,lambda ref,key=id(state):_CACHE.pop(key,None)),shared)
            return shared
    result = capture_unified_field_runtime(state)
    with _LOCK:
        _CACHE[id(state)] = (weakref.ref(state, lambda ref, key=id(state): _CACHE.pop(key, None)), result)
        _INTERN[result.source_identity] = result
    return result
