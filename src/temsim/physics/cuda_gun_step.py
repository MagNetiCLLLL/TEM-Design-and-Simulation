"""Scoped CUDA magnetic sampling for the unchanged population DG update.

The electric predictor, discrete gradient, global iteration barrier and step
acceptance stay in the shared native iteration. Only its magnetic sample is
supplied by the explicitly selected, captured CUDA field adapter. No worker
pool divides this population into independent convergence decisions.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from temsim.immutable_json import freeze_json, thaw_json
from temsim.physics.compute_backend import (
    BACKEND_CPU, BACKEND_CUDA, BACKEND_PREFER_GPU, BACKEND_REQUIRE_GPU,
    normalise_backend,
)


SCHEMA = "scoped-cuda-magnetic-population-dg-v1"
_GPU_POLICIES = (BACKEND_CUDA, BACKEND_PREFER_GPU, BACKEND_REQUIRE_GPU)
_BINDING = ContextVar("cuda_gun_magnetic_population_step", default=None)


@dataclass(frozen=True)
class CudaGunBinding:
    """A prepared immutable source adapter and its trace-local cancellation."""
    sampler: object
    execution_backend: str
    execution_identity: object
    cancelled: object = None


def _coordinator_implementation():
    """Bind the actual DG formulas as well as the dispatch implementation."""
    paths = (Path(__file__), Path(__file__).with_name("static_energy_lorentz.py"),
             Path(__file__).with_name("discrete_gradient.py"),
             Path(__file__).with_name("relativistic_lorentz.py"))
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare_cuda_gun_binding(provider, *, execution_backend, cancelled=None):
    """Prepare only an explicit GPU policy; failures do not retry on CPU."""
    policy = normalise_backend(execution_backend)
    if policy not in _GPU_POLICIES:
        return None
    from temsim.physics.cuda_coil_potential import prepare_cuda_instrument_magnetic
    sampler = prepare_cuda_instrument_magnetic(provider, execution_backend=policy)
    if sampler is None:
        return None
    identity = freeze_json({"schema": SCHEMA, "requested_backend": policy,
        "coordinator_implementation": _coordinator_implementation(),
        "magnetic_execution": sampler.execution_identity})
    return CudaGunBinding(sampler, policy, identity, cancelled)


def gun_field_execution_identity(execution_backend, binding=None):
    """Keep CPU/CUDA and device/kernel choices out of one shared Gun cache."""
    policy = normalise_backend(execution_backend)
    if binding is not None:
        return binding.execution_identity
    return freeze_json({"schema": SCHEMA, "requested_backend": policy,
        "actual_field_dispatch": BACKEND_CPU,
        "coordinator_implementation": _coordinator_implementation()})


@contextmanager
def cuda_gun_execution(binding):
    """Scope one prepared sampler; nested CPU work masks an outer GPU binding."""
    token = _BINDING.set(binding)
    outcome = "raised"
    try:
        yield
        outcome = "returned"
    finally:
        _BINDING.reset(token)
        if binding is not None:
            from temsim.physics.backend_execution import record_backend
            record_backend("gun_magnetic_sampling", binding.execution_backend,
                "CUDA field dispatch / CPU discrete-gradient coordinator",
                outcome=outcome,
                reason="Explicit captured field adapter; receipt distinguishes actual CUDA and reference queries",
                execution_identity=thaw_json(binding.execution_identity),
                field_evidence=thaw_json(binding.sampler.execution_receipt()))


def current_cuda_gun_binding():
    """Expose only the current execution scope, never a physical input."""
    return _BINDING.get()


def try_cuda_gun_step(phase, dt, magnetic, electric, tolerance, iterations,
                      *, native_iteration):
    """Use one full-population native iteration with the selected CUDA field."""
    binding = _BINDING.get()
    if binding is None or magnetic is not binding.sampler:
        return None
    # The common routine calls the exact original discrete_gradient_update and
    # computes one maximum residual across every live row on every iteration.
    # GPU exceptions propagate before any candidate phase can be returned.
    return native_iteration(phase, dt, magnetic, electric,
        tolerance=tolerance, maximum_iterations=iterations,
        _cancelled=binding.cancelled)
