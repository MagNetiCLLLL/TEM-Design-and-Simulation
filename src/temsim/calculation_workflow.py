"""Explicit calculation intent, separate from saved physical instrument inputs."""

from temsim.immutable_json import json_digest


WORKFLOW_LABELS = {
    "full": "Complete calculation",
    "rays": "Ray Diagram",
    "sample": "Sample interactions",
    "sample_region": "Detailed sample paths and X-rays",
    "eds": "EDS spectrum",
    "stem": "STEM detector readout",
    "receiver": "Camera / screen image",
    "imaging": "Illuminating Image",
    "energy_filter": "Energy Filter",
}


def validate_workflow(workflow):
    if not isinstance(workflow, str) or workflow not in WORKFLOW_LABELS:
        raise ValueError(f"Unknown calculation workflow: {workflow!r}")
    return workflow


def workflow_signatures(signatures, workflow):
    """Scope a raw physical signature set; callers must not pass scoped keys.

    Page workflows keep a specimen-free optical column reference. Its branches
    cannot be reused as the full pipeline's material-loss quadrature. The exact
    incident identity remains shared across every workflow.
    """
    validate_workflow(workflow)
    result = dict(signatures)
    if workflow != "full":
        result["column"] = json_digest({"column": result["column"],
            "transport_scope": "optical-reference-without-specimen-v1"})
        result["workflow"] = workflow
        result["request"] = json_digest({
            "request": result["request"], "workflow": workflow,
            "workflow_schema": "explicit-page-calculation-v1",
        })
    return result


def admit_workflow(state, workflow):
    """Apply wave admission only to requested readouts; never enable a source."""
    validate_workflow(workflow)
    admit_unified_product(state, workflow)
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    from temsim.physics.source_admission import (
        admit_requested_wave_products, require_gun_wave_source,
    )
    require_physical_gun_source(state.electron_gun)
    if workflow == "full":
        admit_requested_wave_products(state)
    elif workflow == "imaging":
        # Legacy coherent imaging is separate from classical "receiver" work.
        # Particle counts cannot replace its missing coherent tip-origin state.
        require_gun_wave_source(state, product="Illuminating Image")
    elif workflow == "stem" and bool(getattr(state.sample, "stem_wave_enabled", False)):
        require_gun_wave_source(state, product="STEM wave imaging")


def admit_unified_product(state, workflow):
    """Reject products whose internal field consumer has not been migrated.

    This is request-specific: an inserted specimen does not prevent an optical
    reference ray calculation. The actual-source coherent page owns its separate
    potential/material adapter and never calls the legacy imaging workflow.
    """
    from temsim.simulation_modes import is_unified
    if not is_unified(state):
        return
    from temsim.physics.unified_field_runtime import unified_runtime_preflight
    report = unified_runtime_preflight(state)
    if report["issues"]:
        raise ValueError("Finite-Field Optics: " + "; ".join(report["issues"]))
    from temsim.specimen.source import specimen_interactions_active
    from temsim.physics.wave_imaging import tem_wave_imaging_enabled
    unsupported = None
    if workflow == "stem":
        validate_unified_stem_configuration(state)
        return
    if workflow in {"imaging", "energy_filter"}:
        unsupported = WORKFLOW_LABELS[workflow]
    elif workflow != "rays" and specimen_interactions_active(state.sample):
        unsupported = "finite-volume particle specimen transport"
    elif workflow == "full" and (tem_wave_imaging_enabled(state) or state.energy_filter.enabled):
        unsupported = "legacy wave imaging or dispersive Energy Filter transport"
    if unsupported:
        raise ValueError(
            f"Finite-Field Optics does not yet support {unsupported} with the shared potentials. "
            "Use the explicit legacy execution entry for this product. Optical Ray Diagram and "
            "the separately admitted Electron beam calculation retain their own supported scopes; "
            "the model was not changed automatically."
        )


def validate_unified_stem_configuration(state):
    """Admit the declared projected particle readout, not unmigrated products.

    This model retains a Gaussian closure of the executed incident population
    and a local affine raster response. It does not create finite-volume
    material histories or coherent waves from a projected intensity response.
    """
    sample = state.sample
    unsupported = None
    if bool(getattr(sample, "stem_wave_enabled", False)):
        unsupported = "coherent STEM raster propagation"
    elif bool(getattr(sample, "stem_fourdstem_enabled", False)):
        unsupported = "4D-STEM diffraction acquisition"
    elif str(getattr(sample, "stem_particle_model", "")) != "projected_atoms":
        unsupported = "finite-volume material-path STEM raster acquisition"
    elif bool(getattr(state.energy_filter, "enabled", False)):
        unsupported = "dispersive Energy Filter transport in STEM"
    if unsupported:
        raise ValueError(
            f"Finite-Field Optics does not yet support {unsupported}. "
            "The shared-potential STEM entry supports the explicitly selected "
            "projected_atoms particle model with local affine raster response; "
            "no model or optical parameter was changed automatically."
        )
