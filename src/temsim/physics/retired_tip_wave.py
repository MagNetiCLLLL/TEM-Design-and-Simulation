"""Compatibility errors for retired physical-tip coherent execution.

Historical wave data, source sampling and numerical operators remain readable.
New mixed-coherence work starts from executed particles near the specimen.
"""


class RetiredTipCoherentWorkflow(ValueError):
    """A historical Tip-origin coherent execution entry point was requested."""

    code = "TIP_COHERENT_WORKFLOW_RETIRED"


def reject_retired_tip_coherent_workflow():
    """Reject before device checks, cache access or physical preparation."""
    raise RetiredTipCoherentWorkflow(
        "Coherent propagation from the physical Tip has been retired. "
        "Use particle transport followed by mixed-coherence propagation near "
        "the specimen (python -m temsim.mixed_wave_cli). Historical wave "
        "archives remain readable; their Tip-origin calculation cannot resume."
    )
