"""Read-only numerical-profile labels for executed waves and state overlays."""
from collections.abc import Mapping
import math


def _profile(value):
    checkpoint = getattr(value, "checkpoint", value)
    record = checkpoint if isinstance(checkpoint, Mapping) else getattr(checkpoint, "record", {})
    seen = set()
    while isinstance(record, Mapping) and id(record) not in seen:
        seen.add(id(record))
        profile = record.get("unified_numerical_profile")
        if isinstance(profile, Mapping):
            return profile
        record = record.get("upstream")
    return {}


def _profiles(value, ancestors=frozenset()):
    if id(value) in ancestors:
        return []
    members = getattr(value, "members", None)
    if members is None:
        return [_profile(value)]
    ancestors = ancestors | {id(value)}
    return [profile for member in members for profile in _profiles(member, ancestors)]


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.6g}"
    return "unavailable"


def _details(profile):
    loss = profile.get("numerical_projection_loss_per_reference")
    l2 = profile.get("complex_l2_projection_bound_per_reference")
    evidence = ("projection loss unavailable" if loss is None or l2 is None else
                f"projection loss {_number(loss)} per tip electron; projection L2 cost {_number(l2)}")
    representation = ""
    source = (" | Nominal emission energy (preview); configured FWHM not integrated"
              if profile.get("source_nominal_energy_preview") is True else "")
    if profile.get("column_preview_phase_reselection") is True:
        representation = (
            f" | CPU phase preparation cap {_number(profile.get('phase_preparation_maximum_pixels'))} pixels/axis; "
            f"representation L2 estimate {_number(profile.get('estimated_representation_l2_per_reference'))}; "
            f"combined L2 cost {_number(profile.get('combined_projection_and_estimated_l2_per_reference'))} "
            "(estimated, not a certified bound)")
    return (f"Column cap {_number(profile.get('effective_column_maximum_pixels'))} pixels/axis "
            f"| Relative budgets: probability {_number(profile.get('maximum_support_probability_fraction'))}, "
            f"L2 {_number(profile.get('maximum_accumulated_support_l2'))} | {evidence}{source}{representation}")


def _state_details(profile):
    if profile.get("exploratory_preview") is True:
        return _details(profile)
    return "preview disabled" if profile.get("exploratory_preview") is False else "preview status unavailable"


def coherent_preview_text(value, *, member_index=None):
    """Label any preview member without merging independent error budgets.

    A selected strict member still identifies a mixed result containing other
    preview members. Original member records and display weights are untouched.
    """
    profiles = _profiles(value)
    preview_count = sum(profile.get("exploratory_preview") is True for profile in profiles)
    if not preview_count:
        return ""
    selected_profiles = None
    if member_index is not None:
        members = getattr(value, "members", (value,))
        if type(member_index) is not int or not 0 <= member_index < len(members):
            raise ValueError("Preview label needs a valid selected state index")
        selected_profiles = _profiles(members[member_index])
    heading = "Exploratory preview; not accuracy qualified | "
    if len(profiles) == 1:
        return heading + _details(profiles[0]) + " | "
    heading += (f"Mixed result: {preview_count} of {len(profiles)} states use preview; "
                "member budgets and losses are separate | ")
    if selected_profiles is None:
        parts = [f"State {index+1}: " + _state_details(profile) for index, profile in enumerate(profiles)]
    else:
        details = "; ".join(_state_details(profile) for profile in selected_profiles) or "preview status unavailable"
        parts = [f"Inspecting state {member_index+1}: " + details]
    return heading + " | ".join(parts) + " | "
