"""Single model dispatch for column and alignment source particles."""
from temsim.physics.backend_execution import cpu_call
from temsim import input_io


@input_io.using_state_inputs
@cpu_call("gun")
def trace_source_to_exit(state, count=None, *, previous_trace=None):
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    gun = state.electron_gun
    require_physical_gun_source(gun)
    from temsim.physics.incremental_particles import validate_particle_count_growth
    if previous_trace is not None:
        from types import SimpleNamespace
        if (count is not None and count > len(previous_trace.exit_bundle.ray_id)
                and count != gun.ray_count):
            raise ValueError("Append-only growth requires the requested count in the captured input state; prepare a new numerical input snapshot instead of overriding the count on an old source")
        validate_particle_count_growth(SimpleNamespace(gun_trace=previous_trace), state)
    from temsim.vacuum import bind_gun_environment
    bind_gun_environment(state)
    cancelled = getattr(state, "_tuning_cancelled", None)
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    with instrument_gun_field_context(state):
        from dataclasses import replace
        from temsim.physics.incremental_particles import source_metadata, extend_source_trace
        metadata = source_metadata(state) if count is None else None
        extended = extend_source_trace(state, previous_trace, metadata, cancelled=cancelled)
        if extended is not None:
            return extended
        if cancelled is not None:
            result = gun.trace_to_exit(count, cancelled=cancelled)
        elif count is None:
            result = gun.trace_to_exit()
        else:
            result = gun.trace_to_exit(count)
        # The Gun's own cached result remains unchanged. This input-bound
        # wrapper adds an explicit, independently checked growth certificate.
        return replace(result, incremental_source=metadata) if metadata is not None else result
