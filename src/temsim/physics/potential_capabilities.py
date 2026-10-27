"""Read-only inventory of unified-potential coverage, not solver admission.

This reports available adapters and separate material/measurement owners.
Actual executors still enforce source types, time dependence, finite domains,
energy, representation order and numerical budgets at the requested points.
No FEM solve or field-provider replacement occurs while preparing the report.
"""
from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.physical_scene import capture_physical_scene


def potential_capability_inventory(state):
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    from temsim.simulation_modes import uses_field_maps
    scene = capture_physical_scene(state)
    rows = []

    def add(key, family, representation, classical, coherent, *, owner=None, detail=""):
        rows.append({"key": key, "physical_owner": owner or scene.owner_for(key) or key,
            "family": family, "representation": representation, "classical": classical,
            "coherent": coherent, "detail": detail, "production_admission": False})

    for lens in getattr(state, "lenses", ()):
        if not lens.enabled:
            add(lens.key, "round_lens", "disabled optical source; geometry remains", "inactive", "inactive")
            continue
        recipe = getattr(state, "lens_field_map_descriptors", {}).get(lens.key, {}) if uses_field_maps(state) else {}
        if recipe and recipe.get("solver") not in {"axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"}:
            add(lens.key, "round_lens", "imported map", "existing B-map route",
                "requires retained potential adapter", detail="B samples do not establish a unique usable A or phase reference")
        elif recipe:
            add(lens.key, "round_lens", "retained axisymmetric FEM A_phi", "bounded potential Lorentz",
                "bounded fixed-grid benchmark; natural aberrations require a separately qualified reconstruction",
                detail="Nonlinear shared circuits must be captured once at the complete current vector")
        else:
            add(lens.key, "round_lens", "native Gaussian A_phi; optional high-order vacuum expansion",
                "bounded potential Lorentz", "bounded static magnetic fixed-grid/carrier operators",
                detail="Field order, radial domain and whole-column convergence are independent qualifications")
    seen = set()
    for component in (*getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ()),
                      *getattr(state, "deflectors", ())):
        key = component.key
        if key in seen:
            continue
        seen.add(key)
        if key in SHARED_DEFLECTOR_HOSTS:
            add(key, "logical_drive", "summed at physical host", "no independent field",
                "no independent field", owner=SHARED_DEFLECTOR_HOSTS[key],
                detail="Do not sum this channel again after capturing its host")
            continue
        if not component.enabled:
            add(key, "multipole", "disabled optical source; geometry remains", "inactive", "inactive")
            continue
        native = any(hasattr(component, name) for name in (
            "quadrupole_tensor_m2", "quadrupole_strength_m2", "hexapole_strength_components_m3",
            "kick_events", "upper_x_mrad"))
        dynamic = bool(getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False))
        timed_native = dynamic and type(component) in (AcDeflectorComponent, DescanDeflectorComponent)
        status = ("native timed-coil Lorentz and budgeted retarded scalar wave" if timed_native else
                  "requires time-dependent executor" if dynamic else "bounded static magnetic operator")
        add(key, "multipole", "native polynomial A_z" if native else "no native potential adapter",
            status if native else "unsupported", status if native else "unsupported",
            detail=("Executor admission still checks native drive methods, coil pose, flyback crossings, clocks and error budgets; dynamic material scattering remains unavailable"
                    if timed_native else "A frozen-time field query is not a qualified dynamic propagation") if dynamic else
                   "Uniform coil and Gaussian multipole envelopes retain their declared effective-field assumptions")
    gun = getattr(state, "electron_gun", None)
    if gun is not None:
        add("gun_electric", "acceleration", "retained native phi", "bounded Lorentz with scalar energy law",
            "executed FEG handoff and budgeted positive-branch electric carrier",
            detail="Native source/provider admission, original energy reference and amplitude/phase budgets remain required; no spin, reflected branch or arbitrary pulse wavepacket")
        for name in ("deflector", "stigmator"):
            component = getattr(gun, name, None)
            if component is not None:
                add(component.key, "gun_alignment", "native soft-window polynomial A_z",
                    "bounded potential Lorentz", "fixed-grid magnetic benchmark only",
                    detail="No resolved coil winding or Maxwell fringe solve is implied")
        add("gun_wien_monochromator", "crossed_fields", "existing captured gun model",
            "existing declared gun capability", "existing gun executor; no separately admitted column Wien adapter")
    add("sample", "material", "existing specimen scene and interaction ledger",
        "existing particle interaction executor", "existing elastic/inelastic WaveMode executor",
        detail="A column potential does not also own elastic scattering, absorption or stochastic energy loss")
    add("receivers", "measurement", "existing physical planes and absolute flux response",
        "existing particle readout", "existing per-mode intensity readout",
        detail="PSF, pixels and noise act after transport; incoherent modes have no aggregate phase")
    add("nanopulser", "blanker", "separate static kick/stop capability",
        "existing static approximation", "later capability",
        detail="Pulses require time-dependent potentials and timed wavepackets")
    add("energy_filter", "curved_transport", "separate existing relativistic particle route",
        "existing curved-field executor", "later capability",
        detail="Curved reference, retained potential, slit and energy handoff required")
    payload = {"schema": "potential-capability-inventory-v1", "scene_digest": scene.digest,
        "scope": "adapter inventory, not full-column or instrument qualification", "components": rows,
        "numerics": {"particle": ("initial weighted rays", "energy", "maximum axial step", "integration tolerances"),
                     "coherent": ("source modes and energies", "complex grid and physical field of view",
                                  "axial step", "boundary/spectral budgets", "memory limits")}}
    return freeze_json({**payload, "digest": json_digest(payload)})
