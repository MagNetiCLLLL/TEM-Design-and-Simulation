"""Read-only mesh-axis preflight; no potential solution or particle tracing."""
import importlib.util
from pathlib import Path

path = Path(__file__).with_name("run_gun_tail_diagnostic.py")
spec = importlib.util.spec_from_file_location("gun_tail_diagnostic", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.closed_gun_field import closed_field_request, mesh_axes
from temsim.physics.electrode_mesh import MAX_VERTICES
from temsim.physics.instrument_electric import instrument_electric_end_mm, configure_instrument_electric_domain
from temsim.optics.electron_gun.tracing import geometry_step_criteria
from validate_joint_probe import load_candidate, _restore_exact


def main():
    state, provenance = load_candidate(diagnostic.DEFAULT_CANDIDATE)
    original = capture_instrument_snapshot(state)
    gun = state.electron_gun
    original_cells = int(gun._gun_field_cells_per_bore)
    configure_instrument_electric_domain(gun, instrument_electric_end_mm(state))
    records = []
    reference_physics = None
    for cells in (8, 16, 20, 24, 32):
        request = closed_field_request(gun, cells_per_bore=cells)
        physical_request = {key: value for key, value in request.items() if key != "numerics"}
        physics = diagnostic.digest(physical_request)
        if reference_physics is None:
            reference_physics = physics
        if physics != reference_physics:
            raise ValueError("Mesh-only request changed a physical boundary or geometry")
        work = _restore_exact(original)
        work.electron_gun._gun_field_cells_per_bore = cells
        modified = capture_instrument_snapshot(work)
        restored = _restore_exact(modified)
        if restored.electron_gun._gun_field_cells_per_bore != cells:
            raise ValueError("The explicit field-grid input was not retained in the full snapshot")
        restored.electron_gun._gun_field_cells_per_bore = original_cells
        if capture_instrument_snapshot(restored).digest != original.digest:
            raise ValueError("Changing the grid altered another original input")
        row = dict(cells_per_bore=cells, input_digest=modified.digest,
            request_sha256=diagnostic.digest(request), same_physical_request=True,
            physical_request_sha256=physics, numerics=request["numerics"],
            status="NOT_GENERATED", interpretation="original" if cells == original_cells else
            "coarser_than_original" if cells < original_cells else "finer_than_original")
        try:
            r, z = mesh_axes(request)
            row.update(status="AXES_GENERATED_NO_FIELD_SOLVE", radial_nodes=len(r), axial_nodes=len(z),
                vertices=len(r)*len(z), two_dimensional_cells=(len(r)-1)*(len(z)-1),
                one_float64_grid_bytes=8*len(r)*len(z), radial_axis_sha256=diagnostic.array_hash(r),
                axial_axis_sha256=diagnostic.array_hash(z))
        except ValueError as error:
            row.update(status="REJECTED_BY_EXISTING_BUDGET", reason=str(error))
        records.append(row)
    if capture_instrument_snapshot(state).digest != original.digest:
        raise ValueError("Read-only preflight changed the instrument graph")
    report = dict(schema="temsim-gun-grid-preflight-v1", status="COMPLETE", propagation="NOT_RUN",
        field_solution="NOT_RUN", qualification="NOT_QUALIFIED", candidate=provenance,
        original_cells_per_bore=original_cells, existing_vertex_budget=MAX_VERTICES,
        geometry_step_criteria=geometry_step_criteria(), requests=records,
        fixed_geometry=physical_request,
        limitations=["Grid axes and memory for one dense float64 array are not a solver peak-memory estimate.",
            "Potential assembly, sparse factorization and transport costs have not been measured.",
            "The 8-to-16 comparison coarsens the original 16-cell input before returning to it.",
            "Mesh refinement cannot validate the planar-cathode, electrode-shape, space-charge or image-charge approximations.",
            "Existing budgets and the full instrument field domain remain unchanged."])
    output = Path(__file__).with_name("gun-grid-preflight.json")
    if output.exists():
        raise FileExistsError(output)
    diagnostic.write_json(output, report)
    print(output)
    for row in records:
        print(row["cells_per_bore"], row["status"], row.get("vertices"), row.get("reason", ""))


if __name__ == "__main__":
    main()
