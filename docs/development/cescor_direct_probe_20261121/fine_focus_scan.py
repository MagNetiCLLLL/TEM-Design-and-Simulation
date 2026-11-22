"""Measure electronic fine focus and raster calibration on the original 30 mrad branch."""

import json
from hashlib import sha256

import cupy as cp
import numpy as np
from scipy.linalg import polar

from column_current import source_digest
from downstream_pivot import DrivenTrace, inputs, write
from objective_pivot_sensitivity import summary
from refocus_a2 import objective_receipt
from relay_transfer import Native, START, SURFACE
from scan_waist_audit import EVIDENCE, FIELD


def focus_curve(model, drives, offsets_nm):
    # After the specimen upper face the short extension contains vacuum and the same fields.
    # Equal incoming weights are preserved after aperture contact throughout this bracket.
    _, states = model.replay(drives, SURFACE + offsets_nm*1e-9)
    first_live = states[0][1] == -1
    if not np.all(model.t.weights == model.t.weights[0]):
        raise ValueError("This order-statistic implementation requires equal source weights")
    diameters = []
    for phase, loss in states:
        live = loss == -1
        if not np.array_equal(live, first_live):
            raise ValueError("Source IDs changed within the near-specimen focus scan")
        xy = phase[live, :2]
        radii = np.linalg.norm(xy-xy.mean(axis=0), axis=1)
        k = int(np.ceil(.95*len(radii)))-1
        diameters.append(2e9*np.partition(radii, k)[k])
    index = int(np.argmin(diameters))
    if not 0 < index < len(diameters)-1:
        raise ValueError("D95 minimum outside the near-specimen bracket")
    return offsets_nm[index-1:index+2].tolist(), diameters


def measure(reference, ni, step, mixing):
    receipt = objective_receipt(ni)
    offsets = np.arange(-800., 801., 10.)
    drives = np.asarray(reference["currents_A"])
    model = Native(FIELD, step, 32768, ni, SURFACE + offsets*1e-9)
    if source_digest(model.t) != reference["source_digest"]:
        raise ValueError("Gun-exit checkpoint changed")
    bracket, curve = focus_curve(model, drives, offsets)
    metrics, _, contacts = model.measure(drives)
    k = np.searchsorted(offsets, 0.)
    np.testing.assert_allclose(metrics['specimen']['D95_nm'], curve[k], rtol=1e-10)
    live = contacts == -1
    trace = DrivenTrace(model)
    field = model.field(drives)
    if mixing is None:
        # At the reference excitation only, null scan-dependent mechanical incidence angle.
        basis = trace.response(field, np.eye(4), [SURFACE], 1.)[0][0, 2:4]
        if np.linalg.cond(basis[:, 2:]) > 1e4:
            raise ValueError("The lower station cannot null both components of specimen tilt")
        mixing = .5 * np.vstack((np.eye(2), -np.linalg.solve(basis[:, 2:], basis[:, :2])))
    perturbations = []
    for amplitude in (1., .5):
        out, even, masks = trace.response(field, mixing, [SURFACE], amplitude)
        np.testing.assert_array_equal(masks[0], live)
        perturbations.append(dict(amplitude_urad=amplitude, centroid_even_norm_nm=1e9*float(np.linalg.norm(even[0,:2])),
                                  **summary(out[0])))
    row = dict(objective_At=ni, objective_current_A=ni/1160., field_digest=receipt['field_digest'],
               max_step_mm=step, D95_minimum_bracket_nm_from_specimen=bracket,
               zero_scan_surface=metrics['specimen'], pupil_rim_mrad=metrics['rim']['angle_mrad_min_median_max'],
               emitted=len(live), surviving=int(live.sum()), effective=float(live.sum()),
               surviving_ID_sha256=sha256(model.t.ids[live].tobytes()).hexdigest(),
               source_sha256=source_digest(model.t), probe_D95_nm_curve=curve,
               scan_perturbations=perturbations)
    return row, mixing


def main():
    reference, _ = inputs()
    ni0 = reference['objective_At']
    runs, mixing = [], None
    for step in (.0375, .01875):
        for ni in (ni0, round(ni0+.1, 1), round(ni0-.1, 1)):
            row, mixing = measure(reference, ni, step, mixing)
            jac = np.array(row['scan_perturbations'][0]['position_nm_per_urad'])
            if ni == ni0:
                base_jac = jac
            rotation, stretch = polar(jac @ np.linalg.inv(base_jac))
            row['principal_change_in_specimen_pixel_scale_percent'] = ((np.linalg.eigvalsh(stretch)-1)*100).tolist()
            row['change_in_scan_rotation_deg'] = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
            print(step, ni, row['D95_minimum_bracket_nm_from_specimen'], row['principal_change_in_specimen_pixel_scale_percent'],
                  row['change_in_scan_rotation_deg'], flush=True)
            runs.append(row)
            cp.get_default_memory_pool().free_all_blocks()
    write(EVIDENCE/'fine-focus-scan.json', dict(
        reference='illuminated-normal.json / final; original 30 mrad specimen-focus branch.',
        corrector_sha256=reference['corrector_sha256'], source_plane_mm=START*1e3, specimen_upper_mm=SURFACE*1e3,
        pupil_diameter_um=100., objective_turns=1160, station_command_matrix=mixing.tolist(),
        scan_alignment='At nominal NI only, adjust the lower X/Y currents to null the scan-dependent mechanical specimen-angle response, keeping the upper station fixed. Hold this same mixing for both objective perturbations and both integration steps.',
        scope='Local quasistatic mechanical-angle proxy for scanning. No downstream diffraction projection, descan, dynamic raster, wave imaging or hardware calibration.',
        focus_offsets_nm=np.arange(-800.,801.,10.).tolist(), runs=runs))


if __name__ == '__main__':
    main()
