"""Optical tilt and translated mechanical datums stay explicitly distinct."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.lens_pose import component_datums_mm, lens_pose_registration


def _row(key, center=50., **values):
    return dict(key=key, local_start_z_mm=center-5., local_center_z_mm=center,
                local_end_z_mm=center+5., length_mm=10., **values)


def _assembly(rows, origin=450.):
    parts = tuple(SimpleNamespace(key=row['key'], data=row, module_key='column',
                                  center_z_mm=origin+row['local_center_z_mm'])
                  for row in rows)
    return SimpleNamespace(parts=parts, modules=(SimpleNamespace(key='column', parts=parts),))


def test_declared_reference_rotates_about_mechanical_center_then_module_offset_once():
    row = _row('condenser_lens_1', mechanical_profile='magnetic_lens_assembly',
               optical_reference_local_z_mm=52., offset_x_mm=1., offset_y_mm=2.,
               offset_z_mm=3., rotation_y_mrad=100., nominal_excitation_percent=42.)
    assembly = _assembly([row])
    state = SimpleNamespace(_resolved_assembly=assembly)
    saved = deepcopy(row)
    datums = component_datums_mm(assembly.parts[0], state)
    np.testing.assert_allclose(datums.mechanical_center_mm, (1., 2., 503.), atol=1e-13)
    expected = (1.+2.*np.sin(.1), 2., 503.+2.*np.cos(.1))
    np.testing.assert_allclose(datums.optical_reference.point_mm, expected, atol=1e-13)
    np.testing.assert_allclose(datums.optical_reference.normal, (np.sin(.1), 0., np.cos(.1)))
    np.testing.assert_allclose(np.asarray(datums.basis).T @ datums.basis, np.eye(3), atol=1e-15)
    # Field registration operates in SI on the same unposed global datum.
    field = lens_pose_registration(state, row['key'])
    np.testing.assert_allclose(field.rotation_array @ np.array((0., 0., .502))
                               + field.origin_array_m, np.asarray(expected)*1e-3, atol=1e-15)
    assert row == saved
    assert datums.is_virtual is False
    with pytest.raises(FrozenInstanceError):
        datums.is_virtual = True


def test_physical_dph_host_and_parent_pose_are_each_used_once():
    hp = _row('probe_hp2_hexapole', mechanical_profile='hexapole',
              optical_reference_local_z_mm=50., offset_x_mm=2., rotation_y_mrad=100.)
    coil = _row('probe_dph2_deflector_coils', center=51., mechanical_only=True,
                mechanical_part_role='deflector_coil_insert', parent_key=hp['key'],
                optical_reference_local_z_mm=51., offset_x_mm=.5, offset_y_mm=.4,
                offset_z_mm=.2)
    dph = _row('probe_dph2_deflector', center=51., optical_reference_local_z_mm=51.,
               physical_host_key=coil['key'], physical_host_status='modelled')
    assembly = _assembly([hp, coil, dph])
    coil_datums, channel_datums = (component_datums_mm(part, assembly) for part in assembly.parts[1:])
    np.testing.assert_allclose(channel_datums.optical_reference.point_mm,
                               (2.+.5*np.cos(.1)+1.2*np.sin(.1), .4,
                                500.+1.2*np.cos(.1)-.5*np.sin(.1)), atol=1e-13)
    np.testing.assert_array_equal(channel_datums.mechanical_center_mm, coil_datums.mechanical_center_mm)
    # Parent optical-axis errors do not physically swing the winding around
    # the host's centre; both mechanical rows inherit translation only.
    np.testing.assert_allclose(coil_datums.mechanical_center_mm, (2.5, .4, 501.2), atol=1e-13)
    assert not np.allclose(channel_datums.optical_reference.point_mm,
                           channel_datums.mechanical_center_mm)
    assert channel_datums.representation_role == 'control_channel'
    assert channel_datums.optical_reference.is_virtual is False
    assert coil_datums.optical_reference is None  # winding is not another field channel


def test_objective_virtual_reference_is_not_its_material_center_or_two_field_centers():
    objective = _row('objective_lens', center=100., mechanical_profile='magnetic_lens_assembly',
                     mechanical_part_role='optical_parent', virtual_reference_local_z_mm=101.,
                     upper_field_reference_local_z_mm=98., lower_field_reference_local_z_mm=103.,
                     offset_z_mm=.5)
    assembly = _assembly([objective], origin=450.)
    datums = component_datums_mm(assembly.parts[0], assembly)
    assert datums.mechanical_center_mm == (0., 0., 550.5)
    assert datums.optical_reference.point_mm == (0., 0., 551.5)
    assert datums.optical_reference.kind == 'virtual_reference'
    assert datums.optical_reference.is_virtual is True
    assert datums.is_virtual is False  # the physical assembly exists
    assert datums.interaction_planes == ()  # field centres are not thin kicks


def test_paired_coil_planes_retain_three_dimensional_tilt_and_order():
    row = _row('ac_deflector', optical_reference_local_z_mm=50.,
               interaction_centers_local_z_mm=[47., 53.], rotation_x_mrad=200., offset_y_mm=.2)
    assembly = _assembly([row])
    datums = component_datums_mm(assembly.parts[0], assembly)
    for dz, plane in zip((-3., 3.), datums.interaction_planes, strict=True):
        np.testing.assert_allclose(plane.point_mm, (0., .2-dz*np.sin(.2), 500.+dz*np.cos(.2)), atol=1e-13)
        np.testing.assert_allclose(plane.normal, (0., -np.sin(.2), np.cos(.2)))
    np.testing.assert_allclose(datums.optical_reference.point_mm, (0., .2, 500.), atol=1e-13)


def test_aperture_basis_keeps_operating_hole_offset_separate_from_component_pose():
    row = _row('condenser_aperture_2', optical_reference_local_z_mm=50.,
               rotation_y_mrad=100., offset_x_mm=1.)
    assembly = _assembly([row])
    datums = component_datums_mm(assembly.parts[0], assembly)
    opening = np.array(datums.optical_reference.point_mm) + np.asarray(datums.basis) @ (.2, -.3, 0.)
    np.testing.assert_allclose(opening, (1.+.2*np.cos(.1), -.3, 500.-.2*np.sin(.1)), atol=1e-13)
    assert datums.optical_reference.point_mm == (1., 0., 500.)


def test_detector_reference_is_entrance_face_and_mechanical_or_custom_rows_do_not_gain_optics():
    rows = [_row('camera', center=50.25, branch='detection', mechanical_profile='camera_sensor_plane',
                 optical_reference_local_z_mm=50.),
            _row('coil', mechanical_only=True),
            _row('custom_added', mechanical_part_role='custom_mechanical_copy',
                 optical_reference_local_z_mm=50.)]
    assembly = _assembly(rows)
    camera, coil, custom = [component_datums_mm(p, assembly) for p in assembly.parts]
    assert camera.mechanical_center_mm == (0., 0., 500.25)
    assert camera.optical_reference.point_mm == (0., 0., 500.)
    assert camera.optical_reference.normal == (0., 0., 1.)
    assert coil.optical_reference is custom.optical_reference is None


def test_datum_context_rejects_wrong_capture_and_curved_branch():
    assembly = _assembly([_row('lens', mechanical_profile='magnetic_lens_assembly')])
    wrong = _assembly([_row('lens', mechanical_profile='magnetic_lens_assembly')], origin=700.)
    with pytest.raises(ValueError, match='captured assembly'):
        component_datums_mm(assembly.parts[0], wrong)
    branch = _assembly([_row('energy_filter_prism', branch_path_only=True)])
    with pytest.raises(ValueError, match='mounted branch frame'):
        component_datums_mm(branch.parts[0], branch)


def test_mesh_shape_edits_do_not_redefine_a_declared_reference():
    row = _row('condenser_lens_1', mechanical_profile='magnetic_lens_assembly',
               optical_reference_local_z_mm=50.)
    assembly = _assembly([row])
    before = component_datums_mm(assembly.parts[0], assembly)
    row['model_3d'] = dict(base=dict(kind='box', width_mm=20., height_mm=30., length_mm=40.),
                           transform=dict(scale_xy=[2., 3.]), features=[dict(enabled=True, type='hole')])
    assert component_datums_mm(assembly.parts[0], assembly) == before
