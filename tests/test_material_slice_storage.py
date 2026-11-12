"""Bitwise storage and small-array operator checks, without a physical trial."""
import numpy as np
import pytest

from temsim.physics.material_slice_storage import (
    CompactPeriodicPotential,compact_legacy_periodic_potential,periodic_host_storage_plan,
)
from temsim.physics.mixed_stem_operators import ScanPacket,resident_observe,periodic_coefficients
from test_mixed_stem_scan_operators import NumpyDevice


@pytest.mark.parametrize('tiles',[(1,1),(2,4),(3,5),(3,2)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_compact_slices_equal_the_previous_full_tile_bits(tiles,dtype):
    primitive=np.random.default_rng(301).normal(size=(3,5,7)).astype(dtype)
    primitive[0,0,0]=-0.
    original=primitive.copy()
    dense=np.tile(primitive,(1,tiles[1],tiles[0]))
    compact=CompactPeriodicPotential(primitive,tiles)
    assert compact.shape==dense.shape and compact.dtype==dense.dtype
    assert compact.nbytes==primitive.nbytes
    assert compact.logical_nbytes==dense.nbytes
    primitive[:]=123.
    np.testing.assert_array_equal(compact.primitive,original)
    for index in range(len(compact)):
        assert compact[index].tobytes()==dense[index].tobytes()
    assert compact[-1].tobytes()==dense[-1].tobytes()
    assert compact[1,1:3,2:5].tobytes()==dense[1,1:3,2:5].tobytes()
    with pytest.raises(ValueError):
        compact.primitive.setflags(write=True)
    with pytest.raises(TypeError,match='implicitly'):
        np.asarray(compact)
    with pytest.raises(TypeError,match='one integer'):
        compact[:]


@pytest.mark.parametrize('tiles',[(2,4),(3,5),(3,2)])
@pytest.mark.parametrize('centre',[(0.,0.),(.13,-.27)])
def test_compacted_legacy_shift_matches_the_old_whole_volume_array_bits(tiles,centre):
    primitive=np.random.default_rng(702).normal(size=(4,5,7)).astype(np.float32)
    primitive[0,0,0]=-0.
    tx,ty=tiles
    fov=(31.,27.)
    old=np.fft.fftshift(np.tile(primitive,(1,ty,tx)),axes=(-2,-1))
    if any(centre):
        fy,fx=np.meshgrid(np.fft.fftfreq(old.shape[1],fov[1]/old.shape[1]),
                         np.fft.fftfreq(old.shape[2],fov[0]/old.shape[2]),indexing='ij')
        old=np.fft.ifft2(np.fft.fft2(old)*np.exp(-2j*np.pi*(fx*centre[0]+fy*centre[1]))).real
        old=np.tile(old[:,:5,:7],(1,ty,tx))
    compact=compact_legacy_periodic_potential(primitive,tiles,fov_xy=fov,centre_xy=centre)
    assert compact.dtype==old.dtype
    for index in range(len(compact)):
        assert compact[index].tobytes()==old[index].tobytes()


@pytest.mark.parametrize('primitive,tiles',[(np.zeros((0,2,2),np.float32),(2,2)),
    (np.zeros((2,2),np.float32),(2,2)),(np.zeros((1,2,2),int),(2,2)),
    (np.full((1,2,2),np.nan),(2,2)),(np.zeros((1,2,2)),(0,2)),
    (np.zeros((1,2,2)),(True,2)),(np.zeros((1,2,2)),(1.5,2))])
def test_invalid_storage_does_not_acquire_an_approximate_representation(primitive,tiles):
    with pytest.raises(ValueError):
        CompactPeriodicPotential(primitive,tiles)


def test_bloch_reads_the_immutable_primitive_without_expanding_a_slice(monkeypatch):
    material=CompactPeriodicPotential(np.zeros((2,4,4),np.float32),(2,2))
    packet=ScanPacket(np.ones((1,1),complex),np.array([.25]),np.array([0]),np.array([0]),.02)
    monkeypatch.setattr(np,'tile',lambda *_a,**_k:pytest.fail('Bloch expanded a full slice'))
    result,receipt=periodic_coefficients(NumpyDevice(),packet,material,np.array([.5,.5]),
        (8.,8.),(2,2),{'all':(0.,1000.)},interaction=.1,bandwidth_fraction=2/3)
    assert result.item()==pytest.approx(.25,abs=2e-16)
    assert receipt['literal_periodicity']['full_volume_expanded'] is False
    with pytest.raises(ValueError,match='tiles differ'):
        material.literal_periodicity_record((4,1))


def operator_fixture():
    primitive=np.random.default_rng(53).normal(0.,.02,(3,4,4)).astype(np.float32)
    modes=np.asarray([[1.,.2j],[.3j,.4],[.1,.8j],[.2,.1j]],complex)
    modes/=np.linalg.norm(modes,axis=0)
    packet=ScanPacket(modes,np.array([.2,.07]),np.array([-2,-1,0,1]),np.array([0,1,-1,0]),.02)
    slices=np.array([.3,.4,.3])
    positions=np.array([[0.,0.],[.2,-.3]])
    return primitive,packet,slices,positions


def test_resident_and_bloch_outputs_keep_every_bit_and_absolute_current():
    primitive,packet,slices,positions=operator_fixture()
    compact=CompactPeriodicPotential(primitive,(2,2))
    dense=np.tile(primitive,(1,2,2))
    for material in (dense,compact):
        current=resident_observe(NumpyDevice(),packet,material,slices,(8.,8.),positions,
            {'bf':(0.,15.),'adf':(15.,100.)},interaction=.1,bandwidth_fraction=2/3,
            batch_size=1,diffraction=True)
        coefficients,_=periodic_coefficients(NumpyDevice(),packet,material,slices,(8.,8.),(2,2),
            {'bf':(0.,15.),'adf':(15.,100.)},interaction=.1,bandwidth_fraction=2/3)
        if material is dense:
            reference=current,coefficients
        else:
            for actual,expected in zip(current,reference[0]):
                assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
            assert coefficients.tobytes()==reference[1].tobytes()


def test_periodic_host_plan_has_no_depth_times_full_grid_storage():
    from temsim.physics.mixed_stem_scan import _scan_memory_plan
    compact=_scan_memory_plan(2400,3200,8,40000,96,tiles=(16,20))
    dense=_scan_memory_plan(2400,3200,8,40000,96)
    assert dense['host_potential_storage_bytes']>48*1024**3
    assert compact['host_stage_peak_bytes']<4*1024**3
    assert compact['host_potential_storage_bytes']==4*3200*(2400//16)*(2400//20)
    assert compact['host_expanded_slice_bytes']==4*2400**2
    assert compact['host_full_volume_tile_allocated'] is False
    assert dense['host_potential_storage_bytes']==4*3200*2400**2


def test_memory_admission_uses_the_same_compact_plan_and_preserves_limits(monkeypatch):
    from temsim.physics import mixed_stem_scan as scan,host_memory_budget
    required=[]
    monkeypatch.setattr(host_memory_budget,'check_host_allocation',
        lambda size,**kw:required.append((size,kw['requested_maximum_bytes'])))
    cp=type('Device',(),{'cuda':type('Cuda',(),{'runtime':type('Runtime',(),{
        'memGetInfo':staticmethod(lambda:(32*1024**3,32*1024**3))})})})
    checkpoint=type('Checkpoint',(),{'beam':type('Beam',(),{'modes':()})})
    request=scan.MixedStemScanRequest()
    plan=scan._admit_memory(cp,checkpoint,request,2400,3200,40000,96,tiles=(16,20))
    assert required==[(plan['host_stage_peak_bytes'],48*1024**3)]
    assert request.source.maximum_working_bytes==request.maximum_working_bytes==24*1024**3


def test_translated_storage_can_cancel_before_expanding_a_slice(monkeypatch):
    monkeypatch.setattr(np,'tile',lambda *_a,**_k:pytest.fail('Cancelled translation allocated a slice'))
    with pytest.raises(InterruptedError,match='cancelled'):
        compact_legacy_periodic_potential(np.zeros((2,3,4),np.float32),(2,2),
            fov_xy=(8.,6.),centre_xy=(.1,.2),cancelled=lambda:True)


def test_finite_memory_plan_pads_the_whole_physical_specimen_and_keeps_dense_output():
    from types import SimpleNamespace
    from temsim.physics.material_slice_storage import finite_fourier_storage_plan
    scene=SimpleNamespace(size_xy_nm=(6.,8.),centre_xy_nm=(.1,-.2))
    crystal=(np.array([4.,5.,6.1]),np.zeros((1,3)),np.array([14]))
    plan=finite_fourier_storage_plan(scene,crystal,(32.,30.),(60,64),4)
    assert plan['padded_tiles_xy']==[26,26]
    assert plan['padded_grid_yx']==[260,208]
    assert plan['host_output_bytes']==4*4*60*64
    assert plan['host_builder_peak_bytes']==4*4*60*64+8*260*208
    assert plan['device_builder_peak_bytes']==(512+16)*260*208
    np.testing.assert_array_equal(plan['padded_fov_angstrom_xy'],[104.,130.])
    assert np.all(np.array(plan['padded_fov_angstrom_xy'])>=np.array([62.,84.])+40.)


def test_unsupported_crystal_is_rejected_by_the_shared_light_preflight():
    from temsim.physics.mixed_stem_material import validate_analytic_crystal
    with pytest.raises(ValueError,match='rectangular CIF'):
        validate_analytic_crystal(None,None)
    crystal=(np.array([2.,3.,4.]),np.array([[.13,0.,0.]]),np.array([14]),{})
    with pytest.raises(ValueError,match='quarter-lattice'):
        validate_analytic_crystal(None,crystal)
    crystal=(np.array([2.,3.,4.]),np.array([[.5,.75,.33]]),np.array([14]),{})
    from types import SimpleNamespace
    assert not validate_analytic_crystal(SimpleNamespace(thickness_nm=1.),crystal)['radial_core_clamped_fallback']


def test_current_kernel_glue_forwards_actual_energy_backend_limits_and_keeps_compact_storage(monkeypatch):
    from temsim.physics import mixed_stem_scan as scan,finite_lobato_fourier as kernel,mixed_stem_material as material
    from types import SimpleNamespace
    scene=SimpleNamespace(thickness_nm=1.)
    crystal=(np.array([2.,3.,4.]),np.zeros((1,3)),np.array([14]),{})
    primitive=np.arange(12,dtype=np.float32).reshape(2,2,3)
    backend=object()
    calls=[]
    def periodic(*args,**kwargs):
        calls.append(kwargs)
        return primitive,np.array([5.,5.]),{'numerical_potential_error':{'unchanged':True}}
    monkeypatch.setattr(kernel,'build_periodic_lobato_potential',periodic)
    result,_,record=scan._periodic_potential(crystal,scene,(6.,6.),(2,3),6,2,
        energy_kev=123.456,xp=backend,maximum_device_working_bytes=123456)
    assert isinstance(result,CompactPeriodicPotential)
    assert result.primitive.tobytes()==primitive.tobytes()
    assert calls[0]['energy_kev']==123.456 and calls[0]['xp'] is backend
    assert calls[0]['maximum_device_working_bytes']==123456
    assert record['numerical_potential_error']=={'unchanged':True}
    with pytest.raises(ValueError,match='actual current-weighted'):
        scan._periodic_potential(crystal,scene,(6.,6.),(2,3),6,2,xp=backend)
    monkeypatch.setattr(kernel,'build_finite_lobato_potential',lambda *a,**kw:(calls.append(kw) or ('dense','slices','record')))
    assert material.finite_cif_potential(scene,(6.,6.),(6,6),2,maximum_working_bytes=234567,
        crystal=crystal,energy_kev=123.456,xp=backend,maximum_device_working_bytes=123456)==('dense','slices','record')
    assert calls[-1]['energy_kev']==123.456 and calls[-1]['maximum_working_bytes']==234567
