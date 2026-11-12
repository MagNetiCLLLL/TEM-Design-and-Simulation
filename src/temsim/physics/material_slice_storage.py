"""Immutable periodic material storage with one explicitly expanded slice.

Potentials are real projected volt-angstrom values indexed as ``[z, y, x]``.
This module performs no interpolation, normalization or precision conversion.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Integral

import numpy as np


def _tiles(values):
    try:
        values=tuple(values)
    except TypeError as error:
        raise ValueError("Periodic material tiles must be two positive integers (x,y)") from error
    if len(values)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral)
                            or v<=0 for v in values):
        raise ValueError("Periodic material tiles must be two positive integers (x,y)")
    return tuple(map(int,values))


def _primitive_values(primitive):
    values=np.asarray(primitive)
    if (values.ndim!=3 or any(n<=0 for n in values.shape)
            or values.dtype.kind!='f' or values.dtype.itemsize not in (4,8)
            or not np.isfinite(values).all()):
        raise ValueError("Periodic material requires finite float32/float64 primitive[z,y,x]")
    return values


@dataclass(frozen=True,slots=True,init=False)
class CompactPeriodicPotential:
    """A literal repetition of immutable ``primitive[z,y,x]`` values.

    ``shape`` and ``dtype`` describe the full material. ``primitive`` exposes
    its stored values; ``slice_at(i)`` or ``[i]`` creates exactly one full 2-D
    slice using ``np.tile``. Multi-slice indexing and implicit array conversion
    deliberately fail, preventing an accidental full ``NZ*NY*NX`` allocation.
    ``nbytes`` counts storage; ``logical_nbytes`` counts the expanded volume.
    """
    primitive: np.ndarray
    tiles_xy: tuple[int,int]
    primitive_digest: str

    def __init__(self,primitive,tiles_xy):
        values=_primitive_values(primitive)
        tiles=_tiles(tiles_xy)
        contiguous=np.ascontiguousarray(values)
        content=contiguous.tobytes()
        immutable=np.frombuffer(content,dtype=contiguous.dtype).reshape(contiguous.shape)
        object.__setattr__(self,"primitive",immutable)
        object.__setattr__(self,"tiles_xy",tiles)
        object.__setattr__(self,"primitive_digest",sha256(content).hexdigest())

    @property
    def shape(self):
        nz,ny,nx=self.primitive.shape
        tx,ty=self.tiles_xy
        return nz,ny*ty,nx*tx

    @property
    def dtype(self):
        return self.primitive.dtype

    @property
    def ndim(self):
        return 3

    @property
    def nbytes(self):
        return self.primitive.nbytes

    @property
    def logical_nbytes(self):
        return math.prod(self.shape)*self.dtype.itemsize

    @property
    def expanded_slice_nbytes(self):
        return math.prod(self.shape[-2:])*self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def slice_at(self,index):
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral):
            raise TypeError("Expand one integer-indexed material slice; use .primitive for compact access")
        tx,ty=self.tiles_xy
        return np.tile(self.primitive[int(index)],(ty,tx))

    def __getitem__(self,index):
        if isinstance(index,tuple):
            if len(index)!=3:
                raise TypeError("Periodic material indexing requires one slice and two transverse indices")
            return self.slice_at(index[0])[index[1:]]
        return self.slice_at(index)

    def __array__(self,dtype=None,copy=None):
        raise TypeError("Periodic material cannot be implicitly expanded; use .primitive or .slice_at(i)")

    def literal_periodicity_record(self,tiles_xy):
        if _tiles(tiles_xy)!=self.tiles_xy:
            raise ValueError("Bloch crystal tiles differ from the compact material repetition")
        return {"proof":"Literal np.tile repetition of immutable primitive values by construction",
            "primitive_shape_zyx":list(self.primitive.shape),"full_shape_zyx":list(self.shape),
            "tiles_xy":list(self.tiles_xy),"primitive_sha256":self.primitive_digest,
            "stored_bytes":self.nbytes,"expanded_slice_bytes":self.expanded_slice_nbytes,
            "full_volume_expanded":False}


def compact_legacy_periodic_potential(primitive,tiles_xy,*,fov_xy,centre_xy=(0.,0.),cancelled=lambda:False):
    """Retain the previous FFT shift/translation using bounded slice scratch.

    Odd tile counts require an integer primitive roll for the old supercell
    fftshift. A nonzero translation intentionally retains the old float64 FFT
    result, and applies the same full-grid FFT separately to each slice before
    retaining its literal primitive. No all-depth tiled array is constructed.
    """
    tx,ty=_tiles(tiles_xy)
    primitive=_primitive_values(primitive)
    nz,ny,nx=primitive.shape
    offsets=((ny*ty//2)%ny,(nx*tx//2)%nx)
    shifted=primitive if offsets==(0,0) else np.roll(primitive,offsets,axis=(-2,-1))
    centre=np.asarray(centre_xy,dtype=float)
    fov=np.asarray(fov_xy,dtype=float)
    if centre.shape!=(2,) or fov.shape!=(2,) or not np.isfinite(centre).all() or not np.isfinite(fov).all() or np.any(fov<=0):
        raise ValueError("Periodic translation requires finite XY coordinates and a positive FOV")
    if np.any(centre):
        full_y,full_x=ny*ty,nx*tx
        fx,fy=np.meshgrid(np.fft.fftfreq(full_x,fov[0]/full_x),np.fft.fftfreq(full_y,fov[1]/full_y))
        phase=np.exp(-2j*np.pi*(fx*centre[0]+fy*centre[1]))
        translated=np.empty((nz,ny,nx),dtype=np.float64)
        for index in range(nz):
            if cancelled():
                raise InterruptedError("Periodic material translation cancelled before publishing storage")
            layer=np.tile(shifted[index],(ty,tx))
            result=np.fft.ifft2(np.fft.fft2(layer)*phase).real
            translated[index]=result[:ny,:nx]
        shifted=translated
    return CompactPeriodicPotential(shifted,(tx,ty))


def periodic_host_storage_plan(pixels,count,tiles_xy,*,shifted=False,legacy_builder=False,basis_atoms=16):
    """Bound the current primitive builder/copy and one expanded-slice peak."""
    tx,ty=_tiles(tiles_xy)
    if isinstance(pixels,bool) or isinstance(count,bool) or not isinstance(pixels,Integral) or not isinstance(count,Integral) or pixels<=0 or count<=0:
        raise ValueError("Material memory planning requires positive integer grid and depth")
    pixels,count=int(pixels),int(count)
    if pixels%tx or pixels%ty:
        raise ValueError("Periodic material memory grid must be commensurate with its tiles")
    primitive_pixels=pixels*pixels//(tx*ty)
    if not legacy_builder:
        stored=4*count*primitive_pixels
        full_slice=4*pixels*pixels
        builder=max(2*stored,stored+8*primitive_pixels)
        return {"host_potential_storage_bytes":int(stored),
            "host_expanded_slice_bytes":int(full_slice),
            "host_primitive_builder_peak_bytes":int(builder),
            "host_potential_stage_peak_bytes":int(max(builder,stored+full_slice)),
            "periodic_builder_device_peak_bytes":int((512+16*basis_atoms)*primitive_pixels),
            "host_full_volume_tile_allocated":False,
            "periodic_translation_policy":"Analytic primitive Fourier phase; no full-grid host FFT"}
    itemsize=8 if shifted else 4
    stored=itemsize*count*primitive_pixels
    full_slice=itemsize*pixels*pixels
    # The legacy shifted path also holds the original float32 primitive and
    # its float64 translated result while the immutable copy is constructed.
    builder=(20 if shifted else 16)*count*primitive_pixels
    scratch=(96 if shifted else 4)*pixels*pixels
    return {"host_potential_storage_bytes":int(stored),
        "host_expanded_slice_bytes":int(full_slice),
        "host_primitive_builder_peak_bytes":int(builder),
        "host_potential_stage_peak_bytes":int(builder+scratch),
        "host_full_volume_tile_allocated":False}


def finite_fourier_storage_plan(scene,crystal,fov_xy,shape_yx,count):
    """Use the finite builder's whole-specimen plus 20 A per-side padding."""
    repeat=np.asarray(crystal[0],dtype=float)
    ny,nx=map(int,shape_yx)
    extent=np.asarray(fov_xy,dtype=float)
    tiles=np.rint(extent/repeat[:2]).astype(int)
    primitive_xy=np.array([nx//tiles[0],ny//tiles[1]])
    physical=np.asarray(scene.size_xy_nm)*10+2*abs(np.asarray(scene.centre_xy_nm)*10)
    minimum=np.maximum(extent+40,physical+40)
    padded_tiles=np.ceil(minimum/repeat[:2]/2).astype(int)*2
    padded_xy=primitive_xy*padded_tiles
    padded_pixels=int(np.prod(padded_xy))
    output=4*int(count)*ny*nx
    # Every distinct quarter-lattice layer pattern is bounded by the basis;
    # the builder repeats this check with its actual finite pattern count.
    scratch=(512+16*len(crystal[1]))*padded_pixels
    return {"host_output_bytes":output,"host_builder_peak_bytes":output+8*padded_pixels,
        "device_builder_peak_bytes":scratch,"padded_grid_yx":padded_xy[::-1].tolist(),
        "padded_tiles_xy":padded_tiles.tolist(),"padded_fov_angstrom_xy":(padded_tiles*repeat[:2]).tolist()}
