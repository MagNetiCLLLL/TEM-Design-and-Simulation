"""Explicit FP64 CUDA execution of retained ideal-coil potentials.

The radial/fraction Gauss pack, current, closed circuit, annular near-axis
terms and source order are retained. Admitted radial rectangles use analytic
straight paths; saddle/arc circuits and low path orders retain their line
quadrature. Unsupported poses, domains and unsafe numerical ranges use the
original provider and appear in receipts. CUDA failures never trigger retry.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from threading import RLock

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from .compute_backend import (
    BACKEND_CUDA, BACKEND_PREFER_GPU, BACKEND_REQUIRE_GPU, GPUExecutionError,
    cupy_module, gpu_failure_category, normalise_backend,
)
from .potential_snapshot import ElectromagneticPotentialSnapshot
from .frozen_copper_coil_field import FrozenCopperCoilField
from .grounded_particle_step import _INSTRUMENT_FIELD_METHOD


_SNAPSHOT_SAMPLE_METHOD = ElectromagneticPotentialSnapshot.sample
_FROZEN_COPPER_FIELD_METHOD = FrozenCopperCoilField.field_at_global_positions_t


SCHEMA = "cuda-ideal-coil-potential-fp64-v2-analytic-rectangles"
POTENTIAL_MINIMUM_BATCH = 1
MAGNETIC_MINIMUM_BATCH = 128
MAX_CACHE_ENTRIES = 8
MAX_CACHE_DEVICE_BYTES = 256*1024**2
_CACHE = OrderedDict()
_LOCK = RLock()
_PROBED = set()
_GPU_POLICIES = (BACKEND_CUDA, BACKEND_PREFER_GPU, BACKEND_REQUIRE_GPU)


_TRANSVERSE_KERNEL = r'''
extern "C" __global__ void transverse_ab(
    const double* p, const long long count, const double* nodes,
    const double* w, const long long ncount, const int axis_zero, double* result) {
    const long long i=((long long)blockIdx.x)*blockDim.y+threadIdx.y;
    const int lane=threadIdx.x;
    if(i>=count) return;
    const double x=p[3*i],y=p[3*i+1],z=p[3*i+2];
    double ax=0.,ay=0.,az=0.,cx=0.,cy=0.,cz=0.,bx=0.,by=0.,bz=0.;
    if(!(axis_zero && x==0. && y==0.)) {
        for(long long n=lane;n<ncount;n+=32) {
            const double dx=x-nodes[3*n],dy=y-nodes[3*n+1],dz=z-nodes[3*n+2];
            const double norm2=(dx*dx+dz*dz)+dy*dy;
            const double inv=__ddiv_rn(1.,__dsqrt_rn(norm2)),inv3=__ddiv_rn(inv,norm2);
            const double wx=w[3*n],wy=w[3*n+1],wz=w[3*n+2];
            const double vx=wx*inv,vy=wy*inv,vz=wz*inv;
            const double tx=ax+vx,ty=ay+vy,tz=az+vz;
            cx+=(fabs(ax)>=fabs(vx)) ? (ax-tx)+vx : (vx-tx)+ax;
            cy+=(fabs(ay)>=fabs(vy)) ? (ay-ty)+vy : (vy-ty)+ay;
            cz+=(fabs(az)>=fabs(vz)) ? (az-tz)+vz : (vz-tz)+az;
            ax=tx;ay=ty;az=tz;
            bx+=(wy*dz-wz*dy)*inv3;by+=(wz*dx-wx*dz)*inv3;bz+=(wx*dy-wy*dx)*inv3;
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        const double vx=__shfl_down_sync(0xffffffff,ax,offset),vy=__shfl_down_sync(0xffffffff,ay,offset);
        const double vz=__shfl_down_sync(0xffffffff,az,offset);
        const double dx=__shfl_down_sync(0xffffffff,cx,offset),dy=__shfl_down_sync(0xffffffff,cy,offset);
        const double dz=__shfl_down_sync(0xffffffff,cz,offset);
        const double tx=ax+vx,ty=ay+vy,tz=az+vz;
        cx+=dx+((fabs(ax)>=fabs(vx)) ? (ax-tx)+vx : (vx-tx)+ax);
        cy+=dy+((fabs(ay)>=fabs(vy)) ? (ay-ty)+vy : (vy-ty)+ay);
        cz+=dz+((fabs(az)>=fabs(vz)) ? (az-tz)+vz : (vz-tz)+az);
        ax=tx;ay=ty;az=tz;
        bx+=__shfl_down_sync(0xffffffff,bx,offset);
        by+=__shfl_down_sync(0xffffffff,by,offset);bz+=__shfl_down_sync(0xffffffff,bz,offset);
    }
    if(lane==0) {result[6*i]=ax+cx;result[6*i+1]=ay+cy;result[6*i+2]=az+cz;
                result[6*i+3]=bx;result[6*i+4]=by;result[6*i+5]=bz;}
}
'''
_TRANSVERSE_B_KERNEL = r'''
extern "C" __global__ void transverse_b(
    const double* p,const long long count,const double* nodes,
    const double* w,const long long ncount,const int axis_zero,double* result) {
    const long long i=((long long)blockIdx.x)*blockDim.y+threadIdx.y;
    const int lane=threadIdx.x;
    if(i>=count)return;
    const double x=p[3*i],y=p[3*i+1],z=p[3*i+2];
    double bx=0.,by=0.,bz=0.;
    if(!(axis_zero&&x==0.&&y==0.)) {
        for(long long n=lane;n<ncount;n+=32) {
            const double dx=x-nodes[3*n],dy=y-nodes[3*n+1],dz=z-nodes[3*n+2];
            const double norm2=(dx*dx+dz*dz)+dy*dy;
            const double inv=__ddiv_rn(1.,__dsqrt_rn(norm2)),inv3=__ddiv_rn(inv,norm2);
            const double wx=w[3*n],wy=w[3*n+1],wz=w[3*n+2];
            bx+=(wy*dz-wz*dy)*inv3;by+=(wz*dx-wx*dz)*inv3;bz+=(wx*dy-wy*dx)*inv3;
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        bx+=__shfl_down_sync(0xffffffff,bx,offset);by+=__shfl_down_sync(0xffffffff,by,offset);
        bz+=__shfl_down_sync(0xffffffff,bz,offset);
    }
    if(lane==0){result[3*i]=bx;result[3*i+1]=by;result[3*i+2]=bz;}
}
'''
_SEGMENT_KERNEL = r'''
extern "C" __global__ void radial_segments(
 const double* p,const long long count,const double* a,const double* b,
 const double* factor,const long long ncount,const int axis_zero,
 const int with_a,double* result) {
 const long long i=((long long)blockIdx.x)*blockDim.y+threadIdx.y;
 const int lane=threadIdx.x;
 if(i>=count)return;
 const double x=p[3*i],y=p[3*i+1],z=p[3*i+2];
 double ax=0.,ay=0.,az=0.,cx=0.,cy=0.,cz=0.,bx=0.,by=0.,bz=0.;
 if(!(axis_zero&&x==0.&&y==0.)) {
  for(long long n=lane;n<ncount;n+=32) {
   const double dx=b[3*n]-a[3*n],dy=b[3*n+1]-a[3*n+1],dz=b[3*n+2]-a[3*n+2];
   const double x1=x-a[3*n],y1=y-a[3*n+1],z1=z-a[3*n+2];
   const double x2=x-b[3*n],y2=y-b[3*n+1],z2=z-b[3*n+2];
   const double r1=__dsqrt_rn((x1*x1+z1*z1)+y1*y1),r2=__dsqrt_rn((x2*x2+z2*z2)+y2*y2);
   const double dot=(x1*x2+z1*z2)+y1*y2;
   const double den=r1*r2*(r1*r2+dot);
   const double scale=__ddiv_rn(factor[n]*(r1+r2),den);
   bx+=(dy*z1-dz*y1)*scale;by+=(dz*x1-dx*z1)*scale;bz+=(dx*y1-dy*x1)*scale;
   if(with_a) {
    const double length=__dsqrt_rn((dx*dx+dz*dz)+dy*dy);
    const double f=__ddiv_rn(factor[n]*log1p(__ddiv_rn(2.*length,r1+r2-length)),length);
    const double vx=dx*f,vy=dy*f,vz=dz*f;
    const double tx=ax+vx,ty=ay+vy,tz=az+vz;
    cx+=(fabs(ax)>=fabs(vx))?(ax-tx)+vx:(vx-tx)+ax;
    cy+=(fabs(ay)>=fabs(vy))?(ay-ty)+vy:(vy-ty)+ay;
    cz+=(fabs(az)>=fabs(vz))?(az-tz)+vz:(vz-tz)+az;
    ax=tx;ay=ty;az=tz;
   }
  }
 }
 for(int offset=16;offset>0;offset/=2) {
  if(with_a) {
   const double vx=__shfl_down_sync(0xffffffff,ax,offset),vy=__shfl_down_sync(0xffffffff,ay,offset),vz=__shfl_down_sync(0xffffffff,az,offset);
   const double dx=__shfl_down_sync(0xffffffff,cx,offset),dy=__shfl_down_sync(0xffffffff,cy,offset),dz=__shfl_down_sync(0xffffffff,cz,offset);
   const double tx=ax+vx,ty=ay+vy,tz=az+vz;
   cx+=dx+((fabs(ax)>=fabs(vx))?(ax-tx)+vx:(vx-tx)+ax);
   cy+=dy+((fabs(ay)>=fabs(vy))?(ay-ty)+vy:(vy-ty)+ay);
   cz+=dz+((fabs(az)>=fabs(vz))?(az-tz)+vz:(vz-tz)+az);
   ax=tx;ay=ty;az=tz;
  }
  bx+=__shfl_down_sync(0xffffffff,bx,offset);by+=__shfl_down_sync(0xffffffff,by,offset);
  bz+=__shfl_down_sync(0xffffffff,bz,offset);
 }
 if(lane==0) {
  if(with_a){result[6*i]=ax+cx;result[6*i+1]=ay+cy;result[6*i+2]=az+cz;
             result[6*i+3]=bx;result[6*i+4]=by;result[6*i+5]=bz;}
  else {result[3*i]=bx;result[3*i+1]=by;result[3*i+2]=bz;}
 }
}
'''
_ROUND_KERNEL = r'''
extern "C" __global__ void round_ab(
    const double* p,const long long count,const long long* offsets,
    const double* radii,const double* centers,const double* amplitudes,
    const double* poly,const long long* lengths,const int polynomial_columns,
    const double* coefficients,const int* admitted,const int with_a,double* result) {
    const long long i=((long long)blockIdx.x)*blockDim.y+threadIdx.y;
    const int lane=threadIdx.x,coil=blockIdx.y;
    if(i>=count || !admitted[coil]) return;
    const double x=p[3*i],y=p[3*i+1],z=p[3*i+2],s=x*x+y*y;
    double d[12];
    #pragma unroll
    for(int order=0;order<12;++order) d[order]=0.;
    for(long long loop=offsets[coil]+lane;loop<offsets[coil+1];loop+=32) {
        const double radius=radii[loop],u=__ddiv_rn(z-centers[loop],radius),base=1.+u*u;
        double factor=__ddiv_rn(amplitudes[loop],pow(base,1.5));
        const double increment=__ddiv_rn(__ddiv_rn(1.,radius),base);
        #pragma unroll
        for(int order=0;order<12;++order) {
            double value=poly[order*polynomial_columns+lengths[order]-1];
            for(int c=lengths[order]-2;c>=0;--c) value=poly[order*polynomial_columns+c]+value*u;
            d[order]+=factor*value;
            factor*=increment;
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        #pragma unroll
        for(int order=0;order<12;++order) d[order]+=__shfl_down_sync(0xffffffff,d[order],offset);
    }
    if(lane==0) {
        double f=0.,fs=0.,fz=0.;
        for(int order=0;order<6;++order) {
            const double power=pow(s,(double)order),coefficient=coefficients[order];
            f+=coefficient*power*d[2*order];
            if(order) fs+=order*coefficient*pow(s,(double)(order-1))*d[2*order];
            fz+=coefficient*power*d[2*order+1];
        }
        const long long row=((long long)coil)*count+i;
        if(with_a){const long long index=6*row;
            result[index]=-y*f;result[index+1]=x*f;result[index+2]=0.;
            result[index+3]=-x*fz;result[index+4]=-y*fz;result[index+5]=2.*f+2.*s*fs;}
        else{const long long index=3*row;
            result[index]=-x*fz;result[index+1]=-y*fz;result[index+2]=2.*f+2.*s*fs;}
    }
}
'''
KERNEL_SHA256 = sha256((_TRANSVERSE_KERNEL+_TRANSVERSE_B_KERNEL+_SEGMENT_KERNEL+_ROUND_KERNEL).encode("ascii")).hexdigest()
ADAPTER_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def explicit_cuda_coil_policy(value):
    """Auto deliberately keeps the existing canonical CPU field executor."""
    return normalise_backend(value) in _GPU_POLICIES


def _instrument_callback_supported(provider):
    return ("field_at_global_positions_t" not in vars(provider)
            and getattr(provider.field_at_global_positions_t, "__func__", None) is _INSTRUMENT_FIELD_METHOD)


def _snapshot_callback_supported(snapshot):
    return (getattr(snapshot.sample, "__func__", None) is _SNAPSHOT_SAMPLE_METHOD
            and "sample" not in getattr(snapshot, "__dict__", {}))


def _copper_callback_supported(provider):
    return (getattr(provider.field_at_global_positions_t, "__func__", None) is _FROZEN_COPPER_FIELD_METHOD
            and "field_at_global_positions_t" not in getattr(provider, "__dict__", {}))


def _failure(error, stage):
    category = gpu_failure_category(error, stage="execution") or "unknown"
    if category == "context_corrupted":
        from .ray_device_cache import DEVICE_CACHE
        DEVICE_CACHE.quarantine(str(error))
    # Existing Prefer-GPU retry policies recognise OOM/unavailable categories.
    # A field execution failure must escape those outer CPU retry policies.
    failure = GPUExecutionError("coil_field_execution_failure", f"{stage} ({category}): {error}")
    failure.original_category = category
    failure.failure_stage = stage
    return failure


def _device():
    try:
        cp = cupy_module()
        index = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(index)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        record = dict(device_id=index, device_name=name, compute_capability=[int(props["major"]), int(props["minor"])],
            cupy_version=cp.__version__, cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            cuda_driver_version=int(cp.cuda.runtime.driverGetVersion()))
        key = json_digest(record)
        with _LOCK:
            if key not in _PROBED:
                # Actual device allocation and kernel execution, not enumeration.
                if float(cp.arange(4, dtype=cp.float64).sum().get()) != 6.:
                    raise ValueError("CUDA FP64 device probe returned an invalid result")
                _PROBED.add(key)
        return cp, record
    except Exception as error:
        raise _failure(error, "device discovery/probe") from error


def cuda_coil_execution_identity(execution_backend, *, mode="potential"):
    """Numerical execution identity, available before any particle trace cache."""
    if mode not in ("potential", "magnetic"):
        raise ValueError("CUDA coil execution mode must be potential or magnetic")
    policy = normalise_backend(execution_backend)
    minimum = POTENTIAL_MINIMUM_BATCH if mode == "potential" else MAGNETIC_MINIMUM_BATCH
    record = dict(schema=SCHEMA, requested_backend=policy, mode=mode,
        minimum_gpu_batch=minimum, kernel_sha256=KERNEL_SHA256, precision="float64",
        adapter_sha256=ADAPTER_SHA256, numpy_version=np.__version__,
        compiler_options=["--std=c++11", "--fmad=false"],
        quadrature="Unchanged radial/fraction Gauss pack and NI; full axial tails and six round near-axis terms",
        radial_rectangle_path="Analytic finite straight segments when reference path_order>=24 and conservative admission passes; original line Gauss otherwise",
        saddle_arc_path="Original line Gauss nodes and path order",
        outputs=dict(sample="Real A/B/E/phi", sample_fields="Real B/E/phi; A not requested or constructed by CUDA",
                     magnetic="Real B only; no A accumulation"),
        reduction="32-lane tree; compensated transverse A, ordinary B; original source/owner addition order",
        deterministic_reference="Unsupported law/pose, elliptic branch, winding/roundoff boundary or unsafe numerical range",
        cuda_failure="Raise coil_field_execution_failure; never CPU retry",
        admission=dict(absolute_position_limit_m=1.e4, transverse_bore_fraction=.5,
            nonzero_transverse_weight_range=[1.e-100, 1.e100], round_prefactor_range=[1.e-40, 1.e40],
            round_nonzero_normalized_z_min=1.e-80, round_normalized_z_max=1.e4,
            analytic_rectangle_minimum_reference_path_order=24, analytic_rectangle_maximum_segment_to_half_bore_ratio=16.))
    if policy in _GPU_POLICIES:
        from temsim.calculation_manifest import solver_source_identity
        record["solver_source_identity"] = solver_source_identity()
        _, device = _device(); record.update(resolved_backend="CuPy CUDA FP64 ideal-coil potentials", **device)
    else:
        record["resolved_backend"] = "Canonical CPU reference"
    return freeze_json(record)


@dataclass(frozen=True)
class _Round:
    field: object
    first: int
    stop: int
    radii: np.ndarray
    centers: np.ndarray
    factors: np.ndarray
    geometry: np.ndarray
    magnetic_wrapper: bool

    def admission(self, points):
        from .compiled_copper_instrument import copper_near_axis_domain
        if not self.field.implementation_is_supported():
            return "overridden_round_law"
        if self.magnetic_wrapper and not _copper_callback_supported(self.field):
            return "overridden_round_field_wrapper"
        # Preserve material rejection through the original provider first.
        for coil in self.field.potential.coils:
            coil._local(points)
        if not copper_near_axis_domain(points, self.geometry):
            return "round_pose_domain_or_elliptic_branch"
        if not len(points):
            return None
        lower, upper = points[:, 2].min(), points[:, 2].max()
        ratios = np.maximum(abs(lower-self.centers), abs(upper-self.centers))/self.radii
        if np.any(ratios > 1.e4):
            return "round_normalized_z_range"
        # Odd derivative polynomials can contain a tiny nonzero u. Keep the
        # native common-factor/subnormal policy rather than rounding it away.
        tiny_centers = abs(self.centers) < 1.e-40
        if tiny_centers.any():
            for z in np.unique(points[:, 2]):
                distance = abs((z-self.centers[tiny_centers])/self.radii[tiny_centers])
                if np.any((distance != 0.) & (distance < 1.e-80)):
                    return "round_subnormal_polynomial_range"
        return None


@dataclass(frozen=True)
class _Transverse:
    potential: object
    provider: object
    nodes: object
    weights: object
    geometry: np.ndarray
    axis_zero: bool
    minimum_node_radius: float
    ends: object = None
    path_representation: str = "Gauss line quadrature"
    original_path_node_count: int = 0

    def admission(self, points):
        from .transverse_coil_field import _inside_pack, transverse_potential_implementation_is_supported
        if (not transverse_potential_implementation_is_supported(self.potential)
                or self.provider is not None and not self.provider.implementation_is_supported()):
            return "overridden_transverse_law"
        if np.any(_inside_pack(points, self.potential.coils)):
            raise ValueError("Transverse ideal-current source excludes its occupied winding pack from vacuum queries")
        if np.max(abs(points), initial=0.) > 1.e4:
            return "transverse_numerical_range"
        if len(self.nodes) and np.any(np.hypot(points[:, 0], points[:, 1]) >=
                .5*self.minimum_node_radius*(1.-128*np.finfo(float).eps)):
            return "transverse_bore_numerical_admission"
        return None


class CudaCoilSampler:
    """Immutable captured sources with an inspectable lifetime execution ledger."""
    backend = "CuPy CUDA FP64 ideal-coil potentials with declared native references"

    def __init__(self, original, identity, mode):
        from .potential_snapshot import ElectromagneticPotentialSnapshot
        from .instrument_magnetic import InstrumentMagneticField
        from .copper_coil_field_provider import CopperCoilAssemblyPotential
        from .frozen_copper_coil_field import FrozenCopperCoilField
        from .transverse_coil_field import TransverseCoilPotential, FrozenTransverseCoilField
        from .compiled_annular_coil import pack_annular_axis
        from .compiled_transverse_coil import pack_transverse_potential
        from .analytic_rectangle_coil import pack_analytic_rectangle_potential
        from .lens_field_provider import _identity_registration
        self.original, self.execution_identity, self.mode = original, identity, mode
        self._lock, self._counts, self._reasons = RLock(), Counter(), Counter()
        self._failure_evidence = None
        cp, _ = _device(); self._cp = cp
        self._rows, self._round, self._transverse = [], {}, {}
        self._duplicate_source_keys = False
        offsets, radii, centers, factors, geometry = [0], [], [], [], []
        polynomials = lengths = None
        coil_count = 0
        if mode == "potential":
            if type(original) is not ElectromagneticPotentialSnapshot:
                raise ValueError("CUDA ideal-coil potentials require a static immutable snapshot")
            rows = ([(row.source_id, row.kind, row.potential, None) for row in original.contributions]
                    if _snapshot_callback_supported(original) else [])
        else:
            if type(original) is not InstrumentMagneticField:
                raise ValueError("CUDA magnetic execution requires the exact frozen instrument field")
            rows = ([(row.key, "magnetic", row.provider, row) for row in original._sources]
                    if _instrument_callback_supported(original) else [])
            keys = [row[0] for row in rows]
            self._duplicate_source_keys = len(set(keys)) != len(keys)
            if self._duplicate_source_keys:
                # Instrument sums allow repeated labels. Never collapse them
                # through the adapter's per-source lookup dictionaries.
                rows = []
        try:
            with cp.cuda.Device(int(identity["device_id"])):
                self._transverse_kernel = cp.RawKernel(_TRANSVERSE_KERNEL, "transverse_ab", options=("--std=c++11", "--fmad=false"))
                self._transverse_b_kernel = cp.RawKernel(_TRANSVERSE_B_KERNEL, "transverse_b", options=("--std=c++11", "--fmad=false"))
                self._segment_kernel = cp.RawKernel(_SEGMENT_KERNEL, "radial_segments", options=("--std=c++11", "--fmad=false"))
                self._round_kernel = cp.RawKernel(_ROUND_KERNEL, "round_ab", options=("--std=c++11", "--fmad=false"))
                for key, kind, value, source in rows:
                    self._rows.append((key, kind, value, source))
                    if source is not None and source.known_zero:
                        continue
                    potential = value.potential if type(value) in (FrozenCopperCoilField, FrozenTransverseCoilField) else value
                    if type(potential) is CopperCoilAssemblyPotential:
                        field = value if type(value) is FrozenCopperCoilField else FrozenCopperCoilField(potential)
                        magnetic_wrapper = type(value) is FrozenCopperCoilField
                        if (not field.implementation_is_supported()
                                or magnetic_wrapper and not _copper_callback_supported(field)
                                or any(not _identity_registration(coil.registration) for coil in potential.coils)):
                            continue
                        packed = [pack_annular_axis(coil._loops) for coil in potential.coils]
                        if any(not np.isfinite(part).all() for row in packed for part in row):
                            continue
                        rr, zz, ff = [np.concatenate([row[index] for row in packed]) for index in range(3)]
                        nonzero = abs(ff[ff != 0.])
                        if (np.any(rr < 1.e-6) or np.any(rr > 10.)
                                or np.any(nonzero < 1.e-40) or np.any(nonzero > 1.e40)):
                            continue
                        first = coil_count
                        local_geometry = []
                        for coil, row in zip(potential.coils, packed):
                            r, z, factor, polynomial, size = row
                            if polynomials is not None and (not np.array_equal(polynomials, polynomial) or not np.array_equal(lengths, size)):
                                raise ValueError("Round sources retain different polynomial laws")
                            polynomials, lengths = polynomial, size
                            radii.extend(r); centers.extend(z); factors.extend(factor)
                            offsets.append(len(radii)); coil_count += 1
                            row_geometry = (coil.inner_radius_m, coil.center_z_m, coil.length_m)
                            geometry.append(row_geometry); local_geometry.append(row_geometry)
                        self._round[key] = _Round(field, first, coil_count, rr, zz, ff, np.asarray(local_geometry), magnetic_wrapper)
                    elif type(potential) is TransverseCoilPotential:
                        field = value if type(value) is FrozenTransverseCoilField else None
                        if field is not None and not field.implementation_is_supported():
                            continue
                        segments = pack_analytic_rectangle_potential(potential)
                        if segments is not None:
                            nodes, ends, weighted = segments.starts_m, segments.ends_m, segments.weights_t_m
                            domain, axis_zero = np.empty((0, 11)), segments.axis_zero
                            node_radius = segments.minimum_radius_m
                            representation = "Analytic finite straight segments on retained radial/fraction Gauss pack"
                            reference_nodes = segments.original_path_node_count
                        else:
                            packed = pack_transverse_potential(potential)
                            if packed is None:
                                continue
                            nodes, weighted, domain, axis_zero = packed
                            ends = None
                            node_radius = float(np.hypot(nodes[:, 0], nodes[:, 1]).min(initial=np.inf))
                            representation = "Original line Gauss quadrature"
                            reference_nodes = len(nodes)
                        nonzero = abs(weighted[weighted != 0.])
                        if (np.max(abs(nodes), initial=0.) > 1.e4 or node_radius < 1.e-6
                                or np.any(nonzero < 1.e-100) or np.any(nonzero > 1.e100)):
                            continue
                        self._transverse[key] = _Transverse(potential, field, cp.asarray(nodes), cp.asarray(weighted),
                            domain, axis_zero, node_radius, None if ends is None else cp.asarray(ends), representation, reference_nodes)
                self._coil_count = coil_count
                self._round_args = None
                if coil_count:
                    self._round_args = tuple(cp.asarray(array) for array in
                        (np.asarray(offsets, np.int64), np.asarray(radii), np.asarray(centers), np.asarray(factors), polynomials, lengths))
                self._polynomial_columns = 0 if polynomials is None else polynomials.shape[1]
                coefficient = [(-1.)**n/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1)) for n in range(6)]
                self._coefficients = cp.asarray(coefficient)
                self.device_bytes = int(self._coefficients.nbytes)+sum(int(value.nbytes) for value in (self._round_args or ()))
                self.device_bytes += sum(int(row.nodes.nbytes+row.weights.nbytes+(0 if row.ends is None else row.ends.nbytes))
                                         for row in self._transverse.values())
        except Exception as error:
            raise _failure(error, "CUDA ideal-coil source preparation") from error

    @property
    def _sources(self):
        return self.original._sources if self.mode == "magnetic" else ()

    @property
    def cache_token(self):
        # A mutable callback belongs to its retained wrapper object. Equal
        # numerical graphs may share coefficients, never another object's live
        # callbacks. This process-local token is not a scientific identity.
        return (id(self.original),)

    def execution_receipt(self):
        with self._lock:
            return dict(execution_identity=thaw_json(self.execution_identity),
                counter_scope="Lifetime of this cached sampler; field queries are not emitted particle counts",
                counts=dict(self._counts), reference_reasons=dict(self._reasons),
                failure=self._failure_evidence, retained_device_bytes=self.device_bytes,
                transverse_paths={key: dict(representation=row.path_representation, elements=len(row.nodes),
                    original_nonzero_line_gauss_nodes=row.original_path_node_count,
                    retained_pack_order=int(row.potential.pack_order), reference_path_order=int(row.potential.path_order),
                    physical_turn_count_per_radial_coil=int(row.potential.turn_count_per_coil),
                    internal_closed_current_recipe=thaw_json(row.potential.current_segments),
                    domain_owner="Unchanged complete parent winding allocation")
                    for key,row in self._transverse.items()})

    def _reference(self, value, points, source=None, *, include_a=True):
        if self.mode == "potential":
            fields = value.sample(points)
            # Some unsupported native providers expose only a complete
            # potential bundle; retain their own implementation, discard no
            # physical field and never substitute zero for an unrequested A.
            return fields if include_a else {key:field for key,field in fields.items() if key != "A_t_m"}
        if source is not None:
            from .frozen_copper_coil_field import is_all_z_copper_field
            active = (np.ones(len(points), bool) if is_all_z_copper_field(value)
                      else ((points[:, 2] >= source.bounds_m[0, 2]) & (points[:, 2] <= source.bounds_m[1, 2])))
            result = np.zeros_like(points)
            if active.any():
                result[active] = value.field_at_global_positions_t(points[active])
            return {"B_t": result}
        return {"B_t": value.field_at_global_positions_t(points)}

    def _sample(self, positions, *, fields_only=False):
        raw = np.asarray(positions)
        if raw.dtype.kind not in "iuf" or raw.shape[-1:] != (3,) or not np.isfinite(raw).all():
            raise ValueError("CUDA coil positions must be finite real XYZ metres")
        shape = raw.shape; points = np.asarray(raw, dtype=float).reshape(-1, 3)
        include_a = self.mode == "potential" and not fields_only
        width = 6 if include_a else 3
        self._counts["calls"] += 1; self._counts["query_points"] += len(points)
        self._counts["full_potential_calls" if include_a else ("physical_fields_calls" if self.mode == "potential" else "magnetic_only_calls")] += 1
        if self.mode == "potential" and not _snapshot_callback_supported(self.original):
            self._counts["reference_calls"] += 1; self._reasons["overridden_snapshot_callback"] += 1
            # The complete original callback already owns its final gauge.
            fields = self.original.sample(raw)
            return fields if include_a else {key:fields[key] for key in ("B_t", "E_v_per_m", "phi_v")}
        if self.mode == "magnetic" and not _instrument_callback_supported(self.original):
            # Check on every call: a later class/instance override must not be
            # bypassed by an earlier packed source graph or a larger batch.
            self._counts["reference_calls"] += 1; self._reasons["overridden_instrument_callback"] += 1
            value = np.asarray(self.original.field_at_global_positions_t(raw), dtype=float)
            if value.shape != raw.shape or not np.isfinite(value).all():
                raise ValueError("Overridden instrument callback returned invalid B_t")
            return {"B_t": value}
        if self.mode == "magnetic" and getattr(self, "_duplicate_source_keys", False):
            self._counts["reference_calls"] += 1; self._reasons["duplicate_magnetic_source_keys"] += 1
            return {"B_t": self.original.field_at_global_positions_t(raw)}
        if self.mode == "magnetic" and len(points) < MAGNETIC_MINIMUM_BATCH:
            self._counts["reference_calls"] += 1; self._reasons["small_magnetic_batch"] += 1
            return {"B_t": self.original.field_at_global_positions_t(raw)}
        if not len(points):
            result = {"B_t": np.zeros(shape)}
            if self.mode == "potential":
                result.update(E_v_per_m=np.zeros(shape), phi_v=np.zeros(shape[:-1]))
                if include_a:
                    result["A_t_m"] = np.zeros(shape)
            return result
        native, admitted_round, admitted_transverse = {}, {}, {}
        mask = np.zeros(self._coil_count, np.int32)
        # Complete native domain validation before launching any device work.
        for key, kind, value, source in self._rows:
            if source is not None and source.known_zero:
                continue
            reason = "native_electric_provider" if kind == "electric" else "unsupported_or_posed_source"
            if key in self._round:
                reason = self._round[key].admission(points)
                if reason is None:
                    adapter = self._round[key]; admitted_round[key] = adapter
                    mask[adapter.first:adapter.stop] = 1
                    continue
            elif key in self._transverse:
                reason = self._transverse[key].admission(points)
                if reason is None:
                    adapter = self._transverse[key]
                    if not len(adapter.nodes) or (adapter.axis_zero and np.all(points[:, :2] == 0.)):
                        native[key] = {"B_t": np.zeros_like(points)}
                        if include_a:
                            native[key]["A_t_m"] = np.zeros_like(points)
                        self._counts["exact_zero_source_calls"] += 1
                    else:
                        admitted_transverse[key] = adapter
                    continue
            native[key] = self._reference(value, points, source, include_a=include_a)
            self._counts["reference_source_calls"] += 1; self._reasons[reason] += 1
        device_values, round_values = {}, None
        cp = self._cp
        try:
            with cp.cuda.Device(int(self.execution_identity["device_id"])):
                if admitted_round or admitted_transverse:
                    device = cp.asarray(points)
                    if admitted_round:
                        output = cp.empty((self._coil_count, len(points), width), dtype=cp.float64)
                        offsets, radii, centers, factors, polynomials, lengths = self._round_args
                        self._round_kernel(((len(points)+3)//4, self._coil_count), (32, 4),
                            (device, np.int64(len(points)), offsets, radii, centers, factors, polynomials, lengths,
                             np.int32(self._polynomial_columns), self._coefficients, cp.asarray(mask), np.int32(include_a), output))
                        round_values = output
                    for key, adapter in admitted_transverse.items():
                        output = cp.empty((len(points), width), dtype=cp.float64)
                        if adapter.ends is not None:
                            self._segment_kernel(((len(points)+3)//4,), (32, 4),
                                (device, np.int64(len(points)), adapter.nodes, adapter.ends, adapter.weights,
                                 np.int64(len(adapter.nodes)), np.int32(adapter.axis_zero), np.int32(include_a), output))
                        else:
                            kernel = self._transverse_kernel if include_a else self._transverse_b_kernel
                            kernel(((len(points)+3)//4,), (32, 4),
                                (device, np.int64(len(points)), adapter.nodes, adapter.weights,
                                 np.int64(len(adapter.nodes)), np.int32(adapter.axis_zero), output))
                        device_values[key] = output
                    if round_values is not None:
                        round_values = round_values.get()
                    for key, value in device_values.items():
                        device_values[key] = value.get()
                    self._counts["gpu_calls"] += 1
        except Exception as error:
            self._failure_evidence = dict(stage="allocation_launch_or_copy", type=type(error).__name__, message=str(error))
            raise _failure(error, "CUDA ideal-coil allocation/launch/readback") from error
        result = {"B_t": np.zeros_like(points)}
        if self.mode == "potential":
            result.update(E_v_per_m=np.zeros_like(points), phi_v=np.zeros(len(points)))
            if include_a:
                result["A_t_m"] = np.zeros_like(points)
        for key, kind, value, source in self._rows:
            if source is not None and source.known_zero:
                continue
            if key in admitted_round:
                adapter = admitted_round[key]; values = round_values[adapter.first].copy()
                for index in range(adapter.first+1, adapter.stop):
                    values += round_values[index]
                fields = dict(A_t_m=values[:, :3], B_t=values[:, 3:]) if include_a else {"B_t": values}
                self._counts["gpu_source_calls"] += 1
            elif key in device_values:
                values = device_values[key]
                fields = dict(A_t_m=values[:, :3], B_t=values[:, 3:]) if include_a else {"B_t": values}
                self._counts["gpu_source_calls"] += 1
                self._counts["gpu_analytic_rectangle_source_calls" if self._transverse[key].ends is not None
                             else "gpu_gauss_source_calls"] += 1
            else:
                fields = native[key]
            names = (("phi_v", "E_v_per_m") if kind == "electric" else
                     (("A_t_m", "B_t") if include_a else ("B_t",)))
            for name in names:
                if fields[name].shape != result[name].shape or not np.isfinite(fields[name]).all():
                    if key in admitted_round or key in device_values:
                        self._failure_evidence = dict(stage="finite_result", source_id=key, field=name)
                        raise GPUExecutionError("coil_field_execution_failure", "CUDA ideal-coil field returned invalid "+name)
                    raise ValueError("Native reference returned an invalid "+name)
                result[name] += fields[name]
        if include_a:
            result["A_t_m"] += self.original.gauge.vector_potential_shift_t_m(np.asarray(raw, float)).reshape(-1, 3)
        if any(not np.isfinite(value).all() for value in result.values()):
            raise GPUExecutionError("coil_field_execution_failure", "Combined ideal-coil field exceeds finite numerical range")
        return {key: value.reshape(shape[:-1] if key == "phi_v" else shape) for key, value in result.items()}

    def sample(self, positions_m):
        if self.mode != "potential":
            raise ValueError("A magnetic-only capture does not define electric or gauge potentials")
        with self._lock:
            return self._sample(positions_m)

    def sample_fields(self, positions_m):
        """Return physical B/E/phi, without computing or inventing CUDA A.

        Unsupported native providers retain their own reference interface;
        their bundled A, if any, is discarded rather than replaced by zero.
        Canonical handoff and magnetic-gauge checks must request sample().
        """
        if self.mode != "potential":
            raise ValueError("A magnetic-only capture does not define E or phi")
        with self._lock:
            return self._sample(positions_m, fields_only=True)

    def field_at_global_positions_t(self, positions_m):
        with self._lock:
            return self._sample(positions_m, fields_only=True)["B_t"]

    sample_b = field_at_global_positions_t


def _prepare(original, execution_backend, mode):
    if not explicit_cuda_coil_policy(execution_backend):
        return None
    if mode == "potential" and type(original) is not ElectromagneticPotentialSnapshot:
        raise ValueError("CUDA ideal-coil potentials require a static immutable snapshot")
    if mode == "magnetic":
        from .instrument_magnetic import InstrumentMagneticField
        if type(original) is not InstrumentMagneticField:
            raise ValueError("CUDA magnetic execution requires the exact frozen instrument field")
    identity = cuda_coil_execution_identity(execution_backend, mode=mode)
    if ((mode == "potential" and not _snapshot_callback_supported(original))
            or (mode == "magnetic" and not _instrument_callback_supported(original))):
        # Unknown outer callbacks may depend on the particular retained object.
        # Never substitute an equal-digest object's callback from a cache hit.
        return CudaCoilSampler(original, identity, mode)
    numerical = (original.numerical_digest if mode == "potential" else original.numerical_identity)
    # Unknown/custom synthetic fields use only this exact retained object.
    key = (mode, id(original), numerical, json_digest(identity))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key); return cached
    result = CudaCoilSampler(original, identity, mode)
    with _LOCK:
        previous = _CACHE.setdefault(key, result); _CACHE.move_to_end(key)
        while len(_CACHE) > MAX_CACHE_ENTRIES or sum(row.device_bytes for row in _CACHE.values()) > MAX_CACHE_DEVICE_BYTES:
            _CACHE.popitem(last=False)
        return previous


def prepare_cuda_coil_potential(snapshot, *, execution_backend="Prefer GPU"):
    return _prepare(snapshot, execution_backend, "potential")


def prepare_cuda_instrument_magnetic(provider, *, execution_backend="Prefer GPU"):
    return _prepare(provider, execution_backend, "magnetic")


def cuda_coil_cache_info():
    with _LOCK:
        return dict(entries=len(_CACHE), device_bytes=sum(row.device_bytes for row in _CACHE.values()),
                    maximum_entries=MAX_CACHE_ENTRIES, maximum_device_bytes=MAX_CACHE_DEVICE_BYTES)


def clear_cuda_coil_cache():
    """Release only this cache's references; never drain another job's pool."""
    with _LOCK:
        _CACHE.clear()
