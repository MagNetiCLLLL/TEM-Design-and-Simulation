"""Experimental whole-plane Gaussian Gram integration and convergence checks.

Analytical coherent pair integrals preserve phase, interference and absolute
probability. Sparse omissions have explicit Gaussian phase-space bounds; those
bounds do not cover semiclassical physics, propagation, or floating-point error.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from scripts.experimental_gaussian_beamlets import GaussianBeamlets
from temsim.immutable_json import json_digest
from temsim.optics.electron_gun.tip_coherence import wavelength_m

_WORKING_BYTES = 512*1024**2


def _load_packets(path):
    with np.load(path, allow_pickle=False) as data:
        count = len(data['coefficients'])
        if 720*count > _WORKING_BYTES:
            raise MemoryError('Gaussian Gram input/whitening estimate exceeds working budget')
        packets = GaussianBeamlets(**{name: data[name] for name in data.files})
    if packets.count == 0:
        raise ValueError('Gaussian Gram integration needs at least one packet')
    return {name: getattr(packets, name) for name in
            ('centres_m', 'momenta', 'precision_m1', 'coefficients', 'wavelength_m')}


def _pair_batches(tree, queries, radius, maximum_pairs=250_000):
    """Bound Python neighbor lists before allocating them, including dense cases."""
    first = 0
    while first < len(queries):
        stop = min(first+512, len(queries))
        while True:
            lengths = tree.query_ball_point(queries[first:stop], radius, workers=1, return_length=True)
            if int(np.sum(lengths)) <= maximum_pairs:
                break
            if stop-first == 1:
                raise MemoryError('One Gaussian neighbor list exceeds the bounded pair budget')
            stop = first+max(1, (stop-first)//2)
        neighbors = tree.query_ball_point(queries[first:stop], radius, workers=1)
        ii = np.repeat(np.arange(first, stop, dtype=np.int64), lengths)
        jj = np.concatenate(neighbors).astype(np.int64, copy=False)
        yield ii, jj
        first = stop


def _validate_radii(radii):
    values = np.asarray(radii, float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
        raise ValueError('Pair radii must be finite, positive and strictly increasing')
    return values


def packet_provenance(path):
    """Bind observed packets to their declared executed input and coherent mode."""
    path = Path(path).resolve()
    if not path.name.endswith('-packets.npz'):
        raise ValueError('Expected a replay *-packets.npz with its companion receipt')
    receipt_path = path.with_name(path.name.replace('-packets.npz', '-receipt.json'))
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    source_path = Path(receipt['input_manifest'])
    if not source_path.is_absolute() and not source_path.exists():
        source_path = Path(__file__).resolve().parents[1]/source_path
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != receipt['input_manifest_sha256']:
        raise ValueError('Executed input manifest differs from the replay receipt')
    source = json.loads(source_bytes)
    manifest_identity = source.pop('manifest_digest')
    if json_digest(source) != manifest_identity:
        raise ValueError('Executed input manifest digest is invalid')
    matches = [(index, row) for index, row in enumerate(source['modes'])
               if row['mode_id'] == receipt['mode_id']]
    if len(matches) != 1:
        raise ValueError('Replay coherent mode must identify exactly one executed input mode')
    mode_index, row = matches[0]
    if row['energy_kev'] != receipt['energy_kev'] or row['weight'] != receipt['weight']:
        raise ValueError('Replay energy/weight does not match its executed coherent mode')
    if source['plane_z_mm'] != receipt['start_z_mm']:
        raise ValueError('Replay entrance differs from the executed input plane')
    arrays = {}
    for key, item in row['arrays'].items():
        array_path = source_path.parent/item['file']
        if array_path.parent != source_path.parent or array_path.is_symlink():
            raise ValueError('Invalid executed input array path')
        digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Executed coherent input array checksum changed')
        arrays[key] = {name:item[name] for name in ('sha256','shape','nbytes')}
    with np.load(path, allow_pickle=False) as packets:
        count = len(packets['coefficients'])
        wavelength = float(packets['wavelength_m'])
    if count != receipt['packet_count']:
        raise ValueError('Packet count differs from its replay receipt')
    expected = float(wavelength_m(receipt['energy_kev']*1000))
    if wavelength != expected:
        raise ValueError('Packet wavelength differs from its coherent-mode energy')
    mode_identity = {'mode_id':row['mode_id'], 'mode_index':mode_index,
        'energy_kev':row['energy_kev'], 'weight':row['weight'],
        'reference_plane':source['reference_plane'], 'plane_z_mm':source['plane_z_mm'],
        'axial_reference':row['axial_reference'], 'scattering_history':row.get('scattering_history', []),
        'arrays':arrays}
    return {'packet_file':str(path), 'packet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'receipt_file':str(receipt_path), 'receipt_sha256':hashlib.sha256(receipt_bytes).hexdigest(),
        'mode_id':row['mode_id'], 'mode_index':mode_index, 'mode_weight':row['weight'],
        'energy_kev':row['energy_kev'], 'wavelength_m':wavelength, 'packet_count':count,
        'output_plane_z_mm':receipt['stop_z_mm'], 'executed_input_mode_identity':json_digest(mode_identity),
        'state_file_sha256':receipt.get('state_file_sha256'),
        'binding_scope':'Current packet hash plus verified replay receipt and executed input arrays; legacy replay receipts do not contain a pre-existing output-packet hash.'}


def require_matched_modes(a, b):
    for key in ('mode_id', 'mode_index', 'energy_kev', 'mode_weight', 'wavelength_m',
                'output_plane_z_mm', 'executed_input_mode_identity'):
        if a[key] != b[key]:
            raise ValueError('Convergence comparison requires matching '+key)
    if a['state_file_sha256'] is not None and b['state_file_sha256'] is not None:
        if a['state_file_sha256'] != b['state_file_sha256']:
            raise ValueError('Convergence comparison has different physical states')

@njit(cache=True)
def overlap(qi, pi, ci, qj, pj, cj, k):
    """Integral conj(g_i)*g_j for normalized two-dimensional packets.

    Midpoint coordinates keep the exponent independent of a large lab origin.
    Re(A)>0 fixes the Gaussian determinant square root by continuation from a
    positive real matrix; its principal root is valid for this 2D determinant.
    """
    dx, dy = qj[0]-qi[0], qj[1]-qi[1]
    c00, c01, c11 = cj[0,0]-ci[0,0].conjugate(), cj[0,1]-ci[0,1].conjugate(), cj[1,1]-ci[1,1].conjugate()
    s00, s01, s11 = cj[0,0]+ci[0,0].conjugate(), cj[0,1]+ci[0,1].conjugate(), cj[1,1]+ci[1,1].conjugate()
    a00, a01, a11 = -1j*k*c00, -1j*k*c01, -1j*k*c11
    det = a00*a11-a01*a01
    bx = 1j*k*(pj[0]-pi[0]-.5*(s00*dx+s01*dy))
    by = 1j*k*(pj[1]-pi[1]-.5*(s01*dx+s11*dy))
    constant = 1j*k*(-.5*((pj[0]+pi[0])*dx+(pj[1]+pi[1])*dy)
        +.125*(c00*dx*dx+2*c01*dx*dy+c11*dy*dy))
    exponent = constant+.5*(a11*bx*bx-2*a01*bx*by+a00*by*by)/det
    di = ci[0,0].imag*ci[1,1].imag-ci[0,1].imag**2
    dj = cj[0,0].imag*cj[1,1].imag-cj[0,1].imag**2
    return 2*k*(di*dj)**.25/np.sqrt(det)*np.exp(exponent)

@njit(cache=True)
def _norm_pairs(q, p, c, coefficients, z, ii, jj, k, radii2):
    sums = np.zeros(len(radii2))
    corrections = np.zeros(len(radii2))
    absolute = np.zeros(len(radii2))
    counts = np.zeros(len(radii2), dtype=np.int64)
    for item in range(len(ii)):
        i, j = ii[item], jj[item]
        distance2 = 0.
        for axis in range(4): distance2 += (z[i,axis]-z[j,axis])**2
        value = 2*(coefficients[i].conjugate()*coefficients[j]
                    *overlap(q[i],p[i],c[i],q[j],p[j],c[j],k)).real
        for r in range(len(radii2)):
            if distance2 <= radii2[r]:
                term = value-corrections[r]
                new = sums[r]+term
                corrections[r] = (new-sums[r])-term
                sums[r] = new
                absolute[r] += abs(value)
                counts[r] += 1
    return sums, absolute, counts

def whiten(q,p,c,k):
    # A common canonical shear removes the large common quadratic carrier.
    # Subsequent arbitrary whitening changes neither the pair exponent nor
    # Loewner covariance inequalities used for the omission bound.
    ref = len(q)//2+int(round(math.sqrt(len(q))))//2
    eig, vec = np.linalg.eigh(c[ref].imag)
    sq = (vec*np.sqrt(eig))@vec.T
    invsq = (vec/np.sqrt(eig))@vec.T
    d = c.real-c[ref].real
    inverse = np.linalg.inv(c.imag)
    v = np.empty((len(q),4,4))
    v[:,:2,:2] = sq@inverse@sq/2
    v[:,:2,2:] = sq@inverse@d@invsq/2
    v[:,2:,:2] = v[:,:2,2:].swapaxes(1,2)
    v[:,2:,2:] = invsq@(d@inverse@d+c.imag)@invsq/2
    z = np.concatenate((np.sqrt(k)*q@sq, np.sqrt(k)*(p-q@c[ref].real)@invsq),axis=1)
    eig, vec = np.linalg.eigh(np.mean(v,axis=0))
    transform = (vec/np.sqrt(eig))@vec.T
    v = transform@v@transform
    # Inflation protects the analytic inequality against ordinary eigensolver
    # rounding. This is not a formal interval-arithmetic floating-point proof.
    largest = float(np.linalg.eigvalsh(v)[:,-1].max())*(1+1e-10)+1e-12
    return z@transform, largest

def validate_formula():
    k = 2.7
    qi,pi = np.array([.2,-.3]),np.array([.3,-.1])
    qj,pj = np.array([-.4,.2]),np.array([-.2,.6])
    ci=np.array([[.4+1.2j,.15+.1j],[.15+.1j,-.3+.8j]])
    cj=np.array([[-.2+.9j,-.1+.2j],[-.1+.2j,.5+1.5j]])
    axis=np.linspace(-10,10,501)
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    points=np.stack((xx,yy),axis=-1)
    def density(q,p,c):
        delta=points-q
        n=(k/np.pi)**.5*np.linalg.det(c.imag)**.25
        return n*np.exp(1j*k*(np.einsum('...i,i->...',delta,p)+.5*np.einsum('...i,ij,...j->...',delta,c,delta)))
    exact=overlap(qi,pi,ci,qj,pj,cj,k)
    numerical=np.sum(density(qi,pi,ci).conj()*density(qj,pj,cj))*(axis[1]-axis[0])**2
    reverse=overlap(qj,pj,cj,qi,pi,ci,k)
    self_overlap=overlap(qi,pi,ci,qi,pi,ci,k)
    error=float(abs(exact-numerical))
    assert error<1e-12 and abs(reverse-exact.conjugate())<1e-14 and abs(self_overlap-1)<1e-14
    return {'independent_uniform_quadrature_absolute_error':error,
        'quadrature_shape':[501,501], 'hermitian_error':float(abs(reverse-exact.conjugate())),
        'self_overlap_error':float(abs(self_overlap-1)),
        'analytic_real_imag':[float(exact.real),float(exact.imag)]}

def measure_norm(path, radii=(10.,14.,18.), batch=512):
    started=time.perf_counter()
    packets=_load_packets(path)
    q,p,c,coeff=(packets[key] for key in ('centres_m','momenta','precision_m1','coefficients'))
    k=2*np.pi/float(packets['wavelength_m'])
    z,L=whiten(q,p,c,k)
    tree=cKDTree(z)
    radii=_validate_radii(radii)
    total=np.zeros(len(radii)); absolute=np.zeros(len(radii));counts=np.zeros(len(radii),np.int64)
    for ii,jj in _pair_batches(tree,z,radii[-1]*(1+1e-12)):
        mask=jj>ii
        values,abs_values,local_counts=_norm_pairs(q,p,c,coeff,z,ii[mask],jj[mask],k,radii*radii)
        total+=values;absolute+=abs_values;counts+=local_counts
    diagonal=float(np.vdot(coeff,coeff).real)
    l1=float(abs(coeff).sum())
    rows=[]
    for radius,value,abs_value,count in zip(radii,total,absolute,counts):
        # For normalized pure Gaussian packets:
        # |<i,j>| <= exp[-deltaZ^T(Vi+Vj)^-1 deltaZ/4].
        # Vi<=L I => this <= exp[-distance^2/(8L)].
        # Sum over all excluded ordered pairs is bounded by ||coeff||_1^2.
        tail=l1*l1*math.exp(-(radius*(1-1e-12))**2/(8*L))
        rows.append({'whitened_radius':float(radius),'coherent_norm':float(diagonal+value),
            'analytic_omitted_pair_absolute_norm_bound':tail,'pair_count':int(count),
            'sum_absolute_pair_real_contributions':float(abs_value)})
    return {'schema':'experimental-gaussian-global-coherent-norm-v1','packet_file':str(path),
        'scope':'Global continuous-space analytic Gaussian Gram sum; no physical coefficients changed. Excludes propagation/decomposition/model error.',
        'packet_count':len(q),'coefficient_l1':l1,'diagonal_only_norm':diagonal,
        'maximum_whitened_covariance_eigenvalue_bound':L,
        'tail_bound_formula':'||coefficients||_1^2 exp(-R^2/(8L)); all normalized Gaussian pairs outside the queried phase-space ball',
        'floating_point_note':'Double precision with compensated per-batch summation; analytic truncation bound is not an interval bound on floating-point special functions.',
        'radius_results':rows,'elapsed_s':time.perf_counter()-started}

@njit(cache=True)
def cross_sum(qa,pa,ca,aa,qb,pb,cb,ab,za,zb,ii,jj,k,radii2):
    values=np.zeros(len(radii2),np.complex128)
    corrections=np.zeros(len(radii2),np.complex128)
    absolute=np.zeros(len(radii2))
    counts=np.zeros(len(radii2),np.int64)
    for item in range(len(ii)):
        i,j=ii[item],jj[item]
        distance2=0.
        for axis in range(4): distance2+=(za[i,axis]-zb[j,axis])**2
        value=aa[i].conjugate()*ab[j]*overlap(qa[i],pa[i],ca[i],qb[j],pb[j],cb[j],k)
        for r in range(len(radii2)):
            if distance2<=radii2[r]:
                term=value-corrections[r]
                new=values[r]+term
                corrections[r]=(new-values[r])-term
                values[r]=new
                absolute[r]+=abs(value)
                counts[r]+=1
    return values,absolute,counts

def validate_cross():
    rng=np.random.default_rng(1637)
    n,k=7,2.7
    q=rng.normal(size=(n,2))*.3;p=rng.normal(size=(n,2))*.5
    c=np.empty((n,2,2),complex)
    for i in range(n):
        a=rng.normal(size=(2,2))*.25;c[i]=(a+a.T)/2+1j*np.eye(2)*(1+.1*i)
    a=(rng.normal(size=n)+1j*rng.normal(size=n))/3
    z=np.zeros((n,4))
    ii=np.repeat(np.arange(3,dtype=np.int64),4);jj=np.tile(np.arange(4,dtype=np.int64),3)
    value=cross_sum(q[:3],p[:3],c[:3],a[:3],q[3:],p[3:],c[3:],a[3:],z[:3],z[3:],ii,jj,k,np.array([1.]))[0][0]
    axis=np.linspace(-9,9,451);yy,xx=np.meshgrid(axis,axis,indexing='ij');points=np.stack((xx,yy),axis=-1)
    waves=[]
    for i in range(n):
        delta=points-q[i];norm=(k/np.pi)**.5*np.linalg.det(c[i].imag)**.25
        waves.append(a[i]*norm*np.exp(1j*k*(delta@p[i]+.5*np.einsum('...i,ij,...j->...',delta,c[i],delta))))
    wa=sum(waves[:3]);wb=sum(waves[3:]);area=(axis[1]-axis[0])**2
    numerical=np.vdot(wa,wb)*area
    whole_norm=float(np.vdot(wa+wb,wa+wb).real*area)
    split_norm=float(np.vdot(wa,wa).real*area+np.vdot(wb,wb).real*area+2*value.real)
    self_i=np.repeat(np.arange(n,dtype=np.int64),n);self_j=np.tile(np.arange(n,dtype=np.int64),n)
    self_value=cross_sum(q,p,c,a,q,p,c,a,z,z,self_i,self_j,k,np.array([1.]))[0][0]
    assert abs(value-numerical)<2e-12 and abs(whole_norm-split_norm)<2e-12 and abs(self_value-whole_norm)<2e-12
    return {'independent_cross_quadrature_absolute_error':float(abs(value-numerical)),
        'self_split_norm_error':abs(whole_norm-split_norm),'self_cross_vs_quadrature_error':float(abs(self_value-whole_norm)),
        'quadrature_shape':[451,451],'test_packet_count':n}

def measure_cross(path_a,path_b,norm_a,norm_b,radii=(12.,18.,22.)):
    started=time.perf_counter()
    a,b=_load_packets(path_a),_load_packets(path_b)
    if float(a['wavelength_m'])!=float(b['wavelength_m']):raise ValueError('Different wavelength modes cannot be compared as one coherent field')
    k=2*np.pi/float(a['wavelength_m'])
    qa,pa,ca,aa=(a[key] for key in ('centres_m','momenta','precision_m1','coefficients'))
    qb,pb,cb,ab=(b[key] for key in ('centres_m','momenta','precision_m1','coefficients'))
    if 720*(len(qa)+len(qb)) > _WORKING_BYTES:
        raise MemoryError('Combined Gaussian Gram whitening exceeds working budget')
    if not all(math.isfinite(n) and n > 0 for n in (norm_a,norm_b)):
        raise ValueError('Cross-field comparison requires finite positive measured norms')
    q=np.concatenate((qa,qb));p=np.concatenate((pa,pb));c=np.concatenate((ca,cb))
    z,L=whiten(q,p,c,k);za,zb=z[:len(qa)],z[len(qa):]
    del q,p,c
    tree=cKDTree(zb)
    radii=_validate_radii(radii)
    values=np.zeros(len(radii),complex);absolute=np.zeros(len(radii));counts=np.zeros(len(radii),np.int64)
    for ii,jj in _pair_batches(tree,za,radii[-1]*(1+1e-12)):
        local,local_abs,local_counts=cross_sum(qa,pa,ca,aa,qb,pb,cb,ab,za,zb,ii,jj,k,radii*radii)
        values+=local;absolute+=local_abs;counts+=local_counts
    l1a=float(abs(aa).sum());l1b=float(abs(ab).sum())
    rows=[]
    for radius,value,abs_value,count in zip(radii,values,absolute,counts):
        tail=l1a*l1b*math.exp(-(radius*(1-1e-12))**2/(8*L))
        difference2=norm_a+norm_b-2*value.real
        rows.append({'whitened_radius':float(radius),'cross_inner_product_real':float(value.real),
            'cross_inner_product_imag':float(value.imag),'analytic_omitted_pair_absolute_cross_bound':tail,
            'pair_count':int(count),'absolute_pair_sum':float(abs_value),
            'global_complex_difference_l2_squared':float(difference2),
            'global_relative_complex_l2_vs_a':float(np.sqrt(max(difference2,0)/norm_a)),
            'global_relative_complex_l2_vs_b':float(np.sqrt(max(difference2,0)/norm_b)),
            'normalized_overlap_magnitude':float(abs(value)/np.sqrt(norm_a*norm_b)),
            'overlap_phase_rad':float(np.angle(value)),
            'piston_fit_applied':False})
    return {'schema':'experimental-gaussian-global-cross-v1','packet_files':[str(path_a),str(path_b)],
        'scope':'Global continuous-space complex inner product; no piston/translation/scale fit or renormalization. Numerical convergence within the semiclassical representation, not proof of exact full-wave accuracy.',
        'norm_a':norm_a,'norm_b':norm_b,'maximum_whitened_covariance_eigenvalue_bound':L,
        'tail_bound_formula':'L1a*L1b*exp(-R^2/(8L)) for excluded cross pairs',
        'floating_point_note':'Compensated summation inside batches; analytic omission bounds exclude floating-point special-function errors.',
        'radius_results':rows,'elapsed_s':time.perf_counter()-started}

@njit(cache=True)
def pair_moments(qi,pi,ci,qj,pj,cj,k):
    dx,dy=qj[0]-qi[0],qj[1]-qi[1]
    a00=-1j*k*(cj[0,0]-ci[0,0].conjugate())
    a01=-1j*k*(cj[0,1]-ci[0,1].conjugate())
    a11=-1j*k*(cj[1,1]-ci[1,1].conjugate())
    det=a00*a11-a01*a01
    inv00,inv01,inv11=a11/det,-a01/det,a00/det
    bx=1j*k*(pj[0]-pi[0]-.5*((cj[0,0]+ci[0,0].conjugate())*dx+(cj[0,1]+ci[0,1].conjugate())*dy))
    by=1j*k*(pj[1]-pi[1]-.5*((cj[0,1]+ci[0,1].conjugate())*dx+(cj[1,1]+ci[1,1].conjugate())*dy))
    mx=.5*(qi[0]+qj[0])+inv00*bx+inv01*by
    my=.5*(qi[1]+qj[1])+inv01*bx+inv11*by
    value=overlap(qi,pi,ci,qj,pj,cj,k)
    result=np.empty(6,np.complex128)
    result[0]=value;result[1]=value*mx;result[2]=value*my
    result[3]=value*(inv00+mx*mx);result[4]=value*(inv01+mx*my);result[5]=value*(inv11+my*my)
    return result

@njit(cache=True)
def _moment_pairs(q,p,c,a,z,ii,jj,k,radius2):
    result=np.zeros(6);correction=np.zeros(6);absolute=np.zeros(6);count=0
    for item in range(len(ii)):
        i,j=ii[item],jj[item]
        distance2=0.
        for axis in range(4):distance2+=(z[i,axis]-z[j,axis])**2
        if distance2>radius2:continue
        value=pair_moments(q[i],p[i],c[i],q[j],p[j],c[j],k)*a[i].conjugate()*a[j]
        for axis in range(6):
            real=2*value[axis].real
            term=real-correction[axis];new=result[axis]+term
            correction[axis]=(new-result[axis])-term;result[axis]=new;absolute[axis]+=abs(real)
        count+=1
    return result,absolute,count

def validate_moments():
    k=2.7;qi=np.array([.2,-.3]);qj=np.array([-.4,.2]);pi=np.array([.3,-.1]);pj=np.array([-.2,.6])
    ci=np.array([[.4+1.2j,.15+.1j],[.15+.1j,-.3+.8j]])
    cj=np.array([[-.2+.9j,-.1+.2j],[-.1+.2j,.5+1.5j]])
    axis=np.linspace(-10,10,501);yy,xx=np.meshgrid(axis,axis,indexing='ij');points=np.stack((xx,yy),axis=-1)
    def wave(q,p,c):
        delta=points-q;n=(k/np.pi)**.5*np.linalg.det(c.imag)**.25
        return n*np.exp(1j*k*(delta@p+.5*np.einsum('...i,ij,...j->...',delta,c,delta)))
    product=wave(qi,pi,ci).conj()*wave(qj,pj,cj)
    numerical=np.array([np.sum(product*f)*(axis[1]-axis[0])**2 for f in (1.,xx,yy,xx*xx,xx*yy,yy*yy)])
    analytical=pair_moments(qi,pi,ci,qj,pj,cj,k)
    error=np.abs(numerical-analytical)
    assert error.max()<1e-12
    return {'independent_quadrature_max_absolute_error':float(error.max()),'moment_order':['norm','x','y','xx','xy','yy'],'per_moment_absolute_error':error.tolist()}

def measure_moments(path,radius=18.):
    _validate_radii((radius,))
    started=time.perf_counter();data=_load_packets(path)
    q,p,c,a=(data[key] for key in ('centres_m','momenta','precision_m1','coefficients'))
    k=2*np.pi/float(data['wavelength_m']);z,L=whiten(q,p,c,k);tree=cKDTree(z)
    total=np.zeros(6);absolute=np.zeros(6);count=0
    for ii,jj in _pair_batches(tree,z,radius*(1+1e-12)):
        mask=jj>ii
        values,absvalues,n=_moment_pairs(q,p,c,a,z,ii[mask],jj[mask],k,radius*radius)
        total+=values;absolute+=absvalues;count+=n
    weights=abs(a)**2;covariance=np.linalg.inv(c.imag)/(2*k)
    diagonal=np.array([weights.sum(),np.dot(weights,q[:,0]),np.dot(weights,q[:,1]),
        np.dot(weights,covariance[:,0,0]+q[:,0]**2),np.dot(weights,covariance[:,0,1]+q[:,0]*q[:,1]),np.dot(weights,covariance[:,1,1]+q[:,1]**2)])
    raw=total+diagonal;norm=raw[0];centre=raw[1:3]/norm
    second=np.array([[raw[3],raw[4]],[raw[4],raw[5]]])/norm;cov=second-np.outer(centre,centre)
    l1=float(abs(a).sum());tail=l1*l1*np.exp(-(radius*(1-1e-12))**2/(8*L))
    # A^-1 is bounded by 1/lambda_min(Re A). Remove a common real quadratic
    # carrier from p,C when bounding b; this leaves every pair b and A intact.
    common=c[len(q)//2].real
    pmax=float(np.linalg.norm(p-q@common.T,axis=1).max())
    cmax=float(np.linalg.svd(c-common,compute_uv=False)[:,0].max())
    qmax=float(np.linalg.norm(q,axis=1).max())
    imin=float(np.linalg.eigvalsh(c.imag)[:,0].min())*(1-1e-10)
    mean_bound=qmax+(pmax+cmax*qmax)/imin
    inverse_bound=1/(2*k*imin)
    bounds=np.array([tail,tail*mean_bound,tail*mean_bound,
                     tail*(inverse_bound+mean_bound**2),tail*(inverse_bound+mean_bound**2),tail*(inverse_bound+mean_bound**2)])
    return {'schema':'experimental-gaussian-global-spatial-moments-v1','packet_file':str(path),'packet_count':len(q),
        'scope':'Continuous whole-plane coherent moments from analytical Gaussian pair integrals; excludes semiclassical model error. Normalized statistical moments use measured total norm, while displayed fields and coefficients are not renormalized.',
        'whitened_radius':radius,'maximum_whitened_covariance_eigenvalue_bound':L,'included_pair_count':count,
        'coherent_norm':float(norm),'centre_m':centre.tolist(),'covariance_m2':cov.tolist(),
        'sigma_xy_nm':(np.sqrt(np.diag(cov))*1e9).tolist(),'rms_radius_nm':float(np.sqrt(np.trace(cov))*1e9),
        'principal_sigma_nm':(np.sqrt(np.linalg.eigvalsh(cov))*1e9).tolist(),
        'raw_moments_norm_x_y_xx_xy_yy':raw.tolist(),'analytic_omitted_raw_moment_absolute_bounds':bounds.tolist(),
        'sum_absolute_offdiagonal_moment_contributions':absolute.tolist(),
        'pair_complex_mean_global_norm_bound_m':mean_bound,'pair_inverse_matrix_bound_m2':inverse_bound,
        'floating_point_note':'Analytic omission bounds exclude double-precision accumulation and special-function rounding.',
        'elapsed_s':time.perf_counter()-started}


def self_validation():
    return {'pair_formula':validate_formula(), 'cross_sum':validate_cross(),
            'spatial_moments':validate_moments()}


def _read_norm(path, provenance):
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    observed=result.get('provenance')
    if observed is None or observed['packet_sha256'] != provenance['packet_sha256']:
        raise ValueError('Norm receipt must be generated by this CLI for the same packet bytes')
    if result['schema'] != 'experimental-gaussian-global-coherent-norm-v1':
        raise ValueError('Expected a whole-plane coherent norm receipt')
    return result['radius_results'][-1]['coherent_norm']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--working-mib',type=int,default=512)
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('self-test',help='Independent quadrature, self-overlap and split checks')
    check.add_argument('--output',type=Path)
    for name in ('norm','moments'):
        command=commands.add_parser(name)
        command.add_argument('packets',type=Path)
        command.add_argument('--output',type=Path,required=True)
    cross=commands.add_parser('cross',help='Matched-mode convergence; no phase/position fit')
    cross.add_argument('a',type=Path);cross.add_argument('b',type=Path)
    cross.add_argument('--norm-a',type=Path);cross.add_argument('--norm-b',type=Path)
    cross.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.working_mib < 32:
        parser.error('Working budget must be at least32 MiB')
    global _WORKING_BYTES
    _WORKING_BYTES=args.working_mib*1024**2
    if args.command=='self-test':
        result={'scope':'Small independent mathematical validation only',**self_validation()}
    elif args.command in ('norm','moments'):
        provenance=packet_provenance(args.packets)
        result=(measure_norm(args.packets) if args.command=='norm' else measure_moments(args.packets))
        result['provenance']=provenance
    else:
        pa,pb=packet_provenance(args.a),packet_provenance(args.b)
        require_matched_modes(pa,pb)
        na=(_read_norm(args.norm_a,pa) if args.norm_a else
            measure_norm(args.a)['radius_results'][-1]['coherent_norm'])
        nb=(_read_norm(args.norm_b,pb) if args.norm_b else
            measure_norm(args.b)['radius_results'][-1]['coherent_norm'])
        result=measure_cross(args.a,args.b,na,nb)
        result['provenance_a'],result['provenance_b']=pa,pb
    result['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['maximum_working_bytes']=_WORKING_BYTES
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
