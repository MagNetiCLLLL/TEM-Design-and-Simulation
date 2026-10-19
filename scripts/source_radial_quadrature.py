"""Offline qualification: batched true-margin roots and angular-CDF quadrature.

No source or optics is replaced. The callback must execute the actual requested
base/case/sign state and return continuous signed aperture margins (>=0 passes).
Finite scanning is NOT a proof that arbitrarily narrow extra pass bands do not
exist; this limitation remains explicit even when every located root converges.
This is not the default production emitter or a replacement aperture model.
The callback must be elementwise and consistent across batch compositions:
solver root tolerance does not bound the transport callback's numerical error.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import elementwise
from numpy.polynomial.legendre import leggauss


class BoundaryResolutionError(ValueError):
    pass


class BatchMargins:
    """Cache exact (base ID,u) evaluations and coalesce all active-root calls."""
    def __init__(self, callback, base_count, channel_keys):
        self.callback, self.base_count = callback, base_count
        self.channel_keys = tuple(channel_keys)
        self.cache, self.batches = {}, []

    def __call__(self, base, u):
        base, u = np.asarray(base), np.asarray(u, dtype=float)
        if (base.ndim != 1 or base.shape != u.shape or not np.isfinite(u).all()
                or base.dtype.kind not in 'iu'
                or np.any((u < 0) | (u > 1) | (base < 0) | (base >= self.base_count))):
            raise ValueError('Invalid base IDs or angular CDF coordinates')
        base = base.astype(int, copy=False)
        keys = [(int(b), float(x)) for b, x in zip(base, u)]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            b = np.array([k[0] for k in missing], dtype=int)
            x = np.array([k[1] for k in missing])
            values = np.asarray(self.callback(b.copy(), x.copy()), dtype=float)
            if values.shape != (len(missing), len(self.channel_keys)) or not np.isfinite(values).all():
                raise BoundaryResolutionError('Actual callback returned nonfinite or incorrectly shaped margins')
            self.batches.append(dict(rows=len(missing), base_count=len(np.unique(b))))
            self.cache.update((k, v.copy()) for k, v in zip(missing, values))
        return np.array([self.cache[k] for k in keys]).reshape(len(keys), len(self.channel_keys))


@dataclass
class RadialPartition:
    base_index: int
    edges: np.ndarray
    midpoint_margins: np.ndarray
    diagnostics: dict


@dataclass
class RadialSolution:
    partitions: list
    channel_keys: tuple
    roots: list
    diagnostics: dict
    evaluator: BatchMargins


@dataclass
class GaussTable:
    base_index: np.ndarray
    cell_index: np.ndarray
    angular_cdf: np.ndarray
    weight: np.ndarray
    expected_pass: np.ndarray
    diagnostics: dict


def solve_radial_partitions(callback, base_count, channel_keys, *, scan_intervals=8,
                            scan_refinements=1, u_tol=1e-9, maxiter=12,
                            require_certified=False):
    """Locate all sampled sign changes globally, then batch Chandrupatla roots.

    Each callback receives (base_indices, angular_cdf) arrays and returns
    (rows, channels) margins for all case/sign channels. No per-base root loop
    calls physics. maxiter caps expensive global iterations, not each root's
    scalar evaluations. Every accepted root retains a real bracket <=u_tol.
    Multiple transitions/pass intervals are retained, never coerced to one cut.
    """
    if require_certified:
        raise BoundaryResolutionError('Finite margin scans cannot certify absence of unsampled narrow pass intervals')
    if (type(base_count) is not int or base_count < 1 or not channel_keys or isinstance(channel_keys,str)
            or len(set(channel_keys)) != len(channel_keys) or type(scan_intervals) is not int
            or scan_intervals < 2 or type(scan_refinements) is not int or not 0 <= scan_refinements <= 4
            or isinstance(u_tol,(bool,np.bool_)) or not isinstance(u_tol,(int,float,np.floating))
            or not np.isfinite(u_tol) or not 1e-14 <= u_tol < .01 or type(maxiter) is not int or maxiter < 1):
        raise ValueError('Invalid radial boundary solver parameters')
    evaluator = BatchMargins(callback, base_count, channel_keys)
    intervals = scan_intervals*2**scan_refinements
    grid = np.linspace(0., 1., intervals+1)
    scan_base = np.repeat(np.arange(base_count), len(grid))
    values = evaluator(scan_base, np.tile(grid, base_count)).reshape(base_count, len(grid), -1)
    if np.any((values[:, :-1] == 0) & (values[:, 1:] == 0)):
        raise BoundaryResolutionError('Non-isolated zero-margin interval is unresolved')
    opposite=((values[:, :-1]<0)&(values[:, 1:]>0))|((values[:, :-1]>0)&(values[:, 1:]<0))
    bi, gi, ci = np.nonzero(opposite)
    roots = []
    # Roots exactly on a scanned node have zero residual; verify a narrow actual
    # bracket too, so a rounded-to-zero sample is not accepted without evidence.
    zb, zg, zc = np.nonzero(values == 0)
    root_b, root_c = list(zb), list(zc)
    root_x = list(grid[zg]); root_lo = list(grid[zg]); root_hi = list(grid[zg])
    root_f = [0.]*len(zb); root_nit = [0]*len(zb)
    if len(bi):
        def function(u, base, channel):
            return evaluator(base, u)[np.arange(len(u)), np.asarray(channel, dtype=int)]
        result = elementwise.find_root(function, (grid[gi], grid[gi+1]), args=(bi, ci),
            tolerances=dict(xatol=u_tol, xrtol=0., fatol=0., frtol=0.), maxiter=maxiter)
        if not np.all(result.success):
            failed = np.flatnonzero(~result.success)
            raise BoundaryResolutionError(f'Actual batched root solve failed: statuses={result.status[failed].tolist()}, base={bi[failed].tolist()}, channels={ci[failed].tolist()}')
        root_b.extend(bi); root_c.extend(ci); root_x.extend(result.x)
        root_lo.extend(result.bracket[0]); root_hi.extend(result.bracket[1]); root_f.extend(result.f_x); root_nit.extend(result.nit)
    b, c, x = np.array(root_b, dtype=int), np.array(root_c, dtype=int), np.array(root_x)
    lo, hi = np.array(root_lo), np.array(root_hi)
    if len(b):
        # Exact floating zero can stop SciPy with a wide bracket. One global
        # physical batch verifies a narrow bracket around every such result.
        need = ((hi-lo) > u_tol) | ((hi == lo) & (x > 0) & (x < 1))
        if need.any():
            low = np.maximum(0., x[need]-.49*u_tol); high = np.minimum(1., x[need]+.49*u_tol)
            both = evaluator(np.r_[b[need], b[need]], np.r_[low, high])
            m = int(need.sum()); channels = c[need]
            fl, fh = both[np.arange(m), channels], both[m+np.arange(m), channels]
            if np.any(((fl>0)&(fh>0)) | ((fl<0)&(fh<0)) | ((fl == 0) & (fh == 0))):
                raise BoundaryResolutionError('Root residual vanished but a narrow actual sign bracket could not be verified')
            lo[need], hi[need] = low, high
        if np.any(hi-lo > u_tol*(1+1e-12)):
            raise BoundaryResolutionError('Final actual root brackets exceed angular-CDF tolerance')
        final = evaluator(b, x)[np.arange(len(b)), c]
        for k in range(len(b)):
            roots.append(dict(base_index=int(b[k]), channel_key=channel_keys[c[k]], channel_index=int(c[k]),
                u=float(x[k]), bracket=[float(lo[k]), float(hi[k])], bracket_width=float(hi[k]-lo[k]),
                actual_margin_residual=float(final[k]), iterations=int(root_nit[k])))
    edges = [np.unique(np.r_[0., x[b == i], 1.]) for i in range(base_count)]
    midpoint_b = np.concatenate([np.full(len(e)-1, i, dtype=int) for i, e in enumerate(edges)])
    midpoint_u = np.concatenate([e[:-1]+np.diff(e)/2 for e in edges])
    midpoint_margins = evaluator(midpoint_b, midpoint_u)
    parts=[]; offset=0
    for i, e in enumerate(edges):
        v=midpoint_margins[offset:offset+len(e)-1];offset+=len(e)-1
        if np.any(v == 0):
            raise BoundaryResolutionError('An open cell midpoint lies on an unresolved margin boundary')
        passed=v>0
        pass_counts=np.sum(passed & np.vstack((np.ones((1,len(channel_keys)),dtype=bool),~passed[:-1])),axis=0)
        transitions=np.count_nonzero((values[i,:-1]>0)!=(values[i,1:]>0),axis=0)
        parts.append(RadialPartition(i,e,v,dict(pass_intervals_by_channel=pass_counts.tolist(),
            sampled_transition_count=transitions.tolist(),multiple_pass_intervals=bool(np.any(pass_counts>1)))))
    return RadialSolution(parts,tuple(channel_keys),roots,dict(
        completeness='SAMPLED_BRACKETS_ONLY_NOT_CERTIFIED', max_scan_gap=1/intervals,
        scan_intervals=intervals, maxiter=maxiter, angular_cdf_tolerance=u_tol,
        all_located_roots_converged=True, root_count=len(roots),
        boundary_uncertainty_sum=sum(r['bracket_width'] for r in roots)/base_count,
        uncertainty_scope='Located continuous roots only; excludes missed narrow intervals, callback transport errors and Gauss truncation.',
        callback_batches=evaluator.batches.copy()),evaluator)


def gauss_table(solution, *, order=2):
    """Every cell has full CDF measure; all cases share exactly this table.

    Each base has measure1/M. If channels include separate +/- source states,
    the caller gives each sign probability1/2: this function never double-counts
    signs or chooses a physical survival mask on behalf of later propagation.
    """
    if type(order) is not int or order not in (2,4):
        raise ValueError('Use two or four Gauss nodes in every CDF cell')
    nodes,gw=leggauss(order); count=len(solution.partitions)
    bases=[];cells=[];coordinates=[];weights=[];expected=[]
    for part in solution.partitions:
        widths=np.diff(part.edges)
        if np.any(widths <= 0) or part.edges[0]!=0 or part.edges[-1]!=1:
            raise ValueError('Each radial partition must cover [0,1] with positive cells')
        for j,(lo,width) in enumerate(zip(part.edges[:-1],widths)):
            bases.extend([part.base_index]*order);cells.extend([j]*order)
            coordinates.extend(lo+width*(nodes+1)/2)
            weights.extend(width*gw/(2*count))
            expected.extend([part.midpoint_margins[j]>0]*order)
    table=GaussTable(np.array(bases,dtype=int),np.array(cells,dtype=int),np.array(coordinates),
        np.array(weights),np.asarray(expected),dict(order=order,base_count=count,
            weights='CDF cell width * Gauss probability / base_count; no loss renormalisation',
            sign_probability='External caller multiplies each +/- sign by 1/2',
            boundary_completeness=solution.diagnostics['completeness']))
    mass=np.bincount(table.base_index,weights=table.weight,minlength=count)
    if np.any(table.weight <= 0) or not np.allclose(mass,1/count,atol=8*np.finfo(float).eps/count,rtol=0):
        raise BoundaryResolutionError('Radial quadrature failed source-measure conservation')
    return table


def validate_node_membership(solution,table):
    """One globally batched actual check; reject detected unbracketed structure."""
    margin=solution.evaluator(table.base_index,table.angular_cdf)
    if np.any((margin>=0)!=table.expected_pass):
        raise BoundaryResolutionError('Gauss nodes reveal a missed boundary; refine the actual scan')
    return margin


def integrate_table(table,values):
    values=np.asarray(values,dtype=np.longdouble)
    if values.shape[0]!=len(table.weight) or not np.isfinite(values).all():
        raise ValueError('Integrands must be finite actual values at every table row')
    weights=table.weight.astype(np.longdouble).reshape((-1,)+(1,)*(values.ndim-1))
    return np.asarray(np.sum(weights*values,axis=0),dtype=float)
