"""Opt-in, executed finite source-image regions; never exact common point foci.

The historical three-adjacent-plane counter is intentionally unchanged.
This observer only computes centered moments; physical phase arrays and their
original weights remain unchanged. It neither propagates nor adjusts a lens.
"""
from __future__ import annotations
import numpy as np


def _transitions(values, noise):
    signs=np.where(values>noise,1,np.where(values < -noise,-1,0))
    known=np.flatnonzero(signs)
    return [dict(left=int(a),right=int(b),before=int(signs[a]),after=int(signs[b]))
            for a,b in zip(known[:-1],known[1:]) if signs[a]!=signs[b]]


def measure_finite_waist_regions(checkpoints,masks,weights,lower_mm,upper_mm,*,
                               maximum_spacing_mm,source_ids=None,minimum_rays=5):
    """Observe one finite 2D region per isolated radial minimum, or fail closed.

    Search sides are executed planes inside the component interval, limited by
    neighboring radial sign reversals. Both sides must be wider than the same
    measured center in every transverse direction. Csym must be negative
    definite on the left and positive definite on the right. The accepted
    bracket must contain exactly one negative-to-positive transition of each
    sorted Csym eigenvalue and no opposite transition. Directional envelope
    maxima before this bracket are recorded separately, not counted as waists.
    Repeated directional refocusing inside the bracket is ambiguous even when
    the trace has only one minimum.
    """
    report=dict(schema='executed-finite-waist-regions-v1',status='UNRESOLVED',
        regions=[],radial_candidates=[],unresolved=[],
        definition='Finite two-dimensional source-image region with an executed representative plane; no exact common stationary plane is asserted',
        legacy_counter_replaced=False,source_or_field_modified=False)
    try:
        lo,hi,gap=map(float,(lower_mm,upper_mm,maximum_spacing_mm))
        if not np.isfinite([lo,hi,gap]).all() or lo>=hi or gap<=0 or minimum_rays<5:
            raise ValueError('Finite ordered bounds, positive explicit spacing and at least five rays required')
        z=np.asarray(checkpoints.z_mm,float);w0=np.asarray(weights,float);live=np.asarray(masks)
        arrays=[np.asarray(getattr(checkpoints,k),float) for k in ('x_m','y_m','tx_rad','ty_rad')]
        ids=np.arange(len(w0)) if source_ids is None else np.asarray(source_ids)
        if (z.ndim!=1 or len(z)<3 or np.any(~np.isfinite(z)) or np.any(np.diff(z)<=0)
                or w0.ndim!=1 or np.any(~np.isfinite(w0)) or np.any(w0<0)
                or not np.isfinite(w0.sum()) or w0.sum()<=0
                or live.dtype.kind!='b' or live.shape!=(len(z),len(w0))
                or any(a.shape!=live.shape for a in arrays)
                or ids.shape!=w0.shape or ids.dtype.kind not in 'iu'
                or len(np.unique(ids))!=len(ids)):
            raise ValueError('Invalid executed phase arrays, masks, original weights or IDs')
        edge_noise=64*np.finfo(float).eps*max(1.,abs(lo),abs(hi))
        if any(not np.any(abs(z-edge)<=edge_noise) for edge in (lo,hi)):
            raise ValueError('Exact component interval boundaries must be observed')
        selected=np.flatnonzero((z>=lo-edge_noise)&(z<=hi+edge_noise));zz=z[selected]
        if len(zz)<3 or np.max(np.diff(zz))>gap+edge_noise:
            raise ValueError('Executed observation spacing exceeds the explicit bound')
        lm=live[selected]&(w0>0)[None,:]
        if np.any(~lm[:-1]&lm[1:]):raise ValueError('Stopped positive-current particle reappears')
        if np.any(lm!=lm[0]):raise ValueError('Population changes inside the component interval; do not merge across interception')
        cohort=lm[0]
        if cohort.sum()<minimum_rays:raise ValueError('Insufficient positive-current support')
        phase=np.stack([a[np.ix_(selected,cohort)] for a in arrays],axis=-1)
        if np.any(~np.isfinite(phase)):raise ValueError('Nonfinite current-carrying coordinates')
        weights_local=w0[cohort];w=weights_local/weights_local.sum()
        centered=phase-np.einsum('n,kni->ki',w,phase)[:,None,:]
        r,t=centered[:,:,:2],centered[:,:,2:]
        R=np.einsum('n,kni,knj->kij',w,r,r)
        rt=np.einsum('n,kni,knj->kij',w,r,t);C=.5*(rt+rt.transpose(0,2,1))
        T=np.einsum('n,kni,knj->kij',w,t,t)
        re=np.linalg.eigvalsh(R);ce=np.linalg.eigvalsh(C)
        eps=64*np.finfo(float).eps
        # Centered products can cancel at a true waist. Bound round-off with
        # the rotation-invariant Cauchy scale, not the small cancelled result.
        moment_scale=np.sqrt(np.trace(R,axis1=1,axis2=2)*np.trace(T,axis1=1,axis2=2))
        cn=eps*np.maximum.reduce([np.max(abs(ce),axis=1),moment_scale,
                                 np.full(len(zz),np.finfo(float).tiny)])
        trace=np.trace(C,axis1=1,axis2=2)
        transitions=_transitions(trace,2*cn)
        radial=[(j,row) for j,row in enumerate(transitions) if row['before']==-1 and row['after']==1]
        def directional_changes(left, right):
            result=[]
            for axis in range(2):
                changes=_transitions(ce[left:right+1,axis],cn[left:right+1])
                result.append(dict(eigenvalue_order=axis,
                    converging_to_diverging=sum(row['before']==-1 for row in changes),
                    diverging_to_converging=sum(row['before']==1 for row in changes),
                    transition_brackets_mm=[[float(zz[left+row['left']]),float(zz[left+row['right']])] for row in changes]))
            return result
        report.update(bounds_mm=[lo,hi],maximum_observation_gap_mm=float(np.max(np.diff(zz))),
            requested_maximum_spacing_mm=gap,checkpoint_count=len(zz),
            source_ids=ids[cohort].tolist(),original_weights=weights_local.tolist(),
            original_source_current_fraction=float(weights_local.sum()/w0.sum()),
            radial_sign_transition_count=len(transitions),
            interval_directional_sign_transitions=directional_changes(0,len(zz)-1))
        # A radial minimum exactly on a component boundary cannot be assigned
        # to either open interval. Keep it ambiguous rather than dropping it.
        if ((abs(trace[0])<=2*cn[0] and trace[1]>2*cn[1]) or
                (trace[-2]<-2*cn[-2] and abs(trace[-1])<=2*cn[-1])):
            report['unresolved'].append(dict(reason='radial_stationarity_on_component_boundary'))
        for transition_index,candidate in radial:
            a,b=candidate['left'],candidate['right']
            center=a+int(np.argmin(np.trace(R[a:b+1],axis1=1,axis2=2)))
            basin_lo=transitions[transition_index-1]['right'] if transition_index else 0
            basin_hi=transitions[transition_index+1]['left'] if transition_index+1<len(transitions) else len(zz)-1
            fraction=-trace[a]/(trace[b]-trace[a])
            root=float(zz[a]+fraction*(zz[b]-zz[a]))
            base=dict(radial_bracket_mm=[float(zz[a]),float(zz[b])],
                radial_root_interpolated_mm=root,radial_root_role='Diagnostic interpolation, not an executed point focus',
                representative_z_mm=float(zz[center]),search_basin_mm=[float(zz[basin_lo]),float(zz[basin_hi])])
            report['radial_candidates'].append(base)
            base=dict(base,basin_directional_sign_transitions=directional_changes(basin_lo,basin_hi))
            if b-a>2:
                report['unresolved'].append(dict(base,reason='extended_radial_stationary_plateau'))
                continue
            def growth(index):
                eigen=np.linalg.eigvalsh(R[index]-R[center])
                noise=eps*max(float(np.max(re[index])),float(np.max(re[center])),np.finfo(float).tiny)
                return eigen, bool(np.all(eigen>noise))
            left=next((i for i in range(center-1,basin_lo-1,-1)
                       if zz[i]>lo+edge_noise and np.all(ce[i]<-cn[i]) and growth(i)[1]),None)
            right=next((i for i in range(center+1,basin_hi+1)
                        if zz[i]<hi-edge_noise and np.all(ce[i]>cn[i]) and growth(i)[1]),None)
            if left is None or right is None:
                report['unresolved'].append(dict(base,reason='no_bounded_all_direction_covariance_growth_bracket'))
                continue
            # Any other radial sign reversal lies outside the basin by
            # construction; do not absorb it by choosing a distant wider side.
            bracket=[float(zz[left]),float(zz[right])]
            directional=directional_changes(left,right)
            base=dict(base,directional_sign_transitions=directional)
            if any(row['converging_to_diverging']!=1 or row['diverging_to_converging'] for row in directional):
                report['unresolved'].append(dict(base,reason='missing_or_repeated_directional_transition'))
                continue
            spans=[bound for row in directional for bound in row['transition_brackets_mm']]
            if any(bound[0]<bracket[0] or bound[1]>bracket[1] for bound in spans):
                report['unresolved'].append(dict(base,reason='directional_transition_not_contained_in_selected_region'))
                continue
            report['regions'].append(dict(base,status='EXECUTED_FINITE_2D_WAIST_REGION',
                z_mm=float(zz[center]),bracket_mm=bracket,
                finite_region_span_mm=bracket[1]-bracket[0],
                directional_transition_span_mm=[min(s[0] for s in spans),max(s[1] for s in spans)],
                executed_indices=[int(selected[left]),int(selected[center]),int(selected[right])],
                rms_radius_nm=float(np.sqrt(np.trace(R[center]))*1e9),
                bracket_rms_radius_nm=(np.sqrt(np.trace(R[[left,center,right]],axis1=1,axis2=2))*1e9).tolist(),
                bracket_position_covariances_m2=R[[left,center,right]].tolist(),
                bracket_symmetric_position_slope_covariances_m_rad=C[[left,center,right]].tolist(),
                bracket_covariance_eigenvalues_m_rad=ce[[left,center,right]].tolist(),
                bracket_covariance_growth_m2=[growth(left)[0].tolist(),growth(right)[0].tolist()],
                representative_slope_covariance_rad2=T[center].tolist(),
                ray_count=int(cohort.sum()),source_current_fraction=float(weights_local.sum()/w0.sum()),
                kind='current-weighted finite two-dimensional source-image region',
                exact_common_stationary_plane_claimed=False))
        report['status']='UNRESOLVED' if report['unresolved'] else 'RESOLVED_AT_BOUNDED_SAMPLING'
        return report
    except (AttributeError,TypeError,ValueError,FloatingPointError) as error:
        report['unresolved'].append(dict(reason='invalid_or_incomplete_finite_region_evidence',message=str(error)))
        return report
