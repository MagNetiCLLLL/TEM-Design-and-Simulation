"""Time-domain, three-momentum transport to the first annular metal surface."""
import math

import numpy as np
from numba import njit, prange
from scipy.constants import c, e, m_e

from temsim.physics.discrete_gradient import discrete_gradient_update

REST_EV = m_e*c*c/e


@njit(cache=True)
def field_at(x,y,z,field):
    raxis,zaxis,voltage,cut,lookup,origin,inverse,gradient,phi0,core,core_slope = field
    s = x*x+y*y
    i = np.searchsorted(raxis,np.sqrt(s),side='right')-1
    j = np.searchsorted(zaxis,z,side='right')-1
    if i<0 or j<0 or i+1>=len(raxis) or j+1>=len(zaxis):
        raise ValueError('Particle outside solved electric field')
    ds = raxis[i+1]**2-raxis[i]**2
    dz = zaxis[j+1]-zaxis[j]
    u,v = (s-raxis[i]**2)/ds,(z-zaxis[j])/dz
    if cut[i,j] >= 0:
        parent=2*cut[i,j]+int(v>u)
        for slot in range(2):
            it=lookup[parent,slot]
            if it<0:
                continue
            a,b=s-origin[it,0],z-origin[it,1]
            h=inverse[it,0,0]*a+inverse[it,0,1]*b
            k=inverse[it,1,0]*a+inverse[it,1,1]*b
            if min(h,k)>=-1e-9 and h+k<=1+1e-9:
                gs,gz=gradient[it]
                return phi0[it]+gs*a+gz*b,-2*x*gs,-2*y*gs,-gz
        return 0.,0.,0.,0.
    a,b,v0,v1=voltage[i,j],voltage[i+1,j],voltage[i,j+1],voltage[i+1,j+1]
    lo,hi=(1-u)*a+u*b,(1-u)*v0+u*v1
    gs0,gs1=(b-a)/ds,(v1-v0)/ds
    if s<core[j]:
        gs0=core_slope[j]; lo=voltage[0,j]+s*gs0
    if s<core[j+1]:
        gs1=core_slope[j+1]; hi=voltage[0,j+1]+s*gs1
    gs=(1-v)*gs0+v*gs1
    return (1-v)*lo+v*hi,-2*x*gs,-2*y*gs,-(hi-lo)/dz


@njit(cache=True)
def bfield(x,y,z,b):
    # The fit has support on the full +/-8 mm square. Steps inside the metal
    # are stopped on its first surface before it can contribute a trajectory.
    if max(abs(x),abs(y))>.008 or z<-.002 or z>.452:
        raise ValueError('Particle outside recomputed volume-field table')
    z0=-.002; dz=.002
    j=min(int((z-z0)/dz),len(b)-1)
    t=z-(z0+j*dz)
    x=x/.008; y=y/.008
    xp=(1.,x,x*x,x**3,x**4); yp=(1.,y,y*y,y**3,y**4)
    out=np.zeros(3); k=0
    for degree in range(5):
        for a in range(degree+1):
            f=xp[a]*yp[degree-a]
            for w in range(3):
                p=b[j,k,w]
                out[w]+=f*(((p[0]*t+p[1])*t+p[2])*t+p[3])
            k+=1
    return out[0],out[1],out[2]


@njit(cache=True)
def first_hit(x,y,z,dx,dy,dz,parts):
    first=2.; owner=-1
    for i in range(len(parts)):
        za,zb,inner,outer=parts[i]
        if max(z,z+dz)<za or min(z,z+dz)>zb:
            continue
        lo,hi=0.,1.
        if dz!=0.:
            a,b=(za-z)/dz,(zb-z)/dz
            lo,hi=max(0.,min(a,b)),min(1.,max(a,b))
        if lo>hi or lo>=first:
            continue
        xx,yy=x+lo*dx,y+lo*dy
        r2=xx*xx+yy*yy
        if inner*inner<=r2<=outer*outer:
            candidate=lo
        else:
            qa=dx*dx+dy*dy; qb=xx*dx+yy*dy
            if qa==0.:
                continue
            outside=r2>outer*outer
            qc=r2-(outer*outer if outside else inner*inner)
            disc=qb*qb-qa*qc
            if disc<0. or (outside and qb>=0.):
                continue
            root=math.sqrt(disc)
            if outside:
                t=qc/(-qb+root)
            elif qb>=0.:
                t=-qc/(qb+root)
            else:
                t=(-qb+root)/qa
            candidate=lo+t
        if lo<=candidate<=hi and candidate<first:
            first,owner=candidate,i
    return first,owner


@njit(cache=True)
def sag(x,y):
    radius=1e-7; r=np.hypot(x,y); half=5*np.pi/180
    if r < radius*np.cos(half):
        return -r*r/(radius+np.sqrt(max(0.,radius**2-r*r)))
    return -radius*(1-np.sin(half))-(r-radius*np.cos(half))/np.tan(half)


@njit(cache=True)
def momentum_energy(ux,uy,uz):
    u2=ux*ux+uy*uy+uz*uz
    return REST_EV*u2/(1+np.sqrt(1+u2))


@njit(cache=True)
def advance(x,y,z,ux,uy,uz,dt,phi,e0,ef,bf):
    g0=math.sqrt(1+ux*ux+uy*uy+uz*uz)
    a,b,d=ux-e*dt/(m_e*c)*e0[0],uy-e*dt/(m_e*c)*e0[1],uz-e*dt/(m_e*c)*e0[2]
    for _ in range(32):
        total=g0+math.sqrt(1+a*a+b*b+d*d)
        move=c*dt/total
        dx,dy,dz=move*(ux+a),move*(uy+b),move*(uz+d)
        mx,my,mz=x+.5*dx,y+.5*dy,z+.5*dz
        _,ex,ey,ez=field_at(mx,my,mz,ef)
        bx,by,bz=bfield(mx,my,mz,bf)
        phi1=field_at(x+dx,y+dy,z+dz,ef)[0]
        na,nb,nd=discrete_gradient_update(ux,uy,uz,total,dx,dy,dz,ex,ey,ez,bx,by,bz,phi,phi1,dt)
        err=(na-a)**2+(nb-b)**2+(nd-d)**2
        if err<=(5e-11)**2*max(1e-18,ux*ux+uy*uy+uz*uz,na*na+nb*nb+nd*nd):
            return True,dx,dy,dz,na,nb,nd,phi1
        a,b,d=na,nb,nd
    return False,0.,0.,0.,ux,uy,uz,phi


@njit(cache=True,parallel=True,nogil=True)
def transport(phase,ef,bf,parts,z_plot,n_plot,step_scale):
    n=len(phase); state=np.zeros((n,10)); event=np.full(n,-9,np.int32)
    paths=np.full((n_plot,len(z_plot),2),np.nan,np.float32)
    for k in prange(n):
        x,y,z,ux,uy,uz=phase[k]
        phi=field_at(x,y,z,ef)[0]
        ref=momentum_energy(ux,uy,uz)-phi
        clock=0.; peak_error=0.; slowest=1e9; max_radius=0.; obs=0
        for it in range(100000):
            _,ex,ey,ez=field_at(x,y,z,ef)
            g=math.sqrt(1+ux*ux+uy*uy+uz*uz)
            speed=c*math.sqrt(ux*ux+uy*uy+uz*uz)/g
            acc=e/m_e*math.sqrt(ex*ex+ey*ey+ez*ez)
            ds=step_scale*min(9e-6 if z<.002 else 5e-5 if z<.040 else .00022,
                              .045*(1e-7+math.sqrt(x*x+y*y+z*z)))
            dt=2*ds/(speed+math.sqrt(speed*speed+2*acc*ds))
            success=False
            for _ in range(22):
                success,dx,dy,dz,ax,ay,az,phi1=advance(x,y,z,ux,uy,uz,dt,phi,(ex,ey,ez),ef,bf)
                if success:
                    break
                dt*=.5
            if not success:
                event[k]=-8
                break
            peak_error=max(peak_error,abs(momentum_energy(ax,ay,az)-phi1-ref))
            f,hit=first_hit(x,y,z,dx,dy,dz,parts)
            if z+dz<0. and z+dz < sag(x+dx,y+dy)-2e-15:
                left,right=0.,1.
                for _ in range(32):
                    mid=(left+right)/2
                    if z+mid*dz < sag(x+mid*dx,y+mid*dy):right=mid
                    else:left=mid
                if right<f: f,hit=right,-2
            if z+dz>=.45 and dz>0. and (.45-z)/dz<f:
                f,hit=(.45-z)/dz,-1
            final=f<=1.
            frac=min(f,1.)
            if k<n_plot and dz>0.:
                while obs<len(z_plot) and z_plot[obs]<=z+frac*dz:
                    h=(z_plot[obs]-z)/dz
                    if 0<=h<=frac:
                        paths[k,obs,0]=x+h*dx; paths[k,obs,1]=y+h*dy
                    obs+=1
            x+=frac*dx; y+=frac*dy; z+=frac*dz
            clock+=frac*dt
            ux+=frac*(ax-ux); uy+=frac*(ay-uy); uz+=frac*(az-uz)
            max_radius=max(max_radius,np.hypot(x,y)); phi=phi1
            if .0015<=z<=.025: slowest=min(slowest,momentum_energy(ux,uy,uz))
            if final:
                event[k]=hit
                break
        state[k]=np.array([x,y,z,ux,uy,uz,clock,peak_error,slowest,max_radius])
    return event,state,paths
