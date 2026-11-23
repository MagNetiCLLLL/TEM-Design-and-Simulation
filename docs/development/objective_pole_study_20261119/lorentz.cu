// Double precision stationary relativistic Lorentz integration in laboratory coordinates.
#ifndef CORRECTOR_FIELD_PARAMETERS
#define CORRECTOR_FIELD_PARAMETERS
#define CORRECTOR_FIELD_ARGUMENTS
#endif

struct Ray { double x,y,tx,ty,time; };

__device__ Ray advance(Ray p, Ray derivative, double h) {
    return {p.x+h*derivative.x, p.y+h*derivative.y, p.tx+h*derivative.tx,
            p.ty+h*derivative.ty, p.time+h*derivative.time};
}

__device__ Ray derivative(Ray p, int node, const double *field, const double *electric,
                          const double *r2, int nr, double energy, double phi0,
                          double rest, double factor, double reference_pc CORRECTOR_FIELD_PARAMETERS) {
    const double *f=field+20*node;
    double x=p.x,y=p.y,s=x*x+y*y;
    double bz=f[0]+s*(-f[2]/4+s*(f[4]/64+s*(-f[6]/2304+s*(f[8]/147456-s*f[10]/14745600))));
    double radial=-f[1]/2+s*(f[3]/16+s*(-f[5]/384+s*(f[7]/18432+s*(-f[9]/1474560+s*f[11]/176947200))));
    double n=f[12]-s*f[16]/16, skew=f[13]-s*f[17]/16;
    double pol=y*(x*x-y*y/3),quad=-x*(x*x/3-y*y);
    radial-=(pol*f[16]+quad*f[17])/8;
    double bx=x*radial+2*n*x*y-skew*(x*x-y*y);
    double by=y*radial+n*(x*x-y*y)+2*skew*x*y;
    bz+=pol*(f[14]-s*f[18]/16)+quad*(f[15]-s*f[19]/16);
    int lo=0,hi=nr-1;
    while(hi-lo>1) { int mid=(lo+hi)/2; if(r2[mid]<=s) lo=mid; else hi=mid; }
    const double *ep=electric+(node*(nr-1)+lo)*4;
    double sr=s-r2[lo], kin=energy+(ep[0]+sr*ep[1]-phi0);
    double ez=ep[2]+sr*ep[3], pc=sqrt(kin*(kin+2*rest));
    double tx=p.tx,ty=p.ty,norm2=1+tx*tx+ty*ty,norm=sqrt(norm2);
    double ga=-2*factor*(reference_pc/pc)*norm, ea=-(kin+rest)*norm2/(pc*pc);
    return {tx,ty,ga*(ty*bz-(1+tx*tx)*by+tx*ty*bx)+ea*(-2*x*ep[1]-tx*ez),
            ga*((1+ty*ty)*bx-tx*bz-tx*ty*by)+ea*(-2*y*ep[1]-ty*ez),
            norm*(kin+rest)/(299792458.*pc)};
}

extern "C" __global__ void trace(double *phase, const double *energy, const double *phi0,
        const double *fields, const double *electric, const double *r2, int nr,
        const double *h, const double *radius2, int *loss, double *peak_r2,
        double *minimum_margin, int first, int last, int count, double rest, double factor, double ref_pc
        CORRECTOR_FIELD_PARAMETERS) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count || loss[j]!=-1) return;
    double *a=phase+5*j;
    Ray p={a[0],a[1],a[2],a[3],a[4]};
    double peak=peak_r2[j], margin=minimum_margin[j];
    for(int i=first;i<last;i++) {
        int node=2*i; double step=h[i];
        Ray k1=derivative(p,node,fields,electric,r2,nr,energy[j],phi0[j],rest,factor,ref_pc CORRECTOR_FIELD_ARGUMENTS);
        Ray p2=advance(p,k1,step/2);
        Ray k2=derivative(p2,node+1,fields,electric,r2,nr,energy[j],phi0[j],rest,factor,ref_pc CORRECTOR_FIELD_ARGUMENTS);
        Ray p3=advance(p,k2,step/2);
        Ray k3=derivative(p3,node+1,fields,electric,r2,nr,energy[j],phi0[j],rest,factor,ref_pc CORRECTOR_FIELD_ARGUMENTS);
        Ray p4=advance(p,k3,step);
        Ray k4=derivative(p4,node+2,fields,electric,r2,nr,energy[j],phi0[j],rest,factor,ref_pc CORRECTOR_FIELD_ARGUMENTS);
        Ray next=p;
        next=advance(next,k1,step/6); next=advance(next,k2,step/3);
        next=advance(next,k3,step/3); next=advance(next,k4,step/6);
        double endpoints=fmax(p.x*p.x+p.y*p.y,next.x*next.x+next.y*next.y);
        double swept=fmax(endpoints,fmax(p2.x*p2.x+p2.y*p2.y,
                    fmax(p3.x*p3.x+p3.y*p3.y,p4.x*p4.x+p4.y*p4.y)));
        peak=fmax(peak,swept);
        if(!isfinite(next.tx+next.ty+next.x+next.y) || swept>=r2[nr-1]) {loss[j]=-2;break;}
        if(endpoints>radius2[i]) {loss[j]=i;break;}
        // The native centered-cylinder contact law tests each chord between accepted
        // nodes. Intermediate RK predictors bound clearance but are not accepted states.
        margin=fmin(margin,sqrt(radius2[i])-sqrt(swept)); p=next;
    }
    a[0]=p.x;a[1]=p.y;a[2]=p.tx;a[3]=p.ty;a[4]=p.time;
    peak_r2[j]=peak;minimum_margin[j]=margin;
}
