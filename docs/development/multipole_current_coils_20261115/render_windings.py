"""Render every installed turn centre line in representative multipole CAD."""
from pathlib import Path
import math
import tomllib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from temsim.multipole_winding import multipole_winding_definition

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def curve(coil,radial,delta,reference):
    width,height=coil['window_width_mm'],coil['window_height_mm']
    sections=[]
    for cy,cz,angle in ((width/2,-height/2,-np.pi/2),(width/2,height/2,0.),
                        (-width/2,height/2,np.pi/2),(-width/2,-height/2,np.pi)):
        t=np.linspace(angle,angle+np.pi/2,7)
        sections.append(np.column_stack((cy+delta*np.cos(t),cz+delta*np.sin(t))))
    yz=np.concatenate(sections);yz=np.vstack((yz,yz[0]))
    az=math.radians(coil['azimuth_deg']);r=coil['radial_start_mm']+radial
    return np.column_stack((r*math.cos(az)-yz[:,0]*math.sin(az),
        r*math.sin(az)+yz[:,0]*math.cos(az),yz[:,1]+coil['center_z_mm']-reference))


def main():
    path=ROOT/'configs/instruments/column/C3_ProbeCorrector.toml'
    original=path.read_bytes()
    rows={p['key']:p for p in tomllib.loads(original.decode('utf-8-sig'))['parts']}
    figure=plt.figure(figsize=(15,6.1));figure.subplots_adjust(left=.02,right=.94,bottom=.13,top=.70,wspace=.06)
    for index,(key,name) in enumerate((('probe_hp1_hexapole','HP1'),('probe_qpol_quadrupole','QPol'),
                                      ('probe_hpol_hexapole','HPol')),1):
        part=rows[key];d=multipole_winding_definition(part)
        ax=figure.add_subplot(1,3,index,projection='3d')
        reference=part['optical_reference_local_z_mm'];drawn=0
        for group,colour in (('x','#c55d20'),('y','#6d59a3')):
            paths=[]
            for coil,positions in zip(d['coils'],d['turn_positions']):
                if coil['group']!=group:continue
                for radial,delta in positions:
                    paths.append(curve(coil,radial,delta,reference));drawn+=1
            if paths:ax.add_collection3d(Line3DCollection(paths,colors=colour,linewidths=.65 if index==1 else .25,alpha=.85))
        assert drawn==d['total_turn_count']
        radius=max(c['radial_end_mm'] for c in d['coils'])+1
        height=max(c['outer_height_mm'] for c in d['coils'])/2+1
        ax.set(xlim=(-radius,radius),ylim=(-radius,radius),zlim=(-height,height),
            xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z from reference (mm)')
        ax.set_box_aspect((2*radius,2*radius,2*height));ax.view_init(elev=27,azim=-51);ax.tick_params(labelsize=8)
        if index==1:
            detail=f'6 coils; {d["turn_count_per_coil"]} turns/coil\nAngular datum {d["basis_angle_deg"]:.4f} deg'
        else:
            detail=f'{d["coil_count"]} coils; {d["turn_count_per_coil"]} turns/coil\nTwo independent current groups'
        ax.set_title(name+'\n'+detail,fontsize=11,pad=12)
    figure.suptitle('Installed ideal-current multipole winding CAD: complete saved turn centre lines',fontsize=15,y=.97)
    figure.text(.5,.895,'HP1 retains its physical N=2 per coil; auxiliary QPol/HPol use explicit fine-wire engineering estimates.',ha='center',fontsize=11,color='#475569')
    figure.text(.5,.035,'All saved turns shown. Insulation and passive supports hidden for clarity; no CAD turns are omitted.\n'
        'Engineering reconstruction, not OEM geometry. Existing transport boundaries and optical controls remain unchanged.',
        ha='center',fontsize=10,color='#64748b',linespacing=1.5)
    output=HERE/'validation/installed-multipole-windings.png'
    figure.savefig(output,dpi=150,bbox_inches='tight');plt.close(figure)
    assert path.read_bytes()==original
    print(output)


if __name__=='__main__':main()
