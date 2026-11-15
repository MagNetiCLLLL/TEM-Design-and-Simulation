"""Render actual installed wire counts with clearly labelled display cutaways."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from temsim.cad_mesh import meshes_from_solids
from temsim.cad_transverse_winding import transverse_winding_solids
from temsim.module_manifest import read_document
from temsim.transverse_winding import transverse_winding_definition

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
rows={p['key']:p for p in read_document(ROOT/'configs/instruments/column/C3_ProbeCorrector.toml')['parts']}
fig=plt.figure(figsize=(12,6.2));fig.subplots_adjust(left=.03,right=.97,bottom=.15,top=.83,wspace=.08)
for column,key in enumerate(('probe_dp22_coils','condenser_stigmator'),1):
    part=rows[key];definition=transverse_winding_definition(part)
    specs=transverse_winding_solids(part)
    if column==2:
        # Show one complete radial coil, not a reduced turn count. Other seven
        # coils remain present in the saved CAD and complete-default receipts.
        specs=tuple(s for s in specs if s.parameters['coil_index']==1)
    meshes=meshes_from_solids(specs,8)
    ax=fig.add_subplot(1,2,column,projection='3d')
    for spec,mesh in zip(specs,meshes):
        vertices=mesh.vertices.copy();vertices[:,2]-=part['local_center_z_mm']
        if column==2:
            vertices[:,0]-=(definition['coils'][0]['radial_start_mm']+definition['coils'][0]['radial_end_mm'])/2
        triangles=vertices[mesh.faces]
        if spec.material=='insulator':
            # Remove only the near-side insulation from the illustration so
            # buried conductors are readable. Source geometry is unchanged.
            triangles=triangles[triangles[:,:,1].mean(axis=1)>0]
        if not len(triangles):continue
        color=(.83,.43,.20,.90) if spec.material=='copper_winding' else (.70,.82,.90,.08)
        ax.add_collection3d(Poly3DCollection(triangles,facecolor=color,edgecolor='none',rasterized=True))
    if column==1:
        ax.set(xlim=(-13,13),ylim=(-13,13),zlim=(-8,8));ax.set_box_aspect((26,26,16))
        ax.set_title(f'DP22: all 4 saddle coils\n{definition["turn_count_per_coil"]} turns/coil; {definition["total_turn_count"]} total turns',fontsize=11)
        ax.view_init(elev=28,azim=-53)
    else:
        ax.set(xlim=(-4.5,4.5),ylim=(-5,5),zlim=(-11,11));ax.set_box_aspect((9,10,22))
        ax.set_title(f'Condenser stigmator: one complete radial coil\n{definition["turn_count_per_coil"]} turns shown; 8 coils / {definition["total_turn_count"]} total in CAD',fontsize=11)
        ax.view_init(elev=20,azim=-50)
    ax.set_xlabel('Local X (mm)',fontsize=9);ax.set_ylabel('Local Y (mm)',fontsize=9);ax.set_zlabel('Z offset (mm)',fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle('Installed fine-wire engineering CAD: closed conductors inside fixed winding packs',fontsize=13)
fig.text(.5,.055,'Display cutaway of insulation only; no physical turn count was reduced. Dimensions are engineering assumptions, not OEM measurements.',
         ha='center',fontsize=9)
destination=HERE/'validation/installed-fine-wire-cad.png'
fig.savefig(destination,dpi=150,bbox_inches='tight');plt.close(fig)
print(destination)
