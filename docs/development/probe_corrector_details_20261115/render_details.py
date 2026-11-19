"""Display-only cutaway comparison from the complete native CAD meshes."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from temsim.cad_mesh import meshes_from_solids
from temsim.cad_multipoles import multipole_solids
from temsim.cad_probe_winding import TURN_FIELD,AREA_FIELD,STATUS_FIELD,NOTE_FIELD,probe_winding_dimensions
from temsim.module_manifest import read_document

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/probe-corrector-details-20261115'


def main():
    document=read_document(ROOT/'configs/instruments/column/C3_ProbeCorrector.toml')
    parts={p['key']:p for p in document['parts']};original=deepcopy(parts['probe_hp2_hexapole'])
    example=deepcopy(original)
    example.update({TURN_FIELD:6,AREA_FIELD:.10,STATUS_FIELD:'engineering_reconstruction_not_oem',
                    NOTE_FIELD:'Explicit illustrative CAD rendering only; not default or measured winding data'})
    records=[];fig=plt.figure(figsize=(13,6))
    fig.subplots_adjust(left=.03,right=.94,bottom=.12,top=.80,wspace=.05)
    for index,(label,row) in enumerate((('Default: bulk coils and outer insulation',original),
                                      ('Illustration: 6 turns per coil, metal area 0.10 mm2',example))):
        specs=multipole_solids(row,parts);meshes=meshes_from_solids(specs,angular_segments=24)
        ax=fig.add_subplot(1,2,index+1,projection='3d')
        for spec,mesh in zip(specs,meshes,strict=True):
            vertices=np.array(mesh.vertices,copy=True);vertices[:,2]-=row['local_center_z_mm']
            triangles=vertices[np.asarray(mesh.faces)]
            triangles=triangles[triangles[:,:,1].mean(axis=1)<=0]
            if not len(triangles):continue
            alpha=(1. if spec.material=='copper_winding' else .16 if spec.region=='return_yoke'
                   else .80 if spec.material=='soft_magnetic' else .045)
            color=(*spec.color[:3],alpha)
            ax.add_collection3d(Poly3DCollection(triangles,facecolor=color,edgecolor='none',rasterized=True))
        ax.set(xlim=(-39,39),ylim=(-39,39),zlim=(-10,10),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z offset (mm)')
        ax.set_box_aspect((78,78,25));ax.view_init(elev=30,azim=-65)
        ax.set_title(label,fontsize=10,pad=12);ax.tick_params(labelsize=8)
        records.append(dict(label=label,material_regions=len(specs),
                            turn_regions=sum(s.region.startswith('winding_turn_') for s in specs),
                            full_solids=[asdict(s) for s in specs]))
    fig.suptitle('HP2 internal CAD: radial windings, original poles/yoke and fixed host envelope',fontsize=13)
    fig.text(.5,.015,'Transparent half cutaway for display only. Complete CAD is retained; actual HP turn/area defaults remain undefined.',
             ha='center',fontsize=9)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'probe-internal-comparison.png',dpi=170,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    (OUT/'illustrative-cad-inputs.json').write_text(json.dumps(dict(
        scope='Illustrative CAD only; default HP turns, conductor area and current remain undefined',
        default_region_count=31,example_turns_per_coil=6,example_metal_area_mm2=.10,
        example_witness=probe_winding_dimensions(example),records=records),indent=2)+'\n',encoding='utf-8')
    assert parts['probe_hp2_hexapole']==original


if __name__=='__main__':main()
