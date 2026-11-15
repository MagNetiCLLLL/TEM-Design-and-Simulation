"""Bound all installed multipole current packs and co-located winding gaps."""
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import re
import tomllib

from temsim.cad_mesh import _manifold
from temsim.cad_multipole_winding import multipole_winding_solids
from temsim.multipole_winding import (
    TURN_FIELD,is_primary_multipole,multipole_winding_envelopes,supported_multipole_winding,
)
from temsim.transverse_winding import transverse_winding_envelopes

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def cylindrical_bounds(coils):
    if coils[0]['family']=='saddle':
        return (min(c['inner_radius_mm'] for c in coils),max(c['outer_radius_mm'] for c in coils),
                min(c['start_z_mm'] for c in coils),max(c['end_z_mm'] for c in coils))
    return (min(c['radial_start_mm'] for c in coils),
            max(math.hypot(c['radial_end_mm'],c['outer_width_mm']/2) for c in coils),
            min(c['center_z_mm']-c['outer_height_mm']/2 for c in coils),
            max(c['center_z_mm']+c['outer_height_mm']/2 for c in coils))


def main():
    documents=[];records=[];pairs=[];dipoles=[]
    for path in sorted((ROOT/'configs/instruments/column').glob('C3*.toml')):
        raw=path.read_bytes();document=tomllib.loads(raw.decode('utf-8-sig'))
        hosts=[]
        for part in document['parts']:
            if not supported_multipole_winding(part):continue
            coils=multipole_winding_envelopes(part);bounds=cylindrical_bounds(coils)
            angles=sorted(c['azimuth_deg']%360 for c in coils)
            separation=min(b-a for a,b in zip(angles,angles[1:]+[angles[0]+360]))
            half=max(math.degrees(math.atan2(c['outer_width_mm']/2,c['radial_start_mm'])) for c in coils)
            clearance=separation-2*half
            assert clearance>0
            records.append(dict(path=path.relative_to(ROOT).as_posix(),component=part['key'],
                cylindrical_bounds_mm=bounds,minimum_angular_pack_gap_deg=clearance,
                source_snapshot_sha256=sha256(raw).hexdigest()))
            hosts.append((part,coils,bounds))
        for (a,ca,ba),(b,cb,bb) in combinations(hosts,2):
            axial_overlap=min(ba[3],bb[3])-max(ba[2],bb[2])
            if axial_overlap<=0:continue
            gap=max(ba[0],bb[0])-min(ba[1],bb[1])
            assert gap>0,(path,a['key'],b['key'],gap)
            pairs.append(dict(path=path.relative_to(ROOT).as_posix(),components=[a['key'],b['key']],
                axial_pack_overlap_mm=axial_overlap,global_radial_pack_gap_mm=gap,
                rectangular_corner_included=True))
        # Co-located DPH windings stay radially inside the primary HP windings.
        # This is winding-pack clearance, not whole-instrument manufacturability.
        for part,coils,bounds in hosts:
            if not is_primary_multipole(part):continue
            for other in document['parts']:
                if not re.search(r'_dph[12]_coils$',other['key']):continue
                dph=transverse_winding_envelopes(other)
                if not dph:continue
                db=cylindrical_bounds(dph)
                if min(bounds[3],db[3])<=max(bounds[2],db[2]):continue
                gap=bounds[0]-db[1]
                assert gap>0,(path,part['key'],other['key'],gap)
                dipoles.append(dict(path=path.relative_to(ROOT).as_posix(),
                    multipole=part['key'],deflector=other['key'],radial_winding_pack_gap_mm=gap))
        if hosts:documents.append((path,document))
    assert len(records)==40 and len(pairs)==4
    path,document=next((p,d) for p,d in documents if p.name=='C3_ImageCorrector.toml')
    rows={p['key']:p for p in document['parts']}
    q=dict(rows['image_qpol_quadrupole']);h=dict(rows['image_hpol_hexapole'])
    q[TURN_FIELD]=h[TURN_FIELD]=63
    qa=multipole_winding_solids(q)[0];ha=multipole_winding_solids(h)[0]
    qm,hm=_manifold(qa.shape,16),_manifold(ha.shape,16)
    intersection=(qm^hm).volume()
    assert intersection==0.
    receipt=dict(status='passed',scope='Analytical complete-pack radial/angular/axial bounds and one bounded metal intersection; not whole-instrument manufacturing validation',
        installed_multipole_hosts=len(records),co_located_multipole_pairs=len(pairs),
        minimum_co_located_pack_gap_mm=min(p['global_radial_pack_gap_mm'] for p in pairs),
        bounded_metal_check=dict(components=[q['key'],h['key']],turns_per_coil=63,
            tested_wire_index=1,tested_coil_index=1,angular_segments=16,
            individual_volume_mm3=[qm.volume(),hm.volume()],intersection_volume_mm3=intersection),
        all_installed_host_bounds=records,co_located_pack_pairs=pairs,HP_DPH_winding_pack_pairs=dipoles,
        particle_or_wave_executed=False)
    output=HERE/'validation/winding-clearance.json'
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in {'all_installed_host_bounds','co_located_pack_pairs','HP_DPH_winding_pack_pairs'}}))


if __name__=='__main__':main()
