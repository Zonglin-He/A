from common152 import *
import shutil

def storage152():
 c=rd(H/'experiment/DESIGN_LOCK.json')['resource'];u=1024**3;used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());free=shutil.disk_usage(R).free;assert used<=c['payload_cap_GiB']*u,('payload_cap',used);assert free>=c['free_floor_GiB']*u,('free_floor',free);assert free+used>=(c['free_floor_GiB']+c['payload_cap_GiB']+c['safety_GiB'])*u,('scope_room',free,used);return used,free
