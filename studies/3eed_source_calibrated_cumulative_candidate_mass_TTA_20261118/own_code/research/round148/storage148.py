"""Scoped disk safety, with no old data/results/weights deletion."""
from common148 import *
import shutil

def storage148():
 c=rd(H/'experiment/DESIGN_LOCK.json')['resource'];unit=1024**3;used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());free=shutil.disk_usage(R).free;assert used<=c['payload_cap_GiB']*unit,('payload_cap',used);assert free>=c['free_floor_GiB']*unit,('free_floor',free);assert free+used>=(c['free_floor_GiB']+c['payload_cap_GiB']+c['safety_GiB'])*unit,('scope_room',free,used);return used,free
