"""Measured scoped payload bound; no cleanup or old-file deletion."""
from common147 import *
import shutil

def storage147():
 conf=rd(H/'experiment/DESIGN_LOCK.json')['resource'];unit=1024**3;used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());free=shutil.disk_usage(R).free;cap=conf['new_private_payload_cap_GiB'];floor=conf['free_floor_GiB'];safety=conf['safety_GiB'];assert used<=cap*unit,('new_payload_cap',used);assert free>=floor*unit,('free_floor',free);assert free+used>=(floor+cap+safety)*unit,('remaining_scope_room',free,used);return used,free
