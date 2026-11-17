"""Scope-sized storage bound; never deletes old artifacts or model/data files."""
from common146 import *
import shutil

def storage146():
 conf=rd(H/'experiment/RESOURCE_ADDENDUM_LOCK.json');assert rd(H/'PREPARATION.json')['backfilled']==166;unit=1024**3;used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());free=shutil.disk_usage(R).free;assert used<=conf['new_private_payload_cap_GiB']*unit,('new_scope_payload_cap',used);assert free>=conf['remaining_disk_floor_GiB']*unit,('disk_floor',free);assert free+used>=(conf['remaining_disk_floor_GiB']+conf['new_private_payload_cap_GiB']+conf['safety_GiB'])*unit,('room_for_remaining_scope_cap',free,used);return used,free
