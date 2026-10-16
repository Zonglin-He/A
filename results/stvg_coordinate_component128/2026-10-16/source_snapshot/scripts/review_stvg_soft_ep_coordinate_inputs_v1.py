"""Complete job/output and marked-array custody readback, including invalid/empty jobs."""
from pathlib import Path
import sys,json,re,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def main():
 import numpy as np
 verify_runtime();assembly=read(BASE/'ROOT_INPUT_ASSEMBLY.json');assert assembly['count']==128
 groups=collections.defaultdict(list)
 for r in roster():groups[r.get('cached_parent',str((BASE/'confirmation').relative_to(ROOT)))].append(r['probe_alias'])
 total=valid=invalid=freshjobs=arrays=atcap=valid_atcap=0;bindings=[]
 for rel,ids in groups.items():
  parent=ROOT/rel;jobsdoc=read(parent/'TEACHER_JOB_LOCK.json');jobs=jobsdoc['records'];bar=read(parent/'TEACHER_GLOBAL_BARRIER.json');assert jobsdoc['status']=='locked'and bar['status']=='complete'and not jobsdoc['GT_read']and not bar['GT_read'];assert len(jobs)==len({j['job_id']for j in jobs})==bar['count']==len(bar['files'])
  expected={str((parent/'teacher'/(j['job_id']+'.json')).relative_to(ROOT))for j in jobs};actual={x['path']:x for x in bar['files']};assert set(actual)==expected
  for x in bar['files']:assert sha(ROOT/x['path'])==x['sha256']
  selected=[j for j in jobs if j['case_id']in ids and j['variant']=='new_crop'];inp=read(parent/'CANDIDATE_INPUT_BARRIER.json');assert inp['status']=='complete'and not inp['GT_read'];inf={x['path']:x for x in inp['files']};assert inp['teacher_job_count']==len(jobs)
  records={x['case_id']:x for x in inp['case_records']}
  for cid in ids:
   rec=records[cid];p=ROOT/rec['path'];assert sha(p)==rec['sha256'];detail=read(p);assert detail['case_id']==cid
   casejobs=[j for j in selected if j['case_id']==cid];unknown=detail.get('unknown_intervals',{}).get('new')
   if unknown:assert unknown['status']=='unknown'and unknown['reason'].startswith('unknown_')and not casejobs
   else:assert len(casejobs)==4 and {j['center_slot']for j in casejobs}==set(range(4))and {j['job_id']for j in casejobs}=={f'{cid}_new_crop_{i}'for i in range(4)}
  for j in selected:
   p=parent/'teacher'/(j['job_id']+'.json');z=read(p);assert not z['GT_read']and z['prompt_sha256']==j['prompt_sha256']and hashlib.sha256(j['prompt'].encode()).hexdigest()==j['prompt_sha256']
   assert z['source_case_sha256']==j['source_case_sha256']
   try:
    raw=re.sub(r'^```(?:json)?\s*','',z['raw_output'].strip());raw=re.sub(r'\s*```$','',raw);parsed=json.loads(raw);ident=parsed['id'];assert isinstance(ident,int)and not isinstance(ident,bool)and ident in j['allowed_output_ids'];status='valid';chosen=j['visible_id_to_candidate'].get(str(ident))
   except Exception:status='invalid';chosen=None
   assert z['parsed']['status']==status and z['parsed']['selected_candidate_id']==chosen;total+=1;valid+=status=='valid';invalid+=status=='invalid';atcap+=bool(z.get('output_truncated_at_cap'));valid_atcap+=bool(z.get('output_truncated_at_cap'))and status=='valid'
   if parent==BASE/'confirmation':
    freshjobs+=1;assert z['new_teacher_call'] and z['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
    for v in j['videos']:
     path=ROOT/v['path'];assert v['path']in inf and sha(path)==v['sha256']==inf[v['path']]['sha256'];a=np.load(path,allow_pickle=False);assert a.shape in [(4,3,448,448),(5,3,448,448)]and a.dtype==np.uint8 and hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256'];arrays+=1
  bindings.append(dict(parent_reference_sha256=sha(parent/'REFERENCE_LOCK.json'),teacher_job_lock_sha256=sha(parent/'TEACHER_JOB_LOCK.json'),teacher_global_barrier_sha256=sha(parent/'TEACHER_GLOBAL_BARRIER.json'),selected_case_count=len(ids),selected_job_count=len(selected),selected_job_set_sha256=hashlib.sha256(json.dumps(sorted(j['job_id']for j in selected)).encode()).hexdigest()))
 assert total==assembly['teacher_job_count']
 write(BASE/'ROOT_COMPLETE_INPUT_READBACK.json',dict(status='pass',count=128,all_expected_teacher_outputs_including_invalid_and_source_preserved=True,selected_job_count=total,valid_count=valid,invalid_count=invalid,at_token_cap_count=atcap,valid_complete_JSON_at_cap_count=valid_atcap,at_cap_policy='same locked legacy parser; complete valid JSON remains valid and cap flag disclosed, no posthoc reference change',actual_fresh_job_count=freshjobs,actual_fresh_marked_array_readback_count=arrays,bindings=bindings,reader_sha256=sha(__file__),scope='full job/output SHA set; independent raw ID parser; exact marked NPY file/tensor/shape custody; not a semantic identity guarantee',GT_read=False,CPU_only=True))
 print('input128 full coverage',total,valid,invalid,'fresh',freshjobs,'arrays',arrays)
if __name__=='__main__':main()
