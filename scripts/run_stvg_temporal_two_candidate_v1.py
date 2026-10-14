"""One predeclared fast recipe: Native + Projection, sealed teacher responses.

No new teacher, labels, models or prompt changes. Recompute the exact shared
512-parameter Adam head, verify the full recipe first, then the fixed subset.
"""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import BASE,read,write,sha,guard
OLDT=ROOT/'artifacts/stvg_temporal_caption_query_spatial_opd_v1'

def main():
 sys.addaudithook(guard)
 import torch
 from scripts.stvg_temporal_shared_readout_v1 import fit
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 roster=read(OLDT/'ROSTER_LOCK.json')['records'];jobs=read(OLDT/'CAPTION_JOB_LOCK.json')['records']
 seal=read(OLDT/'TEACHER_GLOBAL_BARRIER.json');bound={x['path']:x['sha256'] for x in seal['files']}
 sb=read(OLDT/'STUDENT_GLOBAL_BARRIER.json');studentbound={x['path']:x['sha256'] for x in sb['files']}
 if seal['status']!='complete' or seal['GT_read'] is not False:raise RuntimeError('teacher not sealed')
 outputs=[]
 for r in roster:
  cid=r['case_id'];cp=ROOT/r['source_case_path'];sp=OLDT/'student'/(cid+'.pt')
  if sha(cp)!=r['source_case_sha256'] or sha(sp)!=studentbound[str(sp.relative_to(ROOT))]:raise RuntimeError('source/temporal student changed')
  case=load_source_case(cp);old=torch.load(sp,map_location='cpu',weights_only=False)
  local=[j for j in jobs if j['case_id']==cid];rewards=[];costs=[];bindings=[]
  for j in local:
   paths=[OLDT/'teacher'/f"{cid}_{j['candidate']['canonical_id']}_{suffix}.json" for suffix in ['caption','alignment']]
   for p in paths:
    if sha(p)!=bound[str(p.relative_to(ROOT))]:raise RuntimeError('teacher response bytes changed')
   rewards.append(read(paths[1])['reward']);costs.append(sum(read(p)['total_seconds']for p in paths));bindings.extend(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths)
  full=fit(case,old['temporal_hidden'],[j['candidate']for j in local],rewards)
  if full['final_interval']!=old['temporal_fit']['final_interval']:raise RuntimeError('full temporal reproduction interval differs')
  for x,y in zip(full['path'],old['temporal_fit']['path']):
   if not torch.equal(x['theta'],y['theta']):raise RuntimeError('full temporal reproduction parameters differ')
  idx=[i for i,j in enumerate(local)if j['candidate']['source_candidate_id']in ['time_Native','time_Projection']]
  if len(idx)!=2:raise RuntimeError('fixed two candidates absent or duplicated')
  t=time.perf_counter();fast=fit(case,old['temporal_hidden'],[local[i]['candidate']for i in idx],[rewards[i]for i in idx]);elapsed=time.perf_counter()-t
  dst=BASE/'temporal_fast'/(cid+'.pt');dst.parent.mkdir(exist_ok=True)
  if dst.exists():raise FileExistsError(dst)
  torch.save(dict(case_id=cid,dataset=r['dataset'],full_fit=full,fast_fit=fast,selected_indices=idx,teacher_response_bindings=bindings,source_case_sha256=r['source_case_sha256'],teacher_requests_full=2*len(local),teacher_requests_fast=2*len(idx),saved_completed_teacher_seconds_full=sum(costs),saved_completed_teacher_seconds_subset=sum(costs[i]for i in idx),cost_scope='historical same-input completed teacher call sums; not fresh end-to-end throughput',actual_cpu_fit_seconds=elapsed,GT_read=False),dst)
  outputs.append(dict(case_id=cid,path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size))
 if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA')
 write(BASE/'TEMPORAL_FAST_GLOBAL_BARRIER.json',dict(status='complete',count=8,files=outputs,actual_fast_head_fits=8,full_head_reproduction_count=8,new_teacher_calls=0,CPU_only=True,GT_read=False,time=time.time()))
 print('Eight full-head reproductions and fixed two-candidate head fits sealed')
if __name__=='__main__':main()
