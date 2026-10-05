"""No-GT complete-query metadata and query-only parse preparation."""
import sys,time,os,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_fixed_full_common_v1 import *
def prepare():
 if (BASE/'DESIGN_LOCK.json').exists():return
 assert read(ROOT/'artifacts/decota_transform_p0_v1/FINAL_COMPLETION.json')['status']=='completed_verified_publication'
 selection=ROOT/'artifacts/decota_identity_commitment_v1/SEARCH_SELECTION.json'
 assert read(selection)['arm']=='top1' and not read(selection)['uses_confirmation']
 protected={str(f.relative_to(ROOT)):sha(f) for f in [selection,ROOT/'methods/CURRENT_METHOD.json',ROOT/'artifacts/tastvg_best_full_v1/STATUS.json']}
 inputs={};details={}
 for ds,n,sources in [('vidstg',10303,732),('hc2',3482,237)]:
  origin=ROOT/'artifacts/tastvg_best_full_v1'/ds/'PLAN.json';old=read(origin);inputs[str(origin.relative_to(ROOT))]=sha(origin)
  rows=[{k:v for k,v in r.items() if k!='tuning_source_exposed'} for r in old['rows']]
  assert len(rows)==n and len({r['source'] for r in rows})==sources and old['conditions']==CONDS
  exposed=set()
  for name in ['tastvg_optuna_v1','tastvg_coordinate_tuning_v2','tastvg_extended_sensitivity_v3','tastvg_decota_c1_same_domain_v1','decota_identity_commitment_v1']:
   f=ROOT/'artifacts'/name/ds/'PLAN.json'
   if f.exists():
    x=read(f);exposed.update(r['source'] for r in x['rows']);inputs[str(f.relative_to(ROOT))]=sha(f)
  for i,r in enumerate(rows):
   assert r['ordinal']==i and r['frame_ids']==r['input']['frame_ids'] and Path(r['input']['video_path']).is_file()
   r['development_source_exposed']=r['source'] in exposed
  for seq in old['orders'].values():assert len(seq)==n and sorted(seq)==list(range(n))
  plan=dict(dataset=ds,rows=rows,orders=old['orders'],conditions=CONDS,queries=n,sources=sources,
   arrivals_per_checkpoint=n*12,full_official_query_cohort=True,expert_availability=1.,historically_exposed=True,
   outside_development_sources=len({r['source'] for r in rows}-exposed),cohort_origin_sha256=sha(origin))
  write(BASE/ds/'PLAN.json',plan);details[ds]={k:v for k,v in plan.items() if k not in ['rows','orders']}
  inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))]=sha(BASE/ds/'PLAN.json')
 cfg=dict(version='decota_fixed_frame_top1_online_full_v1',source_selection_sha256=sha(selection),
  native_when=True,observation_scope='zero-update native interval',observation_budget=4,spatial_expert='frozen Grounding DINO tiny',
  critic='existing admitted Frame-Top1 energy',parameters=1792,spatial_lr=.03,Adam_betas=[.9,.999],Adam_eps=1e-8,steps=10,
  selection='first minimum own loss among 0..10',LN_writeback=1/16,query_reset=True,Adam_reset=True,
  LN_reset='checkpoint/target/condition/order',expert_availability=1.,temporal_parameter_updates=False,production_promoted=False)
 write(ROOT/'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json',cfg)
 inputs['methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json']=sha(ROOT/'methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json')
 design=dict(version='decota_fixed_full_corruption_v1',time=time.time(),datasets=details,
  jobs=[dict(job=j,target=ds,source=src) for j,ds,src in JOBS],online_arrivals=330840,unique_frozen_inputs=165420,
  inputs=inputs,protected=protected,config=cfg,GT_inference=False,no_total_deadline=True,
  hc_annotation_metadata_sha256=read(ROOT/'artifacts/tastvg_best_full_v1/DESIGN_LOCK.json')['hc_annotation_metadata_sha256'])
 write(BASE/'DESIGN_LOCK.json',design)
 status(BASE/'STATUS.json',dict(status='cpu_subject_preparation',pid=os.getpid(),GPU_started=False,predictions=0,GT_read=False,time=time.time()))
 archive('完整官方名单/两个旧固定顺序与方法已锁；正在query-only CPU准备，零新预测')

def subjects(ds):
 import torch
 torch.set_num_threads(4)
 from methods.decota_final_simplified_v1.observations import QuerySubjectParser,visual_query,parse_context
 p=read(BASE/ds/'PLAN.json');out=BASE/ds;known={}
 # Query-only predecessor parses, not predecessor predictions or labels.
 for name in ['tastvg_decota_c1_same_domain_v1']:
  old=read(ROOT/'artifacts'/name/ds/'PLAN.json')
  for r in old['rows']:known[r['input']['caption']]=r['parses']
 subject_cache={}
 if ds=='vidstg':
  b=ROOT/'artifacts/tastvg_best_full_v1'/ds
  bar=read(b/'SUBJECT_BARRIER.json')
  for name,h in bar['files'].items():
   f=b/'subjects'/name;assert sha(f)==h;x=read(f);subject_cache[x['caption_sha256']]=x['parses']
 parser=None;files={};new=0;tick=time.time()
 for i,r in enumerate(p['rows']):
  f=out/'subjects'/f'{i:05}.json';cap=r['input']['caption'];capsha=hashlib.sha256(cap.encode()).hexdigest()
  if not f.exists():
   if cap not in known:
    if parser is None:parser=QuerySubjectParser(ROOT/'.cache/stanza')
    doc=parser.nlp(cap)
    # The very same parsed document feeds the unchanged query-only functions.
    proxy=object.__new__(QuerySubjectParser);proxy.version=parser.version;proxy.nlp=lambda text:doc
    sub=subject_cache.get(capsha) or proxy(cap)
    if isinstance(sub,dict):sub=sub['subject']
    known[cap]=dict(subject=sub,old=visual_query(proxy,cap),context=parse_context(proxy,cap));new+=1
   write(f,dict(ordinal=i,caption_sha256=capsha,parses=known[cap],GT_read=False))
  x=read(f);assert x['caption_sha256']==capsha and x['ordinal']==i and not x['GT_read'];files[f.name]=sha(f)
  if i%100==0:
   status(out/'SUBJECT_STATUS.json',dict(status='running_cpu',pid=os.getpid(),done=i,total=len(p['rows']),new_unique_parses=new,seconds=time.time()-tick,GT_read=False))
   print('FULL_SUBJECTS',ds,i,len(p['rows']),flush=True)
 write(out/'SUBJECT_BARRIER.json',dict(status='sealed',count=len(files),files=files,GT_read=False,time=time.time(),new_unique_parses=new))
 status(out/'SUBJECT_STATUS.json',dict(status='completed',done=len(files),total=len(files),GT_read=False,seconds=time.time()-tick))

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:subjects(sys.argv[1])
