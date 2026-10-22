"""Actual independent complete pipelines,32 existing parents per domain.

Four fixed arms, plus320 ONLY if the predeclared HC63 gate passed. Accuracy
reuses fixed224/expert captures. Four fixed queries/domain independently execute
complete pipelines for timing; no cache cost substitution in these timed runs.
"""
import os,sys,json,time,gc,hashlib,argparse,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.singleview_budget.run import read,write,sha,identity,assert_pause_boundary,OUT as SCREEN
OUT=ROOT/'outputs/single448_recheck_v1';PUBLIC=ROOT/'results/single448_recheck_v1'

def validate():
 p=read(OUT/'plan.json');v={k:x for k,x in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['fingerprint']
 for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
 for f,h in p['assets'].items():assert sha(ROOT/f)==h,f
 for f,h in p['inputs'].items():assert sha(ROOT/f)==h,f
 assert sha(ROOT/'outputs/interval_pool/confirmation_selection.json')==p['selection_sha256'];assert_pause_boundary();return p

def prepare():
 assert not(OUT/'plan.json').exists();assert_pause_boundary();screen=read(SCREEN/'decision.json');items=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection'];old=read(SCREEN/'plan.json');cfg=read(ROOT/'configs/default.json')['datasets'];formal=read(ROOT/'outputs/formal/plan.json');labels={ds:dict(path=j['labels'],sha256=j['labels_sha256'])for ds in items for j in formal['jobs']if j['id']=='tastvg-'+ds};assets={**old['assets'],cfg['vidstg']['source_checkpoint']:sha(ROOT/cfg['vidstg']['source_checkpoint'])}
 sources={**old['sources'],**{str(f.relative_to(ROOT)):sha(f)for f in Path(__file__).parent.glob('*.py')}}
 arms=['Strong','Global6','Single448_0','Single448_1']+(['Single320']if screen['gate']['pass']else[])
 inputs={}
 for ds,rr in items.items():
  assert len(rr)==len({r['parent']for r in rr})==32
  for item in rr:
   for kind,suffix in [('capture','pt'),('posteriors','pt'),('predictions','npz')]:
    f=ROOT/f'outputs/interval_pool/recheck_{ds}/{kind}/{item["query_id"]}.{suffix}';inputs[str(f.relative_to(ROOT))]=sha(f)
 p=dict(schema='single448_real_end_to_end_recheck_v1',inputs=inputs,items=items,configs=cfg,labels=labels,sources=sources,assets=assets,arms=arms,selection_sha256=sha(ROOT/'outputs/interval_pool/confirmation_selection.json'),screen_decision_sha256=sha(SCREEN/'decision.json'),
 protocol='Existing HC32/Vid32 DEVELOPMENT RECHECK, historically exposed, no independent-generalization claim. Fixed448 offset0 primary/offset1 sensitivity, no queryGT choice. Original Strong and Global6 identical224/eight-observation/full-box workflow. Only temporal extra branch differs. All64xarms predictions seal before labels. Full/oldformal/CG stay paused.',
 timing_indices=[0,10,21,31],timing='Four FIXED queries/domain, eight total. Each timed arm independently executes rawRGB decode, native224 both offsets, one ReferDINO eight-observation request without evidence cache, ERef/complete-box propagation, its actual required extra high-res views, physical posterior/MBR/BoxFixed. Resident model and teacher processes; startup/load separately. Cyclic arm order by timing-query index, no RGB/visual/evidence cache substitution in timed runs. Other precision queries reuse existing224/evidence and are NOT called end-to-end timings. Actual query wall includes actual CPU/GPU transfers, parse and final DP; artifact persistence reported separately. First query startup and all-versus-excluding-first timing both reported. No diagnostic per-module synchronization.',
 counts=dict(queries=64,accuracy_query_arm_results=64*len(arms),complete_timing_runs=8*len(arms),teacher_forwards=8*len(arms),experts_per_run=1,observations_per_run=8,training=0))
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(OUT/'plan.json',p)
 for f in sources:
  q=OUT/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
 write(OUT/'state.json',dict(status='prepared',fingerprint=p['fingerprint'],completed=0,total=64));print(p['fingerprint'],arms)

@torch.no_grad()
def run():
 p=validate();assert read(OUT/'state.json')['status']=='prepared'and not(OUT/'sealed.json').exists()
 from stvg.runtime import setup,lease,load_source
 from stvg.predictor import Predictor
 from stvg.decode import decode
 from stvg.inputs import validate_input
 from stvg.teacher_bridge import ResidentTeacher
 from scripts.singleview_budget.capture import capture
 setup();gpu=lease();state=dict(status='running',process=identity(),fingerprint=p['fingerprint'],completed=0,total=64);write(OUT/'state.json',state);files=[];teacher=None;predictor=None;loadcost={};begin=time.perf_counter()
 try:
  for ds,items in p['items'].items():
   tick=time.perf_counter();model=load_source(p['configs'][ds]);torch.cuda.synchronize();model.cpu();loadcost[ds]=dict(model_load_seconds=time.perf_counter()-tick);predictor=Predictor(ds,source=model);teacher=ResidentTeacher(fingerprint=p['fingerprint']);predictor.teacher_service=teacher
   from scripts.singleview_budget.model import forward,posterior
   from stvg.inputs import resolution
   from scripts.tube_fusion.core import restore_geometry
   from stvg.temporal import final_mbr
   from stvg.spatial import evidence_dp
   for i,item in enumerate(items):
    record=item['record'];key=('C'if ds=='hc2'else'V')+f'{i+1:03d}';arms=p['arms'];order=[];values={};costs={};proofs={}
    previous=ROOT/f'outputs/interval_pool/recheck_{ds}'
    capold=torch.load(previous/'capture'/f'{item["query_id"]}.pt',map_location='cpu',weights_only=False)
    postold=torch.load(previous/'posteriors'/f'{item["query_id"]}.pt',map_location='cpu',weights_only=False)
    old=np.load(previous/'predictions'/f'{item["query_id"]}.npz');ids=old['frame_ids'].tolist();g=np.mean(postold['hull_layers'],axis=0);geom=restore_geometry(capold['geometry'])
    values.update(Source=old['Source_Native224'],Source_interval=old['Source_Native224_interval'],frame_ids=ids,positions8=old['positions8'],Global6=old['Hull_MBR'],Global6_interval=old['Hull_MBR_interval'],FixedSpatial=old['Hull_MBR'])
    frames,fids=decode(record['metadata'],ds);validate_input(frames,fids,record['metadata']);assert fids==ids
    if predictor.parser is None:
     from stvg.query import QuerySubjectParser
     predictor.parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
    row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=predictor.parser(record['metadata']['caption'])['subject']))
    model.cuda();torch.manual_seed(20260920);np.random.seed(20260920)
    with resolution(448):high=forward(model,frames,row,(0,1))
    qs=dict(Strong=.5*g+.5*posterior(high,ids))
    for offset in [0,1]:
     one=dict(records=[high['records'][offset]],logits=[high['logits'][offset]])
     qs['Single448_'+str(offset)]=.5*g+.5*posterior(one,ids)
    if 'Single320'in arms:
     torch.manual_seed(20260920);np.random.seed(20260920)
     with resolution(320):medium=forward(model,frames,row,(0,))
     qs['Single320']=.5*g+.5*posterior(medium,ids)
    for arm,q in qs.items():
     span=final_mbr(ids,q)['interval'];values[arm]=evidence_dp(geom.for_interval(span))['final'].numpy();values[arm+'_interval']=span
    del frames;model.cpu();gc.collect();torch.cuda.empty_cache()
    # Full actual independent pipelines only at the predeclared eight timing queries.
    if i in p['timing_indices']:
     index=p['timing_indices'].index(i)+(ds=='vidstg');order=arms[index%len(arms):]+arms[:index%len(arms)]
     for arm in order:
      torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();frames,fids=decode(record['metadata'],ds);validate_input(frames,fids,record['metadata']);rgb=time.perf_counter()-tick
      counts=dict(vis_encoder=0,vid=0,text_encoder=0,ground_encoder=0);handles=[]
      for name in counts:
       def count(m,a,o,name=name):counts[name]+=1
       handles.append(getattr(model,name).register_forward_hook(count))
      try:fresh=capture(predictor,frames,fids,record['metadata'],profile=arm);out=predictor.finish(fresh,'strong');torch.cuda.synchronize();elapsed=time.perf_counter()-tick
      finally:
       for handle in handles:handle.remove()
      expected=2+fresh['cost'].get('actual_extra_views',0);assert all(c==expected for c in counts.values()),counts
      assert fresh['cost']['ReferDINO_query_forwards']==1 and not fresh['cost']['ReferDINO_cache_hit']and len(fresh['planned_positions'])==8
      assert np.array_equal(fresh['Source'],values['Source'])and np.array_equal(fresh['E'],capold['E'])and np.array_equal(out['boxes'],values[arm])and out['interval']==list(values[arm+'_interval'])
      assert np.array_equal(fresh['posterior']['Global6'],g)
      costs[arm]=dict(actual_query_wall_seconds=elapsed,RGB_seconds=rgb,capture_cost=fresh['cost'],finish_cost=out['cost'],peak_GPU_bytes=torch.cuda.max_memory_allocated())
      proofs[arm]=dict(eight_observations=True,independent_complete_pipeline_matches_cached_precision_bitwise=True,actual_encoder_calls=counts,actual_high_views=fresh['cost'].get('actual_extra_views',0),extra_resolution=fresh['cost'].get('extra_resolution',None))
      del frames,fresh,out;gc.collect();torch.cuda.empty_cache()
    pp=OUT/ds/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();np.savez_compressed(pp,**values);io=time.perf_counter()-tick
    r=dict(query=key,id=item['query_id'],dataset=ds,path=str(pp.relative_to(OUT)),sha256=sha(pp),costs=costs,proof=proofs,order=order,persistence_seconds=io,GT_read=False,fingerprint=p['fingerprint']);write(OUT/ds/'receipts'/f'{key}.json',r);files.append(r);state.update(completed=len(files),dataset=ds,query=key);write(OUT/'state.json',state);print(json.dumps(dict(dataset=ds,completed=i+1,total=32,elapsed=time.perf_counter()-begin)),flush=True)
   teacher.close();teacher=None;model.cpu();del model,predictor;predictor=None;gc.collect();torch.cuda.empty_cache()
  validate();write(OUT/'sealed.json',dict(queries=64,files=files,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-begin,load_cost=loadcost));state['status']='sealed_awaiting_CPU_score';write(OUT/'state.json',state)
 except BaseException:
  state.update(status='failed',error=traceback.format_exc());write(OUT/'state.json',state);raise
 finally:
  if teacher is not None:teacher.close()
  if predictor is not None and predictor.source is not None:predictor.source.cpu()
  gpu.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);v=a.parse_args();globals()[v.action]()
