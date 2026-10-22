"""Fixed HC63 single320 budget screen, then locked development rechecks."""
import os,sys,json,time,gc,hashlib,argparse,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,read,write,sha,identity,assert_pause_boundary
OUT=ROOT/'outputs/singleview_budget_v1';PUBLIC=ROOT/'results/singleview_budget_v1'

def validate():
 p=read(OUT/'plan.json');v={k:x for k,x in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['fingerprint']
 for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
 for f,h in p['assets'].items():assert sha(ROOT/f)==h,f
 for f,h in p['boundaries'].items():assert sha(ROOT/f)==h,f
 assert_pause_boundary();return p

def prepare():
 assert not(OUT/'plan.json').exists();assert_pause_boundary();old=read(SUP/'plan.json');full=read(ROOT/'outputs/boxfixed_full_v1/plan.json')
 sources={**old['sources'],**{str(f.relative_to(ROOT)):sha(f)for f in Path(__file__).parent.glob('*.py')}}
 p=dict(schema='singleview320_fixed_budget_v1',items=old['items'],config=old['config'],assets=old['assets'],sources=sources,labels=old['labels'],boundaries={str(f.relative_to(ROOT)):sha(f)for f in [ROOT/'outputs/boxfixed_full_v1/plan.json',SUP/'sealed.json',ROOT/'results/highres_selfhull_v1/report.json',ROOT/'outputs/interval_pool/confirmation_selection.json']},
 protocol='HC63 existing repeatedly exposed development. ONLY new candidate native full-frame320 offset0+self-independent hull+.5Global6, original physical MBR/BoxFixed. Exact original RGB/frames/seed/precision/eight-expert evidence reused. No320 offset1 or other resolution/weight search. Both448 offsets remain separately fixed recheck controls. All63 new predictions seal before independent CPU labels. No full/CG resume.',
 gate=dict(full_v_loss_pp_at_most=1.,t_and_v_vs_Global6_positive=True,extra_branch_wall_reduction_at_least=.40,scope='Engineering point-estimate screening, not statistical equivalence. If320fails stop320;44832x2 user-authorized recheck still proceeds.'),
 benchmark_indices=[0,9,18,27,36,45,54,62],benchmark_arms=['320_0','448_0','448_01'],timing='Eight fixed queries, counterbalanced order, resident SAME frozen model; required preprocessing/transfers/two time passes/copies/posterior included. No per-module timing synchronization; only outer boundaries. Model startup/decode/parse separately. Not whole-method end-to-end; separate32x2 stage measures actual independent complete pipelines.',
 full_snapshot=full['fingerprint'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(OUT/'plan.json',p)
 for f in sources:
  q=OUT/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
 write(OUT/'state.json',dict(status='prepared',fingerprint=p['fingerprint'],completed=0,total=63));print(p['fingerprint'])

@torch.no_grad()
def run():
 p=validate();assert not(OUT/'sealed.json').exists();assert read(OUT/'state.json')['status']=='prepared'
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.inputs import resolution,validate_input
 from stvg.query import QuerySubjectParser
 from stvg.temporal import final_mbr
 from stvg.spatial import evidence_dp
 from scripts.tube_fusion.core import restore_geometry
 from scripts.singleview_budget.model import forward,posterior
 setup();lock=lease();state=dict(status='running',process=identity(),fingerprint=p['fingerprint'],completed=0,total=63);write(OUT/'state.json',state);files=[];model=None;start=time.perf_counter();bench=[]
 try:
  tick=time.perf_counter();model=load_source(p['config']);torch.cuda.synchronize();load=time.perf_counter()-tick;parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
  for i,item in enumerate(p['items']):
   key=item['anonymous'];tick=time.perf_counter();record=item['record'];frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata']);rgb=time.perf_counter()-tick
   row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=parser(record['metadata']['caption'])['subject']))
   low=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);geom=restore_geometry(cap['geometry']);g=low['final']['Global6']
   torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
   with resolution(320):out=forward(model,frames,row,(0,))
   h=posterior(out,ids);q=.5*g+.5*h;span=final_mbr(ids,q)['interval'];boxes=evidence_dp(geom.for_interval(span))['final'].numpy();torch.cuda.synchronize();seconds=time.perf_counter()-tick
   values=dict(Single320=boxes,Single320_interval=span,frame_ids=ids,positions8=cap['planned_positions']);pp=OUT/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**values);cp=OUT/'posteriors'/f'{key}.pt';cp.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(**out,posterior=q,GT_read=False),cp)
   rec=dict(query=key,id=item['id'],fingerprint=p['fingerprint'],path=str(pp.relative_to(OUT)),sha256=sha(pp),posterior_path=str(cp.relative_to(OUT)),posterior_sha256=sha(cp),branch_plus_MBR_DP_seconds=seconds,RGB_seconds=rgb,input_shapes=out['input_shapes'],actual320_views=1,new_expert=0,GT_read=False,peak_GPU_bytes=torch.cuda.max_memory_allocated());write(OUT/'receipts'/f'{key}.json',rec);files.append(rec)
   if i in p['benchmark_indices']:
    arms=p['benchmark_arms'];order=arms[(len(bench))%3:]+arms[:(len(bench))%3];b=dict(query=key,order=order,seconds={},shapes={})
    previous=torch.load(SUP/'posteriors'/f'{key}.pt',map_location='cpu',weights_only=False)
    for arm in order:
     size=320 if arm=='320_0'else 448;offsets=(0,1)if arm=='448_01'else(0,);torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.synchronize();tick=time.perf_counter()
     with resolution(size):value=forward(model,frames,row,offsets)
     hp=posterior(value,ids);fused=.5*g+.5*hp;decision=final_mbr(ids,fused);torch.cuda.synchronize();b['seconds'][arm]=time.perf_counter()-tick;b['shapes'][arm]=value['input_shapes']
     expected=out if size==320 else previous
     expected_logits=expected['logits']if arm!='448_0'else[expected['logits'][0]]
     assert all(torch.equal(a,z)for a,z in zip(value['logits'],expected_logits)),'Benchmark code path did not recover bitwise logits'
    bench.append(b);write(OUT/'benchmark_progress.json',dict(files=bench,GT_read=False))
   state['completed']=i+1;write(OUT/'state.json',state);print(json.dumps(dict(completed=i+1,total=63,elapsed=time.perf_counter()-start,query=key)),flush=True);del frames,low,cap,geom,out;gc.collect();torch.cuda.empty_cache()
  validate();write(OUT/'benchmark.json',dict(files=bench,model_load_seconds=load,GT_read=False));write(OUT/'sealed.json',dict(fingerprint=p['fingerprint'],queries=63,files=files,GT_read=False,benchmark_sha256=sha(OUT/'benchmark.json'),seconds=time.perf_counter()-start));state['status']='sealed_awaiting_CPU_score';write(OUT/'state.json',state)
 except BaseException:
  state.update(status='failed',error=traceback.format_exc());write(OUT/'state.json',state);raise
 finally:
  if model is not None:model.cpu()
  lock.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);v=a.parse_args();globals()[v.action]()
