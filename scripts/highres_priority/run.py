"""User-prioritized448 supplement while the full queue is saved/paused.

New immutable authorization/version; original deferred plan and all frozen
production sources remain unchanged. The native two-offset algorithm is reused.
"""
import os,sys,json,time,gc,hashlib,argparse,fcntl,traceback
from pathlib import Path
from contextlib import contextmanager
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.branch_decomposition.supplement import read,write,sha,identity,alive
from scripts.branch_decomposition.lightweight_design import physical_view,single_high_view_fusion
OUT=ROOT/'outputs/branch_decomposition_v1'
SUP=OUT/'supplement448_priority_v2'
PUBLIC=ROOT/'results/highres_priority_v2'

def inactive(x):
 try:
  a=(Path('/proc')/str(x['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  return int(a[19])!=x['startticks'] or a[0]=='Z'
 except (FileNotFoundError,KeyError):return True

def assert_pause_boundary():
 full=read(ROOT/'outputs/boxfixed_full_v1/state.json')
 assert full['status']=='paused_by_user' and full['paused_completed']==6088
 assert all(inactive(x)for x in [full['controller'],*full.get('children',[])])
 pause=read(ROOT/full['pause_receipt']);assert pause['completed']==6088 and pause['all_owned_exited']
 assert read(ROOT/'outputs/local224_v1/state.json')['status']=='complete_published'
 assert not alive(read(ROOT/'outputs/local224_v1/state.json')['process'])

def validate():
 p=read(SUP/'plan.json');v={k:x for k,x in p.items()if k!='fingerprint'}
 assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['fingerprint']
 for rel,h in p['sources'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in p['assets'].items():assert sha(ROOT/rel)==h,rel
 assert sha(OUT/'supplement448/plan.json')==p['deferred_plan_sha256']
 assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['full_plan_sha256']
 assert sha(ROOT/'data/evaluation/development.jsonl')==p['manifest_sha256']
 assert sha(OUT/'sealed.json')==p['existing_one_case_seal_sha256']
 assert sha(ROOT/'results/tube_fusion_hc64/report.json')==p['old_report_sha256']
 for entry in read(OUT/'sealed.json')['files']:
  assert sha(OUT/entry['path'])==entry['sha256']and sha(OUT/entry['posterior_path'])==entry['posterior_sha256']
 return p

def prepare():
 assert not(SUP/'plan.json').exists();assert_pause_boundary()
 original=read(OUT/'supplement448/plan.json');p={k:v for k,v in original.items()if k!='fingerprint'}
 p.update(schema='priority_448_supplement_v2',authorization='Human now requests448 rerun and analysis while full evaluation remains explicitly saved/paused. This supersedes only scheduling; all original inputs/method/precision/hashes remain exact.',deferred_plan_sha256=sha(OUT/'supplement448/plan.json'))
 p['sources']={**p['sources'],**{str(f.relative_to(ROOT)):sha(f)for f in Path(__file__).parent.glob('*.py')}}
 p['conditions']=dict(p['conditions'],start='Explicit paused full at6088, original owned processes exited, completed local224; acquire sharedGPU lock. Do not resume full.',analysis='Store six-layer448 logits/hidden already computed by the native two-pass process, no extra heavy pass. Derived Only224Last,High6,High6Fusion and original matched singles on63; no queryGT branch selection or parameter search.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(SUP/'plan.json',p)
 for rel in p['sources']:
  q=SUP/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
 write(SUP/'state.json',dict(status='authorized_priority_ready',fingerprint=p['fingerprint'],completed=0,total=63))
 write(SUP/'authorization_migration.json',dict(old_plan_untouched=True,old_fingerprint=original['fingerprint'],new_fingerprint=p['fingerprint'],changed='Explicit user scheduling authorization and additional saved native outputs/derived cached readouts; no numerical method/precision/input change.',new224=0,new_teacher=0,updates=0))
 print('prepared',p['fingerprint'])

@contextmanager
def prefix_costs():
 import stvg.temporal_capture as tc
 old=tc._prefix_view;seconds=[]
 def timed(*args,**kw):
  torch.cuda.synchronize();tick=time.perf_counter();v=old(*args,**kw);torch.cuda.synchronize();seconds.append(time.perf_counter()-tick);return v
 tc._prefix_view=timed
 try:yield seconds
 finally:tc._prefix_view=old

@torch.no_grad()
def execute(resume=False):
 p=validate();assert not(SUP/'sealed.json').exists(),'Already sealed; never recapture';assert_pause_boundary()
 lock=(SUP/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=read(SUP/'state.json')
 if state['status']!='authorized_priority_ready':
  assert resume and state.get('fingerprint')==p['fingerprint'];assert not alive(state.get('process',{})),'Old owned supplement still alive'
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.inputs import resolution,validate_input
 from stvg.query import QuerySubjectParser
 from stvg.temporal_capture import frozen_temporal_forward
 from stvg.temporal import probabilities,mass_table,final_mbr
 from scripts.tube_fusion.core import restore_geometry
 from stvg.spatial import evidence_dp
 setup();gpu=lease();state=dict(status='running',process=identity(),fingerprint=p['fingerprint'],completed=0,total=63);write(SUP/'state.json',state);files=[];model=None;begin=time.perf_counter()
 try:
  model=load_source(p['config']);parser=QuerySubjectParser(ROOT/'checkpoints/stanza')
  for item in p['items']:
   key=item['anonymous'];rp=SUP/'receipts'/f'{key}.json';pp=SUP/'predictions'/f'{key}.npz';cp=SUP/'posteriors'/f'{key}.pt'
   if rp.exists():
    rec=read(rp);assert rec['fingerprint']==p['fingerprint']and sha(pp)==rec['prediction_sha256']and sha(cp)==rec['posterior_sha256'];files.append(rec);state['completed']=len(files);continue
   assert not pp.exists()and not cp.exists(),'Unreceipted artifact preserved; do not overwrite silently'
   for name,path in item['inputs'].items():assert sha(ROOT/path)==item['hashes'][name]
   record=item['record'];tick=time.perf_counter();frames,ids=decode(record['metadata'],'hc2');validate_input(frames,ids,record['metadata']);decode_seconds=time.perf_counter()-tick;low=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['inputs']['prediction']);assert ids==old['frame_ids'].tolist();torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
   row=dict(input=record['metadata'],frame_ids=ids,parses=dict(subject=parser(record['metadata']['caption'])['subject']))
   with prefix_costs() as prefix_seconds:
    with resolution(448):batch,recs,high=frozen_temporal_forward(model,frames,row,audit=True)
   logits=[z.detach().cpu()for z in high.zero['logits']];grids=probabilities(recs,logits);U,qH=mass_table(grids);assert U==ids;qG=low['final']['Global6'];view=[physical_view(g['ids'],g['ij'][0],g['ij'][1],g['p'],U)for g in grids];post=dict(Only224six=qG,Only448=qH,Fusion=.5*qG+.5*qH,SingleHigh0=view[0],SingleHigh1=view[1],SingleFusion0=single_high_view_fusion(qG,view[0]),SingleFusion1=single_high_view_fusion(qG,view[1]));all_logits=[a['second_logits_all_layers']for a in high.zero['audit']];qh6=np.mean([mass_table(probabilities(recs,[a[layer]for a in all_logits]))[1]for layer in range(6)],axis=0);post.update(Only224Last=low['layers']['G'][-1],High6=qh6,High6Fusion=.5*qG+.5*qh6);torch.cuda.synchronize();high_seconds=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
   if final_mbr(U,post['Fusion'])['interval']!=old['Strong_Reference_interval'].tolist():
    failure=SUP/'failures'/key;failure.mkdir(parents=True,exist_ok=True);torch.save(dict(frame_ids=U,records=recs,logits=logits,posteriors=post,GT_read=False),failure/'raw448.pt');write(failure/'mismatch.json',dict(expected=old['Strong_Reference_interval'].tolist(),actual=final_mbr(U,post['Fusion'])['interval'],fingerprint=p['fingerprint']));raise AssertionError('RestoredStrong interval mismatch; raw evidence preserved; no hash relaxation')
   geom=restore_geometry(cap['geometry']);values=dict(Source=old['Source_Native224'],Source_interval=old['Source_Native224_interval'],frame_ids=old['frame_ids'],positions8=old['positions8'],FixedSpatial=old['Strong_Reference'])
   for arm,q in post.items():
    span=final_mbr(U,q)['interval'];values[arm]=evidence_dp(geom.for_interval(span))['final'].numpy();values[arm+'_interval']=span
   assert np.array_equal(values['Only224six'],old['Global6'])and np.array_equal(values['Fusion'],old['Strong_Reference'])
   pp.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**values);tmp=cp.with_suffix('.pt.tmp');torch.save(dict(frame_ids=U,records=recs,logits=logits,all_logits=all_logits,high_hidden=[a['second_hidden']for a in high.zero['audit']],high_grid_sizes=[v['info']['fea_map_size']for v in high.views],low_grid_sizes=[v['grid_size']for v in cap['temporal_cache']['offsets']],grids=grids,posteriors=post,GT_read=False),tmp);os.replace(tmp,cp)
   rec=dict(query=key,id=item['id'],fingerprint=p['fingerprint'],prediction_path=str(pp.relative_to(SUP)),prediction_sha256=sha(pp),posterior_path=str(cp.relative_to(SUP)),posterior_sha256=sha(cp),Source224_calls=0,ReferDINO_calls=0,updates=0,Source448_offsets=2,GT_read=False,old_Global6_and_Strong_exact=True,decode_seconds=decode_seconds,Source448_seconds=high_seconds,prefix_seconds=prefix_seconds,high_grid_sizes=[v['info']['fea_map_size']for v in high.views],low_grid_sizes=[v['grid_size']for v in cap['temporal_cache']['offsets']],peak_GPU_bytes=peak);write(rp,rec);files.append(rec);state['completed']=len(files);write(SUP/'state.json',state);print(json.dumps(dict(completed=len(files),total=63)),flush=True);del frames,batch,high,logits,low,cap,geom,values,post;gc.collect();torch.cuda.empty_cache()
  validate();assert len(files)==63;write(SUP/'sealed.json',dict(fingerprint=p['fingerprint'],files=files,GT_read=False,queries=63,seconds=time.perf_counter()-begin));state['status']='sealed_awaiting_CPU_score';write(SUP/'state.json',state)
 except BaseException:
  state['status']='failed';state['error']=traceback.format_exc();write(SUP/'state.json',state);raise
 finally:
  if model is not None:model.cpu()
  gpu.close();lock.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--resume',action='store_true');v=a.parse_args()
 prepare()if v.action=='prepare'else execute(v.resume)
