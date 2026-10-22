"""Authorized63-query448-only supplement, gated after published full evaluation.

Preparation/preflight never initialize CUDA. No queue or model is started by
prepare. Run is label-free, resumable only with exact hashes, and never signals
another process or rewrites old outputs.
"""
import os,sys,json,time,gc,hashlib,argparse,fcntl,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/branch_decomposition_v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
from scripts.branch_decomposition.lightweight_design import physical_view,single_high_view_fusion
SUP=OUT/'supplement448'

def identity(pid=None):
 pid=os.getpid()if pid is None else int(pid);return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def alive(x):
 try:return identity(x['pid'])=={k:x[k]for k in ['pid','startticks']}
 except (KeyError,FileNotFoundError,ProcessLookupError):return False

def boundary_ready(state):return state.get('status')=='complete_published'and state.get('jobs')=={'hc2':'scored','vidstg':'scored'}
def validate():
 p=read(SUP/'plan.json');raw={k:v for k,v in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()==p['fingerprint']
 for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
 for name,h in p['assets'].items():assert sha(ROOT/name)==h,name
 assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['full_plan_sha256']
 assert sha(ROOT/'data/evaluation/development.jsonl')==p['manifest_sha256']
 assert sha(OUT/'sealed.json')==p['existing_one_case_seal_sha256']
 assert sha(ROOT/'results/tube_fusion_hc64/report.json')==p['old_report_sha256']
 for entry in read(OUT/'sealed.json')['files']:
  assert sha(OUT/entry['path'])==entry['sha256'] and sha(OUT/entry['posterior_path'])==entry['posterior_sha256']
 return p

def prepare():
 assert not(SUP/'plan.json').exists();old=read(OUT/'plan.json');manifest=ROOT/'data/evaluation/development.jsonl';rows={r['id']:r for r in [json.loads(l)for l in manifest.read_text().splitlines()if l.strip()]};full=read(ROOT/'outputs/boxfixed_full_v1/plan.json');config=read(ROOT/'configs/default.json')['datasets']['hc2'];items=[]
 for it in old['items']:
  if it['anonymous']=='H001':continue
  record=rows[it['id']];assert record['dataset']=='hc2';items.append(dict(it,record=record))
 assert len(items)==63
 helpers=['scripts/tube_fusion/core.py','scripts/tube_fusion/run.py','scripts/tube_temporal/run.py','scripts/tube_temporal/score.py','scripts/interval_pool/score.py','scripts/interval_pool/intervention.py','scripts/interval_pool/confirmation.py','scripts/interval_pool/development.py','scripts/interval_pool/core.py','scripts/joint_boundary_readout/score.py','scripts/joint_boundary_readout/run.py','scripts/joint_boundary_readout/core.py','scripts/temporal_signal_audit/extract.py']
 source_paths=sorted(set(helpers+[str(f.relative_to(ROOT))for f in Path(__file__).parent.glob('*.py')]))
 sources={**full['sources'],**{f:sha(ROOT/f)for f in source_paths}}
 p=dict(schema='authorized_missing448_supplement_v1',authorization='Human approved: allow63 missing448 branches only after current full evaluation finishes; no224/expert/training rerun.',items=items,config=config,sources=sources,assets=full['assets'],full_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),manifest_sha256=sha(manifest),old_report_sha256=sha(ROOT/'results/tube_fusion_hc64/report.json'),existing_one_case_seal_sha256=sha(OUT/'sealed.json'),labels=old['labels'],conditions=dict(start='Full state complete_published, both scored, owned controller/workers exited, acquire sharedGPU lock. Never interrupt or start concurrently.',budget='Exactly63 original queries/two448 offsets each. No224 visual/decoder pass, noReferDINO, nooptimizer.',preserve='Original checkpoint, native FP16 visual prefix/FP32 two-pass time flow, seed20260920, frame IDs, query-only parser, resize448, original hull/MBR. Save native logits per offset, raw grids and normalized posterior.',compatibility='Original224six and full-fusion intervals/complete spatialDP must recover saved arrays exactly; any mismatch stops and preserves evidence, never weakens hashes.',panel='Only224six,Only448,Fusion all64 using one cachedH001; SingleHigh0/SingleHigh1/SingleFusion0/SingleFusion1 on63 only because H001 lacks peroffset448 logits. Offset0 fixed primary, offset1 sensitivity, never choose byGT. Equal weights unchanged.',adoption='No automatic deployment. Complete signed/fixed-space/negative tails/costs; no weight or threshold sweep. A one-view deployment saving is unmeasured until an actual one-view adapter is verified. All data historically exposed.'),GT_read=False)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(SUP/'plan.json',p)
 for rel in source_paths:
  dest=SUP/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
 write(SUP/'state.json',dict(status='authorized_waiting_for_full_publication',fingerprint=p['fingerprint'],completed=0,total=63));print(json.dumps(dict(status='prepared_only',fingerprint=p['fingerprint'],queries=63,models_started=0)))

def preflight():
 p=validate();full=read(ROOT/'outputs/boxfixed_full_v1/state.json');write(SUP/'preflight.json',dict(status='ready'if boundary_ready(full)else'waiting_for_full',full_status=full['status'],jobs=full['jobs'],fingerprint=p['fingerprint'],CUDA_initialized=torch.cuda.is_initialized()));assert not torch.cuda.is_initialized();print('ready'if boundary_ready(full)else'waiting_for_full; no model started')

@torch.no_grad()
def execute(resume=False):
 p=validate();assert not(SUP/'sealed.json').exists(),'Already sealed; run independent CPU scoring, never recapture';full=read(ROOT/'outputs/boxfixed_full_v1/state.json');assert boundary_ready(full),'Do not run before full evaluation is scored and published'
 assert not alive(full['controller'])and not any(alive(x)for x in full.get('children',[]))
 for ds in ['hc2','vidstg']:
  assert not alive(read(ROOT/f'outputs/boxfixed_full_v1/{ds}/worker.json'));seal=read(ROOT/f'outputs/boxfixed_full_v1/{ds}/sealed.json');assert seal['queries']==(1902 if ds=='hc2'else 10303)
 lock=(SUP/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=read(SUP/'state.json')
 if state['status']!='authorized_waiting_for_full_publication':
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
   with resolution(448):batch,recs,high=frozen_temporal_forward(model,frames,row,audit=False)
   logits=[z.detach().cpu()for z in high.zero['logits']];grids=probabilities(recs,logits);U,qH=mass_table(grids);assert U==ids;qG=low['final']['Global6'];view=[physical_view(g['ids'],g['ij'][0],g['ij'][1],g['p'],U)for g in grids];post=dict(Only224six=qG,Only448=qH,Fusion=.5*qG+.5*qH,SingleHigh0=view[0],SingleHigh1=view[1],SingleFusion0=single_high_view_fusion(qG,view[0]),SingleFusion1=single_high_view_fusion(qG,view[1]));torch.cuda.synchronize();high_seconds=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
   if final_mbr(U,post['Fusion'])['interval']!=old['Strong_Reference_interval'].tolist():
    failure=SUP/'failures'/key;failure.mkdir(parents=True,exist_ok=True);torch.save(dict(frame_ids=U,records=recs,logits=logits,posteriors=post,GT_read=False),failure/'raw448.pt');write(failure/'mismatch.json',dict(expected=old['Strong_Reference_interval'].tolist(),actual=final_mbr(U,post['Fusion'])['interval'],fingerprint=p['fingerprint']));raise AssertionError('RestoredStrong interval mismatch; raw evidence preserved; no hash relaxation')
   geom=restore_geometry(cap['geometry']);values=dict(Source=old['Source_Native224'],Source_interval=old['Source_Native224_interval'],frame_ids=old['frame_ids'],positions8=old['positions8'],FixedSpatial=old['Strong_Reference'])
   for arm,q in post.items():
    span=final_mbr(U,q)['interval'];values[arm]=evidence_dp(geom.for_interval(span))['final'].numpy();values[arm+'_interval']=span
   assert np.array_equal(values['Only224six'],old['Global6'])and np.array_equal(values['Fusion'],old['Strong_Reference'])
   pp.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**values);tmp=cp.with_suffix('.pt.tmp');torch.save(dict(frame_ids=U,records=recs,logits=logits,grids=grids,posteriors=post,GT_read=False),tmp);os.replace(tmp,cp)
   rec=dict(query=key,id=item['id'],fingerprint=p['fingerprint'],prediction_path=str(pp.relative_to(SUP)),prediction_sha256=sha(pp),posterior_path=str(cp.relative_to(SUP)),posterior_sha256=sha(cp),Source224_calls=0,ReferDINO_calls=0,updates=0,Source448_offsets=2,GT_read=False,old_Global6_and_Strong_exact=True,decode_seconds=decode_seconds,Source448_seconds=high_seconds,peak_GPU_bytes=peak);write(rp,rec);files.append(rec);state['completed']=len(files);write(SUP/'state.json',state);print(json.dumps(dict(completed=len(files),total=63)),flush=True);del frames,batch,high,logits,low,cap,geom,values,post;gc.collect();torch.cuda.empty_cache()
  validate();assert len(files)==63;write(SUP/'sealed.json',dict(fingerprint=p['fingerprint'],files=files,GT_read=False,queries=63,seconds=time.perf_counter()-begin));state['status']='sealed_awaiting_CPU_score';write(SUP/'state.json',state)
 except BaseException:
  state['status']='failed';state['error']=traceback.format_exc();write(SUP/'state.json',state);raise
 finally:
  if model is not None:model.cpu()
  gpu.close();lock.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--preflight',action='store_true');g.add_argument('--run',action='store_true');a.add_argument('--resume',action='store_true');x=a.parse_args()
 if x.prepare:prepare()
 elif x.preflight:preflight()
 else:execute(x.resume)
