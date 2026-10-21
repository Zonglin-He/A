"""Two separately sealed CPU stages. No GT imports or heavy network calls.

The independent diagnostic decides only whether this entire prelocked prototype
proceeds; never uses labels for individual predictions/feedback/rollback.
"""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import OLD,prepare,read,sha,atomic,savept
from scripts.propagation_opd.run import check_idle
from scripts.trajectory_fast.evidence import TrajectoryEvidence
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/propagation_v0_hc64'
ARMS=('OPD','FixedKD','Direct');CONTROLS=('MemoryGreedy','NoMemoryGreedy','EvidenceDP')


def lock_inputs():
 _,records,items,parent=prepare();records,items=records[:64],items[:64];cfg=read(HERE/'config.json')
 assert cfg['queries']==64 and cfg['seed']==41 and cfg['rounds']==3 and cfg['paths']==2 and cfg['max_candidates']==3
 paths=sorted([p for p in HERE.glob('*.py')if not p.name.startswith('test_')]+[HERE/'config.json'])
 assert all((HERE/p).is_file()for p in ('bank.py','policy.py','run.py','diagnose.py','score.py'))
 paths+= [ROOT/p for p in ('scripts/trajectory_fast/evidence.py','scripts/trajectory_opd/evidence.py','scripts/propagation_opd/run.py','scripts/state_opd/run.py','scripts/state_opd/evaluate.py')]
 identity=dict(config=cfg,sources={str(p.relative_to(ROOT)):sha(p)for p in paths},frozen_sources=parent['sources'],
  parent_fingerprint=parent['fingerprint'],parent_lock_sha256=sha(ROOT/'outputs/state_opd_hc128/lock.json'),loader_identity=parent['loader_identity'],
  formal_state_sha256=parent['formal_state_sha256'],registered_config_sha256=parent['registered_config_sha256'],manifest_sha256=parent['manifest_sha256'],
  reference_seal_sha256=parent['reference_seal_sha256'],native_mask_seal_sha256=parent['native_mask_seal_sha256'],inputs=items,ids=[r['id']for r in records],GT_read=False,CPU_only=True)
 fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();lock={**identity,'fingerprint':fp}
 if (TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==lock,'Do not mix code/config/input identities'
 else:
  atomic(TRIAL/'lock.json',lock)
  for p in paths:
   q=TRIAL/'source_snapshot'/p.relative_to(ROOT);q.parent.mkdir(exist_ok=True,parents=True);q.write_bytes(p.read_bytes())
 return cfg,records,items,lock


def _path(trial,phase,rel):
 p=(Path(trial)/phase/rel).resolve();assert p.is_relative_to(Path(trial).resolve());return p


def validate_one(trial,phase,key,item,lock):
 folder=Path(trial)/phase;p=folder/'predictions'/f'{key}.npz';rr=read(p.with_suffix('.json'))
 assert rr['fingerprint']==lock['fingerprint']and rr['GT_read']is False and rr['sha256']==sha(p)
 for stem in ('bank','trace','evidence'):assert sha(_path(trial,phase,rr[stem+'_path']))==rr[stem+'_sha256']
 for field in ('cache_sha256','native_mask_sha256','reference_sha256'):assert rr[field]==item[field]
 bank=torch.load(_path(trial,phase,rr['bank_path']),map_location='cpu',weights_only=False)
 trace=torch.load(_path(trial,phase,rr['trace_path']),map_location='cpu',weights_only=False)
 assert bank['GT_read']is False
 active=torch.as_tensor(bank['active']);valid=torch.as_tensor(bank['valid']);E=torch.as_tensor(bank['E'])
 assert valid.shape==(len(E),3)and valid[:,0].all()and not valid[~active,1:].any()
 assert torch.equal(bank['boxes'][:,0],E)
 with np.load(p,allow_pickle=False)as f:
  assert np.array_equal(f['ERefStrong'],E.numpy())
  names=CONTROLS if phase=='stage1'else ARMS
  for name in names:
   out=trace['controls'][name]if phase=='stage1'else trace[name]
   final=torch.as_tensor(out.get('final',out.get('boxes')))
   assert np.array_equal(f[name],final.numpy())and torch.equal(final[~active],E[~active])
   assert torch.isfinite(final).all()and(final[:,2:]>0).all()
   member=(bank['boxes']==final[:,None,:]).all(-1)&valid
   assert member.any(-1).all(), 'Final output must be an actual valid candidate'
  if phase=='stage1':
   initial=trace['initial_student'];assert torch.equal(initial['boxes'],E)and len(trace['diagnostic_paths'])==2
   paths=trace['diagnostic_paths']+[initial]
   paths += [trace['controls'][a]for a in ('MemoryGreedy','NoMemoryGreedy')]
  else:
   paths=[]
   for arm in ARMS:
    out=trace[arm];assert len(out['rounds'])<=3 and out['costs']['updates']<=3
    assert out['costs']['heavy_forwards']==out['costs']['decoder_calls']==0
    assert torch.as_tensor(out['initialtheta']).eq(0).all()
    paths+=out['trace']['all_rollouts']
  for path in paths:
   assert torch.equal(path['boxes'][~active],E[~active])and not path['actions'][~active].any()
   assert torch.equal(path['memory_before'][~active],path['memory_after'][~active])
 return rr


def validate_phase(trial,phase,lock=None):
 trial=Path(trial).resolve();assert trial==TRIAL.resolve();lock=read(trial/'lock.json')if lock is None else lock
 seal=read(trial/phase/'sealed.json');assert seal['queries']==64 and seal['GT_read']is False
 assert seal['fingerprint']==lock['fingerprint']and seal['lock_sha256']==sha(trial/'lock.json')
 assert set(seal['files'])==set(lock['ids'])
 for p,h in {**lock['sources'],**lock['frozen_sources']}.items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
 assert sha(ROOT/'configs/default.json')==lock['registered_config_sha256']
 assert sha(ROOT/'data/evaluation/development.jsonl')==lock['manifest_sha256']
 assert sha(ROOT/'outputs/state_opd_hc128/lock.json')==lock['parent_lock_sha256']
 assert sha(OLD/'scope_hc2/sealed.json')==lock['reference_seal_sha256']
 assert sha(OLD/'research/native_masks/sealed.json')==lock['native_mask_seal_sha256']
 receipts={}
 for item in lock['inputs']:
  key=item['id'];r=validate_one(trial,phase,key,item,lock);e=seal['files'][key]
  p=_path(trial,phase,e['path']);assert sha(p)==e['sha256']and sha(p.with_suffix('.json'))==e['receipt_sha256']
  assert sha(OLD/'student_cache'/f'{key}.pt')==item['cache_sha256']
  assert sha(OLD/'research/native_masks'/item['mask_path'])==item['native_mask_sha256']
  assert sha(OLD/item['reference_path'])==item['reference_sha256'];receipts[key]=r
 assert not torch.cuda.is_initialized()
 return receipts


def main(stage='evidence',run=False,resume=False):
 cfg,records,items,lock=lock_inputs();phase='stage1'if stage=='evidence'else'stage2';folder=TRIAL/phase
 if not run:print(json.dumps(dict(status='prepared_not_run',fingerprint=lock['fingerprint'],stage=stage)));return
 check_idle();assert not (folder/'sealed.json').exists(),'Already sealed, do not overwrite'
 assert resume or not(folder/'run.json').exists(),'Explicit identical --resume required'
 from scripts.propagation_v0.bank import CandidateBank
 from scripts.propagation_v0.policy import rollout,fit,gumbel_noise,evidence_dp
 if phase=='stage2':
  validate_phase(TRIAL,'stage1',lock)
  gate=read(TRIAL/'gate.json');assert gate['passed']is True,'Diagnostic gate did not pass; do not train'
  assert gate['fingerprint']==lock['fingerprint']and gate['stage1_seal_sha256']==sha(TRIAL/'stage1/sealed.json')
 torch.set_num_threads(1);files={};begin=time.perf_counter()
 atomic(folder/'run.json',dict(status='running',PID=os.getpid(),fingerprint=lock['fingerprint'],GT_read=False,CPU_only=True))
 try:
  for rec,item in zip(records,items):
   key=item['id'];dest=folder/'predictions'/f'{key}.npz'
   if dest.with_suffix('.json').exists():files[key]=validate_one(TRIAL,phase,key,item,lock);continue
   assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
   start=time.perf_counter();costs={};times={}
   if phase=='stage1':
    cp=OLD/'student_cache'/f'{key}.pt';mp=OLD/'research/native_masks'/item['mask_path']
    assert sha(cp)==item['cache_sha256']and sha(mp)==item['native_mask_sha256']
    cache=torch.load(cp,map_location='cpu',weights_only=False);pack=torch.load(mp,map_location='cpu',weights_only=False)
    assert cache['GT_read']is pack['GT_read']is False
    t=time.perf_counter();ev=TrajectoryEvidence(pack,cache,device='cpu');times['evidence_seconds']=time.perf_counter()-t
    t=time.perf_counter();bank=CandidateBank(ev,cache,cfg);times['bank_seconds']=time.perf_counter()-t
    ep=folder/'evidence'/f'{key}.pt';bp=folder/'banks'/f'{key}.pt';savept(ep,ev.private_state());savept(bp,bank.private_state())
    with np.load(OLD/item['reference_path'])as f:gold={k:f[k].copy()for k in f.files}
    assert np.array_equal(gold['E_Ref'],bank.E.numpy())
    result={k:gold[k]for k in ('frame_ids','positions8','Strong_interval')}
    result.update(ERefStrong=gold['E_Ref'],SourceStrong=gold['Source'],GeometryOPD=gold['GeometryOPD'])
    theta=torch.zeros(6,dtype=torch.float64);t=time.perf_counter()
    with torch.no_grad():
     initial=rollout(bank,theta,cfg,'student_greedy');assert torch.equal(initial['boxes'],bank.E)
     paths=[rollout(bank,theta,cfg,'student_sample',noise=gumbel_noise(item['key'],0,j,len(bank.E)))for j in range(2)]
     mem=rollout(bank,theta,cfg,'teacher_greedy');nomem=rollout(bank,theta,cfg,'no_memory_teacher_greedy');dp=evidence_dp(bank)
    controls=dict(MemoryGreedy=mem,NoMemoryGreedy=nomem,EvidenceDP=dp)
    for a,o in controls.items():result[a]=o.get('final',o.get('boxes')).detach().numpy()
    trace=dict(initial_student=initial,diagnostic_paths=paths,controls=controls);times['diagnostic_sequence_seconds']=time.perf_counter()-t
    del cache,pack,ev
   else:
    r1=read(TRIAL/'stage1/predictions'/f'{key}.json');bp=TRIAL/'stage1'/r1['bank_path'];ep=TRIAL/'stage1'/r1['evidence_path']
    bank=CandidateBank.from_state(torch.load(bp,map_location='cpu',weights_only=False))
    with np.load(TRIAL/'stage1/predictions'/f'{key}.npz')as f:result={k:f[k].copy()for k in f.files}
    trace={}
    for arm in ARMS:
     out=fit(bank,item['key'],arm,cfg);trace[arm]=out;result[arm]=out['final'].numpy();costs[arm]=out['costs']
   tp=folder/'traces'/f'{key}.pt';savept(tp,trace);dest.parent.mkdir(parents=True,exist_ok=True)
   temp=dest.with_suffix('.npz.tmp')
   with temp.open('wb')as h:np.savez(h,**result)
   os.replace(temp,dest)
   rr=dict(fingerprint=lock['fingerprint'],sha256=sha(dest),trace_path=str(tp.relative_to(folder)),trace_sha256=sha(tp),
    bank_path=os.path.relpath(bp,folder),bank_sha256=sha(bp),evidence_path=os.path.relpath(ep,folder),evidence_sha256=sha(ep),
    cache_sha256=item['cache_sha256'],native_mask_sha256=item['native_mask_sha256'],reference_sha256=item['reference_sha256'],
    costs=costs,timings=times,total_seconds=time.perf_counter()-start,GT_read=False,development_gate_used=phase=='stage2',model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,CPU_only=True)
   atomic(dest.with_suffix('.json'),rr);files[key]=validate_one(TRIAL,phase,key,item,lock)
   atomic(folder/'progress.json',dict(completed=len(files),total=64,GT_read=False));print(json.dumps(dict(phase=phase,completed=len(files),seconds=rr['total_seconds'])),flush=True)
   del bank,trace,result;gc.collect()
  assert len(files)==64
  for item in items:files[item['id']]=validate_one(TRIAL,phase,item['id'],item,lock)
  for p,h in {**lock['sources'],**lock['frozen_sources']}.items():assert sha(ROOT/p)==h,p
  assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
  atomic(folder/'sealed.json',dict(fingerprint=lock['fingerprint'],queries=64,dataset='hc2',GT_read=False,lock_sha256=sha(TRIAL/'lock.json'),
   files={k:dict(path='predictions/'+k+'.npz',sha256=r['sha256'],receipt_sha256=sha(folder/'predictions'/f'{k}.json'))for k,r in files.items()}))
  atomic(folder/'run.json',dict(status='completed_sealed',queries=64,fingerprint=lock['fingerprint'],seconds=time.perf_counter()-begin,GT_read=False))
 except BaseException:
  atomic(folder/'failure.json',dict(fingerprint=lock['fingerprint'],completed=len(files),traceback=traceback.format_exc(),GT_read=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['evidence','adapt'],default='evidence');p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.stage,a.run,a.resume)
