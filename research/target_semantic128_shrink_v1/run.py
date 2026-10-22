"""Predeclared eight numeric configurations, four families, parent-heldout confirmation."""
import os,sys,time,json,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put
from research.target_semantic128_shrink_v1.core import KAPPAS,BETAS,arms
A=R/'outputs/target_semantic128_collect_v3';D=R/'outputs/target_semantic128_shrink_v1';P=R/'results/target_semantic128_shrink_v1'
BASE=['NativeG6','Loop2','Loop1'];SHRINK=['ShrinkK'+str(int(k))for k in KAPPAS];WEAK=['WeakB'+f'{int(b*100):03d}'for b in BETAS];ARMS=BASE+[x+s for x in SHRINK for s in ['', 'Random','WrongTime','Reflected']]+WEAK

def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(A/'sealed.json')==p['acquisition_seal_sha256']and sha(A/'plan.json')==p['acquisition_plan_sha256']
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 return p

def prepare():
 assert not(D/'plan.json').exists();lock=read(D/'design_lock.json');assert all(sha(R/f)==h for f,h in lock['code'].items());a=read(A/'plan.json');seal=read(A/'sealed.json');assert seal['count']==256 and seal['GT_read']is False;paper=read(R/'outputs/tastvg_paper_v1/plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/native_event_semantic_prototype_v1/core.py','research/native_event_target_semantic_v1/score.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='target_semantic128_shrink_v1',design_lock_sha256=sha(D/'design_lock.json'),primary_family='OneRoundPrototypeShrink',four_families=['Loop2','Loop1','SelectedShrink','SelectedWeakEvidence'],arms=ARMS,kappa_grid=list(KAPPAS),beta_grid=list(BETAS),items=a['items'],count=256,acquisition_plan_sha256=sha(A/'plan.json'),acquisition_seal_sha256=sha(A/'sealed.json'),code=code,paper_code=a['paper_code'],provider_sources=a['provider_sources'],labels={ds:dict(path=j['labels'],sha256=j['labels_sha256'])for ds,j in paper['jobs'].items()},
 authorization='Latestuser cancels07cutoff, expands128data and finite tuning, proposes one-round/shrinkage versus separate weaker-feedback. Spatial/singleexpert8 unchanged.',
 hypothesis='Second-round prototype re-estimation may amplify mistaken initial softmembership. Crossfit muPlus/muMinus shrink toward other-parity unconditional mu0 with kappa effective frame pseudo-count. Covariance/likelihood/ROI/classes unchanged. Weak beta is an independent scale control, never stacked with shrink.',
 split='Each direction128 inputSHA-selected distinct officialparents, onequeryeach.64develop/64confirmation. Prospectivecandidate holdout only; allofficialfull historicallyexposed. No confirmation score before frozen config.',
 selection='Tune kappa family and beta-control independently, THREE strengths each on same64develop/domain only. Feasible: EACHdomain actualt/v gains beyondSource/G6 >0, bothSource short<=5s/long>5s t positive, timewins>harms, severe t/v counts no higher thanLoop2. Amongfeasible maximize equally weighted two-domain mean complete-v gain overG6; ties lowernumericstrength. Ifnonefeasible, freeze maximumsameutility only as explicit unqualified diagnostic, notfulladmission. No repeated selection/stacking/extra capacity.',
 confirmation='Exactlyfour frozen families plusSource/G6 and selectedShrink Random/WrongTime/Reflected matched controls. Unselected config outputs sealed but NEVER confirmationGT-scored. Oneconfirmationonly, no retuning. Fulladmission requires developeligible+eachconfirmationdomain Source/G6 actualt/fullvpositive; Source short/longpositive; wins>harms; per-domain seriousloss counts <=Loop2 and total seriousloss strictlyfewer ifLoop2hasany; selectedShrink t mean exceeds Random/WrongTime/Reflected inbothdomains. Compare selectedWeak separately; do not claim prototype accuracy improvement unless evidence beyond weakening supportsit. No lowerlearningcapacity/gating/nativeabstention accepted as sole success.',
 unchanged='Originaltarget1.5context, oppositeoffset prototypes, classSUM/inclusiveSUM, native q0prior, originalphysical MBR andcompleteboxDP/same8evidence. No GT in inference and no perquery choice byGTduration.',bootstrap_seed=20261023,new_model=0,new_expert=0,new_observations=0,new_training=0,GT_in_inference=False,all256_candidate_predictions_seal_before_GT=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','code','paper_code','provider_sources','labels']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def predict():
 import numpy as np,torch
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 torch.set_num_threads(1);p=verify();seal=read(A/'sealed.json');files={};start=time.perf_counter();put(D/'state.json',dict(status='predicting_CPU_no_GT',fingerprint=p['fingerprint'],pid=os.getpid()))
 for item in p['items']:
  key=item['alias'];e=seal['files'][key];assert sha(A/e['path'])==e['sha256']and sha(A/e['capture_path'])==e['capture_sha256'];old=torch.load(A/e['path'],map_location='cpu',weights_only=False);cap=torch.load(A/e['capture_path'],map_location='cpu',weights_only=False);probs=old['semantic_p'];qs,tr=arms(probs['TargetContext'],old['q0'],old['random_p'],probs['WrongTimeContext'],probs['ReflectedContext']);assert set(qs)==set(ARMS)and np.array_equal(qs['Loop2'],old['original_loop2'])and np.array_equal(qs['Loop1'],old['original_loop1']);ids=cap['frame_ids'];geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arr={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS};assert np.array_equal(arr['NativeG6'],old['NativeG6'])and iv['NativeG6']==old['NativeG6_interval'];arr['Source']=old['Source'];iv['Source']=old['Source_interval'];f=D/'predictions'/f'{key}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=old['positions8']);tf=D/'traces'/f'{key}.pt';tf.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,steps=tr,GT_read=False,original_one_two_and_native_exact=True),tf);files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tf.relative_to(D)),trace_sha256=sha(tf));put(D/'progress.json',dict(completed=len(files),total=256,seconds=time.perf_counter()-start,pid=os.getpid(),GT_read=False));print(key,iv['Loop2'],iv['ShrinkK4'],flush=True)
 verify();put(D/'sealed.json',dict(count=256,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,seconds=time.perf_counter()-start,new_model=0));put(D/'state.json',dict(status='sealed_before_development_GT',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,pid=os.getpid()));raise
