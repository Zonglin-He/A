"""CPU cached whole-interval trial; actual predictions seal before scoring."""
import os,sys,time,json,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import sha,read,write,arr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
from stvg.tube_replay import occupancy_prior
from scripts.interval_state_readout.core import detrend,segment_gain,choose
OUT=ROOT/'outputs/interval_state_readout_v1';PUBLIC=ROOT/'results/interval_state_readout_v1'
ARMS=['StateHidden','StateSpatial','StateJoint','StatePermuted','StateGlobal','PriorMatched','RandomMatched','OldJoint10'];NAMES=['Source','Baseline',*ARMS]

def diagnosis():
 """HC64 development-only GT case audit of the already scored old trial."""
 old=read(ROOT/'outputs/joint_boundary_readout_v2/plan.json');result=read(ROOT/'results/joint_boundary_readout_v2/report.json');lp=ROOT/'data/evaluation/development_labels.jsonl';labels={g['id']:g for g in [json.loads(l)for l in lp.read_text().splitlines()if l.strip()]};rows=[]
 from scripts.joint_boundary_readout.core import iou
 for item,r in zip(old['items'],result['queries_anonymous']):
  if item['cohort']!='HC64' or r['values']['Baseline']['t']>=.5 or r['candidate_best_tIoU']<.5:continue
  z=np.load(ROOT/'outputs/joint_boundary_readout_v2/predictions'/f"{item['anonymous']}.npz");truth=labels[item['id']]['interval'];q=iou(z['candidate_intervals'],truth);k=int(q.argmax());f=z['candidate_features'];u=z['candidate_utilities'];cap=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);obs=np.asarray(cap['frame_ids'])[cap['planned_positions']]
  rows.append(dict(query=item['anonymous'],base_tIoU=float(q[0]),best_candidate_tIoU=float(q[k]),best_candidate_index=k,prior_expected_iou_loss=float(u[0]-u[k]),old_baseline_features=f[0].tolist(),old_correct_features=f[k].tolist(),old_correct_evidence_rank=int((f[:,1]>f[k,1]).sum()+1),old_correct_prior_rank=int((u>u[k]).sum()+1),positive_evidence_but_prior_blocks=bool(f[k,1]>f[0,1]and u[k]+.25*f[k,1]<=u[0]+.25*f[0,1]),correct_evidence_zero=bool(f[k,1]==0),observations_inside_GT=int(((obs>=truth[0])&(obs<truth[1])).sum()),baseline_start_before_GT=bool(z['Baseline_interval'][0]<truth[0]),candidate_count=len(u)))
 assert len(rows)==8
 write(OUT/'development_case_audit.json',dict(status='diagnosed_before_new_readout',GT_read=True,development_only=True,HC32_Vid32_new_case_GT_not_inspected=True,rows=rows,summary=dict(cases=8,correct_candidate_top_evidence=sum(r['old_correct_evidence_rank']==1 for r in rows),positive_evidence_but_prior_blocks=sum(r['positive_evidence_but_prior_blocks']for r in rows),correct_evidence_zero=sum(r['correct_evidence_zero']for r in rows),baseline_starts_early=sum(r['baseline_start_before_GT']for r in rows))))

def validate(p):
 for k,v in p['sources'].items():assert sha(ROOT/k)==v,k
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256'];assert all(sha(ROOT/k)==v for k,v in formal['sources'].items())
def prepare():
 assert not(OUT/'plan.json').exists();diagnosis();old=read(ROOT/'outputs/joint_boundary_readout_v2/plan.json');items=[]
 for item in old['items']:
  path=ROOT/'outputs/joint_boundary_readout_v2/predictions'/f"{item['anonymous']}.npz";items.append(dict(item,old_trial_prediction=str(path.relative_to(ROOT)),old_trial_prediction_sha256=sha(path)))
 p=dict(schema='interval_state_readout_v1',items=items,labels=old['labels'],sources={str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*.py')},development_case_audit_sha256=sha(OUT/'development_case_audit.json'),formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),GT_read=False,
 protocol=dict(diagnosis='Eight HC64 known old-baseline-wrong and candidate-rescuable cases inspected before design; recheck cases not newly inspected. All cohorts historically exposed.',candidate_pool='Same old12 view MBR plus incumbent; no new oracle boundary proposals.',
 representation='Continuous1.5x full-timeline complete-box ROI on actual cached text-conditioned motion memory; normalized ROI minus normalized global motion; fixed per-query intercept/time/time-squared detrending. Same construction for area-preserving spatial permutation. Last native hidden similarly normalized and detrended.',
 whole_interval='Physical-cell-weighted prefix/event/suffix mean-state explained energy; H from native hidden, S from target motion gain minus global-motion gain.',
 decision='Strict improvement over incumbent in BOTH H and S; maximize weaker ordinal-rank improvement, native utility tie-break only. No mixing weight, no confidence threshold. Hidden/spatial-only, permuted/global matched structural controls.',
 matched='PriorMatched and fixed-seed RandomMatched change on exactly the label-free StateJoint intervention queries. Same candidate budget; no GT gating.',
 budget='One structural revision, no weight/temperature search. Single CPU thread/nice15; no RGB, model/decoder/head/expert/GPU/update.',
 adoption='HC64 must have positive t/fullv/fixedv and outperform StateHidden/StatePermuted in t, Source-relative severe v harm <=baseline+2; then both frozen32 groups positive t and nonnegative fullv, pooled equal-domain t lower95>0 and v lower95>-.5pp. No recheck-driven redesign.'))
 validate(p);write(OUT/'plan.json',p)

@torch.no_grad()
def representation(cap,boxes):
 T=len(cap['frame_ids']);curves={k:np.zeros((T,256))for k in ['target','permuted','global','hidden']}
 for o in cap['temporal_cache']['offsets']:
  ix=np.asarray(o['frame_positions'],int);h,w=o['grid_size'];n=h*w;i=o['inputs'];memory=i['encoded_feature'][-n:].permute(1,0,2).double();legal=~i['encoded_mask'][:,-n:];M,_=occupancy_prior(boxes[ix],(h,w),legal);P,_=occupancy_prior(boxes[ix],(h,w),legal,permuted=True)
  global_f=(memory*legal[:,:,None]).sum(1)/legal.sum(-1).clamp_min(1)[:,None];fg=(memory*M[:,:,None]).sum(1)/M.sum(-1).clamp_min(1e-12)[:,None];pg=(memory*P[:,:,None]).sum(1)/P.sum(-1).clamp_min(1e-12)[:,None]
  normalize=lambda x:x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  curves['target'][ix]=arr(normalize(fg)-normalize(global_f));curves['permuted'][ix]=arr(normalize(pg)-normalize(global_f));curves['global'][ix]=arr(global_f);curves['hidden'][ix]=arr(o['hidden'][-1,0])
 ids=np.asarray(cap['frame_ids'],float);return {k:detrend(v,ids)for k,v in curves.items()}

def execute():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);assert not(OUT/'sealed.json').exists();files=[];start=time.perf_counter()
 for j,item in enumerate(p['items']):
  tick=time.perf_counter()
  for name in ['capture','old_trial_prediction']:assert sha(ROOT/item[name])==item[name+'_sha256']
  cap=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['old_trial_prediction']);io=time.perf_counter()-tick;geom=restore_geometry(cap['geometry']);aux=evidence_dp(geom.for_full_timeline());ids=cap['frame_ids'];intervals=old['candidate_intervals'];u=old['candidate_utilities']
  tick=time.perf_counter();curves=representation(cap,arr(aux['final']));gains={k:segment_gain(v,ids,intervals)for k,v in curves.items()};featuretime=time.perf_counter()-tick;H=gains['hidden'];S=gains['target']-gains['global'];P=gains['permuted']-gains['global'];G=gains['global']
  selected=dict(StateHidden=choose(u,H,S,'hidden'),StateSpatial=choose(u,H,S,'spatial'),StateJoint=choose(u,H,S),StatePermuted=choose(u,H,P),StateGlobal=choose(u,H,G));joint=selected['StateJoint'];others=np.arange(1,len(u));selected['PriorMatched']=int(others[np.argmax(u[1:])])if joint and len(others)else 0;selected['RandomMatched']=int(np.random.default_rng(20261022+j).choice(others))if joint and len(others)else 0
  values={k:old[k]for k in ['Source','Source_interval','Baseline','Baseline_interval','FixedSpatial','frame_ids','positions8','candidate_intervals','candidate_utilities']};values.update(OldJoint10=old['Joint10'],OldJoint10_interval=old['Joint10_interval']);memo={tuple(values['Baseline_interval']):values['Baseline']};costs={}
  for arm,k in selected.items():
   span=intervals[k];key=tuple(span);tick=time.perf_counter()
   if key not in memo:memo[key]=arr(evidence_dp(geom.for_interval(span.tolist()))['final'])
   values[arm]=memo[key];values[arm+'_interval']=span;costs[arm]=time.perf_counter()-tick
  restored=evidence_dp(geom.for_interval(values['Baseline_interval'].tolist()));assert np.array_equal(arr(restored['final']),values['Baseline'])
  values.update(**{'gain_'+k:v for k,v in gains.items()},spatial_gain=S,permuted_gain=P)
  dest=OUT/'predictions'/f"{item['anonymous']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**values)
  fp=OUT/'representations'/f"{item['anonymous']}.npz";fp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(fp,**curves)
  files.append(dict(query=item['anonymous'],cohort=item['cohort'],path=str(dest.relative_to(OUT)),sha256=sha(dest),representation_path=str(fp.relative_to(OUT)),representation_sha256=sha(fp),selected=selected,input_IO_seconds=io,representation_and_gain_seconds=featuretime,DP_seconds=costs,GT_read=False))
  write(OUT/'progress.json',dict(completed=j+1,total=128,GT_read=False))
  if j%16==0:print(j+1,'/128',round(time.perf_counter()-start,2),flush=True)
 validate(p);assert not torch.cuda.is_initialized();write(OUT/'sealed.json',dict(files=files,plan_sha256=sha(OUT/'plan.json'),GT_read=False,network_calls=0,CUDA_initialized=False,seconds=time.perf_counter()-start));print('sealed128',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','execute']);x=a.parse_args();globals()[x.action]()
