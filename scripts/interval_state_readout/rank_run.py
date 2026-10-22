"""Existing frozen actionness head on sealed hidden; no backbone/decoder/GPU."""
import os,sys,time,importlib.util,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import read,write,sha,arr
from scripts.interval_state_readout.run import validate
from scripts.interval_state_readout.rank_core import likelihood,choose,posterior_sd,roi_reliability
from scripts.tube_fusion.core import restore_geometry,tensor_digest
from stvg.spatial import evidence_dp
OUT=ROOT/'outputs/rank_interval_readout_v1';PUBLIC=ROOT/'results/rank_interval_readout_v1'
ARMS=['ActionOnly','ActionJoint','ActionPermuted','ActionRaw','JointRaw','UnconstrainedAction','UnconstrainedJoint','PriorMatched','RandomMatched','OldJoint10','StateJoint'];NAMES=['Source','Baseline',*ARMS]

def prepare():
 assert not(OUT/'plan.json').exists();old=read(ROOT/'outputs/interval_state_readout_v1/plan.json');items=[]
 for i in old['items']:
  path=ROOT/'outputs/interval_state_readout_v1/predictions'/f"{i['anonymous']}.npz";ap=ROOT/'outputs/action_interval_readout_v1/predictions'/f"{i['anonymous']}.npz";items.append(dict(i,state_prediction=str(path.relative_to(ROOT)),state_prediction_sha256=sha(path),action_prediction=str(ap.relative_to(ROOT)),action_prediction_sha256=sha(ap)))
 weights=read(ROOT/'outputs/temporal_signal_audit_v1/plan.json')['weights'];sources={str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*.py')}
 p=dict(schema='rank_interval_readout_v1',prior_action_seal_sha256=sha(ROOT/'outputs/action_interval_readout_v1/sealed.json'),items=items,labels=old['labels'],sources=sources,weights=weights,formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),GT_read=False,
 protocol=dict(reason='Frozen actionness has relative event ranking but absolute Bernoulli likelihood over-shrinks under domain calibration error. Replace absolute probabilities with candidate-inside/outside ordinal evidence, not new weights.',head='Reuse content-bound sealed original action_embed logits; zero new head/backbone/decoder/encoder/expert call.',
 loss_calibration='Official vidstg/hcstvg2 experiments EOS_COEF=.3; negative-weighted BCE has logit_eta=logit_weighted+log(.3) at population optimum. Raw controls retained. Checkpoints omit training cfg, so this is a configuration-based correction, not proof of exact calibration.',
 likelihood='Physical-cell-weighted pairwise ordering of cached actionness inside vs outside each same13 candidate. No absolute event probability, no EOS correction effect; raw/debiased arms are explicit identical aliases. ActionJoint weights frames by max(ROI visual-attention enrichment/(1-area),0), not by an event pseudo-label.',
 prior='Candidate must improve likelihood over incumbent and lose at most one incumbent posterior IoU SD in native expected IoU. No scalar mixing weight. Unconstrained controls remove this guard.',
 matched='ActionOnly shares all hidden and candidate info; ActionPermuted changes only area-matched spatial placement; raw logits isolate loss-offset assumption; prior/random matched intervene on exactly ActionJoint queries.',
 candidate='Same12 per-layer/offset MBR plus Global6; all old Source/Baseline/candidates unchanged.',
 budget='One fixed head-based structural trial, no temperature/threshold/weight sweep; one CPU thread/nice15. zero new head calls; previous sealed action_logits reused, no CUDA or new heavy network.',
 adoption='HC64 ActionJoint t/fullv/fixedv positive, better t than ActionOnly and ActionPermuted, Source-relative severe v <=baseline+2; both32 rechecks t>0 and fullv>=0, pooled t lower95>0/fullv lower95>-.5pp. Temporal-only ActionOnly assessed separately, never mislabeled joint contribution.',
 exposure='HC64 failures and aggregate diagnostics used for design; prior HC32/Vid32 results and rescue cases have already been inspected in this task. All cohorts are reused exposed development; not fresh rechecks or independent confirmation. New predictions seal before independent new GT scores. No per-query GT gating.'))
 validate(p);write(OUT/'plan.json',p)

@torch.no_grad()
def execute():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);assert not(OUT/'sealed.json').exists();models={};states={};meta={};rows=[];begin=time.perf_counter();maxerror=0.
 for j,item in enumerate(p['items']):
  start=time.perf_counter()
  for name in ['capture','old_trial_prediction','state_prediction','signal','pack','action_prediction']:assert sha(ROOT/item[name])==item[name+'_sha256']
  cap=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['old_trial_prediction']);state=np.load(ROOT/item['state_prediction']);z=np.load(ROOT/item['signal']);pack=torch.load(ROOT/item['pack'],map_location='cpu',weights_only=False);io=time.perf_counter()-start
  ids=np.asarray(cap['frame_ids']);cached=ROOT/'outputs/action_interval_readout_v1/predictions'/f"{item['anonymous']}.npz";previous=np.load(cached);logits=previous['action_logits'];tick=time.perf_counter();headtime=time.perf_counter()-tick;tick=time.perf_counter();intervals=old['candidate_intervals'];u=old['candidate_utilities'];sd=posterior_sd(ids,pack['hull'],old['Baseline_interval']);cal=logits+np.log(.3);real=roi_reliability(z);perm=roi_reliability(z,True)
  scores=dict(ActionOnly=likelihood(ids,intervals,cal),ActionJoint=likelihood(ids,intervals,cal,real),ActionPermuted=likelihood(ids,intervals,cal,perm),ActionRaw=likelihood(ids,intervals,logits),JointRaw=likelihood(ids,intervals,logits,real));selected={a:choose(u,v,sd)for a,v in scores.items()};selected['UnconstrainedAction']=choose(u,scores['ActionOnly'],sd,False);selected['UnconstrainedJoint']=choose(u,scores['ActionJoint'],sd,False);other=np.arange(1,len(u));active=selected['ActionJoint']!=0;selected['PriorMatched']=int(other[np.argmax(u[1:])])if active else 0;selected['RandomMatched']=int(np.random.default_rng(20261022+j).choice(other))if active else 0;scoretime=time.perf_counter()-tick
  values={k:old[k]for k in ['Source','Source_interval','Baseline','Baseline_interval','FixedSpatial','frame_ids','positions8','candidate_intervals','candidate_utilities']};values.update(OldJoint10=old['Joint10'],OldJoint10_interval=old['Joint10_interval'],StateJoint=state['StateJoint'],StateJoint_interval=state['StateJoint_interval']);geom=restore_geometry(cap['geometry']);memo={tuple(old['Baseline_interval']):old['Baseline']};cost={}
  for arm,k in selected.items():
   span=intervals[k];key=tuple(span);tick=time.perf_counter()
   if key not in memo:memo[key]=arr(evidence_dp(geom.for_interval(span.tolist()))['final'])
   values[arm]=memo[key];values[arm+'_interval']=span;cost[arm]=time.perf_counter()-tick
  values.update(action_logits=logits,action_corrected_logits=cal,roi_reliability=real,permuted_reliability=perm,posterior_sd=np.asarray(sd),**{'score_'+k:v for k,v in scores.items()});dest=OUT/'predictions'/f"{item['anonymous']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,**values)
  rows.append(dict(query=item['anonymous'],cohort=item['cohort'],path=str(dest.relative_to(OUT)),sha256=sha(dest),GT_read=False,selected=selected,input_IO_seconds=io,head_seconds=headtime,score_seconds=scoretime,DP_seconds=cost,CPU_head_calls=0));write(OUT/'progress.json',dict(completed=j+1,total=128,GT_read=False))
  if j%16==0:print(j+1,'/128',round(time.perf_counter()-begin,2),flush=True)
 validate(p);assert not torch.cuda.is_initialized();write(OUT/'sealed.json',dict(files=rows,GT_read=False,plan_sha256=sha(OUT/'plan.json'),CPU_frozen_head_calls=0,reused_previous_head_predictions=True,manual_float64_head_maxerror=read(ROOT/'outputs/action_interval_readout_v1/sealed.json')['manual_float64_head_maxerror'],heavy_network_calls=0,GPU_calls=0,parameter_updates=0,seconds=time.perf_counter()-begin));print('sealed128',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','execute']);x=a.parse_args();globals()[x.action]()
