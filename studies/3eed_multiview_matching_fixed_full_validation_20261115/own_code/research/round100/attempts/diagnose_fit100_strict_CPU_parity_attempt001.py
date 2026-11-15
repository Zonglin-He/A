"""CPU-only mechanistic analysis of sealed R99 heads; never changes predictions/weights."""
from pathlib import Path
import json,hashlib,numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round100';D=H/'fit_diagnostic_R99';B=R/'crossplatform/round99';OLD=R/'crossplatform/round98';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(p):z=np.load(p,allow_pickle=False);return {k:z[k] for k in z.files}
def score(a,f,st,r0):
 raw=F.linear(a,torch.as_tensor(st['weight']),torch.as_tensor(st['bias']))+torch.as_tensor(f['late']);q=F.normalize(raw,dim=-1);u=torch.as_tensor(f['u']);mask=torch.as_tensor(f['mask']).bool();s=q@u.T/.07;return (torch.logsumexp(s[:,mask],-1)-torch.logsumexp(s[:,~mask],-1)).double()+r0
def center(x):return x-x.mean()
def main():
 torch.set_num_threads(4);D.mkdir(exist_ok=False);coll=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json');rows=[];maxp=0.;mismatch=0;allsummary=[]
 with torch.no_grad():
  for src in ['waymo','drone','quad']:
   S=load(R/coll['modules'][src]['state_path']);prefix=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU()).eval().requires_grad_(False)
   for i in [0,2]:prefix[i].weight.copy_(torch.as_tensor(S[f'match.{i}.weight']));prefix[i].bias.copy_(torch.as_tensor(S[f'match.{i}.bias']))
   initial={'weight':S['match.4.weight'],'bias':S['match.4.bias']}
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;cohort=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');labels={r['id_PRIVATE']:r for r in rd(R/'research/round99/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))}
    for ix,r in enumerate(cohort):
     sid=r['id'];f3=load(B/name/'features_PRIVATE'/(sid+'.npz'));fc=load(OLD/name/'features_PRIVATE'/(sid+'.npz'));ev=load(OLD/name/'evidence_PRIVATE'/(sid+'.npz'));ref=load(B/name/'samples_PRIVATE'/(sid+'.npz'));old=load(OLD/name/'samples_PRIVATE'/(sid+'.npz'));gpu={**dict(zip(old['arm_names'],old['scores'])),**dict(zip(ref['arm_names'],ref['scores']))};ff=[{k:f3[str(v)+'_'+k] for k in ['h','u','late','mask']} for v in range(3)]+[{k:fc['C_'+k] for k in ['h','u','late','mask']}];aa=[prefix(torch.as_tensor(f['h'])) for f in ff];r0=torch.as_tensor(ev['r0']);delta=torch.as_tensor(ev['delta']);source_scores=[score(a,f,initial,r0) for a,f in zip(aa,ff)];targets=[s+delta for s in source_scores];states={'source':initial,'one_view':load(OLD/name/'states_PRIVATE'/(sid+'_lm.npz')),'three_view':load(B/name/'states_PRIVATE'/(sid+'_joint3.npz'))};mse={};sc={};cpu_bad=False
     for label,st in states.items():
      ss=[score(a,f,st,r0) for a,f in zip(aa,ff)];sc[label]=ss;mse[label]={v:float(center(t-s).square().mean()/2) for v,t,s in zip(['F','A','B','C'],targets,ss)}
      for vi,key in [(0,{'source':'prepared_frozen','one_view':'lm_F','three_view':'joint3_F'}[label]),(3,{'source':'frozen_C','one_view':'lm_C','three_view':'joint3_C'}[label])]:
       p=ss[vi].softmax(0).numpy();err=float(abs(p-gpu[key]).max());maxp=max(maxp,err);assert err<1e-6;cpu_bad|=int(p.argmax())!=int(gpu[key].argmax())
     mismatch+=int(cpu_bad);qa=int(gpu['lm_FC'].argmax());qb=int(gpu['joint3_FC'].argmax());margins={state:{v:float(gpu[pref+'_'+v][qb]-gpu[pref+'_'+v][qa]) for v in ['F','C','FC']} for state,pref in [('one_view','lm'),('three_view','joint3')]};fchange=int(gpu['lm_F'].argmax())!=int(gpu['joint3_F'].argmax());cchange=int(gpu['lm_C'].argmax())!=int(gpu['joint3_C'].argmax());metric=labels[sid]['arms'];row={'source':src,'target':tgt,'id_PRIVATE':sid,'mse':mse,'cpu_readout_Top1_mismatch':bool(cpu_bad),'FC_Top1_changed':qa!=qb,'F_Top1_changed':fchange,'C_Top1_changed':cchange,'FC_change_without_F_or_C_Top1_change':qa!=qb and not fchange and not cchange,'probability_pair_margins':margins,'task_changes':{v:metric['joint3_'+v]['IoU']-metric['lm_'+v]['IoU'] for v in ['F','C','FC']},'activation_drift_F_to_C_mean':float(((aa[0]-aa[3]).norm(dim=-1)/aa[0].norm(dim=-1).clamp_min(1e-12)).mean())};rows.append(row)
    part=rows[-128:]
    for state in states:
     for view in ['F','A','B','C']:allsummary.append({'source':src,'target':tgt,'state':state,'view':view,'n':128,'mean_centered_score_fit_error':float(np.mean([r['mse'][state][view] for r in part]))})
    print('FIT_DIAGNOSTIC100',name,flush=True)
 assert len(rows)==768;dump(D/'ALL_ROWS_PRIVATE.json',rows);dump(D/'ERROR_SUMMARY_ROWS.json',allsummary);changed=[r for r in rows if r['FC_Top1_changed']];out={'state':'PASS_CPU_HEAD_REPLAY_DIAGNOSTIC_NOT_NEW_TTA','n':768,'max_CPU_GPU_probability_error':maxp,'CPU_GPU_Top1_mismatch_inputs':mismatch,'FC_Top1_changes':len(changed),'FC_changes_without_F_or_C_Top1_changes':sum(r['FC_change_without_F_or_C_Top1_change'] for r in changed),'mIoU_FC_better_n':sum(r['task_changes']['FC']>1e-12 for r in rows),'mIoU_FC_worse_n':sum(r['task_changes']['FC']< -1e-12 for r in rows),'no_new_GT_or_weights_updates':True,'FINAL_access':False};dump(D/'AUDIT.json',out);dump(D/'SEALED.json',{'files':{p.name:sha(p) for p in D.glob('*.json')}});print(out)
if __name__=='__main__':main()
