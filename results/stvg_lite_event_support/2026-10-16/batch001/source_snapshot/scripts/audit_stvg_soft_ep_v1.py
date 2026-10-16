#!/usr/bin/env python3
"""Independent CPU readback of sealed Soft EP scores and saved optimizer witnesses."""
import argparse,hashlib,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
ARMS=('F','T','S_M0','TS_M0','S_M1','TS_M1','S_M2','TS_M2')
METRICS=('m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_supervised_GT','m_sIoU_unsupervised_GT')
PAIRS=(('S_M2_minus_S_M1','S_M2','S_M1'),('TS_M2_minus_TS_M1','TS_M2','TS_M1'),('S_M2_minus_S_M0','S_M2','S_M0'),('TS_M2_minus_TS_M0','TS_M2','TS_M0'),('S_M1_minus_S_M0','S_M1','S_M0'),('TS_M1_minus_TS_M0','TS_M1','TS_M0'),('S_M0_minus_F','S_M0','F'),('TS_M0_minus_T','TS_M0','T'),('S_M1_minus_F','S_M1','F'),('TS_M1_minus_T','TS_M1','T'),('S_M2_minus_F','S_M2','F'),('TS_M2_minus_T','TS_M2','T'))
TOL=1e-6; SIGN=1e-10
def req(x,m):
 if not x: raise RuntimeError(m)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x): return x.item() if isinstance(x,np.generic) else ({k:clean(v) for k,v in x.items()} if isinstance(x,dict) else [clean(v) for v in x] if isinstance(x,list) else x)
def arr(x): return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x)
def err(a,b):
 a,b=np.asarray(a),np.asarray(b); return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
def treeeq(a,b):
 import torch
 if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(treeeq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(treeeq(x,y) for x,y in zip(a,b))
 return a==b
def loadpt(p):
 import torch,numpy as n
 safe=[n.ndarray,n.dtype]
 for mod in (getattr(n,'core',None),getattr(n,'_core',None)):
  ma=getattr(mod,'multiarray',None) if mod is not None else None
  for k in ('_reconstruct','scalar'):
   if ma is not None and hasattr(ma,k): safe.append(getattr(ma,k))
 safe += [v for t in (n.float16,n.float32,n.float64,n.int8,n.int16,n.int32,n.int64,n.uint8,n.bool_) for v in (n.dtype(t).type,type(n.dtype(t)))]
 with torch.serialization.safe_globals(safe): return torch.load(p,map_location='cpu',weights_only=True)
def xy(b): return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
def iou(a,b):
 x,y=xy(a),xy(b); inter=np.maximum(0,np.minimum(x[...,2:],y[...,2:])-np.maximum(x[...,:2],y[...,:2])).prod(-1)
 aa=np.maximum(0,x[...,2:]-x[...,:2]).prod(-1);bb=np.maximum(0,y[...,2:]-y[...,:2]).prod(-1)
 return inter/np.maximum(aa+bb-inter,1e-7)
def file_rows(rows):
 out={}
 for x in rows:
  p=ROOT/x['path'];req(p.is_file() and sha(p)==x['sha256'],f"file SHA mismatch: {x['path']}")
  if 'bytes'in x:req(p.stat().st_size==x['bytes'],f"byte mismatch: {x['path']}")
  req(x['path'] not in out,'duplicate barrier path');out[x['path']]=x
 return out
def scalar_audit(base,expected_ids):
 rows=read(base/'POSTSEAL_SCORE_ROWS.json'); sm=read(base/'POSTSEAL_SCORE_SUMMARY.json')['summary']; pp=read(base/'POSTSEAL_PAIRED_CONTRASTS.json')['paired_contrasts']
 rr=rows['rows'];req(len(rr)==256,'score row count');by={(r['case_id'],r['arm']):r for r in rr};req(len(by)==256 and {a for _,a in by}==set(ARMS) and {c for c,_ in by}==set(expected_ids),'duplicate/missing arm or roster rows');req(all(r['dataset']==expected_ids[r['case_id']] for r in rr),'row dataset differs from roster')
 maxe=0.; summaries=[];paired=[]
 for ds in ('hc2','vidstg'):
  ids=sorted({r['case_id'] for r in rr if r['dataset']==ds});req(len(ids)==16,f'{ds} cases')
  for arm in ARMS:
   for metric in METRICS:
    vals=[];unknown=[]
    for c in ids:
     v=by[c,arm]['metrics'][metric]
     if v is None:unknown.append(c)
     else:req(math.isfinite(float(v)),f'nonfinite {metric}');vals.append(float(v))
    summaries.append(dict(dataset=ds,arm=arm,metric=metric,n_valid=len(vals),n_unknown=len(unknown),unknown_case_ids=unknown,mean=math.fsum(vals)/len(vals) if vals else None,median=statistics.median(vals) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None))
  for name,left,right in PAIRS:
   for metric in METRICS:
    vals={};unknown=[]
    for c in ids:
     a,b=by[c,left]['metrics'][metric],by[c,right]['metrics'][metric]
     if a is None or b is None:unknown.append(c)
     else:vals[c]=100*(float(a)-float(b))
    paired.append(dict(dataset=ds,contrast=name,comparison_arm=left,baseline_arm=right,metric=metric,n_pairs=len(vals),unknown_count=len(unknown),unknown_case_ids=unknown,positive_case_ids=sorted(c for c,v in vals.items() if v>SIGN),negative_case_ids=sorted(c for c,v in vals.items() if v< -SIGN),neutral_case_ids=sorted(c for c,v in vals.items() if abs(v)<=SIGN),neutral_tolerance_pp=SIGN,mean_delta_pp=math.fsum(vals.values())/len(vals) if vals else None,per_source_delta_pp=vals))
 def cmp(actual,expected,key):
  nonlocal maxe
  aa={tuple(x[k] for k in key):x for x in actual};ee={tuple(x[k] for k in key):x for x in expected};req(aa.keys()==ee.keys(),f'{key} membership mismatch')
  for ix,a in aa.items():
   e=ee[ix];req(a.keys()==e.keys(),f'scalar schema mismatch {ix}')
   for k,v in e.items():
    if isinstance(v,float):
     d=abs(v-a[k]);maxe=max(maxe,d);req(d<=1e-12,f'scalar mismatch {ix}/{k}: {d}')
    elif isinstance(v,dict) and k=='per_source_delta_pp':
     req(v.keys()==a[k].keys(),f'per-source membership {ix}');
     for c,x in v.items():d=abs(x-a[k][c]);maxe=max(maxe,d);req(d<=1e-12,f'paired delta mismatch {ix}/{c}')
    else:req(v==a[k],f'scalar field mismatch {ix}/{k}')
 cmp(sm,summaries,('dataset','arm','metric'));cmp(pp,paired,('dataset','contrast','metric'))
 tails=[dict(dataset=x['dataset'],contrast=x['contrast'],metric=x['metric'],positive=len(x['positive_case_ids']),negative=len(x['negative_case_ids']),neutral=len(x['neutral_case_ids']),unknown=x['unknown_count']) for x in paired]
 return dict(status='pass',row_count=len(rr),summary_count=len(sm),paired_count=len(pp),max_abs_reaggregation_error=maxe,tails=tails)
def rollout_audit(r,ev,weights,cfg,pos,key,step):
 ro=r['rollout'];mean=arr(ro['mean']).astype(np.float64);samples=arr(ro['samples']).astype(np.float64);saved=arr(ro['rewards']).astype(np.float64);used=arr(ro['used_rewards']).astype(np.float64);q=arr(ro['weights']).astype(np.float64)
 req(ro['permutations'] is None and mean.shape==(len(pos),4) and samples.shape==(len(pos),32,4),'rollout shapes/control')
 seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{step}'.encode()).hexdigest()[:15],16);req(int(ro['seed'])==seed,'rollout seed')
 bx=arr(r['boxes_before']).astype(np.float64)[pos];mlog=np.log(bx/(1-bx));meanerr=err(mean,mlog);anti=err(samples[:,:16]+samples[:,16:],2*mean[:,None,:]);rewarderr=err(iou(1/(1+np.exp(-samples)),ev[:,None,:]),saved)
 req(meanerr<=TOL and anti<=TOL and rewarderr<=2e-6,'rollout mean/antithetic/IoU reward')
 req(err(used,saved)<=TOL,'used reward differs from unshuffled reward');logits=used/float(cfg['tau']);logits-=logits.max(-1,keepdims=True);qq=np.exp(logits);qq/=qq.sum(-1,keepdims=True);qerr=err(qq,q);req(qerr<=TOL and np.max(np.abs(q.sum(-1)-1))<=TOL,'softmax q')
 req(err(r['reliability'],weights)<=TOL,'outer anchor reliability');kl=np.zeros(len(pos));logp=-((samples-mean[:,None,:])**2).sum(-1)/(2*cfg['sigma']**2)-2*math.log(2*math.pi*cfg['sigma']**2);pa=kl-((q-1/32)*logp).sum(-1);loss=float((weights*pa).mean());losserr=max(err(pa,r['per_anchor_loss']),abs(loss-float(r['loss'])))
 referr=err(iou(bx,ev),r['reference_iou_before']);req(losserr<=TOL and referr<=TOL,'Gaussian loss or central reference IoU');info=bool(np.any(used!=used[:,:1]));req(bool(ro['informative'])==info,'informative flag')
 return dict(reward=max(meanerr,anti,rewarderr),q=qerr,loss=losserr,reference_iou=referr,informative=info)
def _optimizer(row,names):
 g=arr(row['gradient']).astype(np.float64).reshape(-1); d=arr(row['original_adam_delta']).astype(np.float64).reshape(-1)
 before,proposed,after=(row[k] for k in ('optimizer_before','optimizer_proposed','optimizer_after')); group=proposed['param_groups'][0]
 beta1,beta2=group['betas'];lr=float(group['lr']);eps=float(group['eps']);ids=group['params'];req(len(ids)==len(names),'Adam parameter order mismatch')
 me=ve=de=0.;off=0
 for i,(idx,name) in enumerate(zip(ids,names)):
  width=arr(row['state_before'][name]).size; gg=g[off:off+width];off+=width;old=before['state'].get(idx);new=proposed['state'][idx]
  mp=np.zeros(width) if old is None else arr(old['exp_avg']).astype(np.float64).ravel();vp=np.zeros(width) if old is None else arr(old['exp_avg_sq']).astype(np.float64).ravel()
  m=(beta1*mp+(1-beta1)*gg).astype(np.float32);v=(beta2*vp+(1-beta2)*gg*gg).astype(np.float32);me=max(me,err(m,new['exp_avg']));ve=max(ve,err(v,new['exp_avg_sq']))
  t=float(new['step']);req(t==(float(old['step'])+1 if old is not None else 1.),'Adam step counter')
  calc=(-lr/(1-beta1**t)*m.astype(np.float64)/(np.sqrt(v.astype(np.float64)/(1-beta2**t))+eps)).astype(np.float32);de=max(de,err(calc,d[off-width:off]))
 req(off==len(g)==len(d),'Adam flat vector size')
 return me,ve,de
def _fit_audit(cid,mode,f,ref,settings,cfg_expected,expected_key,sourcefit=None):
 reused=sourcefit is not None
 req(f.get('GT_read') is False and f['mode']==mode and f['active_parameters']==1792,'fit identity')
 pos=[int(a['position']) for a in ref['anchors'] if float(a['weight'])>0]; req(f['observation_positions']==pos,'fit anchor order')
 if sourcefit is not None:
  req(f.get('reused_fit_from_sha256')==sourcefit[0],'reuse SHA marker')
  x=dict(f);x.pop('reused_fit_from_sha256');req(treeeq(x,sourcefit[1]),'reused fit differs from primary saved fit')
  f=sourcefit[1]
 cfg=f['config'];req(cfg==cfg_expected and cfg['steps']==5 and cfg['samples']==32 and f['settings']==settings and f['seed_key']==expected_key,'fit config/key')
 path=f['path'];req(len(path)==5,'fit path length');names=list(path[0]['state_before']);req(sum(arr(v).size for v in path[0]['state_before'].values())==1792,'state parameter scope')
 ev=np.array([a['box_cxcywh'] for a in ref['anchors'] if float(a['weight'])>0],dtype=np.float64);weights=np.array([a['weight'] for a in ref['anchors'] if float(a['weight'])>0],dtype=np.float64)
 maxr=maxq=maxloss=maxmom=maxvar=maxdelta=maxstate=maxlam=maxlamabs=maxrho=maxcg=maxnorm=0.;proposals=accepted=rejected=skipped=cgnum=normnum=0;cgrows=[];normrows=[];prevstate=f['initial_state'];prevopt=None;prevboxes=None
 for step,r in enumerate(path):
  req(r['step']==step and treeeq(r['state_before'],prevstate),'state chain before');
  if prevboxes is not None:maxstate=max(maxstate,err(r['boxes_before'],prevboxes));req(err(r['boxes_before'],prevboxes)<=TOL,'box chain')
  if prevopt is not None:req(treeeq(r['optimizer_before'],prevopt),'optimizer chain before')
  state_after=r.get('state_after',r['state_before']);opt_after=r.get('optimizer_after',r['optimizer_before'])
  roll=rollout_audit(r,ev,weights,cfg,pos,expected_key,step) if 'rollout'in r else None
  if roll:
   maxr=max(maxr,roll['reward']);maxq=max(maxq,roll['q']);maxloss=max(maxloss,roll['loss']);maxvar=max(maxvar,roll['reference_iou'])
  if r.get('skipped'):
   skipped+=1;req('gradient' not in r and 'optimizer_proposed' not in r,'skipped row has update')
   if roll:req(not roll['informative'] and r.get('reason')=='no_information','informative row skipped')
   else:req(r.get('reason')=='stopped_after_rejected_step','skip without rollout has wrong reason')
   prevstate=state_after;prevopt=opt_after;prevboxes=r.get('boxes_after',r['boxes_before']);continue
  req(roll is not None and roll['informative'],'proposal without informative rollout')
  me,ve,de=_optimizer(r,names);maxmom=max(maxmom,me,ve);maxdelta=max(maxdelta,de);req(max(me,ve,de)<=TOL,'Adam moments/bias-corrected delta mismatch');proposals+=1
  direction=arr(r['filtered_delta']).astype(np.float32).reshape(-1);d=arr(r['original_adam_delta']).astype(np.float32).reshape(-1)
  if mode=='M1':req(err(direction,d)<=TOL,'M1 direction must equal Adam')
  if mode=='M2' and r.get('solver'):
   s=r['solver'];solution=arr(s['solution']).astype(np.float64).reshape(-1);jd=arr(s['Jd']).astype(np.float64).reshape(-1);jf=arr(s['Jfiltered']).astype(np.float64).reshape(-1);dn=float(np.linalg.norm(d.astype(np.float64)));jdn=float(np.linalg.norm(jd));lam=float(s['lambda']);mu=float(s['mu'])
   req(abs(mu-float(settings['mu']))<=TOL,'CG mu differs from frozen setting')
   if s['status']=='near_zero_response_identity':req(lam==0 and err(solution,d)<=TOL and float(s['rho'])==1. and err(jf,jd)<=TOL,'near-zero CG identity')
   else:
    le=mu*float(np.dot(d,d))/float(np.dot(jd,jd));maxlamabs=max(maxlamabs,abs(lam-le));maxlam=max(maxlam,abs(lam-le)/max(1.,abs(le)));req(abs(lam-le)<=TOL*max(1.,abs(le)),'lambda normalization mismatch')
   rho=float(np.linalg.norm(solution))/max(dn,1e-30);maxrho=max(maxrho,abs(float(s['rho'])-rho));req(abs(float(s['rho'])-rho)<=TOL,'parameter rho mismatch')
   ratioerr=abs(float(s['response_norm_ratio'])-float(np.linalg.norm(jf))/max(jdn,1e-30));maxrho=max(maxrho,ratioerr);req(ratioerr<=TOL,'response norm ratio mismatch');rhs=arr(s['cg_rhs']);cr=arr(s['cg_residual']);
   if 'normal_applied'in s: cres=rhs-arr(s['normal_applied']);maxcg=max(maxcg,err(cres,cr),abs(float(np.linalg.norm(cr))-float(s['solver_residual_abs'])),abs(float(np.linalg.norm(cr))/max(float(np.linalg.norm(rhs)),1e-30)-float(s['solver_residual_rel'])))
   else: cres=cr;maxcg=max(maxcg,float(np.linalg.norm(cr)),float(s['solver_residual_abs']),float(s['solver_residual_rel']))
   req(maxcg<=TOL,'saved CG residual witness mismatch');req(err(direction,solution)<=TOL,'M2 solution/direction mismatch');cgnum+=1;cgrows.append(dict(case_id=cid,step=step,mu=mu,lambda_value=lam,rho=float(s['rho']),iterations=int(s['iterations']),status=s['status'],residual_rel=float(s['solver_residual_rel'])))
  nc=r.get('norm_control')
  if nc:
   od=arr(nc['original_direction']);sd=arr(nc['soft_direction']);md=arr(nc['matched_direction']);want=d.astype(np.float64)*(np.linalg.norm(sd.astype(np.float64))/max(np.linalg.norm(d.astype(np.float64)),1e-30));ne=err(md,want);maxnorm=max(maxnorm,ne,float(abs(np.linalg.norm(md)-np.linalg.norm(sd))));req(ne<=TOL and err(od,d)<=TOL and err(sd,direction)<=TOL,'equal-norm process direction');
   for k in ('matched_U_RMS','soft_U_RMS'):req(math.isfinite(float(nc[k])),'nonfinite norm-matched RMS')
   normrows.append(dict(case_id=cid,mode=mode,step=step,matched_U_RMS=float(nc['matched_U_RMS']),soft_U_RMS=float(nc['soft_U_RMS']),matched_anchor_IoU_mean=float(arr(nc['matched_anchor_IoU']).mean()),soft_anchor_IoU_mean=float(arr(nc['soft_anchor_IoU']).mean()))) ;normnum+=1
  good=bool(r['accepted']);accepted+=int(good);rejected+=int(not good);req(bool(r['rollback'])==(not good),'rollback flag')
  if good:
   scale=float(r['accepted_scale']);req(scale in cfg.get('backtracking',settings['backtracking']),'accepted scale outside backtracking')
   off=0;upd=0.
   for name,old in r['state_before'].items():
    n=arr(old).size;delta=(direction[off:off+n].reshape(arr(old).shape)*np.float32(scale)).astype(arr(old).dtype);want=arr(old)+delta;upd=max(upd,err(want,r['state_after'][name]));off+=n
   maxstate=max(maxstate,upd);req(upd<=TOL,'accepted phi update mismatch');req(treeeq(r['optimizer_after'],r['optimizer_proposed']),'accepted Adam moments not retained')
   acc=[x for x in r['candidates'] if x['accepted']];req(acc and abs(float(acc[-1]['scale'])-scale)<1e-12 and all(not x['accepted'] for x in r['candidates'][:-1]),'accepted candidate/backtracking record')
  else:
   req(treeeq(r['state_after'],r['state_before']) and treeeq(r['optimizer_after'],r['optimizer_before']),'rejected update did not fully rollback');req(not any(x['accepted'] for x in r['candidates']),'rejected candidate marked accepted')
  prevstate=state_after;prevopt=opt_after;prevboxes=r['boxes_after']
 req(f['optimizer_steps']==accepted and f['proposed_optimizer_steps']==proposals,'fit optimizer counts')
 req(treeeq(f['initial_state'],path[0]['state_before']) and treeeq(f['final_state'],prevstate),'fit endpoint state mismatch')
 req(treeeq(f['fit_final_state'],prevstate),'fit_final_state alias mismatch');req(err(prevboxes,f['final_cached_decoder_boxes'])<=TOL,'last saved boxes differ from final decoder cache')
 return dict(case_id=cid,mode=mode,reused=reused,new_fit_invocation=not reused,step_count=5,proposals=proposals,accepted=accepted,rejected=rejected,skipped=skipped,max_reward_error=maxr,max_q_error=maxq,max_gaussian_loss_error=maxloss,max_reference_iou_error=maxvar,max_adam_moment_error=maxmom,max_adam_delta_error=maxdelta,max_phi_update_error=maxstate,max_lambda_relative_error=maxlam,max_lambda_abs_error=maxlamabs,max_rho_error=maxrho,max_cg_residual_error=maxcg,max_equal_norm_error=maxnorm,cg_steps=cgrows,equal_norm_steps=normrows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('base',nargs='?',default='artifacts/stvg_soft_ep_opd_v1');a=ap.parse_args();base=Path(a.base);base=(base if base.is_absolute() else ROOT/base).resolve();req(ROOT in base.parents,'base must be inside repository')
 from scripts.stvg_evidence_common_v1 import load_case
 from scripts.run_stvg_table1_student004 import stable_fit_key
 rt=read(base/'RUNTIME.json');req(rt.get('status')=='locked' and rt.get('GT_read') is False,'runtime lock');
 for p,h in {**rt.get('pins',{}),**rt.get('protected',{})}.items():req(sha(ROOT/p)==h,'runtime pin '+p)
 design=read(base/'DESIGN_LOCK.json');roster=read(base/'ROSTER_LOCK.json');refdoc=read(base/'REFERENCE_LOCK.json');ids=[r['probe_alias'] for r in roster['records']];id_ds={r['probe_alias']:r['dataset'] for r in roster['records']};refs={r['case_id']:r for r in refdoc['records']};req(len(ids)==32 and len(set(ids))==32 and set(ids)==set(refs),'roster/reference set')
 sbp=base/'STUDENT_GLOBAL_BARRIER.json';sb=read(sbp);rp=base/'ROOT_READBACK.json';rr=read(rp);gp=base/'GLOBAL_PREDICTION_BARRIER.json';gb=read(gp)
 req(sb.get('status')=='complete' and sb.get('count')==32 and sb.get('GT_read') is False and sb.get('runtime_sha256')==sha(base/'RUNTIME.json'),'student seal');req(rr.get('status')=='pass' and rr.get('count')==32 and rr.get('GT_read') is False and rr.get('bindings')=={'runtime_sha256':sha(base/'RUNTIME.json'),'barrier_sha256':sha(sbp)},'pre-GT root readback');req(gb.get('status')=='complete' and gb.get('count')==32 and gb.get('GT_read') is False and gb.get('student_barrier_sha256')==sha(sbp) and gb.get('root_readback_sha256')==sha(rp),'prediction seal');req(gb.get('scorer_sha256')==sha(ROOT/'scripts/measure_stvg_soft_ep_v1.py'),'sealed scorer SHA')
 expected={str((base/'student'/f'{c}.pt').relative_to(ROOT)) for c in ids};sf=file_rows(sb['files']);gf=file_rows(gb['files']);gpt={p for p in gf if p.endswith('.pt')};req(set(sf)==expected and gpt==expected and all(sf[p]['sha256']==gf[p]['sha256'] for p in expected),'student/seal tensor file set')
 rows=read(base/'POSTSEAL_SCORE_ROWS.json');summary=read(base/'POSTSEAL_SCORE_SUMMARY.json');paired=read(base/'POSTSEAL_PAIRED_CONTRASTS.json');complete=read(base/'CPU_COMPLETION.json');binding={'runtime_sha256':sha(base/'RUNTIME.json'),'design_lock_sha256':sha(base/'DESIGN_LOCK.json'),'roster_sha256':sha(base/'ROSTER_LOCK.json'),'reference_lock_sha256':sha(base/'REFERENCE_LOCK.json'),'parent_student_barrier_sha256':sha(PARENT/'STUDENT_GLOBAL_BARRIER.json'),'student_barrier_sha256':sha(sbp),'root_readback_sha256':sha(rp),'prediction_seal_sha256':sha(gp)}
 for d in (rows,summary,paired,complete):req(d.get('status')=='complete' and d.get('GT_read') is True and d.get('bindings')==binding,'postseal output binding')
 sc=scalar_audit(base,id_ds);req(sc['status']=='pass','scalar reaggregation')
 parentbar=read(PARENT/'STUDENT_GLOBAL_BARRIER.json');pf=file_rows(parentbar['files']);parentdesign=read(ROOT/'artifacts/stvg_soft_ep_opd_v1/DESIGN_LOCK.json');fitrecords=[];sources={};reused=0
 for rec in roster['records']:
  cid=rec['probe_alias'];case,sp=load_case(rec);ref=refs[cid];path=ROOT/str((base/'student'/f'{cid}.pt').relative_to(ROOT));z=loadpt(path)
  req(rec['dataset']==z['dataset']==ref['dataset']==case['dataset'] and sha(sp)==z['source_file_sha256'] and case['source_case_sha256']==ref['source_case_sha256']==z['source_case_sha256'] and z['GT_read'] is False and z['runtime_sha256']==sha(base/'RUNTIME.json') and z['reference_records']==ref and list(map(int,z['frame_ids']))==list(map(int,case['frame_ids'])),'source/reference identity '+cid)
  pp=PARENT/'student'/f'{cid}.pt';prel=str(pp.relative_to(ROOT));req(prel in pf and sha(pp)==pf[prel]['sha256'] and z['parent_student_sha256']==pf[prel]['sha256'],'parent SHA '+cid);parent=loadpt(pp);req(parent['source_case_sha256']==case['source_case_sha256'],'parent source '+cid)
  for dst,src in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]:req(np.array_equal(arr(z['arms'][dst]['boxes']),arr(parent['arms'][src]['boxes'])) and list(z['arms'][dst]['interval'])==list(parent['arms'][src]['interval']),'M0 reuse '+cid+'/'+dst)
  for mode in ('M1','M2'):
   f=z['fits'][mode]
   if f is None:
    req(not ref['anchors'] and z['readouts'][mode] is None,'unexpected no-fit '+cid+'/'+mode)
    for arm,source in ((f'S_{mode}','F'),(f'TS_{mode}','T')):
     req(np.array_equal(arr(z['arms'][arm]['boxes']),arr(z['arms'][source]['boxes'])) and z['arms'][arm]['interval']==z['arms'][source]['interval'],'empty-reference no-update mismatch '+cid+'/'+arm)
    fitrecords.append(dict(case_id=cid,mode=mode,reused=False,no_fit=True,reason='empty_reference'))
    continue
   sourcefit=None
   if mode=='M1' and f.get('reused_fit_from_sha256'):
    primary=loadpt(ROOT/'artifacts/stvg_soft_ep_opd_v1/student'/f'{cid}.pt');sourcefit=(sha(ROOT/'artifacts/stvg_soft_ep_opd_v1/student'/f'{cid}.pt'),primary['fits']['M1']);reused+=1
    req(treeeq(z['readouts']['M1'],primary['readouts']['M1']) and all(np.array_equal(arr(z['arms'][arm]['boxes']),arr(primary['arms'][arm]['boxes'])) and z['arms'][arm]['interval']==primary['arms'][arm]['interval'] for arm in ('S_M1','TS_M1')),'reused M1 readout/arm mismatch '+cid)
   setting_design=parentdesign if sourcefit else design
   from scripts.run_stvg_table1_student004 import stable_fit_key
   expected_key=stable_fit_key(rec['dataset'],rec['query_key_sha256'])
   fr=_fit_audit(cid,mode,f,ref,setting_design['response_settings'],setting_design['configs'][rec['dataset']],expected_key,sourcefit)
   if not treeeq(f['final_cached_decoder_boxes'],z['readouts'][mode]['boxes']):raise RuntimeError('full decoder readout differs from saved fit cache '+cid+'/'+mode)
   fitrecords.append(fr)
  sources[cid]=dict(source_case_sha256=z['source_case_sha256'],source_file_sha256=z['source_file_sha256'],parent_student_sha256=z['parent_student_sha256'])
 out=base/'ROOT_INDEPENDENT_CPU_AUDIT.json';req(not out.exists(),'refusing to overwrite audit');prev=[{'path':f'artifacts/stvg_soft_ep_opd_v1/ROOT_INDEPENDENT_CPU_AUDIT_ATTEMPT{i:03d}.json','sha256':sha(base/f'ROOT_INDEPENDENT_CPU_AUDIT_ATTEMPT{i:03d}.json')} for i in (1,2) if (base/f'ROOT_INDEPENDENT_CPU_AUDIT_ATTEMPT{i:03d}.json').is_file()];audit=dict(schema='stvg-soft-ep-independent-cpu-audit-v1',status='pass',reader_sha256=sha(__file__),scope='saved scalar reaggregation and optimizer/graph-witness arithmetic only',case_count=32,arm_row_count=len(rows['rows']),summary_count=len(summary['summary']),paired_count=len(paired['paired_contrasts']),new_GT_read=False,model_loaded=False,GPU_used=False,backward_replayed=False,full_J_reconstructed=False,graph_operator_scope='CG residuals are checked against the saved normal-applied/CG witness, not independently reconstructed from a full Jacobian',scalar_audit=sc,fit_count=sum(not x.get('no_fit',False) for x in fitrecords),no_fit_count=sum(x.get('no_fit',False) for x in fitrecords),new_fit_invocation_count=sum(x.get('new_fit_invocation',False) for x in fitrecords),reused_fit_count=sum(x.get('reused',False) for x in fitrecords),proposal_count=sum(x.get('proposals',0) for x in fitrecords),accepted_count=sum(x.get('accepted',0) for x in fitrecords),rejected_count=sum(x.get('rejected',0) for x in fitrecords),skipped_count=sum(x.get('skipped',0) for x in fitrecords),cg_witness_steps=sum(len(x.get('cg_steps',[])) for x in fitrecords),equal_norm_steps=sum(len(x.get('equal_norm_steps',[])) for x in fitrecords),fits=fitrecords,source_bindings=sources,prior_attempts=prev,bindings=binding,time=time.time())
 out.write_text(json.dumps(clean(audit),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'pass','base':str(base.relative_to(ROOT)),'fits':len(fitrecords),'reused':reused,'audit':str(out.relative_to(ROOT))}))
if __name__=='__main__':main()
