"""Sealed cached DINO versus matched source-GT anchor propagation report."""
from pathlib import Path
import sys,argparse,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_anchor_dino_followup_v2 import BASE,D,OUT,OLD,PREVIOUS,SETS,read,put,load,sha

def audit_export():
 import torch,numpy as np
 from scipy.stats import t
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 from vg_tta.stvg_joint_anchor_projection_v2 import interpolation_matrix
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();assert read(D/'PREDICTION_BARRIER.json')['count']==24;rows=[];checks=0
 for f,h in {**read(BASE/'RUNTIME_LOCK.json')['pins'],**read(D/'RUNTIME_LOCK.json')['pins']}.items():assert sha(ROOT/f)==h;checks+=1
 def xy(b):
  b=np.asarray(b);return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
 for r in read(BASE/'PLAN_LOCK.json')['source']:
  alias=r['alias'];z=load(OLD/'features/source'/(alias+'.pt'));y=load(OLD/'source_supervision'/(alias+'.pt'));q=load(D/'inputs'/(alias+'.pt'));p=load(D/'predictions'/(alias+'.pt'));old=load(BASE/'predictions'/(alias+'.pt'));ids=np.asarray(z['frame_ids']);iv=z['native_interval'];planned=load(PREVIOUS/'observations'/(alias+'.pt'))['planned_positions'];row=dict(frame_ids=z['frame_ids'],input=z['input']);entry=dict(alias=alias,dataset=r['dataset'],sets={});common_support=None
  for name in SETS:
   o=q[name];nodes=(o['W']!=0).any(0);Wa=interpolation_matrix(old['N'],z,list(range(len(ids))));outside=((Wa[:,nodes]!=0).sum(1)==0)if nodes.any()else np.ones(len(ids),dtype=bool);a=p['arms'][name];states=dict(N=old['N'],A=a['A'],P=old['P']['final'],AP=a['AP']['final']);metrics={};anchor={};counts={}
   if not len(o['W']):assert torch.equal(states['A'],states['N'])and torch.equal(states['AP'],states['P']);checks+=2
   dd={k:decode_state(x,z)for k,x in states.items()};assert np.array_equal(dd['A']['boxes'][outside],dd['N']['boxes'][outside]);checks+=1
   for arm,dec in dd.items():
    assert dec['interval']==iv and torch.equal(states[arm][0],old['N'][0]);checks+=2
    bb=dec['boxes'][o['positions']];target=o['target'].numpy();err=float(np.abs(bb-target).max())if len(bb)else None
    if arm in ['A','AP']and err is not None:assert err<2e-6;checks+=1
    gt_pos=[pos for pos in o['positions']if ids[pos]in y['dense_source_GT']];truth=np.asarray([y['dense_source_GT'][int(ids[pos])]for pos in gt_pos]).reshape(-1,4)
    if gt_pos:
     pb=xy(dec['boxes'][gt_pos])*np.asarray([z['input']['width'],z['input']['height']]*2);trueIoU=float(box_iou(pb,truth).mean())
    else:trueIoU=None
    anchor[arm]=dict(n=len(bb),supervision_max_error=err,supervision_IoU=float(box_iou(xy(bb),xy(target)).mean())if len(bb)else None,actual_GT_IoU=trueIoU,precise_GT_count=len(gt_pos))
    dt=DenseTube(dec['boxes'],row,y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2');m=dt.score(iv);aoff=official(dec['boxes'],row,y['dense_source_GT'],y['span'],iv,r['dataset'])
    for k in ['v','t','s']:assert abs(m[k]-aoff[k])<2e-9;checks+=1
    m.update(R03=float(m['v']>=.3),R05=float(m['v']>=.5));inside=(dt.fids>=iv[0])&(dt.fids<iv[1])&(dt.fids>=ids[0])&(dt.fids<=ids[-1]);obs=np.isin(dt.fids,ids[planned]);native=np.isin(dt.fids,ids);outsidep=np.isin(dt.fids,ids[outside]);masks=dict(unobserved_native=inside&~obs&native,unobserved_interpolated=inside&~obs&~native,outside_footprint_native=inside&~obs&native&outsidep)
    for k,mask in masks.items():
     m[k+'_s']=float(dt.iou[mask].mean())if mask.any()else None
     if k in counts:assert counts[k]==int(mask.sum())
     counts[k]=int(mask.sum())
    metrics[arm]=m
   if name=='DINO_common':common_support=counts
   if name=='GT_common':assert counts==common_support and o['positions']==q['DINO_common']['positions'];checks+=2
   entry['sets'][name]=dict(reference_count=len(o['W']),support_counts=counts,metrics=metrics,anchor=anchor,reused_from=p['reused_from'].get(name),trace=a['AP']['trace'])
  rows.append(entry)
 keys=list(rows[0]['sets']['DINO_all']['metrics']['N']);means={};paired={};quality={}
 def stats(v,difference=False):
  v=np.asarray([x for x in v if x is not None]);n=len(v);mu=float(v.mean())if n else None;out=dict(mean=mu,n=n)
  if difference:
   rad=float(t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n))if n>1 else None;out.update(CI95=[mu-rad,mu+rad]if rad is not None else None,values=v.tolist(),positive=int((v>1e-8).sum()),negative=int((v< -1e-8).sum()),neutral=int((np.abs(v)<=1e-8).sum()))
  return out
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];means[ds]={};paired[ds]={};quality[ds]={}
  for name in SETS:
   mm={a:{k:stats([r['sets'][name]['metrics'][a][k]for r in rr])for k in keys}for a in ['N','A','P','AP']};means[ds][name]=mm;pp={}
   for a,b in [('A','N'),('AP','A'),('AP','P'),('AP','N')]:
    pp[a+' - '+b]={k:stats([r['sets'][name]['metrics'][a][k]-r['sets'][name]['metrics'][b][k]if r['sets'][name]['metrics'][a][k]is not None else None for r in rr],True)for k in keys}
   paired[ds][name]=pp
  for a in ['A','AP']:
   quality[ds][a]={k:stats([r['sets']['GT_common']['metrics'][a][k]-r['sets']['DINO_common']['metrics'][a][k]if r['sets']['GT_common']['metrics'][a][k]is not None else None for r in rr],True)for k in keys}
 result=dict(status='complete',count=24,source_only=True,all_sources_retained=True,means=means,paired=paired,matched_quality_GT_minus_DINO=quality,new_training=0,new_target_inference=0,new_expert_calls=0,rows=rows,not_deployable_or_Direct_OPD_comparison=True)
 put(D/'RESULTS.json',result);pub={k:v for k,v in result.items()if k!='rows'};pub['cases']=[]
 for r in rows:
  f=OUT/'dino_cases'/(r['alias']+'.json');put(f,r);pub['cases'].append(dict(alias=r['alias'],dataset=r['dataset'],file='dino_cases/'+f.name))
 put(OUT/'DINO_RESULTS.json',pub);root=dict(status='pass',count=24,checks=checks,same_common_positions_and_support=True,A_AP_same_final_reference_accuracy=True,all_time_exact_Native=True,official_DenseTube_exact=True,all_branch_predictions_sealed_before_full_GT_scoring=True,no_training=True)
 put(D/'ROOT_COMPLETION.json',root);put(OUT/'DINO_ROOT_COMPLETION.json',root)
 for f in ['PLAN_LOCK.json','CPU_CONTRACT_COMPLETION.json','GPU_COMPLETION.json']:
  put(OUT/('DINO_'+f),read(D/f))
 report(result)
 print('DINO ROOT',checks)
 for ds in ['hc2','vidstg']:print(ds,{name:{a:means[ds][name][a]['v']['mean']*100 for a in ['N','A','P','AP']}for name in SETS})

def report(r):
 lines=['# Cached DINO follow-up after strong-anchor source qualification','','Parent source-GT gate passed before this branch. No new expert calls, training, target inference or Student fits. Same source24, Native time, frozen priors and deterministic no-added-noise propagation.','','DINO_all retains all cached admitted anchors. DINO_common and GT_common use identical original DINO positions with source GT; no IoU filtering. Comparing common arms changes reference box values only. DINO_all versus GT_planned changes support too and must not be called a pure reference-quality comparison. All queries, including empty sets, remain. A is constrained anchor correction plus existing interpolation; AP adds frozen denoiser propagation with the same final anchor precision. These are diagnostics, not deployable rules.','','## Full-cohort vIoU','','Percent,12/domain. N/P are the original no-anchor controls; no rerun.','','| Reference set | HC A | HC AP | Vid A | Vid AP |','|---|---:|---:|---:|---:|']
 for name in SETS:lines.append('| '+name+' | '+' | '.join(f"{r['means'][ds][name][a]['v']['mean']*100:.6f}"for ds in ['hc2','vidstg']for a in ['A','AP'])+' |')
 lines+=['','## Additional propagation over the same anchor correction','','Source-paired difference AP-A in percentage points,95% t interval and positive/negative/neutral counts. Native decoder and benchmark interpolation supports are separate; outside-footprint support is reference-set-specific, identical between the two common sets.','','| Set / metric | HC | Vid |','|---|---|---|']
 for name in SETS:
  for k in ['v','unobserved_native_s','outside_footprint_native_s','unobserved_interpolated_s']:
   vv=[]
   for ds in ['hc2','vidstg']:
    a=r['paired'][ds][name]['AP - A'][k];ci=a['CI95'];vv.append(f"{a['mean']*100:+.6f} [{ci[0]*100:+.6f},{ci[1]*100:+.6f}] {a['positive']}/{a['negative']}/{a['neutral']}"if ci else 'unknown')
   lines.append('| '+name+' / '+k+' | '+' | '.join(vv)+' |')
 lines+=['','## Same-support GT minus DINO reference-quality intervention','','| Arm / metric | HC | Vid |','|---|---|---|']
 for arm in ['A','AP']:
  for k in ['v','unobserved_native_s','outside_footprint_native_s']:
   vv=[]
   for ds in ['hc2','vidstg']:
    a=r['matched_quality_GT_minus_DINO'][ds][arm][k];ci=a['CI95'];vv.append(f"{a['mean']*100:+.6f} [{ci[0]*100:+.6f},{ci[1]*100:+.6f}]"if ci else 'unknown')
   lines.append('| '+arm+' / '+k+' | '+' | '.join(vv)+' |')
 lines+=['','## Interpretation boundary','','A/AP anchor errors relative to their own supervision are <2e-6, including DINO. Therefore differences are not caused by one arm failing to apply its supplied sparse boxes. DINO reference geometry against source GT is separately recorded in per-case anchor metrics; low IoU does not identify wrong persons without actual semantic pixel review. Hard anchoring of erroneous DINO is an intervention, not a proposed deployment method or proof about the old soft mixture guidance.','','Unanchored P can itself improve trajectories: AP-A is the total benefit of adding the frozen propagation procedure to exact anchoring, not solely the causal contribution of sparse conditions. The parent interaction (AP-P)-(A-N) remains necessary for interpreting whether anchors add propagation beyond the prior alone. No inference that diffusion is uniquely necessary; no deterministic learned corrector was compared here. No training, new target TTA, dense OPD or production promotion follows.']
 (OUT/'DINO_REPORT.md').write_text('\n'.join(lines)+'\n')

def render():
 from PIL import Image,ImageDraw
 from scripts.run_decota_paper_main_v1 import frames_for
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 files=[]
 for ds in ['hc2','vidstg']:
  r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));q=load(D/'predictions'/(r['alias']+'.pt'));pp=load(PREVIOUS/'observations'/(r['alias']+'.pt'))['planned_positions'];frames,ids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert ids==z['frame_ids'];canvas=Image.new('RGB',(1120,768),(22,22,22));draw=ImageDraw.Draw(canvas)
  for j,(name,a)in enumerate([('DINO_common','A'),('DINO_common','AP'),('GT_common','A'),('GT_common','AP')]):
   d=decode_state(q['arms'][name][a]['final']if a=='AP'else q['arms'][name][a],z)
   for k,pos in enumerate(pp):
    im=Image.fromarray(frames[pos]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy));cx,cy,w,h=d['boxes'][pos];draw.rectangle([ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height],outline=(255,210,40),width=2);draw.text((ox+4,j*192+4),f"{r['alias']} {name}/{a} slot{k+1}",fill='white')
  f=BASE/'root_view'/(ds+'_dino.png');f.parent.mkdir(exist_ok=True);canvas.save(f);files.append(str(f))
 put(D/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_only=True,no_GT_overlay=True,new_semantic_identity_labels=0));print('\n'.join(files))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['audit_export','render']);arg=a.parse_args();globals()[arg.command]()
