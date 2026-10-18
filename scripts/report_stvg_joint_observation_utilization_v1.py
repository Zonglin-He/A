"""Independent sealed source-only observation diagnosis and anonymous export."""
from pathlib import Path
import sys,argparse,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_observation_utilization_v1 import BASE,OUT,PARENT,OLD,REPAIR,ARMS,read,put,load,sha,condition

def audit_export():
 import torch,numpy as np
 from scipy.stats import t
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 torch.set_num_threads(4);assert not torch.cuda.is_initialized()
 assert read(BASE/'PREDICTION_BARRIER.json')['logical_predictions']==240
 p=read(BASE/'PLAN_LOCK.json');checks=0;rows=[];support=[]
 for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h;checks+=1
 for r in p['source']:
  z=load(OLD/'features/source'/(r['alias']+'.pt'));y=load(OLD/'source_supervision'/(r['alias']+'.pt'));oo=load(BASE/'observations'/(r['alias']+'.pt'));obs=oo['observations'];ids=np.array(z['frame_ids']);iv=z['native_interval'];planned=ids[oo['planned_positions']];row=dict(frame_ids=z['frame_ids'],input=z['input']);c=condition(z,'cpu');dgt=y['dense_source_GT']
  supports={};counts={};output=dict(alias=r['alias'],dataset=r['dataset'],clocks={})
  def score_decoded(d):
   nonlocal checks
   dense=DenseTube(d['boxes'],row,dgt,y['span'],clip=r['dataset']=='hc2');a=dense.score(d['interval']);b=official(d['boxes'],row,dgt,y['span'],d['interval'],r['dataset'])
   for k in ['v','t','s']:assert abs(a[k]-b[k])<2e-9;checks+=1
   a.update(R03=float(a['v']>=.3),R05=float(a['v']>=.5),fixed_time_v=dense.score(iv)['v'])
   inrange=(dense.fids>=ids[0])&(dense.fids<=ids[-1])&(dense.fids>=iv[0])&(dense.fids<iv[1]);observed=np.isin(dense.fids,planned);native=np.isin(dense.fids,ids)
   masks=dict(observed=inrange&observed,unobserved_native=inrange&~observed&native,unobserved_interpolated=inrange&~observed&~native)
   for k,m in masks.items():
    a[k+'_s']=float(dense.iou[m].mean())if m.any()else None
    if k in counts:assert counts[k]==int(m.sum())
    counts[k]=int(m.sum())
   return a
  output['Native']=score_decoded(dict(boxes=z['boxes'][0].numpy(),interval=iv));output['Native_codec']=score_decoded(decode_state(c['native'][0],z))
  for clock in ['fixed','free']:
   q=load(BASE/'predictions'/clock/(r['alias']+'.pt'));arms={};baseline=q['arms']['none']['final'];output['clocks'][clock]={}
   if clock=='free':assert torch.equal(baseline,load(PARENT/'predictions'/(r['alias']+'.pt'))['path']['final']);checks+=1
   for name in ARMS:
    a=q['arms'][name];xs=a['final'];assert xs.shape==(4,33,4);sc=[]
    if clock=='fixed':assert torch.equal(xs[:,0],c['native'][:,0].expand(4,-1));checks+=1
    if len(obs[name]['time'])==0:assert torch.equal(xs,baseline);checks+=1
    if name in q['reused_from']:assert torch.equal(xs,q['arms'][q['reused_from'][name]]['final']);checks+=1
    for x in xs:
     decoded=decode_state(x,z)
     if clock=='fixed':assert decoded['interval']==iv;checks+=1
     sc.append(score_decoded(decoded))
    trace=a['trace'];diagnostic=dict(reference_count=len(obs[name]['time']),reused_from=q['reused_from'].get(name,'previous_sigma213'if clock=='free'and name=='none'else None),state_max_abs_difference_from_none=float((xs-baseline).abs().max()),state_L2_difference_from_none=float((xs-baseline).norm()),trace_available=bool(trace))
    if trace:
     diagnostic.update(first_active=trace[0]['active_count'],last_active=trace[-1]['active_count'],min_active=min(v['active_count']for v in trace),mean_responsibility=float(np.mean([v['responsibility_mean']for v in trace])),sum_delta_norm=sum(v['delta_norm']for v in trace),max_delta=max(v['delta_max']for v in trace),nonzero_guidance_levels=sum(v['gradient_norm']>0 for v in trace),max_temporal_gradient=max(v['temporal_gradient_norm']for v in trace),clipped_components=sum(v['clipped_components']for v in trace))
     if clock=='fixed':assert diagnostic['max_temporal_gradient']==0;checks+=1
    output['clocks'][clock][name]=dict(samples=sc,guidance=diagnostic,trace=trace)
   assert obs['DINO_common']['positions']==obs['GT_common']['positions'];checks+=1
  output['support_counts']=counts;output['reference_counts']={k:len(o['time'])for k,o in obs.items()};rows.append(output)
 keys=['v','t','s','R03','R05','fixed_time_v','observed_s','unobserved_native_s','unobserved_interpolated_s'];means={};paired={}
 def vals(rr,clock,arm,style,k):
  return [r[arm][k]if arm.startswith('Native')else(r['clocks'][clock][arm]['samples'][0][k]if style=='sample0'else None if r['clocks'][clock][arm]['samples'][0][k]is None else float(np.mean([s[k]for s in r['clocks'][clock][arm]['samples']])))for r in rr]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==12;means[ds]={};paired[ds]={}
  for clock in ['fixed','free']:
   means[ds][clock]={};paired[ds][clock]={}
   for style in ['sample0','mean_per_sample']:
    vv={arm:{k:vals(rr,clock,arm,style,k)for k in keys}for arm in ['Native','Native_codec']+ARMS}
    means[ds][clock][style]={arm:{k:dict(mean=float(np.mean([v for v in vs if v is not None]))if any(v is not None for v in vs)else None,n=sum(v is not None for v in vs))for k,vs in ks.items()}for arm,ks in vv.items()};pair={}
    for a,b in [('none','Native_codec'),('DINO_all','none'),('DINO_common','none'),('GT_common','none'),('GT_common','DINO_common'),('GT_planned','none')]:
     mm={}
     for k in keys:
      delta=np.array([x-y for x,y in zip(vv[a][k],vv[b][k])if x is not None and y is not None]);n=len(delta);rad=float(t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n))if n>1 else None;mu=float(delta.mean())if n else None
      mm[k]=dict(mean=mu,n=n,CI95=[mu-rad,mu+rad]if rad is not None else None,positive=int((delta>1e-8).sum()),negative=int((delta< -1e-8).sum()),neutral=int((np.abs(delta)<=1e-8).sum()),values=delta.tolist());assert mm[k]['positive']+mm[k]['negative']+mm[k]['neutral']==n;checks+=1
     pair[a+' - '+b]=mm
    paired[ds][clock][style]=pair
 result=dict(status='complete',count=24,source_only=True,new_training=0,new_target_inference=0,source_sparse_GT_oracle=True,means=means,paired=paired,rows=rows,source_validation_exposed=True,primary='fixed clock sample0 unobserved native frames, all queries retained; missing-support means separately count available queries',no_automatic_next_stage=True)
 OUT.mkdir(exist_ok=True,parents=True);put(BASE/'RESULTS.json',result);export_cases(result)
 root=dict(status='pass',count=24,independent_checks=checks,all48_outputs_sealed_before_full_source_GT_scoring=True,official_DenseTube_agreement=True,fixed_clock_time_and_tIoU_identical_Native=True,free_none_exact_old_sigma213=True,all_empty_observation_outputs_exact_none=True,matched_common_positions_identical=True,old_pins_unchanged=True,new_training=0,new_target_inference=0)
 put(BASE/'ROOT_COMPLETION.json',root);put(OUT/'ROOT_COMPLETION.json',root)
 for n in ['CPU_CONTRACT_COMPLETION.json','GPU_QUALIFICATION_COMPLETION.json','GPU_COMPLETION.json','PREDICTION_BARRIER.json']:put(OUT/n,read(BASE/n))
 put(OUT/'PROTOCOL.json',{k:v for k,v in p.items()if k!='source'});put(OUT/'SUPPORT_COUNTS.json',read(BASE/'OBSERVATION_INPUT_BARRIER.json')['counts'])
 write_report(result,root)
 print('ROOT',checks)
 for ds in ['hc2','vidstg']:
  for clock in ['fixed','free']:
   print(ds,clock,{a:{k:means[ds][clock]['sample0'][a][k]['mean']for k in ['v','t','unobserved_native_s','unobserved_interpolated_s']}for a in ARMS})

def export_cases(result):
 public={k:v for k,v in result.items()if k!='rows'};public['cases']=[]
 for r in result['rows']:
  f=OUT/'cases'/(r['alias']+'.json');put(f,r);public['cases'].append(dict(alias=r['alias'],dataset=r['dataset'],file='cases/'+f.name))
 put(OUT/'RESULTS.json',public)


def write_report(result,root):
 means=result['means'];paired=result['paired'];lines=['# Frozen source24: sparse observation utilization','','No new training. Repaired HC epoch30 / Vid epoch20, the same 24 source-validation queries and four epsilon paths, sigma .213 to .03 / 16 steps. These sources are held out only for the auxiliary prior, not established unseen to the pretrained Student. No target inference, Student adaptation, new DINO calls, or method promotion.','','## Conditions and controls','','Five conditions are evaluated at both clocks. none has no observation; DINO_all uses all original admitted boxes; DINO_common and GT_common share precisely the same positions with exact source GT, changing only box values; GT_planned uses available source GT at the original four planned anchors, including otherwise missing DINO positions. No exterior GT is fabricated. GT_common/GT_planned are source-only oracle controls, and the common-support DINO arm is also GT-assisted diagnosis. DINO_all versus GT_planned is not a pure reference-quality intervention. All 24 queries and empty-observation cases remain.','','HC has 43 actual DINO / 33 common / 37 planned GT observations; Vid has 41 / 21 / 23. GT_planned is nonempty in 12 HC and 9 Vid queries, common support in 11 HC and 7 Vid; empty cases remain no-update guidance controls.','','Fixed clock pins Native temporal state at initialization, every clean estimate, and every proposal; its observation membership is the exact physical integer Native interval. Free clock preserves the old event-relative floating hard mask. The latter can disable observations as the interval changes. This is still the original event-relative 32-knot representation, not a fixed-video identity model.','','Guidance uses the clean denoised estimate and its complete derivative to noisy state. The learned association MLP is not executed; fixed .5 inlier/outlier mixture, original source-trained variance, .02*sigma^2 step and .2 component cap are retained. No tuning for the oracle. Individual posterior paths are decoded separately, never averaged as coordinates or selected by GT.','','## First-path results','','Percent; 12 queries per domain. Full sIoU uses original benchmark support; fixed_time_v uses common Native temporal output.','','| Clock / condition | HC vIoU | HC tIoU | HC sIoU | HC fixed-time vIoU | Vid vIoU | Vid tIoU | Vid sIoU | Vid fixed-time vIoU |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for clock in ['fixed','free']:
  for a in ['Native','Native_codec']+ARMS:
   vv=[means[ds][clock]['sample0'][a][k]['mean']*100 for ds in ['hc2','vidstg']for k in ['v','t','s','fixed_time_v']];lines.append('| '+clock+' / '+a+' | '+' | '.join(f'{x:.6f}'for x in vv)+' |')
 lines+=['','## Fixed-support spatial completion','','The physical support is held identical for every condition and clock: annotated source frames inside the original Native interval and original decoder input range, excluding the four originally planned anchors for the two unobserved supports. Native decoder frames and benchmark interpolation positions are disjoint. Null supports are not converted to zero; their contributing query and frame counts are reported. These geometry metrics do not annotate identity correctness.','','| Clock / condition | HC observed sIoU (n) | HC unobserved decoder sIoU (n) | HC unobserved interpolated sIoU (n) | Vid observed sIoU (n) | Vid unobserved decoder sIoU (n) | Vid unobserved interpolated sIoU (n) |','|---|---:|---:|---:|---:|---:|---:|']
 for clock in ['fixed','free']:
  for a in ['Native_codec']+ARMS:
   vv=[]
   for ds in ['hc2','vidstg']:
    for k in ['observed_s','unobserved_native_s','unobserved_interpolated_s']:
     x=means[ds][clock]['sample0'][a][k];vv.append(f"{x['mean']*100:.6f} ({x['n']})"if x['mean']is not None else 'unknown (0)')
   lines.append('| '+clock+' / '+a+' | '+' | '.join(vv)+' |')
 lines+=['','## Paired differences','','Source-query paired Student-t 95% intervals in percentage points, with positive/negative/neutral counts. All four per-path outcomes and per-query mean-of-four outcomes are included in RESULTS.json; four paths are not four independent sources. No checkpoint or path selection.','','| Clock / comparison / metric | HC mean [CI95], +/−/= | Vid mean [CI95], +/−/= |','|---|---|---|']
 for clock in ['fixed','free']:
  for comp in ['DINO_all - none','GT_common - DINO_common','GT_planned - none']:
   for k in ['v','unobserved_native_s','unobserved_interpolated_s']:
    vv=[]
    for ds in ['hc2','vidstg']:
     a=paired[ds][clock]['sample0'][comp][k];ci=a['CI95'];vv.append(f"{a['mean']*100:+.6f} [{ci[0]*100:+.6f}, {ci[1]*100:+.6f}], {a['positive']}/{a['negative']}/{a['neutral']}"if ci else 'unknown')
    lines.append('| '+clock+' / '+comp+' / '+k+' | '+' | '.join(vv)+' |')
 lines+=['','## Influence and interpretation limits','','Per-query traces expose active reference count, mixture responsibility, full-chain gradient, actual bounded guidance displacement, clipping, and resulting state difference from none. Empty or identical observation sets are explicitly reused, and free none is exact previous sigma.213 reuse. These traces distinguish an ineffective observation interface from a claim that all conditional completion is impossible. Sparse GT may be geometrically correct while receiving almost no inlier responsibility when the current predicted trajectory is far away. Such near-zero influence does not establish a failure of properly trained conditional imputation.','','This fixed small source-development diagnosis tests the frozen prior and current observation likelihood only. It does not establish cross-domain identity association, reliable dense rewards, OPD superiority, or a new TTA result. No automatic training/target/OPD continuation. Pixel panels inspect actual inputs and decoded outputs only and add no semantic identity labels.','','## Cost and verification','',f"Independent scalar/support/seal checks: {root['independent_checks']}. GPU lease {read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds']:.6f}s; cumulative {read(BASE/'GPU_COMPLETION.json')['cumulative_reserved_seconds']:.6f}s under original14400s cap. Includes two old no-observation interface replays; excludes upstream cached video/expert/feature work, CPU audit and publication. It is not online end-to-end latency."]
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def render():
 from PIL import Image,ImageDraw
 import numpy as np
 from scripts.run_decota_paper_main_v1 import frames_for
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 files=[]
 for ds in ['hc2','vidstg']:
  r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));frames,ids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert ids==z['frame_ids'];pos=load(BASE/'observations'/(r['alias']+'.pt'))['planned_positions']
  for clock in ['fixed','free']:
   q=load(BASE/'predictions'/clock/(r['alias']+'.pt'));arms=[('Native',dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval']))]if clock=='fixed'else[]
   arms += [(a,decode_state(q['arms'][a]['final'][0],z))for a in ['none','DINO_all','GT_planned']];canvas=Image.new('RGB',(1120,192*len(arms)),(22,22,22));draw=ImageDraw.Draw(canvas)
   for j,(name,a)in enumerate(arms):
    for k,idx in enumerate(pos):
     im=Image.fromarray(frames[idx]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy));cx,cy,w,h=a['boxes'][idx];draw.rectangle([ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height],outline=(255,210,40),width=2);draw.text((ox+4,j*192+4),f"{r['alias']} {clock}/{name} slot{k+1} event={a['interval'][0]<=ids[idx]<a['interval'][1]}",fill='white')
   f=BASE/'root_view'/(ds+'_'+clock+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);files.append(str(f))
 put(BASE/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_only=True,GT_overlays=False,identity_labels=False));print('\n'.join(files))

def diagnostics():
 import torch,numpy as np
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 result=read(BASE/'RESULTS.json');rows=[]
 def xy(b):
  b=np.asarray(b);return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
 for r in read(BASE/'PLAN_LOCK.json')['source']:
  z=load(OLD/'features/source'/(r['alias']+'.pt'));obs=load(BASE/'observations'/(r['alias']+'.pt'))['observations'];a=obs['DINO_common'];g=obs['GT_common'];item=dict(alias=r['alias'],dataset=r['dataset'],common_references=len(g['time']),DINO_common_GT_IoU=box_iou(xy(a['boxes'].numpy()),xy(g['boxes'].numpy())).tolist(),clocks={})
  for clock in ['fixed','free']:
   q=load(BASE/'predictions'/clock/(r['alias']+'.pt'));rr={}
   for arm in ARMS:
    o=obs[arm];vv=[]
    for x in q['arms'][arm]['final']:
     d=decode_state(x,z);pp=o['positions'];vv.append(float(box_iou(xy(d['boxes'][pp]),xy(o['boxes'].numpy())).mean())if pp else None)
    rr[arm]=dict(reference_count=len(o['time']),decoded_reference_IoU_samples=vv)
   item['clocks'][clock]=rr
  rows.append(item)
 aggregates={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in result['rows']if r['dataset']==ds];aggregates[ds]={}
  for clock in ['fixed','free']:
   aa={}
   for arm in ARMS[1:]:
    gg=[r['clocks'][clock][arm]['guidance']for r in rr if r['clocks'][clock][arm]['guidance']['reference_count']>0];aa[arm]=dict(nonempty_queries=len(gg))
    for k in ['mean_responsibility','sum_delta_norm','max_delta','state_L2_difference_from_none','first_active','last_active','min_active']:
     v=[r[k]for r in gg];aa[arm][k]=dict(min=float(np.min(v)),mean=float(np.mean(v)),max=float(np.max(v)))
    aa[arm]['zero_gradient_queries']=[r['alias']for r in rr if r['clocks'][clock][arm]['guidance']['reference_count']>0 and r['clocks'][clock][arm]['guidance']['nonzero_guidance_levels']==0]
   aggregates[ds][clock]=aa
 summary={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];vv=[v for r in rr for v in r['DINO_common_GT_IoU']];summary[ds]=dict(DINO_common_reference_GT_IoU_mean=float(np.mean(vv)),reference_count=len(vv),nonempty={})
  for clock in ['fixed','free']:
   summary[ds]['nonempty'][clock]={}
   for arm in ['DINO_all','GT_common','GT_planned']:
    ss=[r for r in result['rows']if r['dataset']==ds and r['reference_counts'][arm]>0];summary[ds]['nonempty'][clock][arm]=dict(n=len(ss),sample0_v_delta_mean=float(np.mean([r['clocks'][clock][arm]['samples'][0]['v']-r['clocks'][clock]['none']['samples'][0]['v']for r in ss])))
 numeric=read(BASE/'NUMERIC_INTERFACE_COMPLETION.json');out=dict(summary=summary,status='complete',source_only=True,rows=rows,influence_aggregates=aggregates,numeric_interface=numeric,not_semantic_identity_labels=True)
 put(BASE/'OBSERVATION_DIAGNOSTICS.json',out);put(OUT/'OBSERVATION_DIAGNOSTICS.json',out);put(OUT/'NUMERIC_INTERFACE_COMPLETION.json',numeric)
 lines=['','## Four-path sensitivity and guidance strength','','Mean of four separately decoded paths, vIoU percent; 12 query denominator per domain.','','| Clock / condition | HC | Vid |','|---|---:|---:|']
 for clock in ['fixed','free']:
  for arm in ['none','DINO_all','GT_common','GT_planned']:
   lines.append('| '+clock+' / '+arm+' | '+' | '.join(f"{result['means'][ds][clock]['mean_per_sample'][arm]['v']['mean']*100:.6f}"for ds in ['hc2','vidstg'])+' |')
 lines+=['','Guidance influence over nonempty queries (per-query trace statistics already average the four paths). Latent changes are not pixel units, and are not direct Student parameter updates.','','| Clock / condition | HC mean responsibility / summed step norm | Vid mean responsibility / summed step norm |','|---|---:|---:|']
 for clock in ['fixed','free']:
  for arm in ARMS[1:]:
   lines.append('| '+clock+' / '+arm+' | '+' | '.join(f"{aggregates[ds][clock][arm]['mean_responsibility']['mean']:.6f} / {aggregates[ds][clock][arm]['sum_delta_norm']['mean']:.6f}"for ds in ['hc2','vidstg'])+' |')
 lines+=['','On the fixed clock, GT_planned first-path gains over none are only +0.037860/+0.022777 pp vIoU (HC/Vid), and unobserved decoder gains are +0.028694/+0.074589 pp. The common-support GT-minus-DINO difference is +0.010171/+0.006562 pp vIoU; this is the reference-quality comparison without a coverage change. This does not establish useful correction, and GT observation quality alone does not solve this frozen interface.','','GT influence is not generally zero: fixed-clock GT_planned mean mixture responsibility is approximately .992 HC / .791 Vid, so widespread rejection of correct observations is not the general explanation. Some Vid oracle observations do receive essentially zero responsibility; the mixture can still suppress distant correct evidence. Actual capped steps are small under the unchanged .02*sigma^2 rule. This is a measured weak-response configuration, not evidence that a trained sparse-conditional completion model cannot work.','','Free-clock DINO_all changes first-path vIoU by -.126367/+.022650 pp, while GT_planned changes it by +.360942/+.115666 pp. HC GT_planned tIoU drops .378925 pp; geometric observations therefore do not establish event-boundary evidence. The original moving interval mask and event-relative nodes are still confounded with geometry at this clock. Even oracle free-clock outcomes remain below raw Native overall.','','The zero-guidance numerical interface replay uses the two prelocked first sources, four paths each at both clocks, with autograd enabled but its guidance exactly zero. It quantifies floating-point differences between inference and autograd execution kernels, not a new scored method. The probe leaves all scientific predictions unchanged; near-zero state differences should not be overinterpreted. Detailed maximum errors and actual-guidance displacements are in NUMERIC_INTERFACE_COMPLETION.json.','','## Decision','','The current frozen observation-likelihood interface has not demonstrated practically useful unobserved-frame completion, even with correct sparse source GT. Do not proceed to target adaptation, dense OPD, identity-model training, or solver/noise sweeps from this result. The result localizes a representation/conditioning/response limitation in this implementation without resolving their individual causal contributions. A redesigned sparse-conditional training task or fixed-video identity representation would be a separate proposal requiring its own bounded protocol; neither was run here.','','Additional numerical-probe GPU reservation is '+f"{numeric['new_reserved_seconds']:.6f}s; total current diagnosis {numeric['new_total_reserved_seconds']:.6f}s, cumulative {numeric['cumulative_reserved_seconds']:.6f}s under original14400s. No training." ]
 lines+=['','Nonempty-reference subsets are descriptive and do not replace full denominators. Details and common-reference GT IoU are in OBSERVATION_DIAGNOSTICS.json.']
 f=OUT/'REPORT.md';f.write_text(f.read_text()+'\n'.join(lines)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['audit_export','render','diagnostics']);arg=a.parse_args();globals()[arg.command]()
