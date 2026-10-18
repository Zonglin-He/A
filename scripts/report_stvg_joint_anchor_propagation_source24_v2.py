"""Independent exact-anchor and unobserved-decoder propagation diagnosis."""
from pathlib import Path
import sys,argparse,collections,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_anchor_propagation_source24_v2 import BASE,OUT,OLD,PREVIOUS,read,put,load,sha,condition
ARMS=['Native_raw','N','A','P','AP']

def audit_export():
 import torch,numpy as np
 from scipy.stats import t
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 from vg_tta.stvg_joint_anchor_projection_v2 import interpolation_matrix
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
 torch.set_num_threads(4);assert not torch.cuda.is_initialized();assert read(BASE/'PREDICTION_BARRIER.json')['count']==24
 p=read(BASE/'PLAN_LOCK.json');checks=0;rows=[]
 for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f;checks+=1
 def xy(b):
  b=np.asarray(b);return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
 for r in p['source']:
  alias=r['alias'];z=load(OLD/'features/source'/(alias+'.pt'));y=load(OLD/'source_supervision'/(alias+'.pt'));q=load(BASE/'inputs'/(alias+'.pt'));v=load(BASE/'predictions'/(alias+'.pt'));ids=np.asarray(z['frame_ids']);iv=z['native_interval'];planned=load(PREVIOUS/'observations'/(alias+'.pt'))['planned_positions'];anchor_nodes=(q['W']!=0).any(0);W_all=interpolation_matrix(q['native'],z,list(range(len(ids))));outside=((W_all[:,anchor_nodes]!=0).sum(1)==0)if anchor_nodes.any()else np.ones(len(ids),dtype=bool);outside_ids=ids[outside];row=dict(frame_ids=z['frame_ids'],input=z['input']);counts={};metrics={};anchor={}
  states=dict(N=v['N'],A=v['A'],P=v['P']['final'],AP=v['AP']['final']);decoded={a:decode_state(x,z)for a,x in states.items()};decoded['Native_raw']=dict(boxes=z['boxes'][0].numpy(),interval=iv)
  assert torch.equal(states['A'],q['anchor_corrected']);checks+=1
  if not len(q['W']):assert torch.equal(states['N'],states['A'])and torch.equal(states['P'],states['AP']);checks+=2
  if len(q['W']):
   target=q['target'].numpy()
   for a in ['A','AP']:
    pred=decoded[a]['boxes'][q['positions']];err=float(np.max(np.abs(pred-target)));assert err<2e-6;checks+=1
   assert np.max(np.abs(decoded['A']['boxes'][q['positions']]-decoded['AP']['boxes'][q['positions']]))<2e-6;checks+=1
  # Pure interpolation control cannot change a decoder frame outside its knot footprint.
  assert np.array_equal(decoded['A']['boxes'][outside],decoded['N']['boxes'][outside]);checks+=1
  for a,d in decoded.items():
   assert d['interval']==iv;checks+=1
   if a!='Native_raw':assert torch.equal(states[a][0],q['native'][0]);checks+=1
   dt=DenseTube(d['boxes'],row,y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2');m=dt.score(iv);off=official(d['boxes'],row,y['dense_source_GT'],y['span'],iv,r['dataset'])
   for k in ['v','t','s']:assert abs(m[k]-off[k])<2e-9;checks+=1
   m.update(R03=float(m['v']>=.3),R05=float(m['v']>=.5))
   within=(dt.fids>=ids[0])&(dt.fids<=ids[-1])&(dt.fids>=iv[0])&(dt.fids<iv[1]);obs=np.isin(dt.fids,ids[planned]);native=np.isin(dt.fids,ids);external=np.isin(dt.fids,outside_ids)
   masks=dict(observed=within&obs,unobserved_native=within&~obs&native,unobserved_interpolated=within&~obs&~native,outside_footprint_native=within&~obs&native&external)
   for k,mask in masks.items():
    m[k+'_s']=float(dt.iou[mask].mean())if mask.any()else None
    if k in counts:assert counts[k]==int(mask.sum())
    counts[k]=int(mask.sum())
   metrics[a]=m
   pp=q['positions'];gg=q['target'].numpy();bb=d['boxes'][pp];anchor[a]=dict(n=len(pp),coordinate_RMSE=float(np.sqrt(np.mean((bb-gg)**2)))if pp else None,max_coordinate_error=float(np.max(np.abs(bb-gg)))if pp else None,IoU_mean=float(box_iou(xy(bb),xy(gg)).mean())if pp else None)
  assert metrics['N']['t']==metrics['A']['t']==metrics['P']['t']==metrics['AP']['t'];checks+=1
  if q['positions']:
   assert anchor['A']['coordinate_RMSE']<2e-6 and anchor['AP']['coordinate_RMSE']<2e-6
   assert anchor['A']['coordinate_RMSE']<=anchor['N']['coordinate_RMSE']+2e-6;checks+=3
  rows.append(dict(alias=alias,dataset=r['dataset'],reference_count=len(q['W']),anchor_touched_knots=int(anchor_nodes.sum()),decoder_frames=len(ids),outside_footprint_decoder_count=int(outside.sum()),support_counts=counts,metrics=metrics,anchor=anchor,propagation_trace=dict(P=v['P']['trace'],AP=v['AP']['trace'])))
 keys=list(rows[0]['metrics']['N']);means={};paired={};interactions={};anchor_summary={}
 def summary(values):
  vv=np.asarray([v for v in values if v is not None]);return dict(mean=float(vv.mean())if len(vv)else None,n=len(vv))
 def diff(values):
  vv=np.asarray([v for v in values if v is not None]);n=len(vv);mu=float(vv.mean())if n else None;rad=float(t.ppf(.975,n-1)*vv.std(ddof=1)/np.sqrt(n))if n>1 else None
  return dict(mean=mu,n=n,CI95=[mu-rad,mu+rad]if rad is not None else None,positive=int((vv>1e-8).sum()),negative=int((vv< -1e-8).sum()),neutral=int((np.abs(vv)<=1e-8).sum()),values=vv.tolist())
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==12;means[ds]={a:{k:summary([r['metrics'][a][k]for r in rr])for k in keys}for a in ARMS};anchor_summary[ds]={a:{k:summary([r['anchor'][a][k]for r in rr])for k in ['coordinate_RMSE','IoU_mean']}for a in ARMS};paired[ds]={}
  for a,b in [('A','N'),('P','N'),('AP','A'),('AP','P'),('AP','N')]:
   paired[ds][a+' - '+b]={k:diff([r['metrics'][a][k]-r['metrics'][b][k]if r['metrics'][a][k]is not None else None for r in rr])for k in keys};checks+=len(keys)
  interactions[ds]={k:diff([r['metrics']['AP'][k]-r['metrics']['P'][k]-r['metrics']['A'][k]+r['metrics']['N'][k]if r['metrics']['N'][k]is not None else None for r in rr])for k in keys}
  for comp,dd in paired[ds].items():
   for k,v in dd.items():assert v['positive']+v['negative']+v['neutral']==v['n'];checks+=1
 gate=all(paired[ds]['AP - A'][k]['mean']is not None and paired[ds]['AP - A'][k]['mean']>0 for ds in ['hc2','vidstg']for k in ['unobserved_native_s','outside_footprint_native_s'])
 result=dict(status='complete',count=24,source_only=True,source_GT_oracle=True,added_noise=False,new_training=0,means=means,paired=paired,interaction_AP_P_minus_A_N=interactions,anchor_summary=anchor_summary,rows=rows,DINO_propagation_gate=gate,decision='cached_DINO_comparison_eligible'if gate else 'stop_before_DINO_no_training_target_or_OPD')
 OUT.mkdir(exist_ok=True,parents=True);put(BASE/'RESULTS.json',result);pub={k:v for k,v in result.items()if k!='rows'};pub['cases']=[]
 for r in rows:
  f=OUT/'cases'/(r['alias']+'.json');put(f,r);pub['cases'].append(dict(alias=r['alias'],dataset=r['dataset'],file='cases/'+f.name))
 put(OUT/'RESULTS.json',pub);root=dict(status='pass',count=24,checks=checks,all24_predictions_sealed_before_full_source_GT_metrics=True,all_anchors_within_representational_tolerance=True,A_AP_final_anchors_matched=True,time_exact_Native=True,outside_interpolation_footprint_A_exact_N=True,official_DenseTube_metrics_exact=True,all_old_pins_unchanged=True,no_training=True)
 put(BASE/'ROOT_COMPLETION.json',root);put(OUT/'ROOT_COMPLETION.json',root);put(BASE/'PROPAGATION_DECISION.json',dict(status='eligible'if gate else 'stop',DINO_gate=gate,source_GT_diagnostic_only=True))
 for f in ['CPU_CONTRACT_COMPLETION.json','GPU_COMPLETION.json'] :put(OUT/f,read(BASE/f))
 put(OUT/'PROTOCOL.json',{k:v for k,v in p.items()if k!='source'});report(result,root)
 print('ROOT',checks,'DINO_gate',gate)
 for ds in ['hc2','vidstg']:print(ds,json.dumps(dict(means=means[ds],anchor=anchor_summary[ds],AP_minus_A=paired[ds]['AP - A'])))

def report(r,root):
 means=r['means'];paired=r['paired'];lines=['# Strong sparse-anchor correction versus frozen-prior propagation','','This is a source-only oracle diagnostic on the same12 HC and12 Vid validation queries, already exposed to auxiliary-prior development. Repaired HC30/Vid20 and cached input conditions remain frozen. It is not a deployable TTA result or Direct-versus-OPD comparison. No new training, expert calls, target inference or Student update.','','## Four controls','','N: Native represented by the original32 event-relative knots and decoded with the original interpolation. A: minimal probability-space knot correction satisfying the available sparse GT anchors, with that same interpolation. P: unanchored frozen-prior propagation from N. AP: propagation from the identical A start, with the identical sparse anchors reimposed on every clean estimate and proposal. Native time is exact in every state. Empty-anchor A=N/AP=P exactly.','','Only existing knot columns touched by anchor interpolation may change during projection; all other logits remain byte exact. If needed, bounded convex SLSQP enforces finite sigmoid coordinates, with no unobserved GT input. The GT observation count is37 HC and23 Vid at the original four planned physical locations, not4 fabricated labels/query. Three Vid queries have no available oracle anchor and remain in the full denominator.','','P and AP use the same old16-step .213 to .03 VE algebra, with NO new global noise. This intentionally separates deterministic propagation from artificial noise recovery. It is not standard diffusion sampling or a calibrated posterior; sigma is the frozen model conditioning input. No schedule/solver/strength tuning is performed.','','## Effective anchor positive control','','Query-mean values over nonempty anchors (HC12/Vid9). RMSE uses normalized cxcywh, not pixels. All predicted A/AP anchor coordinates are checked against sparse GT at full decoder output, not just latent knots.','','| Arm | HC anchor RMSE | HC anchor IoU | Vid anchor RMSE | Vid anchor IoU |','|---|---:|---:|---:|---:|']
 for a in ['N','A','P','AP']:
  vv=[r['anchor_summary'][ds][a][k]['mean']for ds in ['hc2','vidstg']for k in ['coordinate_RMSE','IoU_mean']];lines.append('| '+a+' | '+' | '.join(f'{v:.9f}'for v in vv)+' |')
 lines+=['','## Overall fixed-time results','','Percent; full12-query denominator per domain. All tIoU values must equal Native exactly.','','| Arm | HC vIoU | HC tIoU | HC sIoU | Vid vIoU | Vid tIoU | Vid sIoU |','|---|---:|---:|---:|---:|---:|---:|']
 for a in ARMS:lines.append('| '+a+' | '+' | '.join(f"{means[ds][a][k]['mean']*100:.6f}"for ds in ['hc2','vidstg']for k in ['v','t','s'])+' |')
 lines+=['','## Spatial supports','','All supports are fixed from original physical Native time, original decoder frame IDs, original four planned anchors and annotation availability. Unobserved-native excludes all four planned positions. Outside-footprint-native further excludes every actual decoder frame whose interpolation touches an anchor-corrected knot. A is exactly N on this last support by construction. Unknown/empty supports remain null; counts are queries contributing to each support mean. Benchmark interpolation frames are reported separately.','','| Arm | HC unobserved native sIoU (n) | HC outside footprint sIoU (n) | HC interpolated sIoU (n) | Vid unobserved native sIoU (n) | Vid outside footprint sIoU (n) | Vid interpolated sIoU (n) |','|---|---:|---:|---:|---:|---:|---:|']
 for a in ['N','A','P','AP']:
  vals=[]
  for ds in ['hc2','vidstg']:
   for k in ['unobserved_native_s','outside_footprint_native_s','unobserved_interpolated_s']:
    v=means[ds][a][k];vals.append(f"{v['mean']*100:.6f} ({v['n']})"if v['mean']is not None else 'unknown (0)')
  lines.append('| '+a+' | '+' | '.join(vals)+' |')
 lines+=['','## Matched propagation effects','','Source-query paired Student-t95% intervals and positive/negative/neutral counts, percentage points. They describe this source-development cohort; no target confirmation or selected best path.','','| Contrast / metric | HC | Vid |','|---|---|---|']
 for comp in ['A - N','P - N','AP - A','AP - P']:
  for k in ['v','unobserved_native_s','outside_footprint_native_s']:
   vals=[]
   for ds in ['hc2','vidstg']:
    a=paired[ds][comp][k];ci=a['CI95'];vals.append(f"{a['mean']*100:+.6f} [{ci[0]*100:+.6f}, {ci[1]*100:+.6f}] {a['positive']}/{a['negative']}/{a['neutral']}"if ci else 'unknown')
   lines.append('| '+comp+' / '+k+' | '+' | '.join(vals)+' |')
 lines+=['','The difference of differences (AP-P)-(A-N) is reported separately in RESULTS.json. It measures conditional responsiveness relative to the priors own unanchored drift; it does not by itself prove useful propagation, which requires AP>A on unobserved actual frames. Repeated anchoring can also change a nonlinear projection trajectory, so the interaction is not pure proof of learned semantic dependence.','','## Decision and limits','',str(r['decision']),'. Prelocked DINO eligibility requires positive AP-A mean on BOTH unobserved-native and outside-footprint-native supports in BOTH domains, after anchor qualification. It is an engineering continuation decision, not proof from statistical significance.','',f"Independent checks: {root['checks']}. Cached GPU reservation: {read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds']:.6f}s; cumulative {read(BASE/'GPU_COMPLETION.json')['cumulative_reserved_seconds']:.6f}s under original14400s. CPU projections, full metric audit and upstream video/expert/feature work are separate, not online E2E latency.",'','This test gives observations explicit leverage on representable anchors. If AP fails to improve over A, weak anchor utilization no longer explains that comparison; the failure applies to this frozen propagation procedure and source cohort. It does not show that properly trained conditional imputation or every diffusion design cannot work. No learned association/TopK/fixed-video identity representation, new training or dense OPD was introduced. Actual pixel review is engineering only, not semantic identity annotation.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def render():
 from PIL import Image,ImageDraw
 import numpy as np
 from scripts.run_decota_paper_main_v1 import frames_for
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 files=[]
 for ds in ['hc2','vidstg']:
  r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));q=load(BASE/'predictions'/(r['alias']+'.pt'));pp=load(PREVIOUS/'observations'/(r['alias']+'.pt'))['planned_positions'];frames,ids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert ids==z['frame_ids'];canvas=Image.new('RGB',(1120,768),(22,22,22));draw=ImageDraw.Draw(canvas)
  for j,a in enumerate(['N','A','P','AP']):
   d=decode_state(q[a]['final']if a in ['P','AP']else q[a],z)
   for k,pos in enumerate(pp):
    im=Image.fromarray(frames[pos]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy));cx,cy,w,h=d['boxes'][pos];draw.rectangle([ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height],outline=(255,210,40),width=2);draw.text((ox+4,j*192+4),f"{r['alias']} {a} planned slot{k+1}",fill='white')
  f=BASE/'root_view'/(ds+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);files.append(str(f))
 put(BASE/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_only=True,GT_overlay=False,new_identity_labels=0));print('\n'.join(files))

def render_unobserved():
 import numpy as np
 from PIL import Image,ImageDraw
 from scripts.run_decota_paper_main_v1 import frames_for
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 from vg_tta.stvg_joint_anchor_projection_v2 import interpolation_matrix
 files=[]
 for ds in ['hc2','vidstg']:
  r=next(r for r in read(BASE/'PLAN_LOCK.json')['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));q=load(BASE/'predictions'/(r['alias']+'.pt'));inp=load(BASE/'inputs'/(r['alias']+'.pt'));ids=np.asarray(z['frame_ids']);Wa=interpolation_matrix(q['N'],z,list(range(len(ids))));nodes=(inp['W']!=0).any(0);outside=~(Wa[:,nodes]!=0).any(1);iv=z['native_interval'];eligible=np.flatnonzero(outside&(ids>=iv[0])&(ids<iv[1]));assert len(eligible)>=4;pos=eligible[np.linspace(0,len(eligible)-1,4).round().astype(int)]
  frames,frameids,_=frames_for(ds,dict(input=r['input'],frame_ids=r['frame_ids']),collections.OrderedDict());assert frameids==z['frame_ids'];canvas=Image.new('RGB',(1120,768),(22,22,22));draw=ImageDraw.Draw(canvas)
  for j,a in enumerate(['N','A','P','AP']):
   d=decode_state(q[a]['final']if a in ['P','AP']else q[a],z)
   for k,idx in enumerate(pos):
    im=Image.fromarray(frames[idx]);im.thumbnail((280,155));ox,oy=k*280,j*192+34;canvas.paste(im,(ox,oy));cx,cy,w,h=d['boxes'][idx];draw.rectangle([ox+(cx-w/2)*im.width,oy+(cy-h/2)*im.height,ox+(cx+w/2)*im.width,oy+(cy+h/2)*im.height],outline=(255,210,40),width=2);draw.text((ox+4,j*192+4),f"{r['alias']} {a} outside-footprint slot{k+1}",fill='white')
  f=BASE/'root_view'/(ds+'_unobserved.png');canvas.save(f);files.append(str(f))
 put(BASE/'ROOT_UNOBSERVED_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,choice='first source eachdomain, four uniformly indexed outside-footprint actual decoder frames, no GT gain selection',new_identity_labels=0));print('\n'.join(files))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['audit_export','render','render_unobserved']);arg=a.parse_args();globals()[arg.command]()
