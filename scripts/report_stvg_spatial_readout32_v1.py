"""Postseal metric, raw-candidate elimination and whole-layer audits, no tuning."""
from pathlib import Path
import argparse,collections,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_spatial_readout32_v1 import BASE,OLD,PREV
PUB=ROOT/'results/stvg_spatial_readout/2026-10-18/matched32';ARMS=['Source','original_OPD','target_rank_OPD','layer_select_original_state']
def score():
 import numpy as np,torch
 from scipy.special import logsumexp
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit as math_audit
 from vg_tta.stvg_spatial_readout_v1 import context_probe
 from torchvision.ops import box_convert,nms
 p=read(BASE/'PLAN_LOCK.json');bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==32
 zs=[load(ROOT/f['path'])for f in bar['files']];gtall={d:truths(d,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==d]))for d in ['hc2','vidstg']};rows=[];refs=[];layers=[];checks=mathchecks=unchanged=0;costs=collections.defaultdict(list)
 def ious(boxes,truth):
  boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4);t=np.asarray(truth,dtype=np.float64);p=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),1);inter=np.maximum(0,np.minimum(p[:,2:],t[2:])-np.maximum(p[:,:2],t[:2])).prod(-1);union=np.maximum(0,p[:,2:]-p[:,:2]).prod(-1)+np.maximum(0,t[2:]-t[:2]).prod()-inter;return inter/np.maximum(union,1e-7)
 for z in zs:
  ds=z['dataset'];q=z['query_ordinal'];row=read_row(ds,q);gt=gtall[ds][0][q];span=gtall[ds][1][q];iv=z['Native_interval'];n=len(z['frame_ids']);ll=z['original_layers'];assert ll.shape==(6,n,4);assert torch.equal(ll[-1],z['arms']['original_OPD']);assert torch.equal(z['new_fit']['final'],z['arms']['target_rank_OPD']);assert torch.equal(z['arms']['original_OPD'],z['original_complete_output']['boxes']);assert torch.equal(z['arms']['target_rank_OPD'],z['new_complete_output']['boxes']);checks+=4
  prior=load(PREV/'predictions'/(z['alias']+'.pt'));assert torch.equal(z['arms']['original_OPD'],prior['arms']['top1_frame']['fit']['final']);assert iv==prior['Native_interval'];checks+=2
  ma=math_audit(z['new_fit'],z['new_expert']);mathchecks+=1
  assert all(torch.equal(z['initial'][k],v)for k,v in z['new_fit']['initial'].items());assert z['new_fit']['config']==p['configurations'][ds]
  same=z['old_expert']['anchors']['single4']==z['new_expert']['anchors']['single4']
  if same:assert torch.equal(z['arms']['original_OPD'],z['arms']['target_rank_OPD']);unchanged+=1
  anchors=z['old_expert']['anchors']['single4']
  if anchors:
   independent=np.array([np.mean([ious(box[a['position']].numpy(),[a['box'][0]-a['box'][2]/2,a['box'][1]-a['box'][3]/2,a['box'][0]+a['box'][2]/2,a['box'][1]+a['box'][3]/2])[0]for a in anchors])for box in ll]);assert abs(independent-z['layer_scores'].numpy()).max()<2e-6;selected=int(independent.argmax());assert selected==z['selected_layer']
  else:selected=5;assert z['selected_layer']==5 and z['layer_scores']is None
  assert torch.equal(z['arms']['layer_select_original_state'],ll[selected]);checks+=3
  vals={};obs=[z['frame_ids'][i]for i in z['old_expert']['positions4']]
  for arm,boxes in [('Source',z['Source_boxes'])]+list(z['arms'].items()):
   tube=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2');m=official(boxes.numpy(),row,gt,span,iv,ds);assert max(abs(m[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10;checks+=3;m.update(R03=float(m['v']>.3),R05=float(m['v']>.5));mask=np.isin(tube.fids,z['frame_ids'])&~np.isin(tube.fids,obs);m['unobserved_decoder_sIoU']=float(tube.iou[mask].mean())if mask.any()else None;m['unobserved_decoder_count']=int(mask.sum());vals[arm]=m
  per=[]
  for layer in ll:
   tube=DenseTube(layer.numpy(),row,gt,span,clip=ds=='hc2');m=official(layer.numpy(),row,gt,span,iv,ds);assert max(abs(m[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10;per.append(m);checks+=3
  oracle=int(np.array([v['v']for v in per]).argmax());layers.append(dict(source_alias=z['alias'],dataset=ds,selected_layer_one_based=selected+1,empty_reference=not bool(anchors),six_actual_layer_values=per,postseal_GT_oracle_layer_one_based=oracle+1,oracle_v=per[oracle]['v'],selected_v=per[selected]['v'],layer6_v=per[-1]['v'],oracle_only_not_deployable=True))
  rows.append(dict(source_alias=z['alias'],dataset=ds,values=vals,same_references=same,old_admitted=len(anchors),new_admitted=len(z['new_expert']['anchors']['single4']),selected_layer_one_based=selected+1))
  for slot,(pos,raw)in enumerate(sorted(z['raw'].items())):
   d=raw['detection'];fid=z['frame_ids'][pos];oldp=raw['old_probe'];newp=raw['new_probe'];v=dict(source_alias=z['alias'],dataset=ds,slot=slot,context=raw['context'],GT_available=fid in gt,old_accepted=oldp['accepted'],new_accepted=newp['accepted'],old_reason=oldp['reason'],new_reason=newp['reason'],old_pool_count=len(oldp['boxes']),new_pool_count=len(newp['boxes']),raw_candidates=len(d['all_boxes']),identity='unknown',DINO_called=True)
   if raw['context']:
    for rank,stored in [('mixed',oldp),('target',newp)]:
     pp,st=context_probe(d,pos,fid,rank);assert torch.equal(pp['boxes'],torch.as_tensor(stored['boxes']));assert pp['candidate_ids']==stored['candidate_ids'];assert pp['accepted']==stored['accepted'];checks+=3
     # Independent torchvision NMS ranking list, not reusing candidate_stages.
     xy=box_convert(d['all_boxes'].float(),'cxcywh','xyxy').clamp(0,1);sc=d['all_phrase_scores']if rank=='mixed'else d['target_all_scores'];keep=nms(xy,sc,.5).tolist();assert keep==st['nms'];checks+=1
   if fid in gt:
    W=row['input']['width'];H=row['input']['height'];g=np.asarray(gt[fid],float)/np.array([W,H,W,H]);xy=box_convert(d['all_boxes'].float(),'cxcywh','xyxy').clamp(0,1);boxes=box_convert(xy,'xyxy','cxcywh').numpy();qualities=ious(boxes,g);valid=((xy[:,2:]-xy[:,:2])>0).all(1).numpy();validids=np.flatnonzero(valid);best=int(validids[qualities[validids].argmax()])if len(validids)else None
    v['raw_best_IoU']=float(qualities[best])if best is not None else None
    if raw['context']:
     for tag,stage in [('old',raw['old_stage']),('new',raw['new_stage'])]:
      for name,ids0 in [('NMS',stage['nms']),('Top3',stage['old_top3'])]:v[tag+'_'+name+'_best_IoU']=float(qualities[ids0].max())if ids0 else None
      v[tag+'_raw_best_eliminated_at']='no_valid_raw'if best is None else'NMS'if best not in stage['nms']else'Top3'if best not in stage['old_top3']else'preserved'
    for tag,pp in [('old',oldp),('new',newp)]:
     if pp['accepted']:
      j=int(np.asarray(pp['target_scores']).argmax());quality=float(ious(torch.as_tensor(pp['boxes'])[j].numpy(),g)[0]);v[tag+'_selected_IoU']=quality;v[tag+'_all_available_GT_IoU']=quality
     else:v[tag+'_selected_IoU']=None;v[tag+'_all_available_GT_IoU']=0.
   refs.append(v)
  # Preserve planned slots for queries with no legal detector input, not only calls.
  for slot,pos in enumerate(z['old_expert']['positions4']):
   if pos in z['raw']:continue
   fid=z['frame_ids'][pos];v=dict(source_alias=z['alias'],dataset=ds,slot=slot,context=False,GT_available=fid in gt,old_accepted=False,new_accepted=False,old_reason='no_legal_expert_input_no_call',new_reason='no_legal_expert_input_no_call',old_pool_count=0,new_pool_count=0,raw_candidates=0,identity='unknown',DINO_called=False)
   if fid in gt:v.update(raw_best_IoU=None,old_selected_IoU=None,new_selected_IoU=None,old_all_available_GT_IoU=0.,new_all_available_GT_IoU=0.)
   refs.append(v)
  for k,v in z['compute'].items():costs[ds+'_'+k].append(v)
 groups={d:{a:{k:float(np.mean([r['values'][a][k]for r in rows if r['dataset']==d]))for k in ['v','t','s','R03','R05','unobserved_decoder_sIoU']}for a in ARMS}for d in ['hc2','vidstg']};paired={}
 for ds in groups:
  paired[ds]={}
  for a in ARMS[2:]:
   rr=[r for r in rows if r['dataset']==ds];d=np.array([r['values'][a]['v']-r['values']['original_OPD']['v']for r in rr])*100;gen=np.random.default_rng(20261018);ci=np.quantile(d[gen.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]);paired[ds][a]=dict(mean_pp=float(d.mean()),CI95=ci.tolist(),positive=int((d>1e-9).sum()),negative=int((d< -1e-9).sum()),neutral=int((abs(d)<=1e-9).sum()),per_source_signed_pp={r['source_alias']:float(v)for r,v in zip(rr,d)})
 refgroups={}
 for ds in groups:
  rr=[r for r in refs if r['dataset']==ds];known=[r for r in rr if r['GT_available']];ctx=[r for r in known if r['context']];refgroups[ds]=dict(planned_observations=len(rr),actual_DINO_calls=sum(r['DINO_called']for r in rr),exact_GT=len(known),unknown_event_outside=len(rr)-len(known),context_exact_GT=len(ctx),old_admitted=sum(r['old_accepted']for r in rr),new_admitted=sum(r['new_accepted']for r in rr),old_all_GT_mean_IoU=float(np.mean([r['old_all_available_GT_IoU']for r in known])),new_all_GT_mean_IoU=float(np.mean([r['new_all_available_GT_IoU']for r in known])),old_success_at_0p5=sum(r['old_all_available_GT_IoU']>=.5 for r in known),new_success_at_0p5=sum(r['new_all_available_GT_IoU']>=.5 for r in known),elimination={a:{stage:sum(r[a+'_raw_best_eliminated_at']==stage for r in ctx)for stage in ['NMS','Top3','preserved']}for a in ['old','new']},geometric_coverage_at_0p5={stage:sum(r.get(stage) is not None and r[stage]>=.5 for r in ctx)for stage in ['raw_best_IoU','old_NMS_best_IoU','old_Top3_best_IoU','new_NMS_best_IoU','new_Top3_best_IoU']},recovered_Top3_geometric_coverage=sum((r['old_Top3_best_IoU']or 0)<.5 and(r['new_Top3_best_IoU']or 0)>=.5 for r in ctx),lost_Top3_geometric_coverage=sum((r['old_Top3_best_IoU']or 0)>=.5 and(r['new_Top3_best_IoU']or 0)<.5 for r in ctx))
 PUB.mkdir(parents=True,exist_ok=True);put(PUB/'RESULTS.json',dict(status='complete',count=32,groups=groups,paired=paired,rows=rows,time_fixed='same Native for every arm',historical_exposed=True,GT_only_after_all_prediction_seal=True));put(PUB/'CANDIDATE_QUALITY.json',dict(status='complete',count=32,groups=refgroups,observations=refs,geometric_coverage_is_not_identity_annotation=True,all_unannotated_observations_retained=True));put(PUB/'LAYER_DIAGNOSIS.json',dict(status='complete',count=32,rows=layers,GT_oracle_diagnostic_not_online_selection=True));put(PUB/'COST.json',dict(status='complete',count=32,new_DINO_calls=bar['new_DINO_calls'],GPU_lease_seconds=bar['GPU_lease_seconds'],actual={k:dict(total=float(sum(v)),mean=float(np.mean(v)))for k,v in costs.items()},old_OPD_fits_exact_reused_not_reexecuted=True,new_source_captures=32,new_fits_at_most=32,no_end_to_end_speedup_claim=True));put(PUB/'VERIFICATION.json',dict(status='pass',count=32,checks=checks,new_OPD_mathematical_audits=mathchecks,unchanged_reference_full_fit_exact_controls=unchanged,all_DINO_raw_old_pipeline_input_and_output_exact=True,all_six_layer_cached_and_complete_second_pass_equal=True,all_original_parameter_states_and_final_layer6_equal=True,all_target_GT_after_prediction_seal=True))
 lines=['# Fixed32 spatial candidate ranking and six-layer readout','', 'Same exposed16 sources/domain as the previous matched Source-episodic32, original domain OPD budgets and fixed Native time. Two changes tested separately; not combined, no new model training. Original OPD states/final outputs replay exactly. Raw DINO outputs recaptured on original four frames, same tensors feed both ranking pipelines; original probes and input hashes reproduce exactly.','', '| Domain | Arm | vIoU % | tIoU % | sIoU % | Unobserved actual decoder sIoU % |','|---|---|---:|---:|---:|---:|']
 for ds in groups:
  for a in ARMS:
   v=groups[ds][a];lines.append('| '+ds+' | '+a+' | '+' | '.join(f'{100*v[k]:.6f}'for k in ['v','t','s','unobserved_decoder_sIoU'])+' |')
 lines+=['','All signed per-source paired differences, damage tails, R03/R05 and intervals are in RESULTS;10000 paired parent bootstrap, unadjusted. Same initial Source1792 and original LR/steps/noise/tau/seed; no inherited LN. Ranking arm actually re-fits original OPD; layer arm fixes original adapted state, selecting exactly one whole real layer trajectory with original accepted DINO references. Six layers are not independent/calibrated experts. No reference defaults to layer6. Both complete decoder passes and cached six-layer outputs agree.','', 'Candidate diagnosis uses exact annotated physical anchor frames only; no GT box interpolation and no filtering/re-ranking by GT online. Raw best geometric overlap is an oracle availability diagnostic, not evidence a useful semantic selector can find it. NMS and Top3 attrition are separated. Geometry IoU cannot certify correct person; all identity labels unknown. Event-outside frames keep unknown GT and remain part of deployed observations.','', 'Whole-six-layer GT oracle is postseal diagnosis only; it never supplies the selected layer. WBF/coordinate fusion and stacking the two changes are not run. These small development results cannot promote a full method or prove universal conclusions.']
 (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n');put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,checks=checks,unchanged_reference_controls=unchanged));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,GT_read=True));print(json.dumps(dict(groups=groups,paired=paired,reference=refgroups),indent=2))
def render():
 import numpy as np
 from PIL import Image,ImageDraw
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 p=read(BASE/'PLAN_LOCK.json');rr=read(PUB/'RESULTS.json')['rows'];selected=[];out=[]
 for ds in ['hc2','vidstg']:
  first=next(r for r in p['records']if r['dataset']==ds);selected.append((ds,'fixed_first',first))
  for arm in ['target_rank_OPD','layer_select_original_state']:
   worst=min((r for r in rr if r['dataset']==ds),key=lambda x:x['values'][arm]['v']-x['values']['original_OPD']['v']);r=next(r for r in p['records']if r['alias']==worst['source_alias']);selected.append((ds,'posthoc_worst_'+arm,r))
 for ds,kind,r in selected:
  z=load(BASE/'predictions'/(r['alias']+'.pt'));row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,collections.OrderedDict());obs=z['old_expert']['positions4'];pos=obs+[i for i in np.linspace(0,len(ids)-1,8,dtype=int)if i not in obs][:4];canvas=Image.new('RGB',(320*4,220*len(pos)),'white');d=ImageDraw.Draw(canvas)
  for j,i in enumerate(pos):
   for c,a in enumerate(ARMS):
    im=Image.fromarray(frames[i]).convert('RGB').resize((320,190));b=z['Source_boxes'][i]if a=='Source'else z['arms'][a][i];x,y,w,h=map(float,b);ImageDraw.Draw(im).rectangle(((x-w/2)*320,(y-h/2)*190,(x+w/2)*320,(y+h/2)*190),outline='red',width=3);canvas.paste(im,(320*c,220*j));d.text((320*c+4,220*j+192),a+(' observed'if i in obs else' unobserved'),fill='black')
  f=BASE/'root_review'/(ds+'_'+kind+'.png');f.parent.mkdir(exist_ok=True);canvas.save(f);out.append(dict(path=str(f),sha256=sha(f),dataset=ds,selection=kind,source_alias=r['alias'],cells=4*len(pos),private=True))
 put(BASE/'ROOT_VIEW_INPUTS.json',dict(status='ready_for_actual_view',files=out));print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','render']);globals()[p.parse_args().stage]()
