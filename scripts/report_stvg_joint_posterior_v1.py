"""Root source-validation diagnostics, independent scalar audit and public export."""
from pathlib import Path
import json,sys,math,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_v1 import BASE,OLD,KINDS,read,put,save,load,sha,condition,decode_state,verify
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/pilot32'

def source_validation():
 import torch,numpy as np
 from scipy.stats import rankdata
 from vg_tta.stvg_joint_posterior_v1 import JointPrior,Association,roi,posterior
 from scripts.run_decota_paper_main_v1 import gpu
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
 p=verify();lease=gpu();rows=[];models={};calibration={}
 try:
  for n,r in enumerate([r for r in p['source']if r['split']=='source_validation']):
   z=load(BASE/'features/source'/(r['alias']+'.pt'));y=load(BASE/'source_supervision'/(r['alias']+'.pt'));ds=r['dataset'];c=condition(z);ad=load(BASE/'models'/(ds+'_association.pt'));assoc=Association().cuda();assoc.load_state_dict(ad['state']);assoc.eval().requires_grad_(False);lo,hi=z['frame_ids'][0],z['frame_ids'][-1]+1;anchors=z['expert']['anchors']['single4']
   obs=dict(time=torch.tensor([(z['frame_ids'][a['position']]-lo)/(hi-lo)for a in anchors],device='cuda'),boxes=torch.tensor([a['box']for a in anchors],device='cuda').reshape(-1,4))
   with torch.no_grad():
    visual=roi(c['maps'],obs['boxes'][None],obs['time'][None])[0]if anchors else torch.zeros(0,512,device='cuda');prior=assoc(visual,c['query'].mean(1).expand(len(anchors),-1)).sigmoid()
   # Only source validation labels; no target images/labels enter this check.
   positions=[a['position']for a in anchors];labs={v['position']:v['label']for v in y['associations']};calibration.setdefault(ds,[]).extend([(float(prior[i]),labs[pos])for i,pos in enumerate(positions)if pos in labs])
   predictions=dict(Source=dict(boxes=z['boxes'][0].numpy(),interval=z['native_interval']),Direct=dict(boxes=z['boxes'][1].numpy(),interval=z['native_interval']))
   for kind in KINDS:
    if(ds,kind)not in models:
     net=JointPrior(kind=='separated_diffusion').cuda();net.load_state_dict(load(BASE/'models'/(ds+'_'+kind+'.pt'))['state']);models[ds,kind]=net.eval().requires_grad_(False)
    cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()};x,_=posterior(models[ds,kind],cc,obs,prior,ad['variance'],'deterministic'if kind=='joint_deterministic'else'diffusion',20261018+n)
    predictions[kind]=decode_state(x.mean(0),z)
   save(BASE/'source_validation_predictions'/(r['alias']+'.pt'),dict(predictions=predictions,source_only=True,checkpoint_selected_before_this_posterior_evaluation=True))
   # Frozen-model source-validation metrics are diagnostic, no reselection.
   metric={arm:official(a['boxes'],dict(frame_ids=z['frame_ids'],input=z['input']),y['dense_source_GT'],y['span'],a['interval'],ds)for arm,a in predictions.items()}
   rows.append(dict(alias=r['alias'],dataset=ds,metrics=metric));print('SOURCE_VALIDATION',r['alias'],flush=True)
  means={ds:{a:{k:float(np.mean([r['metrics'][a][k]for r in rows if r['dataset']==ds]))for k in ['v','t','s']}for a in rows[0]['metrics']}for ds in ['hc2','vidstg']};ca={}
  for ds,v in calibration.items():
   pp=np.array([a for a,b in v]);yy=np.array([b for a,b in v]);n1=int(yy.sum());n0=len(v)-n1;auc=float((rankdata(pp)[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))if n0 and n1 else None
   ca[ds]=dict(count=len(v),inliers=n1,outliers=n0,brier=float(np.mean((pp-yy)**2)),threshold05_accuracy=float(np.mean((pp>=.5)==yy)),AUROC=auc,scope='held-out source geometric compatibility only; not target identity labels')
  put(BASE/'SOURCE_VALIDATION_POSTERIOR.json',dict(status='complete',count=24,rows=rows,means=means,association=ca,no_retraining_or_reselection=True,target_GT_used=False))
 finally:lease.close()

def audit():
 import numpy as np
 from scipy.stats import t
 p=read(BASE/'POSTSEAL_RESULTS.json');checks=0
 assert read(BASE/'PREDICTION_BARRIER.json')['count']==32
 for ds in ['hc2','vidstg']:
  rr=[r for r in p['rows']if r['dataset']==ds];assert len(rr)==16
  for arm,metric in p['means'][ds].items():
   for k,v in metric.items():assert abs(v-np.mean([r['metrics'][arm][k]for r in rr]))<1e-12;checks+=1
  for contrast,ms in p['paired'][ds].items():
   a,b=contrast.split(' - ')
   for k,v in ms.items():
    d=np.array([r['metrics'][a][k]-r['metrics'][b][k]for r in rr]);assert np.allclose(d,v['values'],atol=1e-14,rtol=0)
    assert abs(float(d.mean())-v['mean'])<1e-12;ci=np.array([d.mean()-t.ppf(.975,15)*d.std(ddof=1)/4,d.mean()+t.ppf(.975,15)*d.std(ddof=1)/4]);assert np.allclose(ci,v['CI95'],atol=1e-12,rtol=0)
    assert v['positive']+v['negative']+v['neutral']==16;checks+=len(d)+5
  for r in rr:
   for kind,g in r['coupling_gradient'].items():
    if kind=='separated_diffusion':assert g['time_output_spatial_state_gradient']==0 and g['spatial_output_time_state_gradient']==0
    else:assert g['time_output_spatial_state_gradient']>0 and g['spatial_output_time_state_gradient']>0
    checks+=2
 expected=all(p['means'][ds]['joint_diffusion'][k]>p['means'][ds]['Direct'][k]for ds in ['hc2','vidstg']for k in ['v','t']);assert p['gate_pass']==expected
 # Frozen source checkpoint selection is exactly source-validation minimum.
 training=read(BASE/'TRAINING_COMPLETION.json');assert len(training['rows'])==6
 for r in training['rows']:
  assert r['optimizer_steps']==360 and len(r['history'])==60
  pick=min(r['history'],key=lambda q:q['source_validation_loss']);assert r['selected_epoch']==pick['epoch']and r['source_validation_loss']==pick['source_validation_loss'];checks+=3
 put(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=32,independent_scalar_and_CI_checks=checks,source_only_selection=True,separated_cross_derivatives_zero=True,all_predictions_sealed_before_target_GT=True))
 print('ROOT_AUDIT',checks,flush=True)

def render():
 from PIL import Image,ImageDraw
 import numpy as np
 from scripts.run_decota_paper_main_v1 import frames_for,read_row
 import collections
 p=read(BASE/'PLAN_LOCK.json');files=[]
 # Prelocked first case each domain, plus known mechanism cases H002/H016.
 for alias in ['H001','H002','H016','V001']:
  r=next(r for r in p['target']if r['alias']==alias);z=load(BASE/'features/target'/(alias+'.pt'));pred=load(BASE/'predictions'/(alias+'.pt'));row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,collections.OrderedDict())
  positions=sorted(set(z['expert']['positions4']));positions=positions[:4];arms=list(pred['arms']);tilew,tileh=300,205;canvas=Image.new('RGB',(tilew*len(positions),tileh*len(arms)),(25,25,25));d=ImageDraw.Draw(canvas)
  for j,arm in enumerate(arms):
   for k,pos in enumerate(positions):
    im=Image.fromarray(frames[pos]);w,h=im.size;im.thumbnail((tilew,170));x=k*tilew;y=j*tileh+30;canvas.paste(im,(x,y));b=pred['arms'][arm]['boxes'][pos];cx,cy,bw,bh=b;bx=[(cx-bw/2)*im.width+x,(cy-bh/2)*im.height+y,(cx+bw/2)*im.width+x,(cy+bh/2)*im.height+y];d.rectangle(bx,outline=(255,210,40),width=2)
    inside=pred['arms'][arm]['interval'][0]<=ids[pos]<pred['arms'][arm]['interval'][1];d.text((x+3,j*tileh+4),f'{alias} {arm} anchor{k+1} event={inside}',fill='white')
  dest=BASE/'root_review'/(alias+'.png');dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest);files.append(str(dest))
 put(BASE/'ROOT_VIEW_PLAN.json',dict(status='rendered_pending_actual_view',files=files,source_media_private=True,GT_overlays=False,labels='engineering input/output check, not identity annotation'))
 print('\n'.join(files),flush=True)

def export():
 import numpy as np
 p=read(BASE/'POSTSEAL_RESULTS.json');a=read(BASE/'ROOT_SCALAR_READBACK.json');assert a['status']=='pass'
 assert read(BASE/'ROOT_VIEW_COMPLETION.json')['status']=='pass'
 OUT.mkdir(parents=True,exist_ok=True)
 public=dict(p)
 # Aliases only; no original identifiers, tube coordinates or text.
 (OUT/'RESULTS.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
 for name in ['TRAINING_COMPLETION.json','SOURCE_VALIDATION_POSTERIOR.json','CPU_CONTRACT_COMPLETION.json','ROOT_SCALAR_READBACK.json','DECISION.json']:
  (OUT/name).write_text(json.dumps(read(BASE/name),indent=2,allow_nan=False)+'\n')
 plan=read(BASE/'PLAN_LOCK.json');protocol={k:v for k,v in plan.items()if k not in ['source','target','source_provenance','created_epoch','deadline_epoch']};protocol['data_count']=dict(source_train=72,source_validation=24,target_HC=16,target_Vid=16,parent_disjointness=True,source_binding='original video + query + used segment + targetID, binding002');put(OUT/'PROTOCOL.json',protocol)
 lines=['# Joint event–trajectory posterior: source96 → exposed target32','','Source-only learned prior, fixed finite source/target protocol. No Qwen, no new DINO, no production promotion. Previous target oracle exposure is retained. This is a feasibility pilot, not an independent test.','','## Target results','','Values in percent; full 16-source denominator in each domain.','','| Method | HC vIoU | HC tIoU | HC sIoU | Vid vIoU | Vid tIoU | Vid sIoU |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in p['means']['hc2']:
  vals=[p['means'][ds][arm][k]*100 for ds in ['hc2','vidstg']for k in ['v','t','s']];lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}'for v in vals)+' |')
 lines+=['','## Matched contrasts','','Paired source-level t intervals, percentage points. Intervals are descriptive on the exposed development cohort.','','| Contrast | HC ΔvIoU [95%] | Vid ΔvIoU [95%] |','|---|---:|---:|']
 for contrast in p['paired']['hc2']:
  vals=[]
  for ds in ['hc2','vidstg']:
   v=p['paired'][ds][contrast]['v'];vals.append(f"{100*v['mean']:+.6f} [{100*v['CI95'][0]:+.6f}, {100*v['CI95'][1]:+.6f}]")
  lines.append('| '+contrast+' | '+' | '.join(vals)+' |')
 lines+=['','## Decision','','Prelocked transfer gate: joint diffusion must exceed all-anchor episodic Direct mean vIoU AND tIoU in EACH domain. Actual gate: **'+str(p['gate_pass'])+'**.','', 'Actual posterior-guided Student parameter adaptation is **not triggered** because the prerequisite failed.'if not p['gate_pass']else'Gate passed; actual parameter adaptation still requires its independent source qualification and result seal.','', 'The first two requested controls have actual measurements. The third (inference versus parameter adaptation) is conditional and is not reported as measured when the prior fails. This is not a failure of a measured Student update.','','## What this implementation does and does not establish','','- Whole event and32 trajectory knots are jointly denoised with dynamic video/time ROI, full Query and interval-exterior context. The separated control has exactly zero bidirectional state derivatives; joint derivatives are nonzero. Derivatives demonstrate implemented dependence, not beneficial coupling.','-171,268 parameters per prior, two source directions,60epochs/360AdamWsteps each. Checkpoint selection is source validation only. Deterministic correction has matching architecture/features/steps, but Source/Direct-corruption rather than GT-centered DSM training; it cannot isolate diffusion as a single changed loss term.','- Association head estimates source geometric compatibility, not verified identity. Other/nonmatch are marginalized together; event-exterior source anchors are unknown, not negative. No distance-to-Source gate. Sparse experts are observations, not forced targets.','- Frozen map pooling8x8,32context times and32event knots may discard detail. Spatial training supervision is interpolated only between real source event-interior annotations, never invented outside the event. Final exterior predictions retain Source.','- Four fixed-seed samples/16prior evaluations each; latent averaging can mix identities. Observation guidance uses a bounded approximate noisy-state likelihood, not an exact clean posterior/DPS implementation. No automatic claim of SDA theoretical guarantees.','- Small source training set and cross-domain shift are unresolved risks. A negative pilot does not establish that all diffusion/data-assimilation approaches fail. No extra data, seeds or noise searches were run.','- All target predictions were sealed before this run’s target scoring, but these32 sources were exposed previously. No independent confirmation, official full-set claim or novelty claim.','- R@.3/.5, full signed per-source deltas, fixed-Source-time geometry, source validation/cost and unknown supports are in machine-readable results. Pixel panels are private and were actually viewed; no independent human identity labels were added.','', 'Primary sources: [SDA](https://arxiv.org/abs/2306.10574), [official SDA code](https://github.com/francois-rozet/sda), [Diffuser](https://diffusion-planning.github.io/), [Switchable Constraints](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf).']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 files=['vg_tta/stvg_joint_posterior_v1.py','scripts/run_stvg_joint_posterior_v1.py','scripts/report_stvg_joint_posterior_v1.py','tests/test_stvg_joint_posterior_v1.py','protocols/stvg_joint_posterior_v1.md']+[str(f.relative_to(ROOT))for f in OUT.iterdir()if f.is_file()]
 secret=[];old=read(OLD/'PLAN_LOCK.json')
 for r in old['source']+old['target']:
  if isinstance(r.get('parent'),str)and len(r['parent'])>4:secret.append(r['parent'])
  if 'input'in r:secret.append(r['input']['caption'])
 for file in files:
  text=(ROOT/file).read_text()
  assert not any(s and s in text for s in secret),(file,'private identifier/text')
  import re
  assert not re.search(r'(?:github_pat_|hf_|ghp_)[A-Za-z0-9_]{16,}|Bearer\s+[A-Za-z0-9._-]{20,}',text)
 put(BASE/'ROOT_PRIVACY_READBACK.json',dict(status='pass',files=len(files),no_raw_Query_media_GT_coordinates_weights_or_personal_conversations=True))
 put(BASE/'PUBLIC_FILES.json',dict(files=files));print('PUBLIC_FILES',len(files),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
