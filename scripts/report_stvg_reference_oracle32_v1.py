"""Independent sealed-fit/math audit and anonymous oracle-only factorial report."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_oracle32_v1 import BASE,OLD,ARMS,read,put,load,sha
PUB=ROOT/'results/stvg_reference_oracle/2026-10-17/matched32'
def audit():
 import numpy as np,torch
 from scipy.stats import t
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit as math_audit
 from methods.decota_final_simplified_v1.objectives import SpatialLoss
 from scripts.stvg_pipeline_speed_v1 import compare_tree
 p=read(BASE/'PLAN_LOCK.json');bar=read(BASE/'PREDICTION_BARRIER.json');scores=read(BASE/'POSTSEAL_RESULTS.json')
 targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
 checks=statechecks=opdchecks=directchecks=grad=rgb=0;audits=[];updates=[];cost=collections.defaultdict(list)
 for rec,saved in zip(p['records'],scores['rows']):
  z=load(BASE/'predictions'/(rec['alias']+'.pt'));inp=load(BASE/'inputs'/(rec['alias']+'.pt'));ds=rec['dataset'];q=rec['query_ordinal'];row=read_row(ds,q);gt=targets[ds][0][q];span=targets[ds][1][q]
  assert z['alias']==saved['source_alias']==rec['alias'];rgb+=z['original_DINO_RGB_checks']
  for name,arm in z['arms'].items():
   fit=arm['fit'];assert arm['GT_assisted_diagnostic']and fit['GT_used'];assert fit['positions']==inp['positions']
   assert sum(v.numel()for v in fit['initial'].values())==1792
   for k,v in z['initial'].items():assert torch.equal(fit['initial'][k],v);statechecks+=v.numel()
   assert torch.equal(fit['before'],z['Source_boxes'])
   assert torch.equal(arm['complete_output']['boxes'],fit['final'])
   if not inp['positions']:
    assert not fit['fit_invoked']and fit['gradient_calls']==0 and torch.equal(fit['final'],z['Source_boxes'])
    assert all(torch.equal(fit['state'][k],v)for k,v in z['initial'].items())
   elif arm['updater']=='OPD':
    # The legacy checker assumes its generic fitter flag. Actual data provenance
    # remains GT-assisted in the immutable payload and all published receipts.
    original_view={**fit,'GT_used':fit['fitter_generic_GT_flag']};ma=math_audit(original_view,inp['experts'][arm['reference']]);opdchecks+=1
    audits.append(dict(source_alias=z['alias'],arm=name,math=ma,GT_assisted_diagnostic=True,legacy_math_view_flag_is_not_data_provenance=True))
   else:
    assert len(fit['trace'])==p['configurations'][ds]['steps']+1
    assert fit['gradient_calls']==p['configurations'][ds]['steps']
    fn=SpatialLoss(inp['experts'][arm['reference']]['anchors']['single4'],z['Source_boxes'])
    for idx,boxes in [(0,z['Source_boxes']),(-1,fit['final'])]:
     err=abs(float(fn(boxes))-fit['trace'][idx]['loss']);assert err<2e-5,err;directchecks+=1
   grad+=fit['gradient_calls'];cost[ds+'_'+name].append(z['compute']['arm_seconds'][name])
   rounds=fit.get('rounds',[])
   updates.append(dict(source_alias=z['alias'],dataset=ds,arm=name,updater=arm['updater'],reference=arm['reference'],common_anchors=len(inp['positions']),empty=not inp['positions'],maximum_rounds=p['configurations'][ds]['steps'],gradient_calls=fit['gradient_calls'],no_information_rounds=sum(r.get('no_information',False)for r in rounds),nonempty_zero_update=bool(inp['positions'])and fit['gradient_calls']==0,GT_assisted_diagnostic=True))
  if inp['positions']:
   b=z['arms']['B']['fit'];d=z['arms']['D']['fit'];assert [r['rollout']['seed']for r in b['rounds']]==[r['rollout']['seed']for r in d['rounds']]
   assert torch.equal(b['rounds'][0]['rollout']['samples'],d['rounds'][0]['rollout']['samples'])
  for arm in ['Source',*ARMS]:
   boxes=z['Source_boxes']if arm=='Source'else z['arms'][arm]['fit']['final']
   tube=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2')
   for sup,iv in [('Native',z['Native_interval']),('projection',z['projection_interval'])]:
    m=tube.score(iv)
    for k in ['v','t','s']:assert abs(m[k]-saved['values'][arm][sup][k])<2e-10;checks+=1
    for th in [.3,.5]:assert float(m['v']>th)==saved['values'][arm][sup]['R_at_'+str(th).replace('.','p')];checks+=1
   assert saved['values'][arm]['Native']['t']==saved['values']['Source']['Native']['t']
   assert saved['values'][arm]['projection']['t']==saved['values']['Source']['projection']['t']
  for e in inp['experts']['GT_oracle']['anchors']['single4']:
   fid=z['frame_ids'][e['position']];rect=xyxy(np.array(e['box']),row['input']['width'],row['input']['height']);assert np.max(abs(rect-gt[fid]))<1e-7
  for e in inp['experts']['DINO']['anchors']['single4']:assert z['frame_ids'][e['position']]in gt
 assert grad==bar['gradient_calls']and rgb==124
 for ds in ['hc2','vidstg']:
  for subset in ['all32','nonempty25']:
   rr=[r for r in scores['rows']if r['dataset']==ds and(subset=='all32'or not r['empty'])]
   for a in ['Source',*ARMS]:
    for s in ['Native','projection']:
     for k in ['v','t','s','R_at_0p3','R_at_0p5']:
      assert abs(np.mean([r['values'][a][s][k]for r in rr])-scores['groups'][ds][subset]['means'][a][s][k])<1e-14;checks+=1
 contrasts={'C-A':('C','A'),'D-B':('D','B'),'B-A':('B','A'),'D-C':('D','C'),'A-Source':('A','Source'),'B-Source':('B','Source'),'C-Source':('C','Source'),'D-Source':('D','Source')}
 pairs=[]
 def describe(delta,aliases):
  x=np.asarray(delta);n=len(x);se=float(x.std(ddof=1)/np.sqrt(n))if n>1 else None;mean=float(x.mean());half=float(t.ppf(.975,n-1)*se)if n>1 else None
  return dict(count=n,mean_pp=mean,CI95_t=None if half is None else[mean-half,mean+half],positive=int((x>1e-10).sum()),negative=int((x<-1e-10).sum()),neutral=int((abs(x)<=1e-10).sum()),per_source_delta_pp=dict(zip(aliases,map(float,x))))
 for ds in ['hc2','vidstg']:
  for subset in ['all32','nonempty25']:
   rr=[r for r in scores['rows']if r['dataset']==ds and(subset=='all32'or not r['empty'])];aliases=[r['source_alias']for r in rr]
   for sup in ['Native','projection']:
    for met in ['v','s']:
     for name,(a,b)in contrasts.items():pairs.append(dict(dataset=ds,subset=subset,time=sup,metric=met,contrast=name,**describe([100*(r['values'][a][sup][met]-r['values'][b][sup][met])for r in rr],aliases)))
     pairs.append(dict(dataset=ds,subset=subset,time=sup,metric=met,contrast='(D-B)-(C-A)',**describe([100*(r['values']['D'][sup][met]-r['values']['B'][sup][met]-r['values']['C'][sup][met]+r['values']['A'][sup][met])for r in rr],aliases)))
 put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,scalar_checks=checks,initial_parameter_coordinate_checks=statechecks,OPD_Gaussian_Adam_path_checks=opdchecks,Direct_endpoint_loss_checks=directchecks,original_DINO_RGB_checks=rgb,nonempty_fit_calls=bar['actual_nonempty_fit_calls'],gradient_calls=grad,GT_assisted_diagnostic=True,math_audits=audits))
 put(PUB/'RESULTS.json',scores);put(PUB/'PAIRED_CONTRASTS.json',dict(status='complete',count=32,paired=pairs,CI='descriptive paired-parent t95 on exposed development cohort; no selection gate'))
 put(PUB/'SUPPORT.json',read(BASE/'SUPPORT_READBACK.json'))
 put(PUB/'FROZEN_PROTOCOL.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 put(PUB/'VERIFICATION.json',{k:v for k,v in read(BASE/'ROOT_READBACK.json').items()if k!='math_audits'})
 put(PUB/'COST.json',dict(new_Source_captures=32,new_DINO_calls=0,new_Temporal_updates=0,logical_arm_outputs=128,no_op_outputs=28,nonempty_fit_calls=bar['actual_nonempty_fit_calls'],gradient_calls=grad,measured_arm_seconds={k:dict(sum=sum(v),mean=sum(v)/len(v))for k,v in cost.items()},timing_scope='fit plus full-decoder output and validation; excludes Source capture, video decode, root audit and publication; not deployment latency'))
 put(PUB/'UPDATE_DIAGNOSTICS.json',dict(status='complete',count=32,rows=updates,summary={ds:{a:dict(gradient_calls=sum(r['gradient_calls']for r in updates if r['dataset']==ds and r['arm']==a),no_information_rounds=sum(r['no_information_rounds']for r in updates if r['dataset']==ds and r['arm']==a),nonempty_zero_update_sources=[r['source_alias']for r in updates if r['dataset']==ds and r['arm']==a and r['nonempty_zero_update']])for a in ARMS}for ds in ['hc2','vidstg']}))
 print(json.dumps(dict(status='pass',scalar_checks=checks,grad=grad,means={ds:{a:100*v['Native']['v']for a,v in z['all32']['means'].items()}for ds,z in scores['groups'].items()},contrasts=[x for x in pairs if x['subset']=='all32'and x['time']=='Native'and x['metric']=='v'and x['contrast']in ['C-A','D-B','B-A','D-C','(D-B)-(C-A)']])))
def render():
 import numpy as np
 from PIL import Image,ImageDraw,ImageFont
 from scripts.run_decota_paper_main_v1 import frames_for,read_row
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 records=read(BASE/'PLAN_LOCK.json')['records'];panels=[]
 for alias in ['H001','H002','V014']:
  r=next(r for r in records if r['alias']==alias);z=load(BASE/'predictions'/(alias+'.pt'));row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,collections.OrderedDict());truth=truths(r['dataset'],dict(parents=[r['query_ordinal']]))[0][r['query_ordinal']]
  observed=z['positions'];remaining=[i for i,f in enumerate(ids)if f in truth and i not in observed];pos=(observed[:2]+remaining[:2]);assert pos
  canvas=Image.new('RGB',(1600,1640),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
  draw.text((8,8),alias+' PRIVATE oracle diagnostic; cyan GT, red Source, green final; fixed positions across all four rows',fill='black',font=font)
  for i,arm in enumerate(ARMS):
   for j,k in enumerate(pos):
    im=Image.fromarray(frames[k]);w,h=im.size;im.thumbnail((390,340));nw,nh=im.size;x=j*400+(400-nw)//2;y=65+i*390;canvas.paste(im,(x,y));d=ImageDraw.Draw(canvas)
    def box(b,col):
     a=np.asarray(b)*np.array([nw/w,nh/h,nw/w,nh/h]);d.rectangle((x+a[0],y+a[1],x+a[2],y+a[3]),outline=col,width=3)
    box(truth[ids[k]],'cyan');box(xyxy(z['Source_boxes'][k].numpy(),w,h),'red');box(xyxy(z['arms'][arm]['fit']['final'][k].numpy(),w,h),'lime')
    d.text((j*400+5,y+342),f'{arm} {ARMS[arm]} '+('observed'if k in observed else'unobserved native'),fill='black',font=font)
  dest=BASE/'private_pixels'/(alias+'.jpg');dest.parent.mkdir(exist_ok=True);canvas.save(dest,quality=94);panels.append(dict(source_alias=alias,path=str(dest.relative_to(ROOT)),sha256=sha(dest),cells=4*len(pos)))
 put(BASE/'PIXEL_MANIFEST.json',dict(status='rendered_pending_actual_view',panels=panels))
 print(json.dumps(panels))
def report():
 assert read(BASE/'ROOT_PIXEL_REVIEW.json')['status']=='pass'
 s=read(PUB/'RESULTS.json');pp=read(PUB/'PAIRED_CONTRASTS.json')['paired'];root=read(BASE/'ROOT_READBACK.json')
 text=['# Matched reference intervention: DINO/GT oracle × Direct/original OPD','',
 '**All four arms are GT-assisted diagnostics, not deployable TTA results.** DINO arms also use exact-GT availability to restrict support. Target-GT coordinates are used only in C/D. No arm,step,reference or method is selected by these evaluation metrics.','',
 'Same exposed HC16/Vid16 development cohort,67 common admitted anchors;25 nonempty sources and7 exact no-ops. No added/interpolated GT positions. All four arms reset to identical frozen Source1792 state and fresh Adam; no LN carry. Native time is fixed across arms; saved frozen Source projection is a secondary CPU readout. DINO references are reused,0 new Teacher calls.','',
 '| Target | Subset | n | Source | A DINO Direct | B DINO OPD | C GT Direct | D GT OPD |','|---|---|---:|---:|---:|---:|---:|---:|']
 for ds,z in s['groups'].items():
  for subset,g in z.items():text.append(f"| {ds} | {subset} | {g['count']} | "+' | '.join(f"{100*g['means'][a]['Native']['v']:.6f}"for a in ['Source',*ARMS])+' |')
 text+=['','Values are vIoU percentages, primary fixed Native interval. Full tIoU/sIoU/R@.3/R@.5 and observed/unobserved-native/interpolated geometry are retained in RESULTS.json. All32 and the same25 support-bearing sources are separate; omitted supervision is not silently filled.','',
 '| Target | Subset | Contrast | Mean ΔvIoU pp | Paired t95 | Positive / negative / zero |','|---|---|---|---:|---|---|']
 for x in pp:
  if x['time']=='Native'and x['metric']=='v'and x['contrast']in ['C-A','D-B','B-A','D-C','(D-B)-(C-A)']:
   ci=x['CI95_t'];text.append(f"| {x['dataset']} | {x['subset']} | {x['contrast']} | {x['mean_pp']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {x['positive']}/{x['negative']}/{x['neutral']} |")
 text+=['','C-A and D-B isolate the reference-coordinate intervention within each updater. B-A and D-C compare the fixed updater configurations on matched starting state/support/time/max rounds. The interaction is (D-B)-(C-A). Intervals are descriptive on already exposed development sources; they do not establish independent confirmation.','',
 '## Mechanistic findings','',
 'Better benchmark reference coordinates improve both updater means: C-A is+11.011453/+6.012458pp HC/Vid, and D-B is+5.917787/+4.386804pp. Direct C-A has14/1/1 positive/negative/zero HC sources and9/0/7 Vid. OPD D-B has11/3/2 and7/2/7; correct references are not sufficient to ensure a beneficial parameter update.','',
 'On DINO support, OPD-Direct is+1.496956pp HC and−0.088995pp Vid; both descriptive t95 intervals cross zero. With oracle references, OPD-Direct is−3.596710pp HC (t95[−9.412440,+2.219020]) and−1.714649pp Vid ([−3.204928,−0.224369]). This fixed-budget intervention does not establish a general superiority or robustness claim for OPD. Direct has higher oracle-reference point estimates in both domains; no production method is changed based on these exposed oracle results.','',
 'H002 now supplies a matched correct-reference failure: Source vIoU31.585462%,GT-Direct46.978912%,GT-OPD8.870489%. GT-OPD observed mean IoU improves57.151969→64.145826%, while unobserved native mean drops47.242920→27.407058%; on the same Native readout support it drops53.261508→25.296456%. One correctly supervised anchor switches to the interaction counterpart by step8,its GT IoU and all32 local sampled rewards become zero; the remaining anchors continue updating. Final three anchor IoUs are0/.948212/.976163. Thus this is not a claim that every observed anchor improves: average observed improvement conceals one destroyed anchor. All GT references were fixed throughout; no best-step or oracle acceptance is introduced.','',
 'A second limitation is local reward coverage. DINO-OPD H005 has40 no-information rounds and preserves Source against a disjoint bad reference; H016 also has40 such rounds under both DINO and GT references and misses a useful correction. GT-Direct H016 gains22.110535pp while GT-OPD makes no update. All sampled IoU rewards in these no-update fits are zero. The same mechanism can suppress wrong supervision and fail to use correct supervision; it is not evidence that OPD recognizes semantic reliability. Vid GT-OPD V012 also has10 no-information rounds. Full UPDATE_DIAGNOSTICS and saved-path metrics retain these cases.','',
 'The DINO row differs from the earlier all-anchor comparison because all39 event-exterior references are excluded from every arm here by GT availability. Its new positive mean cannot justify a deployable event filter; neither the selection mask nor this oracle row is available to target-unlabeled TTA. Earlier Direct versus inherited-OPD scores remain distinct and unchanged.','',
 'Decision: reference quality is a substantial bottleneck in this cohort, and the correctly supervised H002 failure plus flat-reward cases show a separate update/coverage bottleneck. No new critic, identity architecture, optimizer modification or parameter sweep is automatically launched. A deployable reference-reliability mechanism and matched evidence of update value would still be required before method promotion.','',
 'HC lr.01/40 rounds;Vid lr.03/10,Adam(.9,.999),eps1e-8,1792 parameters. Direct preserves original5L1+2GIoU sum/4; OPD preserves original Gaussian32 antithetic,detached IoU/control-variate and per-supported-frame mean,HC sigma.025/tau.05,Vid sigma.1/tau.25. Same seeds in B/D; first samples match exactly. Maximum update rounds are matched, while original flat-reward OPD skips remain and actual gradients are counted. No rollback,best-step selection,response filter,annealing or hyperparameter search. These are fixed-budget configurations,not claims about optimally tuned algorithms.','',
 'All128 logical outputs sealed before comparative evaluation. Source zero output exactly matches the retained Source; nonempty fits match complete two-decoder readouts. All arm starts and Source model hashes are checked. GT assistance is declared in every payload. The legacy OPD arithmetic checker receives its generic original fitter flag in a temporary math-only view; this does not change the true GT-assisted provenance or make an oracle arm target-unlabeled.','',
 f"Independent checks: {root['scalar_checks']} scalars/means, {root['initial_parameter_coordinate_checks']} matched initial coordinates, {root['OPD_Gaussian_Adam_path_checks']} complete OPD Gaussian/Adam path audits, {root['Direct_endpoint_loss_checks']} Direct endpoint losses;124 original DINO RGB SHA matches. Fixed H001/H002/V014 actual output panels are privately viewed; no new independent human semantic labels. Source/Teacher/fit/reference/GT coordinates and media are not public.",'',
 'No Source/Teacher architecture change, critic training, identity model, temporal adaptation, state writeback or old-queue resume. CURRENT/MAIN/PAPER stay unchanged. Benchmark GT remains subject to the annotation uncertainty already disclosed in diagnosis32. This oracle intervention is not a deployment recipe, and its results must not be used as target-GT-based tuning or acceptance gates.']
 (PUB/'REPORT.md').write_text('\n'.join(text)+'\n')
 put(BASE/'REPORT_COMPLETION.json',dict(status='complete',count=32,GT_assisted_diagnostic=True,files={str(f.relative_to(ROOT)):sha(f)for f in PUB.iterdir()}))
def path_export():
 import torch
 d=read(BASE/'H002_SAVED_PATH_DIAGNOSIS.json');assert d['no_new_model']and d['no_GT_best_step_selection']
 # This file contains metrics and local reward diagnostics, never frame IDs or boxes.
 put(PUB/'H002_SAVED_PATH_DIAGNOSIS.json',d)
 cases=[]
 for alias,arm in [('H005','B'),('H016','B'),('H016','D'),('V012','D')]:
  z=load(BASE/'predictions'/(alias+'.pt'));f=z['arms'][arm]['fit'];assert f['gradient_calls']==0
  assert all(torch.count_nonzero(r['rollout']['rewards'])==0 for r in f['rounds'])
  assert torch.equal(f['before'],f['final'])and all(torch.equal(f['state'][k],v)for k,v in f['initial'].items())
  cases.append(dict(source_alias=alias,arm=arm,reference=z['arms'][arm]['reference'],rounds=len(f['rounds']),all_sampled_IoU_rewards_zero=True,parameters_and_boxes_unchanged=True))
 put(PUB/'FLAT_REWARD_READBACK.json',dict(status='pass',cases=cases,GT_assisted_diagnostic=True))
if __name__=='__main__':globals()[sys.argv[1]]()
