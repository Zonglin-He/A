"""Audit source-only checkpoint selection and conditional target five-arm outputs."""
from pathlib import Path
import sys,os,argparse,collections,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dfine_source_capture_v1 import D,P,TARGET

def properties():
 from vg_tta.stvg_dfine_refinement32_v1 import Refiner,prepare_query,pack,roi,distribution_loss,sample_teacher,crn
 torch.set_num_threads(4);torch.manual_seed(20261018);model=Refiner(.5);E=torch.tensor([[.4,.5,.3,.4],[.5,.5,.3,.4]],dtype=torch.float64);maps=torch.randn(2,256,8,9);valid=torch.ones(2,8,9,dtype=torch.bool);z=prepare_query(dict(E=E,maps=maps,valid=valid,frames=[dict(position=0,boxes=np.array([[.4,.5,.3,.4]]))],selected=[0]));batch=pack([z,z],[0,1],'cpu');current=batch['E'].clone();current[:,0]+=.05;st=model(batch,current);assert torch.equal(st[1]['roi_boxes'],st[0]['boxes'])and not torch.equal(st[0]['roi_boxes'],st[1]['roi_boxes']);loss,extra=distribution_loss(st,batch['E'],batch['E'],model.bins);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters());assert len(model.bins)==33 and model.bins[16]==0 and (model.bins[1:]>model.bins[:-1]).all();s=sample_teacher(st[-1]['prob'].detach(),batch['E'],model.bins,crn('properties',2));assert s['boxes'].shape==(2,32,4);put(D/'CPU_PROPERTIES.json',dict(status='pass',bins33_nonuniform_signed=True,stage2_actual_ROI_reread=True,all_head_gradients_finite=True,teacher_draw_shape=[2,32,4],source_GT_only_training=True));print('CPU_PROPERTIES_PASS',flush=True)

def score():
 torch.set_num_threads(4)
 from scripts.run_stvg_dfine_refinement32_v1 import verify
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import iou,xy,summarize_frame
 from vg_tta.stvg_dfine_refinement32_v1 import expected_reward
 from vg_tta.stvg_discriminative_dense32_v1 import candidates
 p=verify();gate=read(P/'SOURCE_GATE.json')
 if not gate['passed']:
  assert not(D/'GLOBAL_PREDICTION_BARRIER.json').exists();put(P/'TARGET_STATUS.json',dict(status='not_run',reason='source qualification did not pass in both domains',OPD_effectiveness_tested=False,target_GT_newly_accessed=False));return
 bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_new_target_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['target']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['target']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));vals=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);x=load(TARGET/'features'/(a+'.pt'));cached=load(TARGET/'feedback'/(a+'.pt'));b=cached['feedback']['candidates'];teacher=o['paths']['Direct_refine2'][0]['teacher'];raw=expected_reward(b,teacher).double();prior=-.5*torch.cat(tuple(cached['feedback']['noise']),1).square().sum(-1);valid=cached['feedback']['real']['valid']if 'real'in cached['feedback']else torch.ones(b.shape[:2],dtype=torch.bool)
  # Existing visibility is independent of GT and recomputed on identical boxes.
  from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
  valid=(cell_overlap(b,*x['valid'].shape[-2:])*x['valid'][:,None]).sum((-1,-2))>1e-12;safe=valid.clone();safe[~safe.any(-1)]=True;prior[~safe]=-torch.inf;pt=(prior+raw/.1).softmax(-1);quality=[];ids=np.array(o['frame_ids']);unobs=~np.isin(np.arange(len(ids)),o['planned_positions']);xx=xy(b.numpy(),plans[ds][q],ds)
  for t in np.flatnonzero(unobs):
   if int(ids[t])not in gt[ds][0][q]:continue
   iq=iou(xx[t],np.array(gt[ds][0][q][int(ids[t])]));v=valid[t].numpy().copy();v[0]=True;quality.append(summarize_frame(iq,raw[t].numpy(),pt[t].numpy(),v,iq[0],o['active'],iq[0],iq[0]))
  for ep in ['v','t','s','unobserved_actual_s']:assert abs(vals['values']['E_MBR'][ep]-old[a]['values']['E_MBR'][ep])<1e-12
  rows.append(dict(alias=a,dataset=ds,active=o['active'],values=vals['values'],geometry=vals['geometry'],interpolation_effects=vals['interpolation_effects'],ranking={k:None if not[v[k]for v in quality if v[k]is not None]else float(np.mean([v[k]for v in quality if v[k]is not None]))for k in ['raw_auc','posterior_auc','strong_pool_raw_top_good','strong_pool_posterior_top_good','strong_pool_no_good_outranks_E_raw','strong_pool_no_good_outranks_E_posterior']},teacher_invalid=[v['teacher']['invalid_fraction']for v in o['paths']['Requery_OPD']],actual_shared_teacher_calls=o['actual_shared_teacher_calls']))
 arms=list(rows[0]['values']);comparisons=[(a,'E_MBR')for a in arms[1:]]+[('Requery_OPD',a)for a in ['Direct_refine2','Fixed_distill','Resample_fixed_teacher']];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,metrics={a:{ep:float(np.mean([r['values'][a][ep]for r in rr]))for ep in ['v','t','unobserved_actual_s']}for a in arms},paired={a+'_minus_'+b:{ep:paired([100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for a,b in comparisons},ranking={k:paired([r['ranking'][k]for r in rr])for k in rr[0]['ranking']})
 put(P/'TARGET_RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,all32_before_new_target_GT=True,independent_confirmation=False));put(P/'TARGET_STATUS.json',dict(status='complete',count=32,five_arm_actual_test=True,new_target_DINO_calls=0))

def audit():
 torch.set_num_threads(4)
 from scripts.run_stvg_dfine_refinement32_v1 import verify
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 p=verify();source=read(P/'SOURCE_TRAINING.json');checks=collections.Counter();sr=[]
 for s in source['rows']:
  ds=s['source_domain'];history=[r for r in s['history']if 'validation_v'in r];best=max(history,key=lambda r:r['validation_v']);assert best['epoch']==s['selected']['epoch'];checks['source_only_checkpoint_selection']+=1
  cp=load(D/'models'/ds/f"epoch{best['epoch']:02}.pt");assert sha(D/'models'/ds/f"epoch{best['epoch']:02}.pt")==s['teacher_checkpoint_sha256'];assert len(cp['state']['bins'])==33;val=load(D/'source_validation'/ds/f"epoch{best['epoch']:02}.pt");lookup={r['alias']:r for r in val['rows']}
  records=[r for r in p['source']if r['dataset']==ds and r['split']=='source_validation']
  dv=np.mean([r['values']['Direct_refine2']['v']-r['values']['E_MBR']['v']for r in val['rows']]);sv=[r['values']['Direct_refine2']['unobserved_actual_s']-r['values']['E_MBR']['unobserved_actual_s']for r in val['rows']if r['values']['Direct_refine2']['unobserved_actual_s']is not None and r['values']['E_MBR']['unobserved_actual_s']is not None];dsval=np.mean(sv);assert abs(100*dv-s['selected']['delta_v_pp'])<1e-12 and abs(100*dsval-s['selected']['delta_unobserved_s_pp'])<1e-12;assert s['source_qualification_pass']==bool(dv>0 and dsval>0);checks['independent_source_gate']+=1
  from vg_tta.stvg_dfine_refinement32_v1 import Refiner,prepare_query,pack
  first=records[0];raw=load(D/'source_features'/(first['alias']+'.pt'));query=prepare_query(raw);m=Refiner(cp['radius']).eval();m.load_state_dict(cp['state']);n=len(query['E']);batch=pack([query]*n,list(range(n)),'cpu')
  with torch.no_grad():replay=m(batch,batch['E'])
  actual=val['packets'][0]['paths']['Direct_refine2'][0]['teacher'];assert (replay[-1]['prob']-actual['prob']).abs().max()<1e-4;checks['selected_checkpoint_CPU_vs_GPU_forward']+=1
  del m,batch,replay
  for packet in val['packets']:
   for t in packet['paths']['Direct_refine2']:
    teacher=t['teacher'];assert torch.equal(teacher['stage_roi_boxes'][1],teacher['stage_boxes'][0]);checks['actual_stage2_ROI_reread']+=1
  for r,o in zip(records,val['packets']):
   assert r['alias']==o['alias'];truth=load(D/'source_supervision'/(r['alias']+'.pt'));independent_metrics(o,dict(input=r['input']),truth['gt'],truth['span'],lookup[r['alias']],checks);checks['source_final_two_stage_readout']+=1
  sr+=val['rows']
  assert len({r['parent']for r in p['source']if r['dataset']==ds})==59;assert not({r['parent']for r in p['source']if r['dataset']==ds}&{r['parent']for r in p['target']if r['dataset']==ds});checks['source_target_parent_disjoint']+=1
 gate=read(P/'SOURCE_GATE.json');assert gate['passed']==all(s['source_qualification_pass']for s in source['rows'])
 if gate['passed']:
  from scripts.score_stvg_opd_p1_v1 import truths
  tr=read(P/'TARGET_RESULTS_PUBLIC.json');lookup={r['alias']:r for r in tr['rows']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['target']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
  for r in p['target']:
   o=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];independent_metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q],lookup[r['alias']],checks)
   for a in ['Fixed_distill','Resample_fixed_teacher','Requery_OPD']:
    for t in o['paths'][a]:assert torch.equal(t['eta_after'],t['eta_before']-.5*t['gradient']);checks['actual_output_policy_step']+=1
 else:assert not(D/'predictions').exists();checks['source_gate_failed_no_target_OPD']=1
 put(P/'SOURCE_VALIDATION_SELECTED_PUBLIC.json',dict(count=32,rows=sr,source_validation_only=True));verify();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(checks),source_models=2,target_actual_test=gate['passed'],source_parent_disjoint=True,original_registries_unchanged=True));print(dict(checks),flush=True)

def report():
 p=read(D/'PLAN_LOCK.json');source=read(P/'SOURCE_TRAINING.json');gate=read(P/'SOURCE_GATE.json');sv=read(P/'SOURCE_VALIDATION_SELECTED_PUBLIC.json');bar=read(D/'SOURCE_FEATURE_BARRIER.json');spent=read(D/'TRAIN_GPU_STOP.json')['cumulative_GPU_seconds'];target=read(P/'TARGET_STATUS.json');has=target['status']=='complete'
 if has:spent=read(D/'TARGET_GPU_STOP.json')['cumulative_GPU_seconds']
 put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['source','target','source_provenance','created_epoch','deadline_epoch']});put(P/'MODEL_PROVENANCE.json',dict(backbone='TA-STVG unchanged pretrained checkpoints, own-domain source/opposite-domain target',feature='input_proj pre-text-fusion, same as cached target32',head='new D-FINE-inspired two-stage conditional edge-distribution module, no GO-LSD or official pretrained D-FINE weights',new_head_models=2,selected=[dict(source_domain=s['source_domain'],checkpoint_sha256=s['teacher_checkpoint_sha256'],parameters=s['parameters'])for s in source['rows']],target_DINO_cache_unchanged=True))
 costs=dict(total_GPU_allocation_seconds=spent,GPU_cap_seconds=3600,source_capture_GPU_seconds=bar['GPU_allocation_seconds'],source_captures_completed=118,total_Source_captures_completed=118,source_DINO_calls=sum(r['new_DINO_calls']for r in bar['costs']),new_target_DINO_calls=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_head_models=2,target_test_ran=has,source_capture_resume_original19_preserved=True,source_individual_costs_missing_for_first19=True,old_target_Source_E_cost_excluded=True,source_validation_external_state_fit_episodes=576,source_validation_external_state_rounds=1152,source_validation_actual_shared_teacher_queries=576);put(P/'COST_BREAKDOWN.json',costs)
 lines=['# D-FINE-inspired E/anchor-conditioned refinement prototype','',
 'Source-only one-seed/one-configuration head training, no change to pretrained STVG/DINO weights. Same target NH16/NV16 exposed development panel and E/MBR if qualification passes. Not complete D-FINE replication; no localization self-distillation.','',
 'Ready local source pool: after excluding current target parents HC59/Vid64. Each domain hash-select59 =43 train+16 validation, one query/parent. No media download, below requested maxima128/32. Historical source/validation/legacy E calibration exposure disclosed; not independent confirmation. Source GT binds actual clip+query+target and never substitutes perfect anchors. Actual DINO inputs include errors/missing references. Source captures118, old matched DINO reused or new source-only calls; no target expert call.','',
 'Input frozen pre-text-fusion256D map; ROI7x7 context1.5 preserves relative spatial structure; source target-anchor ROIs same feature space. Project32/hidden128,2 separate residual132-logit heads over4x33 signed nonuniform E-relative edges. Stage2 explicitly re-reads region decoded from stage1. Radius is source-train95pct in [.25,2]. Training30epochs x6steps x64,parent-balanced, E/real Native/E-candidates; no GT-noise inputs. Interpolated-bin CE plus5L1+2GIoU, both stages averaged. Validate every5epochs, select best source-only complete-v of center-only direct2.','',
 '| Head trained on | Train/val parents | Parameters | Radius | Selected epoch | Source direct2 Δv pp | Source direct2 Δunobserved actual s pp | Source gate |','|---|---:|---:|---:|---:|---:|---:|---|']
 for s in source['rows']:
  b=s['selected'];lines.append(f"| {s['source_domain']} | 43/16 | {s['parameters']} | {s['radius']:.6f} | {b['epoch']} | {b['delta_v_pp']:+.6f} | {b['delta_unobserved_s_pp']:+.6f} | {s['source_qualification_pass']} |")
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 for ds in ['hc2','vidstg']:
  rr=[r for r in sv['rows']if r['dataset']==ds];lines+=['',f'## {ds} selected source validation','']
  for arm in ['Direct_refine2','Teacher_boundary_mean2']:
   for ep in ['v','unobserved_actual_s']:
    q=paired([None if r['values'][arm][ep]is None or r['values']['E_MBR'][ep]is None else 100*(r['values'][arm][ep]-r['values']['E_MBR'][ep])for r in rr]);lines.append(f"{arm}−E {ep}: {q}")
  for category in ['E_good_ge075','E_poor_lt05']:
   q=paired([r['accuracy'][category]['delta_IoU_pp']for r in rr]);lines.append(f"{category}: known nodes={sum(r['accuracy'][category]['nodes']for r in rr)}, query-balanced change={q}")
  lines.append('Independent edge draw invalid fraction mean: '+str(np.mean([v for r in rr for v in r['sample_invalid_fraction']])))
 if has:
  tr=read(P/'TARGET_RESULTS_PUBLIC.json');lines+=['','## Target actual five-arm experiment','','| Output | HC vIoU % | Vid vIoU % | HC actual unobserved sIoU % | Vid actual unobserved sIoU % |','|---|---:|---:|---:|---:|']
  for a in tr['groups']['hc2']['metrics']:lines.append('| '+a+' | '+' | '.join(f"{100*tr['groups'][ds]['metrics'][a][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
  for ds,g in tr['groups'].items():lines+=['',ds+' paired differences: '+str(g['paired']),ds+' raw/posterior E-relative ranking: '+str(g['ranking'])]
 else:lines+=['','## Target not run','',
 'Source qualification did NOT pass in both domains. As prespecified stop before target: no target5-arm model inference, OPD update, new target-GT scoring or target parameter search. This is a source teacher/readout qualification failure, NOT evidence that this new teacher’s target OPD failed. Original cosine/ridge/critic results unchanged.']
 lines+=['','Budgets and costs: '+str(costs),'',
 'Engineering interruption: source capture001 stopped after19 because root edited the locked protocol documentation; original protocol restored byte-exact, new auxiliary source readout note isolated. Original19 files preserved; source capture002 only resumed remaining99. The118.814774s original GPU segment is included, cap/deadline not reset. First19 individual capture times unavailable, explicitly unknown; cumulative held GPU time independently recorded.','',
 'Report source high-quality-E preservation and low-quality-E correction above; these GT strata never enter online selection. Teacher boundary mean is auxiliary source capability readout with variable width, not the center-only target candidate arm or selection criterion. Source gate point thresholds fixed before training; bootstrap describes uncertainty, not target-driven gate changes.','',
 'The shared source validation constructor also computed temporary fixed/resampled/requeried external-state readouts, all retained in SOURCE_VALIDATION_SELECTED_PUBLIC; these do not train the head on validation data and do not change the prelocked direct2 selection/gate. Target test status is reported separately.','',
 'No model/weights/media/GT coordinates/raw candidates/original identifiers or queries published. Code/config/history/full signed source results and any target failures preserved. No automatic expansion, extra epochs, seed/config search or method promotion.','',
 'References: [D-FINE paper](https://arxiv.org/html/2410.13842v1), [official implementation](https://github.com/Peterande/D-FINE), [official nonuniform weighting](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_utils.py). These support borrowed design principles, not STVG effectiveness.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4))
 for s in source['rows']:
  h=[v for v in s['history']if 'validation_delta_v_pp'in v];ax.plot([v['epoch']for v in h],[v['validation_delta_v_pp']for v in h],marker='o',label=s['source_domain'])
 ax.axhline(0,color='k',lw=.5);ax.set(xlabel='Training epoch',ylabel='Source validation direct2 delta vIoU (pp)');ax.legend();fig.tight_layout();fig.savefig(P/'SOURCE_VALIDATION.svg');fig.savefig(D/'SOURCE_VALIDATION.png',dpi=150);plt.close(fig)
 files=['vg_tta/stvg_dfine_refinement32_v1.py','scripts/run_stvg_dfine_source_capture_v1.py','scripts/run_stvg_dfine_refinement32_v1.py','scripts/audit_stvg_dfine_refinement32_v1.py','scripts/continue_stvg_dfine_refinement32_v1.py','protocols/stvg_dfine_refinement32_v1.md','protocols/stvg_dfine_refinement32_source_readout_v1.md']+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()];put(D/'PUBLIC_FILES_FINAL.json',dict(files=files));put(D/'REPORT_COMPLETION.json',dict(status='complete',source_models=2,target_test_ran=has))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['properties','score','audit','report']);v=a.parse_args();globals()[v.stage]()
