"""Post-CPU-completion aggregate audit/report; never scores GT or runs a model."""
from pathlib import Path
import os,sys,json,collections,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=ROOT/'artifacts/stvg_dense_correspondence32_v1';P=ROOT/'results/stvg_soft_correspondence/2026-10-18/online32'
ARMS=['A_sparse','B_raw','C_rank','D_interpolation'];STATES=['Source','Before','After']

def emit(path,value):put(path,value,True)

def avg(x):
 v=[float(y)for y in x if y is not None];return sum(v)/len(v)if v else None

def pack(values):
 valid=[(a,float(v))for a,v in values if v is not None];v=np.array([v for _,v in valid])*100;n=len(v)
 if not n:return dict(parents=0,mean_pp=None,CI95_pp=None,leave_one_scored_parent_out_min_pp=None,positive=0,negative=0,neutral=0,per_parent_pp=[])
 ix=np.random.default_rng(20261018).integers(n,size=(10000,n));return dict(parents=n,mean_pp=float(v.mean()),CI95_pp=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),leave_one_scored_parent_out_min_pp=min(float(np.delete(v,i).mean())for i in range(n))if n>1 else None,positive=int((v>1e-8).sum()),negative=int((v< -1e-8).sum()),neutral=int((abs(v)<=1e-8).sum()),per_parent_pp=[dict(alias=a,delta_pp=float(x))for(a,_),x in zip(valid,v)])

def cohort():
 p=read(D/'PLAN_LOCK.json');plans={ds:read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')['rows']for ds in ['hc2','vidstg']};design=read(ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2/DESIGN_LOCK.json');old=read(ROOT/'artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json')['records'];excluded={ds:set()for ds in plans}
 for r in old:excluded[r['dataset']].add(r['parent'])
 # Original P6 identities are recovered from the already-complete temporal64 readout metadata only.
 tb=read(ROOT/'artifacts/stvg_interval_decision_v1/GLOBAL_READOUT_BARRIER.json')
 for f in tb['files']:
  r=load(ROOT/f['path']);excluded[r['dataset']].add(plans[r['dataset']][r['query_ordinal']]['source'])
 for ds in plans:
  for q in design['datasets'][ds]['development_query_ordinals']:excluded[ds].add(plans[ds][q]['source'])
  selected=[];seen=set()
  for q in design['stages']['P1_'+ds]['orders']['order1']:
   parent=plans[ds][q]['source']
   if parent in excluded[ds]or parent in seen:continue
   seen.add(parent);selected.append((q,parent))
   if len(selected)==16:break
  actual=[(r['query_ordinal'],r['parent'])for r in p['records']if r['dataset']==ds];assert selected==actual and len(set(parent for _,parent in actual))==16;assert p['excluded_parent_counts'][ds]==len(excluded[ds])
  assert all(plans[ds][q]['source']==parent and parent not in excluded[ds]for q,parent in actual)
 assert p['historical_full_GT_exposure']and not p['independent_confirmation']
 result=dict(status='pass',count=32,unique_parents_per_domain=16,reconstructed_first_eligible_original_order1_selection_exact=True,disjoint_recent_rollback32_P6_64_and_original_tuning_parents=True,historically_full_stream_exposed=True,independent_confirmation=False,new_aliases_do_not_reuse_old_identity_labels=True,only_metadata_used_no_current_partial_predictions_or_GT_read=True)
 put(D/'COHORT_READBACK.json',result,True);return result

def run():
 cohort_result=read(D/'COHORT_READBACK.json');assert cohort_result['status']=='pass'
 completion=read(D/'CPU_COMPLETION.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert completion['status']=='complete'and completion['count']==bar['count']==32 and bar['fits']==128 and not bar['GT_read'];assert sha(P/'RESULTS.json')==completion['result_sha256']
 z=read(P/'RESULTS.json');decision=read(P/'DECISION.json');p=read(D/'PLAN_LOCK.json');evaluation=read(D/'EVALUATION_LOCK002.json');assert sha(ROOT/'scripts/score_stvg_dense_correspondence32_v1.py')==evaluation['scorer_sha256']
 from methods.decota_final_simplified_v1.config import MethodConfig
 source_checkpoints={}
 for ds,direction in [('hc2','vid_to_hc1'),('vidstg','hc2_to_vid')]:
  cfg=MethodConfig.for_direction(direction);source_checkpoints[ds]=dict(basename=Path(cfg.checkpoint).name,sha256=cfg.checkpoint_sha256,source_dataset=cfg.source_dataset)
 counts={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in z['rows']if r['dataset']==ds];counts[ds]=dict(no_admitted_reference_queries=sum(r['no_fit']for r in rr),per_arm={a:dict(updated_queries=sum(r['compute'][a]['gradient_calls']>0 for r in rr),zero_update_queries=sum(r['compute'][a]['gradient_calls']==0 for r in rr),with_reference_but_zero_update_queries=sum(not r['no_fit']and r['compute'][a]['gradient_calls']==0 for r in rr),actual_parameter_updates=sum(int(r['compute'][a]['gradient_calls'])for r in rr))for a in ARMS})
 assert len(z['rows'])==32;assert not z['independent_confirmation']and z['original_online_stream_not_reproduced'];checks=0
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds];assert len(rr)==len({r['alias']for r in rr})==len({r['parent_alias']for r in rr})==16
  idx=np.random.default_rng(20261018).integers(16,size=(10000,16))
  for r in rr:
   source=[r['values'][a]['Source']for a in ARMS];assert all(v==source[0]for v in source)
   assert len({r['values'][a][s]['t']for a in ARMS for s in STATES})==1
   assert r['no_fit']==(r['admitted_anchors']==0)
   for a in ARMS:
    v=r['values'][a];d=r['deltas'][a]
    for k,x,y in [('total','After','Source'),('current','After','Before'),('inherited','Before','Source')]:assert abs(d[k]-(v[x]['v']-v[y]['v']))<1e-12;checks+=1
    if r['no_fit']:assert v['After']==v['Before']
    for s in STATES:
     assert v[s]['R30']==float(v[s]['v']>.3)and v[s]['R50']==float(v[s]['v']>.5);checks+=2
  for a in ARMS:
   for state in STATES:
    for met,value in g['metrics'][a][state].items():assert abs(value-sum(r['values'][a][state][met]for r in rr)/16)<1e-12;checks+=1
   for kind,v in g['deltas'][a].items():
    delta=np.array([r['deltas'][a][kind]for r in rr]);assert abs(delta.mean()*100-v['query_mean_pp'])<1e-10 and v['query_mean_pp']==v['parent_mean_pp'];assert np.max(abs(np.quantile(delta[idx].mean(1),[.025,.975])*100-v['CI95_pp']))<1e-9;assert sum(delta<-.05)==v['severe_harm_gt5pp'];checks+=4
  for name,m in g['paired'].items():
   a,b=name.split('_minus_');diff=np.array([r['values'][a]['After']['v']-r['values'][b]['After']['v']for r in rr]);assert abs(diff.mean()*100-m['mean_pp'])<1e-10;assert np.max(abs(np.quantile(diff[idx].mean(1),[.025,.975])*100-m['CI95_pp']))<1e-9;assert abs(min(np.delete(diff,i).mean()*100 for i in range(16))-m['leave_one_parent_out_min_pp'])<1e-10;checks+=3
  ca=g['paired']['C_rank_minus_A_sparse'];cd=g['paired']['C_rank_minus_D_interpolation'];conditions=dict(C_beats_A_and_D=ca['mean_pp']>0 and cd['mean_pp']>0,leave_one_out_positive=ca['leave_one_parent_out_min_pp']>0 and cd['leave_one_parent_out_min_pp']>0,no_large_harm_count_increase=g['deltas']['C_rank']['current']['severe_harm_gt5pp']<=g['deltas']['A_sparse']['current']['severe_harm_gt5pp']+1);assert conditions==decision['groups'][ds]['conditions'];assert all(conditions.values())==decision['groups'][ds]['passed'];checks+=4
 assert decision['passed']==all(v['passed']for v in decision['groups'].values())
 emit(P/'COHORT_AUDIT.json',cohort_result)
 # Additional scalar paired decomposition uses only final saved scores, never raw GT.
 paired={}
 for ds in z['groups']:
  rr=[r for r in z['rows']if r['dataset']==ds];paired[ds]={}
  for b in ['A_sparse','D_interpolation']:
   a='C_rank';out={}
   for k in ['total','current','inherited']:out[k]=pack([(r['alias'],r['deltas'][a][k]-r['deltas'][b][k])for r in rr])
   for support in ['observed','unobserved','far']:
    pairs=[]
    for r in rr:
     aa=r['decoder'][a][support];bb=r['decoder'][b][support];v=None if any(v is None for v in [aa['After'],aa['Before'],bb['After'],bb['Before']])else(aa['After']-aa['Before'])-(bb['After']-bb['Before']);pairs.append((r['alias'],v))
    out['actual_'+support+'_current_IoU']=pack(pairs)
   paired[ds][a+'_minus_'+b]=out
 emit(P/'PAIRED_DECOMPOSITION.json',dict(status='pass',count=32,groups=paired,GT_rescored=False,LOO_removes_saved_score_not_stream_update=True,bootstrap_descriptive_conditional_on_fixed_filtered_stream=True))
 # Pure parameter/decoder movement/ESS readback, with no new annotations or scoring.
 rounds=[];fitcalls=gradientcalls=0;source_states=collections.defaultdict(set)
 for f in bar['files']:
  saved=load(ROOT/f['path']);assert not saved['GT_read'];source_states[saved['dataset']].add(saved['Source_state_sha256']);planned=saved['planned_positions'];dense=[j for j in range(len(saved['frame_ids']))if j not in planned]
  for arm,fit in saved['arms'].items():
   fitcalls+=1;gradientcalls+=fit['gradient_calls'];states=fit['path'];records=fit['rounds'];assert sum(r['updated']for r in records)==fit['gradient_calls'];checks+=1
   for k,r in enumerate(records):
    update=states[k].get('update');move=(states[k+1]['boxes']-states[k]['boxes']).abs().mean(1)
    sparse_ess=r.get('ESS')if arm=='A_sparse'else r.get('sparse_ESS');dense_ess=r.get('dense_ESS')
    sparse_g=r.get('autograd_mean_gradient')if arm=='A_sparse'else r.get('sparse_mean_gradient');dense_g=r.get('dense_mean_gradient')
    val=lambda x:None if x is None else float(x.mean())
    norm=lambda x:None if x is None else float(x.norm())
    rounds.append(dict(alias=saved['alias'],dataset=saved['dataset'],arm=arm,round=k,updated=r['updated'],sparse_ESS=val(sparse_ess),dense_ESS=val(dense_ess),parameter_step_norm=norm(None if update is None else update['raw']),parameter_gradient_norm=norm(None if update is None else update['gradient']),sparse_mean_gradient_norm=norm(sparse_g),dense_mean_gradient_norm=norm(dense_g),observed_actual_decoder_mean_abs_cxcywh_movement=float(move[planned].mean())if planned else None,unobserved_actual_decoder_mean_abs_cxcywh_movement=float(move[dense].mean())if dense else None))
 assert fitcalls==128 and all(len(v)==1 for v in source_states.values())
 emit(P/'ROUND_DIAGNOSTICS.json',dict(status='pass',fit_invocations=fitcalls,actual_parameter_updates=gradientcalls,rows=rounds,raw_parameters_gradients_tensors_excluded=True,GT_read_for_this_readback=False))
 emit(P/'ROOT_PUBLIC_AGGREGATE_AUDIT.json',dict(status='pass',count=32,checks=checks,all_parent_query_denominators_exact=True,all_Source_and_fixed_time_equal=True,all_saved_mean_pair_CI_gate_recomputed=True,extra_paired_current_inherited_decoder_readback=True,actual_round_state_ESS_and_parameter_steps_read_back=True,GT_rescored=False))
 emit(P/'ACTUAL_PROTOCOL.json',dict(count=32,per_domain=16,arms=ARMS,configs=p['configs'],selection=p['selection'],excluded_parent_counts=p['excluded_parent_counts'],historical_full_GT_exposure=p['historical_full_GT_exposure'],independent_confirmation=False,stream=p['stream'],final_time=p['final_time'],dense_coefficient=p['dense_coefficient'],correspondence_temperature=p['correspondence_temperature'],Source_model_state_sha256={ds:next(iter(v))for ds,v in source_states.items()},Source_checkpoints=source_checkpoints,actual_fit_counts=counts,evaluation_scorer_lock002_sha256=evaluation['scorer_sha256'],GPU_lease_seconds=bar['GPU_lease_seconds'],Source_captures=32,DINO_calls=0,fit_invocations=128,actual_parameter_updates=gradientcalls,training=0,cap_seconds=p['GPU_seconds_cap'],resource_enforcement=dict(systemd_MemoryHigh_GiB=16,systemd_MemoryMax_GiB=20,systemd_MemorySwapMax_GiB=1,automatic_stage_host_available_GiB=6,automatic_stage_disk_free_GiB=30,GPU_free_floor_GiB=4,GPU_floor_enforcement='manual root progress checks; not an automatic runner device-floor gate',shared_GPU_lease='cooperating-job serialization only',peak_GPU_memory='not measured')))
 # Narrative does not silently relabel a new filtered stream as original full online.
 lines=['# Four-arm actual online dense correspondence feedback32','','Sixteen unique parent queries/domain, selected by the first eligible original P1 order1 parent after excluding recent rollback32/P6_64/original tuning parents. Historically full-stream exposed; these are new to the recent mechanism panel but not unseen confirmation. Four Source-initialized independent branches carry their own LN state in this newly filtered16-query stream; QR and Adam reset at each query. It does not replay the original full online history.','','A is unchanged sparse OPD; B adds raw frozen correspondence; C adds correspondence rankings with interpolation-derived weight strength; D adds continuous box-interpolation feedback. Dense term and sparse term are separately frame-averaged with coefficient one. Final Native time, Source checkpoints and original domain sigma/tau/lr/steps/writeback/1792 parameters stay fixed. No new DINO or training. Sparse-plus-dense uses two separately normalized losses; it does not match the total gradient budget of A. Actual gradients and Adam movements are measured. New aliases H001/V001 refer to different parents from older panels; no old identity labels are joined.','','## Complete metrics','','| Domain | Arm | State | vIoU % | tIoU % | sIoU % | R03 % | R05 % |','|---|---|---|---:|---:|---:|---:|---:|']
 for ds,g in z['groups'].items():
  for arm in ARMS:
   for state in STATES:
    m=g['metrics'][arm][state];lines.append(f'| {ds} | {arm} | {state} | '+' | '.join(f'{100*m[k]:.6f}'for k in ['v','t','s','R30','R50'])+' |')
 lines+=['','## Prespecified outcome and immediate-versus-inherited effects','','The two-domain gate failed because HC C does not exceed A and its leave-one-scored-parent-out condition fails; the Vid mean/LOO/harm conditions pass. The stop follows this prespecified conjunction, not a rule that every interval crossing zero proves no effect. C−D mean gains and their uncertainty remain. No confirmation, direct-control run, temperature/encoder search or promotion is dispatched.','','| Domain | C minus A total Δv pp | C minus A current Δv pp | C minus A inherited Δv pp |','|---|---:|---:|---:|']
 for ds in z['groups']:
  vv=paired[ds]['C_rank_minus_A_sparse'];lines.append(f'| {ds} | {vv["total"]["mean_pp"]:+.6f} | {vv["current"]["mean_pp"]:+.6f} | {vv["inherited"]["mean_pp"]:+.6f} |')
 lines+=['','Vid C−A total benefit is accompanied by a negative current-query difference and a larger positive inherited-state difference. It cannot be described as a measured immediate dense-update improvement. HC has the same sign split but negative total difference. These are algebraic state-score contrasts under four different coupled histories, not causal shares or proof of long-stream transfer.','','## Current and inherited changes','','Total=After−Source, current=After−Before, inherited=Before−Source. This exact arithmetic does not identify causal cross-video transfer. One query/parent makes query macro equal parent macro. Parent bootstrap10000 draws is descriptive conditional on this coupled online sequence.','','| Domain | Arm | Total pp [CI95] | Current pp [CI95] | Inherited pp [CI95] | Current harms below −5pp |','|---|---|---:|---:|---:|---:|']
 for ds,g in z['groups'].items():
  for a in ARMS:
   d=g['deltas'][a];fmt=lambda k:f'{d[k]["parent_mean_pp"]:+.6f} [{d[k]["CI95_pp"][0]:+.6f}, {d[k]["CI95_pp"][1]:+.6f}]';lines.append(f'| {ds} | {a} | {fmt("total")} | {fmt("current")} | {fmt("inherited")} | {d["current"]["severe_harm_gt5pp"]} |')
 lines+=['','## Effective fit counts','','No admitted reference is distinct from actual zero updates. Queries with references may still have all-equal sparse feedback; dense feedback can be informative in that case. Fixed maximum steps/samples are shared, but actual gradient-call counts are reported per arm.','','| Domain | Queries without references | Arm | Queries with updates | Queries with zero updates | Reference present but no update | Parameter updates |','|---|---:|---|---:|---:|---:|---:|']
 for ds,c in counts.items():
  for a,v in c['per_arm'].items():lines.append(f'| {ds} | {c["no_admitted_reference_queries"]} | {a} | {v["updated_queries"]} | {v["zero_update_queries"]} | {v["with_reference_but_zero_update_queries"]} | {v["actual_parameter_updates"]} |')
 lines+=['','## Same-panel paired comparisons','','| Domain | After comparison | Δv pp [CI95] | Positive / negative / neutral | Leave-one-scored-parent-out minimum pp |','|---|---|---:|---:|---:|']
 for ds,g in z['groups'].items():
  for pair,v in g['paired'].items():lines.append(f'| {ds} | {pair} | {v["mean_pp"]:+.6f} [{v["CI95_pp"][0]:+.6f}, {v["CI95_pp"][1]:+.6f}] | {v["positive"]}/{v["negative"]}/{v["neutral"]} | {v["leave_one_parent_out_min_pp"]:+.6f} |')
 lines+=['','Leave-one-out drops one saved parent score from an average; it does not rerun the online stream with that update removed. PAIRED_DECOMPOSITION.json additionally separates C−A/C−D current and inherited effects, and matched observed/unobserved/far actual-decoder current-IoU changes with signed parents and descriptive intervals.','','## Actual decoder support and unknowns','','Official interpolated tube scores above are distinct from actual decoder-frame geometry below. Known annotation only; missing support is unknown, not zero. All four planned DINO slots, including rejected slots, are excluded from unobserved. Far uses physical distance≥25% input span from any planned slot.','','| Domain | Support | Parents with known frames / 16 | Known frames | Arm | Source s % | Before s % | After s % |','|---|---|---:|---:|---|---:|---:|---:|']
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for sup in ['observed','unobserved','far']:
   nq=sum(r['actual_support_counts'][sup]>0 for r in rr);nf=sum(r['actual_support_counts'][sup]for r in rr)
   for a in ARMS:
    fmt=lambda s:'unknown'if g['actual_decoder'][a][sup][s]is None else f'{100*g["actual_decoder"][a][sup][s]:.6f}';lines.append(f'| {ds} | {sup} | {nq}/16 | {nf} | {a} | {fmt("Source")} | {fmt("Before")} | {fmt("After")} |')
 lines+=['','## Strength, actual updates and cost','','C uses sorted interpolation weights calculated at C\'s own current Student and samples; after branches diverge these are not D\'s actual branch weights. Ties average assigned weights and can lower nominal strength; flat rewards are uniform. Equal weight concentration does not guarantee equal gradient/Adam movement. ROUND_DIAGNOSTICS.json retains every round\'s ESS, actual parameter gradient/step norms and actual decoder observed/unobserved movements; raw tensors/parameter vectors remain private. Dense-rank statistics use each branch\'s current samples and cannot by themselves establish actual tube gain. After updates, B/C/D candidate centers diverge: their rank deltas are not a strict same-candidate comparison of feedback quality; the earlier fixed-candidate69 audit provides that separate control.','','| Domain | Arm | Mean fit sec | Updates/query | Sparse ESS | Dense ESS | Parameter step norm | Dense rank Δ pp |','|---|---|---:|---:|---:|---:|---:|---:|']
 for ds,g in z['groups'].items():
  for a in ARMS:
   v=g['compute'][a];fmt=lambda k:'unknown'if v[k]is None else f'{v[k]:.6f}';lines.append(f'| {ds} | {a} | {fmt("fit_seconds")} | {fmt("gradient_calls")} | {fmt("sparse_ESS")} | {fmt("dense_ESS")} | {fmt("parameter_step_norm")} | {fmt("dense_rank_delta_pp")} |')
 lines+=['','## Engineering evaluation and actual pixels','','Before any current GT scoring, the evaluator state-difference comparison was corrected to preserve float32 update arithmetic, and saved seed/antithetic/sigma/parameter-name audits were added. EVALUATION_LOCK002 binds the final scorer; original evaluation lock and correction description are retained privately. No model output, parameter state, gate or prediction was changed for that correction. Six actual private panels show new H001/V001 four-slot Source/Before/After readouts (72 RGB cells), with original frame and DINO pixel hashes exact. This is engineering inspection only, no GT overlay and no new identity/semantic labels. V001 has no admitted reference and its no-update outputs remain, rather than being dropped. No older same-name case identity label is reused. Source checkpoint basenames and hashes and frozen model state hashes are in ACTUAL_PROTOCOL.json.']
 lines+=['',f'GPU lease {bar["GPU_lease_seconds"]:.6f}s includes Source loading/capture, transfers, verification and fitting, not pure kernel or first-download deployment time. 32 frozen Source captures,128 fit invocations including no-op sources, actual parameter updates '+str(gradientcalls)+',0 new DINO calls,0 training. No-fit sources stay in all denominators. Full two-decoder outputs replayed exactly, analytic likelihood/Adam/state arithmetic independently checked; decoder Jacobian correctness is not independently proved by this audit.','','## Resource enforcement and prelocked decision','','Actual enforcement: systemd MemoryHigh16GiB/MemoryMax20GiB/SwapMax1GiB is automatic; stage resource_guard checks host available6GiB and disk free30GiB. GPU free4GiB was checked manually by root at progress points, not by an automatic device-floor gate in this runner. The shared GPU lease serializes cooperating jobs; it does not itself enforce physical GPU memory admission. No peak GPU usage is asserted.','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','Additional correspondence evidence requires C>A and C>D in both domains, positive leave-one-scored-parent-out means, and C current-harm count≤A+1. No temperature/encoder sweep or automatic follow-up/production promotion occurs. Full signed/neutral/negative/unknown, common event-length/motion/area/query strata and state decomposition remain in RESULTS.json. Offline GT strata do not route online evidence. This filtered, previously exposed finite panel cannot establish original full-stream benefit or unseen confirmation. Actual pixel checks are engineering only; identity correctness is not inferred from IoU and no new semantic labels are created.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 fs=['scripts/run_stvg_dense_correspondence32_v1.py','vg_tta/stvg_dense_correspondence_opd_v1.py','scripts/score_stvg_dense_correspondence32_v1.py','scripts/report_stvg_dense_correspondence32_v1.py','scripts/render_stvg_dense_correspondence32_v1.py','protocols/stvg_dense_correspondence32_v1.md']+[str((P/f).relative_to(ROOT))for f in ['RESULTS.json','AUDIT.json','DECISION.json','COHORT_AUDIT.json','PAIRED_DECOMPOSITION.json','ROUND_DIAGNOSTICS.json','ROOT_PUBLIC_AGGREGATE_AUDIT.json','ACTUAL_PROTOCOL.json','REPORT.md']];emit(D/'PUBLIC_FILES.json',dict(files=fs));print('REPORT_READY',checks,fitcalls,gradientcalls,flush=True)
if __name__=='__main__':run()
