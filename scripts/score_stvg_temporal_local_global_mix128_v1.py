"""Sealed cached-mixture evaluation and predeclared post-hoc explanation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,signal,fcntl,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_local_global_mix128_v1 import D,P,OLD,OP,BASE,ARMS,tensor_sha
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def tails(delta,aliases):
    import numpy as np
    d=np.asarray(delta,float)
    return dict(positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),flat=int((abs(d)<=1e-10).sum()),
        below_minus5=int((d< -5).sum()),below_minus3=int((d< -3).sum()),
        leave_one_out_mean_range_pp=[float(((d.sum()-d)/(len(d)-1)).min()),float(((d.sum()-d)/(len(d)-1)).max())],
        worst=sorted([dict(alias=a,delta_pp=float(v))for a,v in zip(aliases,d)],key=lambda x:x['delta_pp'])[:12])

def decomposition(rows):
    import numpy as np
    names=[r['alias']for r in rows]
    def v(arm):return np.array([r['values']['Mean_'+arm]['v']for r in rows])
    native=v('Native');dd=100*(v('DenseLocal')-native);ss=100*(v('HiResAuxMix')-native)
    mm=100*(v('LocalGlobalMix')-native);old=100*(v('DenseLocalMix')-native)
    dense=dd>1e-10;strong=ss>1e-10;groups={}
    for key,mask in [('both_improve',dense&strong),('Dense_only',dense&~strong),('Strong_only',~dense&strong),('neither',~dense&~strong)]:
        count=int(mask.sum());den=float(np.maximum(dd[mask],0).sum())
        def summary(a):return dict(group_mean_pp=float(a[mask].mean())if count else None,contribution_to_all128_mean_pp=float(a[mask].sum()/128))
        groups[key]=dict(parents=count,aliases=[a for a,b in zip(names,mask)if b],
            Dense_minus_Native=summary(dd),Strong_minus_Native=summary(ss),New_minus_Native=summary(mm),New_minus_Strong=summary(mm-ss),
            Dense_positive_gain_sum_pp=den,capped_positive_Dense_gain_retained_fraction=float(np.minimum(np.maximum(mm[mask],0),np.maximum(dd[mask],0)).sum()/den)if den else None,
            Dense_gain_sign_retained_parents=int((mask&(mm>1e-10)).sum()))
    oldbad=old< -5;newbad=mm< -5;assert int(oldbad.sum())==16
    def aliases(mask):return [a for a,b in zip(names,mask)if b]
    severe=dict(reference='delta to Native, original DenseLocalMix severe16; no online grouping',old_severe=16,
        any_alleviated=int((oldbad&(mm>old+1e-10)).sum()),exited_severe=int((oldbad&~newbad).sum()),
        persistent_severe=int((oldbad&newbad).sum()),worsened=int((oldbad&(mm<old-1e-10)).sum()),
        unchanged=int((oldbad&(abs(mm-old)<=1e-10)).sum()),new_severe_outside_old16=int((~oldbad&newbad).sum()),
        new_total_severe=int(newbad.sum()),old_aliases=aliases(oldbad),exited_aliases=aliases(oldbad&~newbad),
        persistent_aliases=aliases(oldbad&newbad),new_aliases=aliases(~oldbad&newbad),
        per_old_severe=[dict(alias=a,old_delta_pp=float(o),new_delta_pp=float(m),new_minus_old_pp=float(m-o))for a,o,m,b in zip(names,old,mm,oldbad)if b])
    assert sum(g['parents']for g in groups.values())==128
    return dict(groups=groups,severe=severe,grouping_uses_GT_only_after_global_seal=True,no_GT_online_selection=True)

def main():
    import numpy as np,torch
    from main_method.runtime import resource_guard
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
    def reject(event,args):
        if not phase['GT']and event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
            raise PermissionError('mixture independent audit forbids GT before barrier')
    sys.addaudithook(reject)
    p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
    assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
    def guard():resource_guard();assert time.perf_counter()-beg<900 and time.time()<p['deadline_epoch']
    space=oldexact=scalar=utilities=0;maxerr=0.;ii,jj=np.triu_indices(65,1)
    for r in p['records']:
        guard();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');ids=z['frame_ids']
        assert ids==old['frame_ids']and sha(R/z['source_prediction'])==z['source_sha256']
        q=z['outputs']['LocalGlobalMix']['q'];assert q.shape==(65,65)and not np.tril(q).any()and abs(q.sum()-1)<1e-12
        for i,j in zip(ii,jj):
            v=(float(old['outputs']['HiResAuxMix']['q'][i,j])+float(old['outputs']['DenseLocal']['q'][i,j]))/2
            assert v==q[i,j];scalar+=1
        for k in BASE:
            assert np.array_equal(z['outputs'][k]['q'],old['outputs'][k]['q'])and z['outputs'][k]['interval']==old['outputs'][k]['interval'];oldexact+=1
        for k,b in old['trajectories'].items():assert tensor_sha(b)==z['frozen_spatial_hashes'][k];space+=1
        out=final_mbr(ids,q);assert out['interval']==z['outputs']['LocalGlobalMix']['interval']
        u=.5*final_mbr(ids,old['outputs']['HiResAuxMix']['q'])['utility_grid'][ii,jj]+.5*final_mbr(ids,old['outputs']['DenseLocal']['q'])['utility_grid'][ii,jj]
        err=float(np.max(abs(out['utility_grid'][ii,jj]-u)));assert err<1e-11;maxerr=max(maxerr,err)
        sel=np.unique(np.r_[np.linspace(0,2079,13).astype(int),np.flatnonzero((ii==out['indices'][0])&(jj==out['indices'][1]))])
        direct=brute_utility(ids,q,sel);err=float(np.max(abs(direct-out['utility_grid'][ii[sel],jj[sel]])));assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(sel)
    assert space==512 and oldexact==512 and scalar==266240
    audit=dict(status='pass',parents=128,mixture_scalar_checks=scalar,old_readouts_exact=oldexact,frozen_space_hashes_exact=space,
        independent_direct_MBR_sample_utility_checks=utilities,all2080_utility_linearity_per_parent=True,max_utility_error=maxerr,new_GT=False)
    put(D/'AUDIT_BEFORE_GT.json',audit);put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
    raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    prior={r['alias']:r for r in read(OP/'RESULTS_PUBLIC.json')['rows']};rows=[];officialchecks=oldmetric=0
    for r in p['records']:
        guard();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');values={};actual={}
        for space,b in old['trajectories'].items():
            tube=DenseTube(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
            selected=np.isin(tube.fids,np.array(sorted(set(z['frame_ids'])-set(r['physical_frame_ids8']))))
            actual[space]=dict(nodes=int(selected.sum()),sIoU=float(tube.iou[selected].mean())if selected.any()else None,all_temporal_arms_identical=True)
            assert actual[space]==prior[a]['fixed_unobserved_actual_sIoU'][space]
            for arm in ARMS:
                iv=z['outputs'][arm]['interval'];v=tube.score(iv);off=official(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2')
                assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
                values[space+'_'+arm]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
                if arm in BASE:
                    assert all(abs(values[space+'_'+arm][k]-prior[a]['values'][space+'_'+arm][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetric+=1
            assert len({values[space+'_'+arm]['s']for arm in ARMS})==1
        for arm in ARMS:values['Mean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
        expected={k:prior[a]['expected_candidate_quality'][k]for k in BASE}
        expected['LocalGlobalMix']={k:.5*expected['HiResAuxMix'][k]+.5*expected['DenseLocal'][k]for k in ['v','t']}
        rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,fixed_unobserved_actual_sIoU=actual))
    means={arm:{k:avg([r['values']['Mean_'+arm][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for arm in ARMS}
    stats={};expected_stats={};aliases=[r['alias']for r in rows]
    for arm in BASE:
        key='LocalGlobalMix_minus_'+arm
        stats[key]={k:paired([(100 if k!='boundary_MAE'else 1)*(r['values']['Mean_LocalGlobalMix'][k]-r['values']['Mean_'+arm][k])for r in rows])for k in ['v','t','s','boundary_MAE']}
        stats[key]['tails']=tails([100*(r['values']['Mean_LocalGlobalMix']['v']-r['values']['Mean_'+arm]['v'])for r in rows],aliases)
        expected_stats[key]={k:paired([100*(r['expected_candidate_quality']['LocalGlobalMix'][k]-r['expected_candidate_quality'][arm][k])for r in rows])for k in ['v','t']}
    exp=decomposition(rows);signal=stats['LocalGlobalMix_minus_HiResAuxMix']['v']['mean']>1e-10
    oldpan=read(R/'artifacts/stvg_temporal_increment_confirm109_v1/PLAN_LOCK.json')
    allparents={q['input']['source']for q in raw};devparents={r['parent']for r in p['records']};left=allparents-devparents
    feasibility=dict(validation_parents=len(allparents),development_parents=128,remaining_validation_parents=len(left),
        overlaps_previously_used_confirmation=len(left&{r['parent']for r in oldpan['records']}),
        meets_requested_minimum128=len(left)>=128,not_untouched_method_test=True,
        decision='requires_explicit_roster_amendment_if_positive; do not silently shrink or change split')
    decision=dict(development_mean_gain_over_Strong=signal,primary=stats['LocalGlobalMix_minus_HiResAuxMix']['v'],
        decision='freeze_only_mixture_pending_confirmation_roster_and_cost_protocol'if signal else'close_this_fixed_mixture_no_weight_search',
        no_immediate2pp_gate=True,not_OPD=True,not_independent_confirmation=True,no_GT_online_selection=True,
        no_weight_sweep=True,no_automatic_adaptive_observation=True,confirmation_status='not_run',goal_achieved=False)
    put(P/'RESULTS_PUBLIC.json',dict(status='completed_cached_fixed_half_mixture',parents=128,seeds=p['seeds'],means=means,paired=stats,
        expected_quality_derived_linearly_from_old_GT_diagnostic=expected_stats,rows=rows,spatial_frozen=True,new_model_forward=0,not_OPD=True))
    put(P/'COMPLEMENTARITY_PUBLIC.json',exp);put(P/'CONFIRMATION_FEASIBILITY_PUBLIC.json',feasibility);put(P/'DECISION_PUBLIC.json',decision)
    audit.update(new_GT=True,official_Dense_output_checks=officialchecks,old_metrics_exact=oldmetric,GT_provenance=prov,
        all128_sealed_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_actual_sIoU_exact_old=True)
    assert audit['all128_sealed_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
    cost=dict(**read(D/'COST.json'),CPU_audit_and_scoring_seconds=time.perf_counter()-beg,publication_archive_excluded=True,
        source_cache_generation_cost_separate=read(OLD/'COST.json')['GPU_seconds'],standalone_deployment_profile='not_measured; cached combination is not full inference latency')
    put(P/'COST_PUBLIC.json',cost)
    lines=['# One fixed local/global posterior combination: exposed HC128','','New=.5 HiResAuxMix+.5 pure DenseLocal, same65/2080/MBR and frozen E/Live41/42/43. One combination only; no model/RGB/GPU/expert/training/student/spatial update, no OPD. All128 sealed before new GT. Repeatedly used development parents; seed average within parent then equal parents.','','| Output | full vIoU percent | tIoU percent | boundary MAE frames |','|---|---:|---:|---:|']
    for arm,v in means.items():lines.append(f"| {arm} | {100*v['v']:.6f} | {100*v['t']:.6f} | {v['boundary_MAE']:.6f} |")
    lines+=['','| Paired comparison | full-v pp [CI95] | tIoU pp [CI95] | losses<-5pp |','|---|---|---|---:|']
    for k,v in stats.items():lines.append(f"| {k} | {v['v']['mean']:+.6f} {v['v']['CI95']} | {v['t']['mean']:+.6f} {v['t']['CI95']} | {v['tails']['below_minus5']} |")
    lines+=['','## Complementarity and severe tails','','Groups use GT only AFTER sealing, never online selection. Dense-only means pure Dense improves Native but Strong does not. Signed gain and all128 contributions retained; capped-positive retention is explanatory and cannot hide negative signed effects. All group aliases and individual severe changes in COMPLEMENTARITY_PUBLIC.json.','','```json',json.dumps(exp,indent=2),'```','','## Decision / confirmation / cost','','```json',json.dumps(decision,indent=2),json.dumps(feasibility,indent=2),json.dumps(cost,indent=2),'```','','No2pp threshold was imposed on this single development complementarity check. A positive mean relativeStrong is a candidate signal, not statistical/independent proof. Do not sweep weights after failure. No automatic OPD or adaptive observation. Confirmation requires a source-disjoint prelocked panel and separate actual deployment profiling; the old1532s matrix is not deployment latency. Native/Strong/Dense/DenseMix metrics reproduced, all negative/flat parents and no-reference states retained. Expected-quality numbers are linear combinations of old offline GT diagnostic values, not fresh model feedback or a gate. No spatial/CURRENT/MAIN/PAPER change.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],source='existing frozen Source Vid and local/global caches only',
        exposure='HC128 repeatedly exposed development; no independent confirmation',excluded=['captions','parentIDs','media','GT coordinates','weights','raw caches']))
    assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);guard()
    print(json.dumps(dict(decision=decision,severe=exp['severe'],groups={k:{kk:vv for kk,vv in v.items()if kk!='aliases'}for k,v in exp['groups'].items()}),indent=2),flush=True)

if __name__=='__main__':
    def timeout(signum,frame):raise TimeoutError('finite900s cached-mixture scoring')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:put(D/f'SCORING_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:lock.close()
