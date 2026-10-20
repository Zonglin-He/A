"""Independent acquisition/reveal audit, then sealed policy-effect evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,signal,fcntl,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_state_observation128_v1 import D,P,OLD,EARLY,STAGE,HR,POLICIES,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
    import torch
    from main_method.runtime import resource_guard
    from vg_tta.stvg_temporal_state_observation128_v1 import choose,reflected,uniform_order,array_sha
    from vg_tta.stvg_temporal_registered_motion128_v1 import tensor_sha
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
    from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
    torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
    def block(event,args):
        if not phase['GT']and event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):
            raise PermissionError('cached-policy preaudit forbids GT')
    sys.addaudithook(block);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==128 and bar['all_before_GT']and bar['weights_flags_grad_unchanged']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
    assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
    def guard():resource_guard();assert time.perf_counter()-beg<900
    metrics=dict(selection_steps=0,independent_acquisition_scores=0,selected_only_fusion_inputs=0,layer_posteriors=0,independent_final_layer_CDF=0,
        direct_MBR_sample_utilities=0,space_exact=0,native_empty_exact=0,first_round_Fixed_Live_exact=0,logical_policy_budgets=0)
    maxscore=maxcdf=maxutility=0.;ii,jj=np.triu_indices(65,1);state_changes=[]
    for r in p['records']:
        guard();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');e=load(EARLY/'early_inputs'/f'{a}.pt');st=load(STAGE/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids'])
        assert np.array_equal(z['initial_q'],old['outputs']['HiResAuxMix']['q'])and np.array_equal(z['hires_q'],hr['observation']['q'])
        for key,v in z['replays'].items():
            selected=v['selected'];assert selected==sorted(selected)and key==','.join(map(str,selected))and len(selected)in [0,4,8,65]
            if len(selected)==65:assert a in ['RC001','RC002']
            n=[]
            for off in (0,1):
                cap=e['captured_encoder_inputs'][off];h,w=e['original_views'][off]['info']['fea_map_size'];l=h*w
                n.append(cap['src'][-l:].permute(1,2,0).reshape(len(ids[off::2]),256,h,w))
            native=torch.stack([n[i%2][i//2]for i in range(65)]);want=native.clone()
            for t in selected:want[t]=old['motion_features']['DenseLocal'][t].to(want)
            assert tensor_sha(want)==v['motion_sha256']and tensor_sha(old['motion_features']['DenseLocal'][selected])==v['selected_features_sha256']
            mask=np.ones(65,bool);mask[selected]=False;assert torch.equal(want[mask],native[mask])
            for off in (0,1):
                cap=e['captured_encoder_inputs'][off];h,w=e['original_views'][off]['info']['fea_map_size'];l=h*w;x=cap['src'].clone()
                x[-l:]=want[off::2].to(x).flatten(2).permute(2,0,1)
                meta=v['metadata'][str(off)];assert tensor_sha(x)==meta['actual_input_sha256']and meta['appearance_text_input_exact']and meta['position_padding_exact']and meta['actual_encoder_entry_verified']
                assert torch.equal(x[:-l],cap['src'][:-l])and tensor_sha(v['prefixes'][off])==meta['prefix_sha256'];metrics['selected_only_fusion_inputs']+=1
            for k in range(6):
                gs=probabilities(z['records'],[l[k]for l in v['second_pass_all6_logits']]);grid,q=mass_table(gs)
                assert grid==z['frame_ids']and np.array_equal(q,v['layer_qs'][k]);metrics['layer_posteriors']+=1
                if k==5:
                    err=float(abs(independent_cdf(gs,grid)-q).max());assert err<2e-13;maxcdf=max(maxcdf,err);metrics['independent_final_layer_CDF']+=1
            q=.5*z['hires_q']+.5*np.mean(v['layer_qs'],axis=0);assert np.array_equal(q,v['q'])
            out=final_mbr(ids,q);assert out['interval']==v['interval']
            choices=np.unique(np.r_[np.linspace(0,2079,13).astype(int),np.flatnonzero((ii==out['indices'][0])&(jj==out['indices'][1]))])
            err=float(abs(brute_utility(ids,q,choices)-out['utility_grid'][ii[choices],jj[choices]]).max());assert err<1e-11;maxutility=max(maxutility,err);metrics['direct_MBR_sample_utilities']+=len(choices)
        zero=z['replays'][''];assert np.array_equal(zero['q'],z['initial_q'])and all(torch.equal(x,y)for x,y in zip(zero['second_pass_all6_logits'],st['pass_logits'][1]));metrics['native_empty_exact']+=1
        for k,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][k]);metrics['space_exact']+=1
        for policy in POLICIES:
            selected=[];before=z['initial_q']
            for rn,step in enumerate(z['policies'][policy]['history']):
                assert np.array_equal(step['actual_student_q_before'],before)
                if policy=='Uniform8':
                    selected=uniform_order()[:4*(rn+1)];assert selected==step['selected']
                else:
                    qstate=before if policy=='CurrentState8'else reflected(z['initial_q'])if policy=='ReflectedState8'else z['initial_q']
                    assert array_sha(qstate)==step['selection']['state_q_sha256']
                    result,trace=choose(ids,qstate,selected,4);assert result==step['selected'];iv=final_mbr(ids,qstate)['interval']
                    starts=ids[ii];ends=ids[jj]+1;inter=np.maximum(0,np.minimum(ends,iv[1])-np.maximum(starts,iv[0]));u=ends-starts+iv[1]-iv[0]-inter
                    ww=qstate[ii,jj]*(1-inter/u);sw=np.bincount(ii,weights=ww,minlength=65);ew=np.bincount(jj,weights=ww,minlength=65)
                    for tr,saved in zip(trace['trace'],step['selection']['trace']):
                        assert tr['position']==saved['position']and np.array_equal(tr['all_candidate_gains'],saved['all_candidate_gains'])
                        covered_s=np.zeros(65,bool);covered_e=np.zeros(65,bool)
                        for t in selected:
                            lo=max(int(ids[0]),int(ids[t])-6);hi=min(int(ids[-1]),int(ids[t])+8)+1
                            covered_s|=(ids>=lo)&(ids<hi);covered_e|=(ids+1>lo)&(ids+1<=hi)
                        actual=[]
                        for t in range(65):
                            lo=max(int(ids[0]),int(ids[t])-6);hi=min(int(ids[-1]),int(ids[t])+8)+1
                            actual.append(float(sw[(ids>=lo)&(ids<hi)&~covered_s].sum()+ew[(ids+1>lo)&(ids+1<=hi)&~covered_e].sum()))
                        err=float(np.max(abs(np.array(actual)-saved['all_candidate_gains'])));bound=4096*np.finfo(float).eps*max(1.,2*float(ww.sum()));assert err<=bound;maxscore=max(maxscore,err)
                        eligible=np.array([t not in selected for t in range(65)]);assert np.array_equal(eligible,saved['eligible'])
                        assert saved['gain']>=np.array(actual)[eligible].max()-bound
                        selected.append(saved['position']);metrics['selection_steps']+=1;metrics['independent_acquisition_scores']+=65
                    assert selected==result
                assert len(selected)==4*(rn+1)and len(set(selected))==len(selected)
                k=step['replay_key'];rep=z['replays'][k];assert sorted(selected)==rep['selected']and np.array_equal(step['q_after'],rep['q'])
                before=rep['q']
            assert len(selected)==8 and np.array_equal(z['outputs'][policy]['q'],before)
            assert z['outputs'][policy]['interval']==final_mbr(ids,before)['interval'];metrics['logical_policy_budgets']+=1
        first=z['policies']['InitialState8']['history'][0];live=z['policies']['CurrentState8']['history'][0]
        assert first['selected']==live['selected']and np.array_equal(first['q_after'],live['q_after']);metrics['first_round_Fixed_Live_exact']+=1
        f=z['policies']['InitialState8']['final_selected'];c=z['policies']['CurrentState8']['final_selected']
        state_changes.append(dict(alias=a,first4_L1=float(abs(first['q_after']-z['initial_q']).sum()),round2_final_set_changed=set(f)!=set(c),
            final_intersection=len(set(f)&set(c)),Current_unique_frames=z['policies']['CurrentState8']['physical_unique_frames']))
    assert metrics['space_exact']==512 and metrics['logical_policy_budgets']==512
    audit=dict(status='pass',checks=metrics,max_independent_acquisition_score_error=maxscore,max_final_layer_CDF_error=maxcdf,
        max_direct_utility_error=maxutility,selected_local_features_only=True,zero_new_RGB_encoding=True,new_GT=False)
    put(D/'AUDIT_BEFORE_GT.json',audit);put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all128_before_GT=True));phase['GT']=True
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
    raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    prev={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_local_motion_completion128/RESULTS_PUBLIC.json')['rows']}
    rows=[];officialchecks=oldmetric=0
    for r in p['records']:
        guard();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');values={};stage1={k:[]for k in POLICIES};quality={'v':[],'t':[]};actual={}
        for space,b in z['trajectories'].items():
            tube=DenseTube(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
            sel=np.isin(tube.fids,np.array(sorted(set(z['frame_ids'])-set(r['physical_frame_ids8']))))
            actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
            assert actual[space]==prev[a]['fixed_unobserved_actual_sIoU'][space]
            def evaluate(iv):
                nonlocal officialchecks
                v=tube.score(iv);o=official(b,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2')
                assert max(abs(v[k]-o[k])for k in ['v','t','s'])<2e-10;officialchecks+=1
                return dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
            for arm,o in z['outputs'].items():
                values[space+'_'+arm]=evaluate(o['interval'])
                if arm in ['Native','HiResAuxMix','DenseLocalMix']:
                    assert all(abs(values[space+'_'+arm][k]-prev[a]['values'][space+'_'+arm][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetric+=1
            for policy in POLICIES:
                first=z['policies'][policy]['history'][0];iv=z['replays'][first['replay_key']]['interval'];v=evaluate(iv)
                if space!='E':stage1[policy].append(v)
            if space!='E':
                vals=[tube.score([int(z['frame_ids'][i]),int(z['frame_ids'][j])+1])for i,j in zip(ii,jj)]
                for k in quality:quality[k].append(np.array([v[k]for v in vals]))
        for arm in ARMS:values['Mean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
        means1={pol:{k:avg([v[k]for v in vv])for k in ['v','t','s','boundary_MAE']}for pol,vv in stage1.items()}
        curves={k:np.mean(v,0)for k,v in quality.items()}
        expected={arm:{k:float(o['q'][ii,jj]@curve)for k,curve in curves.items()}for arm,o in z['outputs'].items()}
        rows.append(dict(alias=a,values=values,stage1_mean=means1,expected_candidate_quality=expected,fixed_unobserved_actual_sIoU=actual,
            selected_nodes={pol:z['policies'][pol]['final_selected']for pol in POLICIES},
            physical_unique_frames={pol:z['policies'][pol]['physical_unique_frames']for pol in POLICIES}))
    means={arm:{k:avg([r['values']['Mean_'+arm][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for arm in ARMS}
    pairs=[('CurrentState8',b)for b in ['HiResAuxMix','Uniform8','InitialState8','ReflectedState8','Native']]+[('InitialState8','Uniform8'),('InitialState8','HiResAuxMix'),('Uniform8','HiResAuxMix'),('ReflectedState8','HiResAuxMix')]
    from scripts.score_stvg_temporal_local_global_mix128_v1 import tails
    stats={};expect={};aliases=[r['alias']for r in rows]
    for aa,bb in pairs:
        key=aa+'_minus_'+bb;delta=[100*(r['values']['Mean_'+aa]['v']-r['values']['Mean_'+bb]['v'])for r in rows]
        stats[key]={k:paired([(100 if k!='boundary_MAE'else 1)*(r['values']['Mean_'+aa][k]-r['values']['Mean_'+bb][k])for r in rows])for k in ['v','t','s','boundary_MAE']}
        stats[key]['tails']=tails(delta,aliases)
        expect[key]={k:paired([100*(r['expected_candidate_quality'][aa][k]-r['expected_candidate_quality'][bb][k])for r in rows])for k in ['v','t']}
    net=stats['CurrentState8_minus_HiResAuxMix']['v'];location=stats['CurrentState8_minus_Uniform8']['v'];refresh=stats['CurrentState8_minus_InitialState8']['v']
    decision=dict(primary_over_Strong=net,position_value_over_Uniform=location,refresh_over_initial_state=refresh,
        state_selection_candidate=net['mean']>1e-10 and location['mean']>1e-10,
        refresh_candidate=net['mean']>1e-10 and location['mean']>1e-10 and refresh['mean']>1e-10,
        not_OPD=True,not_independent_confirmation=True,no_parameter_sweep=True,no_automatic_next_experiment=True,
        scope='cached-feature partial observation; not fresh RGB acquisition or deployment',goal_achieved=False)
    changes=dict(rows=state_changes,mean_first4_L1=avg([r['first4_L1']for r in state_changes]),
        changed_round2_final_sets=sum(r['round2_final_set_changed']for r in state_changes),same_first4_all128=True,
        mean_current_unique_frames=avg([r['Current_unique_frames']for r in state_changes]))
    put(P/'RESULTS_PUBLIC.json',dict(status='completed_cached_feature_acquisition_probe_not_OPD',parents=128,seeds=p['seeds'],means=means,
        paired=stats,expected_quality=expect,rows=rows,spatial_frozen=True))
    put(P/'DECISION_PUBLIC.json',decision);put(P/'STATE_CHANGE_PUBLIC.json',changes)
    audit.update(new_GT=True,official_Dense_output_checks=officialchecks,old_metrics_exact=oldmetric,GT_provenance=prov,
        all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),scalar_candidate_quality_calls=798720)
    assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
    cost=dict(**read(D/'COST.json'),CPU_audit_and_scoring_seconds=time.perf_counter()-beg,publication_and_archive_excluded=True)
    put(P/'COST_PUBLIC.json',cost)
    lines=['# Current state guided observation: cached local-feature probe','','HC128 exposed unique parents; Source Vid frozen, same65/2080/MBR and fixed spatial E/Live41/42/43. Eight revealed cached Dense nodes in2x4 rounds, no new encoding/RGB/expert/training. Existing frozen fusion/full decoder/head replay is charged. Actual Strong preserved:0.5 old Source448 +0.5 mean6 current224 second-pass posteriors. No posterior weight sweep.','','Selection proxy is posterior risk-weighted boundary coverage, NOT calibrated actual observation value or event probability. CurrentState uses its real after4 posterior for round2; InitialState holds initial q only for selection, not output. Reflected selection uses wrong phase; actual feedback/output are never reflected. Uniform covers fixed8. Identical masks may reuse exact deterministic suffix results; logical observations equal8 but actual joint experiment replays/cache hits disclosed.','','| Output | full vIoU percent | tIoU percent | boundary MAE frames |','|---|---:|---:|---:|']
    for arm,v in means.items():lines.append(f"| {arm} | {100*v['v']:.6f} | {100*v['t']:.6f} | {v['boundary_MAE']:.6f} |")
    lines+=['','| Comparison | full-v pp [CI95] | tIoU pp [CI95] | losses<-5pp |','|---|---|---|---:|']
    for k,v in stats.items():lines.append(f"| {k} | {v['v']['mean']:+.6f} {v['v']['CI95']} | {v['t']['mean']:+.6f} {v['t']['CI95']} | {v['tails']['below_minus5']} |")
    lines+=['','```json',json.dumps(decision,indent=2),json.dumps({k:v for k,v in changes.items()if k!='rows'},indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','All128 predictions/reveal histories sealed before GT; all parents/seed41,42,43 retained, seeds averaged within parent, paired parent bootstrap10000. First4-stage metrics and all final signed outputs, selected physical-node IDs, severe tails are in RESULTS_PUBLIC. Candidate quality is offline diagnostic and not a substitute for actual full-v. Acquisition comparisons use identical local clip budget; full global original inputs remain. Cached fixed-B8 local features are not proof of exact fresh sparse clip packing;64 real hypothetical RGB slots/128 padded slots are not measured encoder time. This is adaptive inference/observation capability, no optimized student policy state, no OPD contribution. No training/mixture/observation-budget scan or automatic confirmation/new experiment. Spatial/CURRENT/MAIN/PAPER unchanged.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='repeatedly used HC128 development',excluded=['parentIDs','captions','media','GT coordinates','weights','raw features']))
    assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);guard()
    print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':
    def timeout(signum,frame):raise TimeoutError('finite900s cached-acquisition audit/score')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:put(D/f'SCORING_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:lock.close()
