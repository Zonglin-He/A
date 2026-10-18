"""Independent all-output/metric/aggregate audit; no new inference or update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import collections,itertools,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_supported_source256_v1 import D,P,PRE,OLD,verify


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official
    p=verify();assert read(D/'CPU_COMPLETION.json')['count']==256
    assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256']
    bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==256 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in target}
    rs=read(P/'RESULTS.json');by={r['alias']:r for r in rs['rows']};cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');checks=collections.Counter();error=0.;cost=collections.Counter()
    for r in p['records']:
        a=r['alias'];ds,q=r['dataset'],r['query_ordinal'];s=load(D/'source'/(a+'.pt'));z=load(D/'predictions'/(a+'.pt'));base=s['Source'].double().numpy();assert np.array_equal(base,z['outputs']['Source_no_correction']);checks['Source_exact']+=1
        di='vidstg_to_hc2'if ds=='hc2'else'hcstvg2_to_vidstg';inp=torch.load(OLD/'inputs'/di/'clean'/f'{q:05}.pt',map_location='cpu',weights_only=False)
        assert torch.equal(s['Source'],inp['native_boxes']);checks['old_Source_cache_exact']+=1
        def chart(b):return np.column_stack((b[:,:2],np.log(b[:,2:])))
        frames=z['frames'];pools=[chart(f['boxes'])-chart(base)[f['position']]for f in frames];setting=cal['source_domains']['vidstg'if ds=='hc2'else'hc2'];scale=np.array(setting['scale']);cap=setting['truncation'];costs=[np.log(np.maximum(f['scores'],1e-12).max())-np.log(np.maximum(f['scores'],1e-12))for f in frames]
        best=float('inf');n=0
        for assign in itertools.product(*[range(-1,len(v))for v in pools]):
            active=[v[k]for v,k in zip(pools,assign)if k>=0];delta=np.mean(active,0)if active else np.zeros(4)
            value=sum(min(cap,float(np.min(np.sum(((v-delta)/scale)**2,1)+cap/16*c)))for v,c in zip(pools,costs));best=min(best,value);n+=1
        assert abs(best-z['estimate']['objective'])<1e-8;checks['estimator_games']+=1;checks['assignments']+=n
        top=[v[f['top1']]for f,v in zip(frames,pools)]
        for arm,delta in [('A_Top1_mean',np.mean(top,0)if top else np.zeros(4)),('E_DINO_support',z['estimate']['delta'])]:
            expected=np.column_stack((base[:,:2]+delta[:2],base[:,2:]*np.exp(delta[2:])));err=float(abs(expected-z['outputs'][arm]).max());assert err<1e-12;error=max(error,err);checks['corrected_outputs']+=1
        row=plans[ds][q];gt,span=target[ds][0][q],target[ds][1][q];ids=np.array(z['frame_ids']);known=np.array([int(fid)in gt for fid in ids]);unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions'])
        for arm,b in z['outputs'].items():
            met=official(b,row,gt,span,z['interval'],ds);saved=by[a]['values'][arm]
            for k in ['v','t','s']:assert abs(met[k]-saved[k])<2e-10;checks['metrics']+=1
            if unobs.any():
                xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*np.array([row['input']['width'],row['input']['height']]*2)
                if ds=='hc2':xy=np.maximum(xy,0)
                x=xy[unobs];g=np.array([gt[int(fid)]for fid in ids[unobs]]);wh=np.maximum(0,np.minimum(x[:,2:],g[:,2:])-np.maximum(x[:,:2],g[:,:2]));inter=wh.prod(1);union=np.maximum(0,x[:,2:]-x[:,:2]).prod(1)+np.maximum(0,g[:,2:]-g[:,:2]).prod(1)-inter
                observed=float(np.divide(inter,union,out=np.zeros_like(inter),where=union>0).mean());assert abs(observed-saved['unobserved_actual_s'])<2e-10
            else:assert saved['unobserved_actual_s']is None
            checks['unobserved_actual_support']+=1
        for k,v in s['compute'].items():cost[k]+=v
    for ds,summary in rs['summary'].items():
        rr=[r for r in rs['rows']if r['dataset']==ds];assert len(rr)==128
        for arm in p['arms']:
            for ep,v in summary['query_mean'][arm].items():
                vv=[r['values'][arm][ep]for r in rr if r['values'][arm][ep]is not None];expected=sum(vv)/len(vv)if vv else None
                assert v is None and expected is None or v is not None and abs(v-expected)<1e-12;checks['aggregates']+=1
        for control in p['arms'][:2]:
            for ep in ['v','unobserved_actual_s']:
                v=np.array(summary['E_minus'][control][ep]['per_parent_pp']);ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));ci=np.quantile(v[ix].mean(1),[.025,.975]);assert max(abs(ci-summary['E_minus'][control][ep]['CI95']))<1e-12;checks['paired_CIs']+=1
    assert cost['new_Student_updates']==cost['new_DINO_frames']==0
    pipeline=read(D/'EXECUTION_RUNTIME.json');wall=(D/'CPU_COMPLETION.json').stat().st_mtime-pipeline['start_epoch']
    actual=dict(cost,pipeline_capture_predict_score_wall_seconds=wall,CPU_audit_not_in_that_wall=True,
        new_training=0,planned_expert_frames=1024,original_cached_expert_inputs=256,baseline='frozen Source',
        cancelled_historical_recovery=dict(checkpoints_preserved=read(PRE/'CANCELLATION_RECEIPT003.json')['retained_count'],included_in_new_method_latency=False))
    put(P/'ACTUAL_COSTS.json',actual);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=256,checks=dict(checks),max_output_error=error,all_signed_results_retained=True,new_Student_updates=0))
    lines=['# Frozen Source-start E expanded evaluation: HC128 + Vid128','','The exact fixed parent/query roster is unchanged. Targets are historically analyzed; this is expanded evaluation, not independent confirmation. Original E32 beta1/16, geometric calibration, four-frame DINO support, candidate pool and full-trajectory correction are frozen. All three arms share final-distribution MBR.','','## Explicit starting-state change','','The human stopped original online-state recovery and selected a frozen Source start. The cancelled recovery retained all71 completed checkpoints and two Before captures. Its partial inherited states are NOT inputs to this experiment. Source_no_correction is the frozen trajectory; A/E are output corrections of that exact trajectory. This measures frozen-start correction and does not establish incremental improvement over inherited online OPD.','','| Target | Source vIoU % | A vIoU % | E vIoU % | E−Source pp | E−A pp | E−Source unobserved sIoU pp | E−A unobserved sIoU pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds,s in rs['summary'].items():
        v=[100*s['query_mean'][arm]['v']for arm in p['arms']]+[s['E_minus'][c][ep]['mean']for ep,c in [('v','Source_no_correction'),('v','A_Top1_mean'),('unobserved_actual_s','Source_no_correction'),('unobserved_actual_s','A_Top1_mean')]];lines.append('| '+ds+' | '+' | '.join(f'{x:.6f}'for x in v)+' |')
    lines+=['','All256 outputs sealed before new GT scoring. GT/old labels never calibrate or route E. No-reference/no-op/unknown and negative cases remain. Missing annotations are unknown, not failures. Per-parent paired10000 bootstrap CIs, positive/negative/neutral counts, <-5/<-20pp harm tails, leave-one-out sensitivity, equal-domain macro CIs and existing shared-bin motion/area/event-length strata are in RESULTS.json. One query per parent makes query and parent means equal.','','## Cost and limits','',f'Exact full capture reuse: {int(cost["reused_full_capture"])}; new frozen Source forwards: {int(cost["new_Source_forwards"])}. ZERO new DINO frames, optimizer steps or model training. Actual capture/output/score wall and cache work are in ACTUAL_COSTS.json; cancelled recovery is disclosed separately and excluded from method latency.','', 'The original E32 failed two-domain no-regression gate remains. No hyperparameter scan, method promotion, domain-specific routing or dense OPD follows automatically. Original E32 parent overlap is HC11/Vid5; original tuning parents were excluded before sampling.','','Independent exhaustive objective/output and metric/aggregate audits passed. Actual fixed pixel views, privacy/export, remote GitHub content verification and archive closure remain separate required root publication actions.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=256,actual_fixed_pixels_privacy_remote_publication_archive_pending=True))
    put(D/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready_for_root_pixels_privacy_remote_publication_archive',count=256))
    assert not torch.cuda.is_initialized()


if __name__=='__main__':main()
