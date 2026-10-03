"""Sealed fixed-box readouts, exhaustive grid oracle and paired source metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *
import numpy as np,collections
from vg_tta.tastvg_temporal_candidate_coverage_v1 import grid_values
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,physical

BASE_FIELDS=['A_v','new_v','actual_gain','A_t','new_t','actual_t_gain','GT_time_v','grid_oracle_v',
    'GT_time_minus_grid','grid_headroom_vs_A','GT_headroom_vs_A','gross_gain','gross_loss']
EXPERT_FIELDS=['old_oracle_v','new_oracle_v','oracle_gain','old_selection_gap','new_selection_gap',
    'old_coverage_gap','new_coverage_gap','grid_minus_old_oracle','grid_minus_new_oracle',
    'old_unique','new_unique','pool_overlap','stratum_fallback_slots','actual_change',
    'native_v','old_fast_gain','new_fast_gain','old_oracle_headroom','new_oracle_headroom',
    'old_candidate_center_range','new_candidate_center_range','old_duration_range','new_duration_range']

def summarize(rows):
    out={}
    for group in ['corruption','clean']:
        out[group]={}
        for sub in ['all','expert','nonexpert']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
            fields=BASE_FIELDS+(EXPERT_FIELDS if sub=='expert' else [])
            z=summary(rr,fields)
            z['counts']=dict(improved=sum(r['actual_gain']>1e-12 for r in rr),
                harmed=sum(r['actual_gain']<-1e-12 for r in rr),
                unchanged=sum(abs(r['actual_gain'])<=1e-12 for r in rr),
                severe_harm_gt5pp=sum(r['actual_gain']<-.05 for r in rr))
            z['correctness']={str(th):dict(correct_to_wrong=sum(r['A_v']>th and r['new_v']<=th for r in rr),
                wrong_to_correct=sum(r['A_v']<=th and r['new_v']>th for r in rr)) for th in [.3,.5]}
            out[group][sub]=z
    return out

def run():
    import torch
    torch.set_num_threads(2);bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['status']=='sealed' and len(bar['files'])==1152
    verify(labels=True);tick=time.monotonic();checks=collections.Counter();maxerr=0.
    write(BASE/'GT_EXPOSURE.json',dict(scope='CPU scoring after both datasets and both panels sealed',
        global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        prediction_barrier_time=bar['time'],new_GT_score_started=time.time(),
        all_inputs_historically_exposed=True,generation_GT_read=False))
    done=0
    for ds in DATASETS:
        p=plan(ds)
        for split in SPLITS:
            labels=read(POOL/ds/f'GT_LABELS_{split}.json');rows=[]
            for c in [c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==split]:
                f=BASE/ds/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                pred=read(f);a=acell(c);x=donor(c);row=p['rows'][c['parent']];ids=row['frame_ids']
                g=labels[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()};span=g['span']
                s=DenseTube(x['A']['boxes'],row,truth,span,ds=='hc2')
                ia=x['A']['physical_interval'];nidx=pred['new_indices'];ni=physical(nidx,ids)
                old=s.score(ia);new=s.score(ni);gt=s.score(span);grid=grid_values(s,ids)
                gv=float(grid['values'][grid['best']]);gi=grid['intervals'][grid['best']]
                r=dict(dataset=ds,split=split,source_id=c['parent'],condition=c['condition'],order=c['order'],
                    arrival=c['arrival'],expert_scheduled=c['scheduled'],A_v=old['v'],A_t=old['t'],
                    new_v=new['v'],new_t=new['t'],actual_gain=new['v']-old['v'],actual_t_gain=new['t']-old['t'],
                    GT_time_v=gt['v'],grid_oracle_v=gv,grid_pair_count=grid['count'],sampled_frames=len(ids),
                    GT_time_minus_grid=gt['v']-gv,grid_headroom_vs_A=gv-old['v'],GT_headroom_vs_A=gt['v']-old['v'],
                    gross_gain=max(new['v']-old['v'],0.),gross_loss=max(old['v']-new['v'],0.),
                    A_state_pre_sha256=pred['persistent_pre_sha'],A_state_post_sha256=pred['persistent_post_sha'])
                assert gv>=max(old['v'],new['v'])-1e-12 and gt['v']>=gv-1e-12
                measured=[(ia,old),(ni,new),(span,gt),(gi,s.score(gi))]
                # Prefix enumeration spot-checks on deterministic, label-independent indices.
                spots=sorted(set([0,grid['best'],grid['count']-1]+np.linspace(0,grid['count']-1,13).astype(int).tolist()))
                for at in spots:
                    er=abs(float(grid['values'][at])-s.score(grid['intervals'][at])['v'])
                    assert er<1e-11;maxerr=max(maxerr,er);checks['prefix_vs_scalar_grid']+=1
                if c['scheduled']:
                    pools={b:pred[b+'_candidates'] for b in ['old','new']};v={};t={}
                    for b,pool in pools.items():
                        assert len(pool)==8
                        values=[s.score(i['physical_interval']) for i in pool]
                        v[b]=[z['v'] for z in values];t[b]=[z['t'] for z in values]
                        measured.extend((q['physical_interval'],z) for q,z in zip(pool,values))
                    od,nd=pred['old_decision'],pred['new_decision']
                    assert abs(v['old'][od['selected']]-old['v'])<1e-12
                    assert abs(v['new'][nd['selected']]-new['v'])<1e-12
                    native=s.score(physical(pred['native_indices'],ids))
                    assert abs(v['old'][0]-native['v'])<1e-12 and abs(v['new'][0]-native['v'])<1e-12
                    length=ids[-1]+1-ids[0]
                    def ranges(pool):
                        q=np.array([i['physical_interval'] for i in pool]);dur=(q[:,1]-q[:,0])/length
                        center=(q.sum(1)/2-ids[0])/length
                        return float(np.ptp(center)),float(np.ptp(dur))
                    oc,ow=ranges(pools['old']);nc,nw=ranges(pools['new'])
                    r.update(old_candidate_v=v['old'],new_candidate_v=v['new'],old_candidate_t=t['old'],new_candidate_t=t['new'],
                        old_scores=od['scores'],new_scores=nd['scores'],old_selected=od['selected'],new_selected=nd['selected'],
                        old_candidate_indices=[q['indices'] for q in pools['old']],new_candidate_indices=[q['indices'] for q in pools['new']],
                        old_oracle_v=max(v['old']),new_oracle_v=max(v['new']),oracle_gain=max(v['new'])-max(v['old']),
                        old_selection_gap=max(v['old'])-old['v'],new_selection_gap=max(v['new'])-new['v'],
                        old_coverage_gap=gt['v']-max(v['old']),new_coverage_gap=gt['v']-max(v['new']),
                        grid_minus_old_oracle=gv-max(v['old']),grid_minus_new_oracle=gv-max(v['new']),
                        old_unique=len({tuple(i['indices']) for i in pools['old']}),new_unique=len({tuple(i['indices']) for i in pools['new']}),
                        pool_overlap=len({tuple(i['indices']) for i in pools['old']}&{tuple(i['indices']) for i in pools['new']}),
                        stratum_fallback_slots=sum(i['fallback'] for i in pools['new']),actual_change=int(ni!=ia),
                        native_v=native['v'],old_fast_gain=old['v']-native['v'],new_fast_gain=new['v']-native['v'],
                        old_oracle_headroom=max(v['old'])-native['v'],new_oracle_headroom=max(v['new'])-native['v'],
                        old_candidate_center_range=oc,new_candidate_center_range=nc,old_duration_range=ow,new_duration_range=nw,
                        source_native_matches_current=pred['source_native_matches_current'])
                    for name in ['old','new']:
                        assert abs(r[name+'_coverage_gap']-r['grid_minus_'+name+'_oracle']-r['GT_time_minus_grid'])<1e-12
                        assert r['grid_minus_'+name+'_oracle']>=-1e-12
                    checks['candidate_cells']+=1
                else:assert ni==ia and r['actual_gain']==0
                for interval,z in measured:
                    q=official(x['A']['boxes'],row,truth,span,interval,ds)
                    for m in ['v','t','s']:
                        er=abs(q[m]-z[m]);assert er<1e-10,(ds,split,c['arrival'],m,er)
                        maxerr=max(maxerr,er);checks['official_scalar_checks']+=1
                rows.append(r);done+=1;checks['enumerated_grid_intervals']+=grid['count']
                if done%96==0:print('SCORE_GRID',done,1152,flush=True)
            out=PUBLIC/split/ds
            write(out/'ROWS.json',rows);write(out/'SUMMARY.json',summarize(rows))
            write(out/'CASES.json',{field:dict(high=sorted([r for r in rows if field in r],key=lambda r:-r[field])[:5],
                low=sorted([r for r in rows if field in r],key=lambda r:r[field])[:5]) for field in
                ['actual_gain','actual_t_gain','GT_time_minus_grid','grid_headroom_vs_A','oracle_gain','grid_minus_old_oracle']})
    assert done==1152 and checks['candidate_cells']==288 and not torch.cuda.is_initialized()
    result=dict(status='pass',arrivals=done,expert_pools=288,checks=dict(checks),max_dense_error=maxerr,
        global_seal_before_GT=True,CUDA_initialized=False,new_model_calls=0,new_expert_calls=0,
        worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'SCORE_ROOT_CHECKS.json',result);write(PUBLIC/'SCORE_ROOT_CHECKS.json',result)
    status(BASE/'STATUS.json',dict(status='scored_pending_independent_root_audit_report_publication',arrivals=done,time=time.time()))
    verify(labels=True);archive('1152全采样网格及288新旧八候选已CPU评分，候选覆盖与端点网格缺口已逐到达分解，待根复核/报告/公开')
    print('CPU_SCORE_COMPLETE',result,flush=True)
if __name__=='__main__':run()
