"""CPU-only residual ceilings with official dense readout verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.tastvg_oracle_event5_common_v1 import *
import numpy as np,collections
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,physical
def summarize(rows,fields):
    result={}
    for group in ['corruption','clean']:
        result[group]={}
        for sub in ['all','expert','nonexpert']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
            usable=[f for f in fields if rr and all(f in r for r in rr)]
            z=source_summary(rr,usable)
            if rr and 'H_selection' in usable:
                gap=z['metrics']['H_selection']['mean'];gain=z['metrics']['T_recovered']['mean']
                z['T_recovery']=dict(ratio_of_macro_means=gain/gap if gap>1e-12 else None,
                    recovered=gain,available=gap,zero_selection_gap_cells=sum(r['H_selection']<=1e-12 for r in rr))
            result[group][sub]=z
    return result
def run():
    import torch
    torch.set_num_threads(2);verified(include_GT=True);tick=time.monotonic();checks=collections.Counter();maxerr=0.;total=0
    fields=['A_v','GT_time_v','GT_space_v','Joint_GT_v','H_temporal','H_spatial','H_joint',
        'temporal_oracle_v','spatial_oracle_v','joint_oracle_v','T_v','H_selection','H_coverage',
        'H_spatial_selection','H_spatial_coverage','joint_minus_temporal','joint_minus_spatial',
        'joint_minus_best_single','joint_synergy','T_recovered','T_remaining']
    for ds in DATASETS:
        p=plan(ds)
        for split in SPLITS:
            labels=read(POOL/ds/f'GT_LABELS_{split}.json');rows=[]
            # Existing predictions were sealed and published before any new GT use.
            for c in [c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==split]:
                x=cached_payload(c);row=p['rows'][c['parent']];g=labels[str(c['parent'])]
                truth={int(k):v for k,v in g['truth'].items()};span=g['span'];b=x['A']['boxes'];ia=x['A']['physical_interval']
                scorer=DenseTube(b,row,truth,span,ds=='hc2');a=scorer.score(ia);gt=scorer.score(span)
                r=dict(dataset=ds,split=split,parent=c['parent'],source_id=c['parent'],condition=c['condition'],
                    order=c['order'],arrival=c['arrival'],expert_scheduled=c['scheduled'],A_v=a['v'],A_t=a['t'],A_s=a['s'],GT_time_v=gt['v'])
                measured=[(b,ia,a,False),(b,span,gt,False)]
                space=official(None,row,truth,span,ia,ds,True);joint=official(None,row,truth,span,span,ds,True)
                assert abs(joint['v']-1)<1e-12 and abs(joint['t']-1)<1e-12
                assert abs(space['v']-a['t'])<1e-12
                r.update(GT_space_v=space['v'],Joint_GT_v=joint['v'],H_temporal=gt['v']-a['v'],
                    H_spatial=space['v']-a['v'],H_joint=joint['v']-a['v'],
                    GT_event_frames=len([f for f in truth if span[0]<=f<span[1]]),
                    GT_frames_outside_sample_support=sum(not scorer.sample_support[0]<=f<=scorer.sample_support[1] for f in truth))
                assert r['H_temporal']>=-1e-12 and r['H_spatial']>=-1e-12
                if c['scheduled']:
                    intervals=[z['physical_interval'] for z in x['temporal_candidates']];targets=x['probes']
                    assert 0<len(intervals)<=8 and len(targets)==9 and ia in intervals
                    scores=[DenseTube(z['boxes'],row,truth,span,ds=='hc2') for z in targets]
                    matrix=np.array([[s.score(i)['v'] for s in scores] for i in intervals])
                    temporal=matrix[:,0];spatial=np.array([s.score(ia)['v'] for s in scores])
                    jt,js=np.unravel_index(np.argmax(matrix),matrix.shape);it=int(np.argmax(temporal));is_=int(np.argmax(spatial))
                    view=read(OLD/ds/'views'/f'{prefix(c)}.json');T=scorer.score(intervals[view['rule']['selected']])
                    r.update(temporal_candidate_v=temporal.tolist(),spatial_candidate_v=spatial.tolist(),
                        joint_candidate_v=matrix.tolist(),temporal_oracle_v=float(temporal[it]),spatial_oracle_v=float(spatial[is_]),
                        joint_oracle_v=float(matrix[jt,js]),T_v=T['v'],T_t=T['t'],temporal_oracle_index=it,
                        spatial_oracle_index=is_,joint_oracle_indices=[int(jt),int(js)],T_index=view['rule']['selected'],
                        temporal_candidate_count=len(intervals),spatial_candidate_count=9,
                        temporal_unique=len(set(map(tuple,intervals))),spatial_unique=len({z['boxes'].numpy().tobytes() for z in targets}))
                    r.update(H_selection=r['temporal_oracle_v']-a['v'],H_coverage=gt['v']-r['temporal_oracle_v'],
                        H_spatial_selection=r['spatial_oracle_v']-a['v'],H_spatial_coverage=space['v']-r['spatial_oracle_v'],
                        joint_minus_temporal=r['joint_oracle_v']-r['temporal_oracle_v'],
                        joint_minus_spatial=r['joint_oracle_v']-r['spatial_oracle_v'],
                        joint_minus_best_single=r['joint_oracle_v']-max(r['temporal_oracle_v'],r['spatial_oracle_v']),
                        joint_synergy=r['joint_oracle_v']-r['temporal_oracle_v']-r['spatial_oracle_v']+a['v'],
                        T_recovered=T['v']-a['v'],T_remaining=r['temporal_oracle_v']-T['v'])
                    for f in ['H_selection','H_coverage','H_spatial_selection','H_spatial_coverage','joint_minus_temporal',
                              'joint_minus_spatial','T_remaining']:assert r[f]>=-1e-12,(ds,c,f,r[f])
                    assert abs(r['H_temporal']-r['H_selection']-r['H_coverage'])<1e-12
                    measured += [(targets[0]['boxes'],intervals[it],scores[0].score(intervals[it]),False),
                        (targets[is_]['boxes'],ia,scores[is_].score(ia),False),
                        (targets[js]['boxes'],intervals[jt],scores[js].score(intervals[jt]),False),(b,intervals[view['rule']['selected']],T,False)]
                    # Full matrix official check on a fixed, pre-specified first clean donor per ds/split.
                    if c['condition']=='clean' and c['order']=='order1' and c['arrival']==0:
                        measured += [(z['boxes'],i,s.score(i),False) for i in intervals for z,s in zip(targets,scores)]
                    checks['candidate_cells']+=1;checks['candidate_combinations']+=matrix.size
                for boxes,interval,z,gs in measured:
                    m=official(boxes,row,truth,span,interval,ds,gs)
                    for k in ['v','t','s']:
                        er=abs(m[k]-z[k]);assert er<1e-10,(ds,c,k,er);maxerr=max(maxerr,er);checks['official_scalar_checks']+=1
                rows.append(r);total+=1
                if total%48==0:print('CPU_ORACLE',total,1152,flush=True)
            out=PUBLIC/'experiment1'/split/ds
            write(out/'ORACLE_ROWS.json',rows);write(out/'SUMMARY.json',summarize(rows,fields))
            write(out/'CASES.json',{f:dict(high=sorted(rows,key=lambda r:-r.get(f,0))[:5],
                low=sorted(rows,key=lambda r:r.get(f,0))[:5]) for f in ['H_temporal','H_spatial','H_selection','H_coverage','H_spatial_selection','joint_minus_best_single']})
    assert total==1152 and checks['candidate_cells']==288 and not torch.cuda.is_initialized()
    verified();audit=dict(status='pass',arrivals=total,checks=dict(checks),max_official_error=maxerr,
        CUDA_initialized=False,new_model_calls=0,worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'EXPERIMENT1_ROOT_AUDIT.json',audit);write(PUBLIC/'experiment1/ROOT_AUDIT.json',audit)
    status(BASE/'STATUS.json',dict(status='CPU_oracle_completed_pending_event5',time=time.time()))
    archive('实验一1152全流及288专家候选上限已CPU完成，JointGT与官方dense对照通过，尚无GT-event新增调用')
    print('EXPERIMENT1_COMPLETE',audit,flush=True)
if __name__=='__main__':run()
