"""CPU development score only after every candidate in this coordinate seals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *
import numpy as np

def run(index,uid):
    verify();c,cb=coordinate_barrier(index);assert uid in [x['trial'] for x in c['candidates']]
    dest=BASE/'trials'/uid;t=read(dest/'CONFIG.json');barrier=read(dest/'PREDICTION_BARRIER.json')
    assert barrier['config_sha256']==sha(dest/'CONFIG.json') and barrier['cells']==64 and not barrier['GT_read']
    for f,h in barrier['files'].items():
        p=ROOT/f;rc=read(p.with_suffix('.json'));assert sha(p)==h==rc['sha256'] and not rc['GT_read']
        if 'bytes' in rc:assert p.stat().st_size==rc['bytes']
        assert rc['time']<=barrier.get('original_seal_time',barrier['time'])<=cb['time']
    if (dest/'CPU_COMPLETION.json').exists():return
    import torch
    torch.set_num_threads(2)
    from scripts.score_decota_spatial_opd_v1 import truths,source_stats
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
    write(dest/'GT_EXPOSURE.json',dict(coordinate=index,coordinate_barrier_sha256=sha(BASE/'coordinates'/f'{index:02}_{COORDINATES[index]}'/'PREDICTION_BARRIER.json'),
        role='historically_exposed_HC2_development_coordinate_selection',time=time.time()))
    dense,spans,provenance=truths('hc2',t);plan=read(PAPER/'hc2/PLAN.json')
    sources=sorted({plan['rows'][q]['source'] for q in t['parents']});ids={s:i for i,s in enumerate(sources)}
    rr=[];previous={};prevhash={};mathchecks=0;densechecks=0
    for order,at,q,p,inputbase in records(uid):
        z=load(p);fit=z['fit'];row=plan['rows'][q];inpfile=inputbase/z['input']['path'];inp=load(inpfile)
        assert sha(inpfile)==z['input']['sha256']==read(inpfile.with_suffix('.json'))['sha256']
        assert z['parent']==q and z['source']==t['source'] and z['dataset']=='hc2' and not z['GT_read'] and not inp['GT_read']
        assert z['previous_payload_sha256']==prevhash.get(order) and fit['config']==t['config']==z['config']
        assert fit['active_parameters']==1792 and fit['selected_step']==t['config']['steps']
        assert len(fit['path'])==t['config']['steps']+1 and not fit['GT_used']
        assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
        if order in previous:
            assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else previous[order][n]) for n,v in fit['initial'].items())
        else:
            # Every coordinate/config/order starts at the same source LN state.
            ref=load(OLD/'trials/hc2/refine_09'/order/'00000.pt')['fit']['initial']
            assert all(torch.equal(fit['initial'][n],ref[n]) for n in ref)
        assert all(torch.equal(z['committed'][n],v) for n,v in commit(fit['initial'],fit['state'],t['config']['writeback']).items())
        assert audit(fit,unpack_expert(inp['expert']))==z['math_audit'];mathchecks+=1
        values={}
        for name,boxes in [('Frozen',inp['native_boxes']),('Before',fit['before']),('After',fit['final'])]:
            v=official(boxes.numpy(),row,dense[q],spans[q],z['interval'],'hc2')
            independent=DenseTube(boxes.numpy(),row,dense[q],spans[q],clip=True).score(z['interval'])
            assert abs(v['v']-independent['v'])<2e-10 and abs(v['t']-independent['t'])<2e-10
            values[name]=v;densechecks+=1
        assert values['Frozen']['t']==values['Before']['t']==values['After']['t']
        r=dict(source_id=ids[row['source']],query_ordinal=q,order=order,arrival=at,condition='clean',trial=uid,
            delta_total_v=values['After']['v']-values['Frozen']['v'],
            delta_current_v=values['After']['v']-values['Before']['v'],
            delta_inherited_v=values['Before']['v']-values['Frozen']['v'],
            delta_total_s=values['After']['s']-values['Frozen']['s'],
            compute=z['compute'],backward_steps=fit['gradient_calls'],observed_frames=len(fit['positions']),empty=fit['empty'])
        for name,v in values.items():
            for metric,val in v.items():r[name+'_'+metric]=val
        for threshold in [.3,.5]:
            before=values['Frozen']['v']>threshold;after=values['After']['v']>threshold
            r['correct_to_wrong_'+str(threshold)]=float(before and not after)
            r['wrong_to_correct_'+str(threshold)]=float(not before and after)
        assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<3e-15
        rr.append(r);previous[order]=z['committed'];prevhash[order]=sha(p)
    assert len(rr)==64 and len(sources)==32
    stats=source_stats(rr,FIELDS);out=PUB/'trials'/uid;out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/'ROWS.jsonl.gz','wt') as f:
        for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
    summary=dict(dataset='hc2',trial=uid,config=t['config'],statistics=stats,
        development_selection=True,historical_exposure=True,confirmation_used_for_selection=False,
        exact_prediction_history_reused=barrier['exact_history_reused'],
        independent_math_state_arrivals=mathchecks,independent_dense_v_t_comparisons=densechecks,
        GPU_fit_seconds=sum(r['compute']['fit_GPU_seconds'] for r in rr),GT_provenance=provenance,time=time.time())
    write(out/'SUMMARY.json',summary)
    write(dest/'CPU_COMPLETION.json',dict(status='complete_development_score_audit',rows=64,
        rows_sha256=sha(out/'ROWS.jsonl.gz'),summary_sha256=sha(out/'SUMMARY.json'),
        coordinate_barrier_time=cb['time'],GT_exposure_time=read(dest/'GT_EXPOSURE.json')['time'],time=time.time()))
    print('HC2_COORDINATE_CPU_DONE',uid,stats['metrics']['delta_total_v'],flush=True)

if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
