"""Actual post-global-seal scoring/root of the common-state diagnostic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1';PUB=ROOT/'results/stvg_opd_optimization_geometry/2026-10-12/common_state'
ARMS=['Full','Direct','Direct_capped']

def stats(rows,field):
    by=collections.defaultdict(list)
    for r in rows:
        if r.get(field) is not None:by[r['source_id']].append(r[field])
    vals=np.array([math.fsum(v)/len(v) for v in by.values()]);rng=np.random.default_rng(20261012);boot=[]
    if not len(vals):return dict(parents=0,mean=None,ci95=None)
    for _ in range(100):boot.extend(np.mean(vals[rng.integers(0,len(vals),(100,len(vals)))],1).tolist())
    return dict(parents=len(vals),mean=math.fsum(vals)/len(vals),ci95=np.quantile(boot,[.025,.975]).tolist(),
      negative_parent_tail_5pp=int(np.sum(vals<-.05)),positive_parent_tail_5pp=int(np.sum(vals>.05)),p05=float(np.quantile(vals,.05)))

def run():
    b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['logical_fits']==5376 and not b['GT_read']
    assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    runtime=read(BASE/'GPU_RUNTIME.json')
    for p,h in runtime['pins'].items():assert sha(ROOT/p)==h,p
    write(BASE/'POSTSEAL_CPU_RUNTIME.json',dict(status='locked_after_all_deploy_global_before_GT',
      global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),scorer_sha256=sha(Path(__file__)),GT_read=False,time=time.time()))
    import torch;torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.stvg_opd_paper_later_common_v1 import stage_definition
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from scripts.analyze_stvg_opd_mechanism_review_v1 import partition
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import mathematical_readback
    records=read(BASE/'COMMON_STATE_COHORT.json')['records'];rows=[];checked=0;rounds=0;caps=0;dense=0;hashes=0;source=None;start=time.time()
    PUB.mkdir(parents=True,exist_ok=True)
    for r in records:
        # Exact Full aliases can retain P0 origin stage names. GT roster is
        # the canonical matched P2 condition, never the alias origin name.
        canonical='P2_'+r['dataset']+('_cross_clean' if r['setting'].endswith('/cross_clean') else '_same_5percent')
        if source!=canonical:
            source=canonical;gt,spans,provenance=truths(r['dataset'],stage_definition(source))
        row=read_row(r['dataset'],r['query_ordinal']);fits={};probes=None
        for arm in ARMS:
            p=BASE/'common_state'/f"{r['index']:05}"/(arm+'.pt');rel=str(p.relative_to(ROOT));rc=read(p.with_suffix('.json'))
            assert sha(p)==b['files'][rel]==rc['sha256'] and not rc['GT_read'];hashes+=1
            z=load(p);assert z['arm']==arm and z['index']==r['index'] and z['GT_read'] is False
            m=mathematical_readback(z['fit']);assert m==z['math_audit']['independent'];checked+=m['scalar_checks'];rounds+=len(z['fit']['rounds'])
            fits[arm]=z['fit']
        before=fits['Full']['before'];initial=fits['Full']['initial']
        assert all(torch.equal(f['before'],before) and all(torch.equal(f['initial'][n],initial[n]) for n in initial) for f in fits.values())
        out={k:r[k] for k in ['dataset','source_id','stage','condition','order','arrival','query_ordinal','setting','development']}
        interval=load(ROOT/r['original_Full']['path'])['interval'];q=r['query_ordinal'];metrics={};tubes={}
        for arm,boxes in [('Before',before)]+[(arm,f['final']) for arm,f in fits.items()]:
            v=official(boxes.numpy(),row,gt[q],spans[q],interval,r['dataset']);ind,fids,ious=dense_metric(boxes.numpy(),row,gt[q],spans[q],interval,r['dataset'])
            assert all(abs(v[k]-ind[k])<2e-10 for k in v);dense+=3
            metrics[arm]=v;tubes[arm]=DenseTube(boxes.numpy(),row,gt[q],spans[q],clip=r['dataset']=='hc2')
            for k,val in v.items():out[arm+'_'+k]=val
        for arm in ARMS:out[arm+'_current_v']=out[arm+'_v']-out['Before_v']
        out['Full_minus_Direct_v']=out['Full_v']-out['Direct_v'];out['Full_minus_Direct_capped_v']=out['Full_v']-out['Direct_capped_v']
        out['Direct_capped_minus_Direct_v']=out['Direct_capped_v']-out['Direct_v']
        fids=tubes['Full'].fids;mask=partition(fids,row['frame_ids'],[row['frame_ids'][p] for p in fits['Full']['positions']],interval)
        event=(fids>=spans[q][0])&(fids<spans[q][1])
        for group,select in mask.items():
            for arm in ARMS:
                a=(tubes[arm].iou-tubes['Before'].iou)[select&event]
                out[arm+'_'+group+'_current_IoU']=float(a.mean()) if len(a) else None
        cosines=[d['joint_cosine'] for d in fits['Full']['geometry']['rounds'] if d['joint_cosine'] is not None]
        out['common_parameter_gradient_cosine']=math.fsum(cosines)/len(cosines) if cosines else None
        cap=[h['update']['output_cap'] for h in fits['Direct_capped']['path'][:-1] if h.get('update')];caps+=len(cap)
        ratios=[c['budget_utilization'] for c in cap if c['budget_utilization'] is not None]
        out['Direct_capped_budget_utilization']=math.fsum(ratios)/len(ratios) if ratios else None
        out['Direct_capped_extra_forwards']=fits['Direct_capped']['geometry']['extra_native_forwards'];rows.append(out)
        if r['index']%32==0:
            status(BASE/'POSTSEAL_CPU_STATUS.json',dict(status='actual_global_sealed_CPU_root_scoring',done=len(rows),total=1792,CPU_only=True,GT_read=True,time=time.time()))
            print('COMMON_STATE_ROOT_PROGRESS',len(rows),1792,checked,flush=True)
    assert len(rows)==1792 and hashes==5376
    with gzip.open(PUB/'ROWS.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    fields=[f'{arm}_{k}' for arm in ARMS for k in ['v','current_v']]+['Full_minus_Direct_v','Full_minus_Direct_capped_v','Direct_capped_minus_Direct_v','common_parameter_gradient_cosine','Direct_capped_budget_utilization']
    summary={}
    for setting in sorted({r['setting'] for r in rows}):
        for pop in ['all_historical','excluding_development']:
            rr=[r for r in rows if r['setting']==setting and (pop=='all_historical' or not r['development'])]
            summary[setting+'/'+pop]=dict(cells=len(rr),statistics={f:stats(rr,f) for f in fields},per_order={o:{f:stats([r for r in rr if r['order']==o],f) for f in fields} for o in sorted({r['order'] for r in rr})})
    write(PUB/'SUMMARY.json',dict(status='complete_actual_postseal_common_state',settings=summary,bootstrap=10000,seed=20261012,
      conditional_on_original_Full_history=True,independent_online_stream_effect_not_claimed=True,historically_exposed=True,unadjusted_CI=True,
      Direct_cap_is_upper_budget_not_exact_norm_matching=True,GT_used_for_online_update=False))
    write(BASE/'COMMON_STATE_CPU_COMPLETION.json',dict(status='pending_actual_independent_statistics_PDF_view_publication_archive',
      cells=1792,complete_fits=5376,complete_rounds=rounds,independent_math_scalars=checked,independent_dense_scalars=dense,
      input_and_prediction_hashes=hashes,actual_cap_checks=caps,GT_read=True,CPU_only=True,files={str(p.relative_to(ROOT)):sha(p) for p in PUB.iterdir() if p.is_file()},elapsed_seconds=time.time()-start,time=time.time()))
    print('COMMON_STATE_CPU_COMPLETE',len(rows),checked,dense,flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'postseal_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
