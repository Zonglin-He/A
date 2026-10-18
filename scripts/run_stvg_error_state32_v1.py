"""CPU-only sparse error correction; freeze predictions before dense evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read, put, load, save, sha
D = ROOT/'artifacts/stvg_error_state32_v1'
OLD = ROOT/'artifacts/stvg_dino_rollback32_v1'
P = ROOT/'results/stvg_error_state/2026-10-18/cached32'
BASES = ['Before', 'Source']
REFS = ['DINO_full', 'GT_planned', 'GT_common', 'DINO_common']
ARMS = ['Base', 'LocalAnchor', 'GlobalError']


def prepare():
    from vg_tta.stvg_error_state_correction_v1 import contract
    assert not (D/'PLAN_LOCK.json').exists()
    p = read(OLD/'PLAN_LOCK.json')
    own = ['vg_tta/stvg_error_state_correction_v1.py', 'scripts/run_stvg_error_state32_v1.py',
           'scripts/score_stvg_error_state32_v1.py', 'protocols/stvg_error_state32_v1.md',
           'vg_tta/stvg_posterior_decision_tree_v1.py', 'scripts/stvg_temporal_marginal_decode_util256_v1.py',
           'vg_tta/tastvg_oracle_event5_v1.py']
    inputs = {}
    for r in p['records']:
        for folder in ['source', 'predictions']:
            f = OLD/folder/(r['alias']+'.pt')
            load(f)
            inputs[str(f.relative_to(ROOT))] = sha(f)
    assert all(len({r['parent'] for r in p['records'] if r['dataset']==ds})==16 for ds in ['hc2','vidstg'])
    put(D/'PLAN_LOCK.json', dict(status='locked_before_new_GT_and_outputs',count=32,records=p['records'],
        bases=BASES,reference_sets=REFS,arms=ARMS,primary_base='Before',primary_clock='final_MBR',
        clocks=['final_MBR','Native','projection','marginal_MAP'],new_models=0,new_DINO_calls=0,
        new_Student_updates=0,new_training=0,CPU_cap_seconds=3600,created_epoch=time.time(),
        original_panel_exposed=True,oracle_not_deployable=True,no_automatic_followup=True))
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in own},inputs=inputs,
        registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},
        strata_sha256=sha(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')))
    put(D/'CPU_CONTRACT_COMPLETION.json',contract())
    put(D/'STATUS.json',dict(status='prepared',stage='DINO_full_seal_pending'))


def verify():
    p=read(D/'PLAN_LOCK.json'); l=read(D/'RUNTIME_LOCK.json')
    for f,h in {**l['pins'],**l['registries']}.items():
        assert sha(ROOT/f)==h,f
    return p


def source(alias):
    import numpy as np
    s=load(OLD/'source'/(alias+'.pt')); z=load(OLD/'predictions'/(alias+'.pt'))
    for folder in ['source','predictions']:
        f=OLD/folder/(alias+'.pt');assert sha(f)==read(D/'RUNTIME_LOCK.json')['inputs'][str(f.relative_to(ROOT))]
    xy=np.asarray(s['native_boxes_xyxy_normalized'],float)
    boxes=np.column_stack(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]))
    return s,z,dict(Source=boxes,Before=z['fit']['before'].double().numpy())


def build(bases,positions,targets):
    from vg_tta.stvg_error_state_correction_v1 import correct
    return {name:correct(b,positions,targets)for name,b in bases.items()}


def dino():
    import numpy as np, torch
    from scipy.special import logsumexp
    from scripts.stvg_temporal_marginal_decode_util256_v1 import mass_table,mode
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    torch.set_num_threads(4); p=verify(); files=[]; started=time.perf_counter()
    for r in p['records']:
        s,z,bases=source(r['alias']); ids=list(map(int,s['frame_ids'])); grids=[]; natives=[]
        for off,o in enumerate(z['projection']['offsets']):
            ij=o['ij'].numpy(); lp=o['logp0'].numpy(); prob=np.exp(lp-logsumexp(lp)); fids=ids[off::2]
            assert ij.shape==(2,len(prob)) and abs(prob.sum()-1)<1e-12
            grids.append(dict(ids=fids,ij=ij,p=prob)); a,b=ij[:,lp.argmax()];natives.append([fids[a],fids[b]+1])
        assert [min(x[0]for x in natives),max(x[1]for x in natives)]==s['Native_interval']
        U,q=mass_table(grids); assert U==ids; mb=final_mbr(ids,q); mm,_=mode(ids,q)
        anchors=z['expert']['anchors']['single4']; pos=[int(a['position'])for a in anchors]
        targets=np.array([a['box']for a in anchors],float).reshape(-1,4)
        out=dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=ids,
            intervals=dict(final_MBR=mb['interval'],Native=s['Native_interval'],projection=z['arms']['Frozen_projection']['interval'],marginal_MAP=mm),
            planned_positions=list(z['expert']['positions4']),admitted_positions=pos,reference_positions=pos,
            reference_targets=targets,bases=bases,corrections=build(bases,pos,targets),q=q,
            GT_read=False,source_prediction_sha256=sha(OLD/'predictions'/(r['alias']+'.pt')))
        files.append(save(D/'dino_full'/(r['alias']+'.pt'),out));assert time.perf_counter()-started<3600
    assert not torch.cuda.is_initialized()
    put(D/'DINO_FULL_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,
        new_models=0,new_experts=0,new_updates=0,CPU_wall_seconds=time.perf_counter()-started))
    put(D/'STATUS.json',dict(status='running',stage='DINO_full_sealed_oracle_pending'),True)


def oracle():
    import numpy as np, torch
    from scripts.score_stvg_opd_p1_v1 import truths
    p=verify();bar=read(D/'DINO_FULL_BARRIER.json');assert bar['count']==32 and not bar['GT_read']
    # Full truth files are parsed by the historical loader. Only these sparse
    # locked positions are passed through the prediction-generation interface.
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    files=[];inp=[];started=time.perf_counter()
    for r in p['records']:
        z=load(D/'dino_full'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];gt=targets[ds][0][q]
        scale=np.array([plans[ds][q]['input']['width'],plans[ds][q]['input']['height']]*2)
        planned=[v for v in z['planned_positions']if z['frame_ids'][v]in gt]
        common=[v for v in z['admitted_positions']if z['frame_ids'][v]in gt]
        def boxes(pos):
            xy=np.array([gt[z['frame_ids'][i]]for i in pos],float).reshape(-1,4)/scale
            return np.column_stack(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]))
        dino_common=np.array([z['reference_targets'][z['admitted_positions'].index(i)]for i in common],float).reshape(-1,4)
        refs=dict(GT_planned=(planned,boxes(planned)),GT_common=(common,boxes(common)),DINO_common=(common,dino_common))
        sparse={k:dict(positions=a,targets=b)for k,(a,b)in refs.items()}
        inp.append(save(D/'oracle_sparse_inputs'/(r['alias']+'.pt'),dict(alias=r['alias'],references=sparse,GT_sparse_used=True)))
        # No full truth or annotation data is accepted by build().
        results={name:build(z['bases'],pos,b)for name,(pos,b)in refs.items()}
        files.append(save(D/'oracle'/(r['alias']+'.pt'),dict(alias=r['alias'],references=sparse,corrections=results,
            GT_sparse_used=True,unobserved_GT_used_for_predictions=False)))
        assert time.perf_counter()-started<3600
    assert not torch.cuda.is_initialized()
    put(D/'ORACLE_SPARSE_INPUT_BARRIER.json',dict(status='complete',count=32,files=inp,full_files_parsed=True,
        only_locked_sparse_GT_used_for_predictions=True))
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,
        DINO_full_barrier_sha256=sha(D/'DINO_FULL_BARRIER.json'),GT_sparse_used=True,
        all_predictions_sealed_before_dense_GT_evaluation=True,CPU_wall_seconds=time.perf_counter()-started))
    put(D/'STATUS.json',dict(status='running',stage='all_predictions_sealed_dense_score_pending'),True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','dino','oracle'])
    stage=parser.parse_args().stage
    try:globals()[stage]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=stage,type=type(e).__name__,error=str(e),partial_preserved=True))
        raise
