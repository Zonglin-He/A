"""Source-calibrate and seal four CPU-only output interventions before target GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read, put, load, save, sha
from vg_tta.stvg_error_state_correction_v1 import encode
from vg_tta.stvg_joint_robust_correction32_v1 import estimate, apply, contract
D = ROOT/'artifacts/stvg_joint_robust_correction32_v1'
P = ROOT/'results/stvg_error_state/2026-10-18/joint_robust32'
OLD = ROOT/'artifacts/stvg_dino_rollback32_v1'
ERROR = ROOT/'artifacts/stvg_error_state32_v1'
CRITIC = ROOT/'artifacts/stvg_joint_event_critic_v1'
BASES = ['Before', 'Source']
ARMS = ['A_mean', 'B_robust_top1', 'C_joint', 'D_C_center']


def input_pool(expert):
    """Original admitted frame set; alternatives from ORIGINAL retained probes."""
    frames = []
    for a in expert['anchors']['single4']:
        pos = int(a['position']); probe = expert['observations'][('original', pos)]['probe']
        b = np.asarray(probe['boxes'], dtype=np.float64).reshape(-1, 4)
        s = np.asarray(probe['target_scores'], dtype=np.float64)
        assert probe['accepted'] and 0 < len(b) <= 3 and np.isfinite(b).all() and (b[:, 2:] > 0).all()
        winner = int(s.argmax()); assert np.array_equal(b[winner], np.asarray(a['box']))
        frames.append(dict(position=pos, boxes=b, scores=s, candidate_ids=list(probe['candidate_ids']),
                           top1=winner, original_frame_accepted=True))
    return frames


def prepare():
    import torch
    assert not (D/'PLAN_LOCK.json').exists()
    old = read(OLD/'PLAN_LOCK.json'); cp = read(CRITIC/'PLAN_LOCK.json')
    inp = {}; baseline_checks = 0; counts = {}
    def bind(f):
        load(f); inp[str(f.relative_to(ROOT))] = sha(f)
    for r in old['records']:
        a = r['alias']
        for f in [OLD/'predictions'/(a+'.pt'), ERROR/'dino_full'/(a+'.pt'),
                  ROOT/'artifacts/stvg_spatial_readout32_v1/predictions'/(a+'.pt')]: bind(f)
        z = load(OLD/'predictions'/(a+'.pt')); x = load(ERROR/'dino_full'/(a+'.pt'))
        rec = load(ROOT/'artifacts/stvg_spatial_readout32_v1/predictions'/(a+'.pt'))
        frames = input_pool(z['expert']); assert z['expert']['positions4'] == x['planned_positions'] == rec['old_expert']['positions4']
        assert [f['position'] for f in frames] == x['admitted_positions']
        assert np.array_equal(x['bases']['Before'], z['fit']['before'].double().numpy())
        for f in frames:
            op = rec['old_expert']['observations'][('original', f['position'])]['probe']
            assert np.array_equal(f['boxes'], np.asarray(op['boxes']))
            assert np.array_equal(f['scores'], np.asarray(op['target_scores']))
            assert f['candidate_ids'] == op['candidate_ids']
            baseline_checks += 1
        counts.setdefault(r['dataset'], []).append(dict(alias=a, frames=len(frames), candidates=sum(len(f['boxes']) for f in frames)))
    source = [r for r in cp['source'] if r['split'] == 'source_train']
    assert len(source) == 72
    for ds in ['hc2', 'vidstg']:
        assert len({r['parent'] for r in source if r['dataset'] == ds}) == 36
        assert not ({r['parent'] for r in cp['source'] if r['dataset'] == ds} &
                    {r['parent'] for r in cp['target'] if r['dataset'] == ds})
    for r in source:
        bind(CRITIC/'source'/(r['alias']+'.pt'))
        bind(ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt'))
    own = ['vg_tta/stvg_joint_robust_correction32_v1.py', 'scripts/run_stvg_joint_robust_correction32_v1.py',
           'scripts/score_stvg_joint_robust_correction32_v1.py', 'protocols/stvg_joint_robust_correction32_v1.md',
           'vg_tta/stvg_error_state_correction_v1.py', 'vg_tta/tastvg_oracle_event5_v1.py']
    put(D/'PLAN_LOCK.json', dict(status='locked_before_calibration_and_target_outputs', records=old['records'], source=source,
        count=32, arms=ARMS, bases=BASES, primary='Before/final_MBR', input_support='original admitted DINO_full frames only',
        calibration='opposite source-train36 only; query-balanced within-query true-correction RMS; parent-balanced95pct cutoff',
        empirical_cutoff_quantile=.95, scale_numerical_floor=1e-8, no_proximity_prior=True, no_new_query_rejection_gate=True,
        created_epoch=time.time(), CPU_cap_seconds=3600, target_GT_before_prediction_seal=False,
        new_models=0, new_experts=0, new_training=0, new_Student_updates=0, no_automatic_followup=True))
    put(D/'RUNTIME_LOCK.json', dict(pins={f:sha(ROOT/f) for f in own}, inputs=inp,
        registries={f:sha(ROOT/f) for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},
        source_split_plan_sha256=sha(CRITIC/'PLAN_LOCK.json')))
    put(D/'INPUT_READBACK.json', dict(status='pass', count=32, old_Top1_exact_checks=baseline_checks, counts=counts,
        original_probe_and_recapture_old_probe_exact=True, original_output_not_replaced_by_recapture=True,
        source_train_count=72, source_validation_used=0, source_target_parent_overlap=0, target_GT_read=False))
    put(D/'CPU_CONTRACT_COMPLETION.json', contract())
    assert not torch.cuda.is_initialized()


def verify(inputs=False):
    p = read(D/'PLAN_LOCK.json'); l = read(D/'RUNTIME_LOCK.json')
    for f, h in {**l['pins'], **l['registries'], **(l['inputs'] if inputs else {})}.items(): assert sha(ROOT/f) == h, f
    return p


def calibrate():
    import torch
    p = verify(inputs=True); rows = []; result = {}
    for r in p['source']:
        z = load(CRITIC/'source'/(r['alias']+'.pt'))
        y = load(ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt'))
        assert z['split'] == y['split'] == 'source_train' and y['source_GT']
        pos = [i for i in z['expert']['positions4'] if z['frame_ids'][i] in y['dense_source_GT']]
        if len(pos) < 2:
            rows.append(dict(alias=r['alias'], dataset=r['dataset'], anchors=len(pos), eligible=False)); continue
        xy = np.array([y['dense_source_GT'][z['frame_ids'][i]] for i in pos], dtype=float)
        xy /= np.array([z['input']['width'], z['input']['height']]*2)
        b = np.column_stack(((xy[:, :2]+xy[:, 2:])/2, xy[:, 2:]-xy[:, :2]))
        residual = encode(b)-encode(z['boxes'][0].double().numpy())[pos]
        centered = residual-residual.mean(0)
        rows.append(dict(alias=r['alias'], dataset=r['dataset'], anchors=len(pos), eligible=True,
                         centered=centered, model_hash=z['model_hash']))
    for ds in ['hc2', 'vidstg']:
        rr = [r for r in rows if r['dataset'] == ds and r['eligible']]
        scale = np.sqrt(np.mean([np.mean(r['centered']**2, axis=0) for r in rr], axis=0))
        scale = np.maximum(scale, p['scale_numerical_floor']); vals = []; weights = []
        for r in rr:
            v = np.sum((r['centered']/scale)**2, axis=1)
            vals.extend(v); weights.extend([1./len(v)]*len(v))
        order = np.argsort(vals, kind='stable'); v = np.asarray(vals)[order]; w = np.asarray(weights)[order]
        k = int(np.searchsorted(np.cumsum(w)/w.sum(), .95, side='left')); cutoff = float(v[k])
        assert cutoff > 0
        hashes = {r['model_hash'] for r in rr}; assert len(hashes) == 1
        result[ds] = dict(scale=scale.tolist(), truncation=cutoff, source_train_queries=36,
            eligible_queries=len(rr), known_planned_anchors=sum(r['anchors'] for r in rr),
            singleton_or_empty_queries=sum(not r['eligible'] for r in rows if r['dataset']==ds),
            model_hash=next(iter(hashes)), quantile=.95, validation_used=False, target_GT_used=False)
    serial = [{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in rows]
    put(D/'SOURCE_CALIBRATION.json',dict(status='locked_before_target_outputs', source_domains=result,
        source_rows=serial, target_to_source=dict(hc2='vidstg',vidstg='hc2'), new_training=0, target_GT_used=False))
    put(D/'STATUS.json',dict(status='running',stage='source_calibration_locked_target_seal_pending'),True)
    assert not torch.cuda.is_initialized()
    print('SOURCE_CALIBRATION', {k:{kk:vv for kk,vv in v.items() if kk!='model_hash'} for k,v in result.items()},flush=True)


def measure():
    import torch
    torch.set_num_threads(4); p=verify(inputs=True); cal=read(D/'SOURCE_CALIBRATION.json')
    assert cal['status']=='locked_before_target_outputs' and not cal['target_GT_used']
    files=[]; started=time.perf_counter(); exact=0
    for i,r in enumerate(p['records']):
        a=r['alias']; z=load(OLD/'predictions'/(a+'.pt')); x=load(ERROR/'dino_full'/(a+'.pt'))
        frames=input_pool(z['expert']); settings=cal['source_domains'][cal['target_to_source'][r['dataset']]]
        scale=np.asarray(settings['scale']); cutoff=settings['truncation']; bases={}
        for base in BASES:
            b=x['bases'][base]; zb=encode(b); pools=[encode(f['boxes'])-zb[f['position']] for f in frames]
            top=[v[f['top1']][None] for v,f in zip(pools,frames)]
            delta=np.mean(np.concatenate(top),axis=0) if top else np.zeros(4)
            A=apply(b,delta); assert np.array_equal(A,x['corrections'][base]['global_error']);exact+=1
            B=estimate(top,scale,cutoff); C=estimate(pools,scale,cutoff)
            Ddelta=C['delta'].copy();Ddelta[2:]=0.
            out=dict(A_mean=A,B_robust_top1=apply(b,B['delta']),C_joint=apply(b,C['delta']),D_C_center=apply(b,Ddelta))
            out['D_C_center'][:,2:]=b[:,2:]  # Identity on size, including floating representation.
            assert np.array_equal(out['D_C_center'][:,:2],out['C_joint'][:,:2])
            assert np.allclose(out['D_C_center'][:,2:],b[:,2:],rtol=0,atol=1e-15)
            for name,est,pp in [('B_robust_top1',B,top),('C_joint',C,pools)]:
                selected=[v[k] for v,k in zip(pp,est['selected']) if k>=0]
                if selected:assert np.max(abs(np.mean(selected,axis=0)-est['delta']))<1e-10
            bases[base]=dict(Base=b.copy(),outputs=out,delta_A=delta,estimate_B=B,estimate_C=C,
                             delta_D=Ddelta,C_and_D_share_selection_exact=True)
        frames_saved=[{k:v for k,v in f.items()}for f in frames]
        files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=x['frame_ids'],
            interval=x['intervals']['final_MBR'],planned_positions=x['planned_positions'],frames=frames_saved,
            bases=bases,source_calibration_domain=cal['target_to_source'][r['dataset']],target_GT_read=False)))
        assert time.perf_counter()-started<p['CPU_cap_seconds'];print('SEALED',a,i+1,flush=True)
    verify(inputs=True);assert not torch.cuda.is_initialized()
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,
        source_calibration_sha256=sha(D/'SOURCE_CALIBRATION.json'),all_target_outputs_before_new_target_GT=True,
        baseline_bit_exact_checks=exact,D_shared_C_selection=True,new_GPU_model_expert_training_Student_updates=0,
        CPU_generation_seconds=time.perf_counter()-started))
    put(D/'STATUS.json',dict(status='running',stage='all32_sealed_target_score_pending'),True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','calibrate','measure']);s=ap.parse_args().stage
    try:globals()[s]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=s,type=type(e).__name__,error=str(e),partial_preserved=True));raise
