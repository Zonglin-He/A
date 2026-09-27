"""Bounded real-M spatial coverage and temporal iterate diagnostics, TA only."""
import argparse
import collections
import fcntl
import gc
import hashlib
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import load, read, save, sha, status, write

OUT = ROOT / 'artifacts/decota_corrective_iteration_v1/probe_v1'
FULL = ROOT / 'artifacts/decota_full_case_iteration_v1'
SOURCE = ROOT / 'artifacts/stvg_fullscale_diagnostics_v1'
COHORTS = ('hcstvg1_test', 'vidstg_test')
PINS = ['scripts/run_decota_corrective_probe_v1.py', 'vg_tta/corrective_probe_v1.py',
        'protocols/decota_corrective_probe_v1.md', 'methods/decota_v1/api.py',
        'vg_tta/decota_tastvg_episode_v1.py', 'vg_tta/tastvg_fullspan_multistep.py',
        'vg_tta/tastvg_baseline_expansion.py', 'vg_tta/tg_spatial_tta_v1.py',
        'methods/decota_refine_uniform_v1/configs.json', 'scripts/score_stvg_fullscale_v1.py']


def rank(value):
    return hashlib.sha256(('corrective-probe-v1:' + str(value)).encode()).hexdigest()


def guard(deadline=None):
    if shutil.disk_usage(ROOT).free < 40 * 2**30:
        raise RuntimeError('Disk below 40GiB; preserve artifacts')
    if deadline is not None and time.time() > deadline:
        raise TimeoutError('Finite substep budget reached; preserve artifacts')


def plan():
    from scripts.run_decota_full_case_iteration_v1 import verify
    from scripts.run_stvg_fullscale_v1 import plan as source_plan
    verify()
    assert read(FULL/'queue_complete.json')['full_reference_done']
    for cohort in COHORTS:
        c = read(FULL/'analysis'/cohort/'complete.json')
        for k, f in [('summary_sha256','summary.json'), ('rows_sha256','rows.json')]:
            assert c[k] == sha(FULL/'analysis'/cohort/f)
    if (OUT/'lock.json').exists():
        p = read(OUT/'lock.json')
        for f, h in p['pins'].items():
            assert sha(ROOT/f) == h, f
        return p
    src = source_plan()
    cases = read(ROOT/'artifacts/decota_corrective_iteration_v1/E0_full_reference/case_selection.json')
    cohorts = {}
    for c in COHORTS:
        mapping = {r['key']: r for r in src['rows'][c]}
        chosen = [(mapping[r['key']], 'reviewed_cases') for r in cases[c]]
        excluded = {r['input']['source'] for r, _ in chosen}
        available = collections.defaultdict(list)
        for r in src['rows'][c]:
            if r['input_unavailable'] is None and r['input']['source'] not in excluded:
                available[r['input']['source']].append(r)
        for s in sorted(available, key=rank)[:16]:
            chosen.append((min(available[s], key=lambda r: rank(r['key'])), 'source_reference'))
        assert len(chosen) == len({r['input']['source'] for r, _ in chosen}) == 21
        rows = []
        for r, split in chosen:
            f = SOURCE/'method/tastvg'/c/f"{r['ordinal']:06d}.pt"
            receipt = read(f.with_suffix('.json'))
            assert sha(f) == receipt['sha256']
            rows.append(dict(key=r['key'], source=r['input']['source'], split=split,
                             query_type=r['query_type'], input=r['input'],
                             cache_path=str(f), cache_sha256=receipt['sha256']))
        cohorts[c] = rows
    p = dict(version='decota_corrective_probe_v1', created=time.time(),
             protocol='protocols/decota_corrective_probe_v1.md',
             source_lock_sha256=sha(SOURCE/'lock.json'), cohorts=cohorts,
             group_mapping=src['legacy_checkpoint_mapping'],
             pins={f:sha(ROOT/f) for f in PINS}, steps=[0,1,2,3,4,5],
             gpu_budget_seconds=3600, no_GT_online=True, production_changed=False,
             max_candidates=[1,3,5], reference_sources_per_direction=16,
             diagnostic_cases_per_direction=5, source_balanced_selection=True,
             original_test_development_exposed=True)
    write(OUT/'lock.json', p)
    return p


def spatial():
    import torch
    from vg_tta.corrective_probe_v1 import post_nms_pool
    from vg_tta.tg_spatial_tta_v1 import choose_candidate
    p = plan(); deadline = time.time()+600; data=[]
    for c, rows in p['cohorts'].items():
        for row in rows:
            guard(deadline); x = load(row['cache_path']); assert not x['GT_used']
            probes=[]
            for z in x['expert']:
                replay=choose_candidate(z['all_boxes'], z['all_phrase_scores'], x['config'])
                for k in ('accepted','reason','margin','box','score','top_boxes','top_scores'):
                    assert replay[k] == z[k], (row['key'],k)
                pool=post_nms_pool(z['all_boxes'],z['all_phrase_scores'],x['config']['nms_iou'])
                assert not pool['boxes'] or pool['boxes'][0] == replay['box']
                probes.append(dict(position=z['position'],frame_id=z['frame_id'],
                    accepted=z['accepted'],reason=z['reason'],margin=z['margin'],phrase=z['text'],pool=pool))
            data.append(dict(key=row['key'],cohort=c,source=row['source'],split=row['split'],
                             query_type=row['query_type'],probes=probes,config=x['config'],
                             called=len(probes),selected_positions=x['keyframes'],
                             cache_path=row['cache_path'],cache_sha256=row['cache_sha256']))
    save(OUT/'spatial/pools.pt',dict(rows=data,GT_used=False,created=time.time()))
    write(OUT/'spatial/barrier.json',dict(queries=len(data),pools_sha256=sha(OUT/'spatial/pools.pt'),
          lock_sha256=sha(OUT/'lock.json'),original_gate_replay_exact=True,GT_used=False,
          new_model_inference=False,created=time.time()))
    print('SPATIAL_POOLS_SEALED', len(data), flush=True)


def temporal():
    import torch
    from scripts.run_decota_refine_v1 import configure, student
    from scripts.run_stvg_fullscale_v1 import query_parser
    from vg_tta.exact_frame_decode_audit_v2 import decode as decode_pixels
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.decota_tastvg_episode_v1 import forward, make_batch, fitted_merge, native_view_indices
    from methods.decota_v1 import fit, decode
    from vg_tta.corrective_probe_v1 import choose_loss_iterate
    p=plan(); lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.time();deadline=began+p['gpu_budget_seconds'];configure()
    old=read(ROOT/'artifacts/decota_refine_v1/lock.json');parser=query_parser();receipts=[]
    configs=read(ROOT/'methods/decota_refine_uniform_v1/configs.json')['tastvg']
    for c, rows in p['cohorts'].items():
        group=p['group_mapping'][c]; cfg=configs[group]; m=student(old,'tastvg',group)
        state0=state_digest(m)
        weights={k:v.detach().cpu() for k,v in m.temp_embed.state_dict().items()}
        save(OUT/'temporal'/c/'source_head.pt',weights)
        for row in rows:
            guard(deadline); start=time.perf_counter(); original=load(row['cache_path'])
            raw,ids=decode_pixels(row['input']);assert ids==original['frame_ids']
            subject=parser(row['input']['caption'])['subject']
            batch=make_batch(raw,ids,row['input'],subject,m)
            base,inputs,records=forward(m,batch)
            b0=base['raw_boxes'].float().cpu(); ni=list(base['predicted_indices'])
            assert torch.equal(b0,original['predictions']['frozen']['boxes']), ('native_boxes',row['key'])
            assert ni==list(original['predictions']['frozen']['indices']), ('native_indices',row['key'])
            outputs={};prefix_audits=[];losses=[]
            for steps in p['steps']:
                head,zs,audit=fit(m.temp_embed,inputs,backbone='tastvg',lr=cfg['lr'],steps=steps,gamma=0.)
                extent=fitted_merge(zs,records,ids)
                decoded=decode(base['temporal_logits'],extent,ids,native_indices=ni)
                outputs[f'steps{steps}']=dict(indices=list(decoded['indices']),extent=list(extent),
                    offset_indices=[list(native_view_indices(z)) for z in zs],
                    offset_logits=[z.cpu() for z in zs],decode=decoded,audit=audit)
                losses.append(audit['audit']['final_loss']);prefix_audits.append(audit['steps'])
                if steps==0:
                    assert list(decoded['indices'])==ni
                    assert all(torch.equal(a,b) for a,b in zip(head.parameters(),m.temp_embed.parameters()))
                del head,zs
            for steps, audits in enumerate(prefix_audits):
                assert audits==prefix_audits[-1][:steps], ('prefix_state_mismatch',row['key'],steps)
            assert outputs['steps5']['indices']==list(original['predictions']['temporal_only']['indices']), ('reference_5step_indices',row['key'])
            old_losses=[z['loss_after'] for z in original['temporal_audit']['steps']]
            assert losses[1:]==old_losses, ('reference_5step_losses',row['key'],losses,old_losses)
            head,zs,audit=fit(m.temp_embed,inputs,backbone='tastvg',lr=0.,steps=5,gamma=0.)
            extent=fitted_merge(zs,records,ids);decoded=decode(base['temporal_logits'],extent,ids,native_indices=ni)
            assert list(decoded['indices'])==ni
            assert all(torch.equal(a,b) for a,b in zip(head.parameters(),m.temp_embed.parameters()))
            outputs['lr0']=dict(indices=list(decoded['indices']),audit=audit)
            del head,zs
            chosen=choose_loss_iterate(losses)
            outputs['loss_min']=dict(indices=outputs[f'steps{chosen}']['indices'],selected_step=chosen,
                                    selection_GT_used=False,loss=losses[chosen])
            assert state_digest(m)==state0
            f=OUT/'temporal'/c/(row['key'].split(':')[1]+'.pt')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            save(f,dict(key=row['key'],source=row['source'],split=row['split'],query_type=row['query_type'],
                cohort=c,frame_ids=ids,boxes=b0,native_indices=ni,outputs=outputs,config=cfg,
                head_inputs=[h.cpu() for h in inputs],records=records,native_logits=base['temporal_logits'],
                GT_used=False,source_model_unchanged=True,native_exact=True,reference5_exact=True,
                prefix_exact=True,seconds=seconds,cache_sha256=row['cache_sha256']))
            receipt=dict(key=row['key'],path=str(f),sha256=sha(f),seconds=seconds)
            write(f.with_suffix('.json'),receipt);receipts.append(receipt)
            status(OUT/'temporal/progress.json',dict(run='running',done=len(receipts),total=42,
                last_key=row['key'],last_seconds=seconds,elapsed_seconds=time.time()-began,updated=time.time()))
            print('TEMPORAL',c,len(receipts),42,row['key'],round(seconds,2),'loss_pick',chosen,flush=True)
            del raw,batch,base,inputs,records,outputs,original,b0
            gc.collect();torch.cuda.empty_cache()
        del m;gc.collect();torch.cuda.empty_cache()
    plan();elapsed=time.time()-began
    write(OUT/'temporal/barrier.json',dict(receipts=receipts,queries=len(receipts),GT_used=False,
        production_changed=False,native_reference_exact=True,prefix_exact=True,
        lock_sha256=sha(OUT/'lock.json'),gpu_lease_seconds=elapsed,created=time.time()))
    status(OUT/'temporal/progress.json',dict(run='completed',done=len(receipts),total=42,
                                          elapsed_seconds=elapsed,updated=time.time()))


def score_results():
    import numpy as np
    from scripts.run_stvg_fullscale_v1 import plan as source_plan, label_payload
    from scripts.score_stvg_fullscale_v1 import score, summarize
    from vg_tta.box_stability_diagnostics_v1 import overlap
    p=plan();sb=read(OUT/'spatial/barrier.json');tb=read(OUT/'temporal/barrier.json')
    assert sb['lock_sha256']==tb['lock_sha256']==sha(OUT/'lock.json')
    assert sha(OUT/'spatial/pools.pt')==sb['pools_sha256']
    for r in tb['receipts']: assert sha(r['path'])==r['sha256']
    labels=label_payload(source_plan())
    spatial_rows=[]
    for x in load(OUT/'spatial/pools.pt')['rows']:
        gt=labels[x['key']];old=load(x['cache_path']);obs=[]
        for z in x['probes']:
            i=z['position'];assert old['frame_ids'][i]==z['frame_id']
            if not gt['valid'][i]:continue
            truth=np.array(gt['boxes'][i],float);pool=z['pool'];b=np.array(pool['boxes'],float).reshape(-1,4)
            ious=overlap(b,np.repeat(truth[None],len(b),axis=0)) if len(b) else np.array([])
            q0=float(overlap(np.array(old['predictions']['frozen']['boxes'][i:i+1]),truth[None])[0])
            entry=dict(position=i,frame_id=z['frame_id'],accepted=z['accepted'],reason=z['reason'],
                       native_IoU=q0,top1_IoU=float(ious[0]) if len(ious) else None,variants={})
            for m in p['max_candidates']:
                valid=np.array(pool['valid'][:m],bool)
                score_ok=np.array(pool['scores'][:m])>=x['config']['phrase_threshold']
                for name,mask in [('post_nms',valid),('score_eligible',valid&score_ok)]:
                    best=float(ious[:m][mask].max()) if mask.any() else 0.
                    entry['variants'][f'{name}_M{m}']=dict(best_IoU=best,recall50=float(best>=.5),recall70=float(best>=.7))
            obs.append(entry)
        spatial_rows.append({k:x[k] for k in ('key','cohort','source','split','query_type','called')} |
                            dict(valid_observations=len(obs),observations=obs))
    temporal_rows=[]
    for receipt in tb['receipts']:
        x=load(receipt['path']);gt=labels[x['key']]
        metrics={name:score(x['boxes'],gt,x['frame_ids'],z['indices'])[0] for name,z in x['outputs'].items()}
        assert len({m['sIoU'] for m in metrics.values()})==1
        temporal_rows.append({k:x[k] for k in ('key','cohort','source','split','query_type')} |
            dict(metrics=metrics,losses=[x['outputs'][f'steps{i}']['audit']['audit']['final_loss'] for i in p['steps']],
                 loss_selected_step=x['outputs']['loss_min']['selected_step']))
    result=dict(created=time.time(),run='completed',measurement='valid',claim='local_diagnostic_not_promotion',
                spatial={},temporal={},GT_used_only_offline=True,gpu_lease_seconds=tb['gpu_lease_seconds'])
    for c in COHORTS:
        for split in ('reviewed_cases','source_reference'):
            key=c+'/'+split; rr=[r for r in temporal_rows if r['cohort']==c and r['split']==split];ss=[r['source'] for r in rr]
            result['temporal'][key]=dict(queries=len(rr),sources=len(set(ss)),methods={name:{
                metric:summarize([r['metrics'][name][metric] for r in rr],ss)
                for metric in ('vIoU_corrected','sIoU','tIoU','span')} for name in rr[0]['metrics']},
                delta_from_steps5={name:{metric:summarize([r['metrics'][name][metric]-r['metrics']['steps5'][metric] for r in rr],ss)
                for metric in ('vIoU_corrected','tIoU')} for name in rr[0]['metrics']},
                loss_selected_steps=dict(collections.Counter(r['loss_selected_step'] for r in rr)))
            sr=[r for r in spatial_rows if r['cohort']==c and r['split']==split]
            panels={}
            for panel in ('all_called_valid_GT','original_gate_accepted_valid_GT'):
                byquery=[]
                for r in sr:
                    obs=[o for o in r['observations'] if panel=='all_called_valid_GT' or o['accepted']]
                    if not obs:continue
                    vals={name:{metric:float(np.mean([o['variants'][name][metric] for o in obs]))
                          for metric in ('best_IoU','recall50','recall70')} for name in obs[0]['variants']}
                    byquery.append(dict(key=r['key'],source=r['source'],observations=len(obs),variants=vals))
                panels[panel]=dict(eligible_queries=len(byquery),ineligible_queries=len(sr)-len(byquery),
                    observations=sum(r['observations'] for r in byquery),variants={name:{
                    metric:summarize([r['variants'][name][metric] for r in byquery],[r['source'] for r in byquery])
                    for metric in ('best_IoU','recall50','recall70')} for name in (byquery[0]['variants'] if byquery else {})})
            result['spatial'][key]=dict(queries=len(sr),sources=len({r['source'] for r in sr}),panels=panels)
    write(OUT/'spatial/rows.json',spatial_rows);write(OUT/'temporal/rows.json',temporal_rows)
    write(OUT/'summary.json',result)
    write(OUT/'complete.json',dict(summary_sha256=sha(OUT/'summary.json'),created=time.time(),
        scope='E3a_real_topM_and_E4a_step_probe_only',appearance_complete=False,eight_signals_complete=False,
        production_changed=False))
    print('PROBE_SCORED',str(OUT/'summary.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','spatial','temporal','score'])
    args=parser.parse_args()
    try:
        if args.stage=='prepare':print('PREPARED', {c:len(r) for c,r in plan()['cohorts'].items()},flush=True)
        elif args.stage=='spatial':spatial()
        elif args.stage=='temporal':temporal()
        else:score_results()
    except BaseException as exc:
        target=OUT/f'failure_{args.stage}_{time.time_ns()}.json'
        write(target,dict(run='failed',stage=args.stage,error=str(exc),traceback=traceback.format_exc(),
              artifacts_preserved=True,production_changed=False,created=time.time()))
        raise
