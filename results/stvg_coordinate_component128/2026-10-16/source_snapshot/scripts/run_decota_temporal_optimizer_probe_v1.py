"""Bounded E4b: exact cached-head replay, single last-step intervention, sealed scoring."""
import argparse
import collections
import fcntl
import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.run_decota_corrective_probe_v1 import plan as prior_plan,OUT as PRIOR,guard
OUT=ROOT/'artifacts/decota_corrective_iteration_v1/temporal_optimizer_v1'
PINS=['scripts/run_decota_temporal_optimizer_probe_v1.py','vg_tta/temporal_optimizer_probe_v1.py',
      'protocols/decota_temporal_optimizer_probe_v1.md','tests/test_temporal_optimizer_probe_v1.py',
      'external/TA-STVG/models/net_utils.py']


def plan():
    old=prior_plan()
    barrier=read(PRIOR/'temporal/barrier.json')
    for r in barrier['receipts']: assert sha(r['path'])==r['sha256']
    audit=read(PRIOR/'independent_audit.json');assert audit['measurement']=='valid'
    if (OUT/'lock.json').exists():
        p=read(OUT/'lock.json')
        for f,h in p['pins'].items():assert sha(f)==h,f
        return p
    pins=[ROOT/f for f in PINS]+[PRIOR/'lock.json',PRIOR/'temporal/barrier.json',PRIOR/'independent_audit.json']
    pins += [PRIOR/'temporal'/c/'source_head.pt' for c in old['cohorts']]
    from vg_tta.temporal_optimizer_probe_v1 import BRANCHES
    p=dict(version='decota_temporal_optimizer_probe_v1',created=time.time(),
           receipts=barrier['receipts'],cohorts=old['cohorts'],branches=BRANCHES,
           pins={str(f):sha(f) for f in pins},GPU_budget_seconds=1800,bootstrap=1000,seed=20260912,
           GT_online=False,production_changed=False,original_test_development_exposed=True)
    write(OUT/'lock.json',p);return p


def make_head(state,device='cuda'):
    import torch
    spec=importlib.util.spec_from_file_location('e4b_official_netutils',ROOT/'external/TA-STVG/models/net_utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    head=module.MLP(state['layers.0.weight'].shape[1],state['layers.0.weight'].shape[0],2,2,dropout=.3)
    head.load_state_dict(state,strict=True)
    return head.to(device).eval().requires_grad_(False)


def decode_output(snapshot,x):
    from methods.decota_v1 import decode
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge,native_view_indices
    z=[v.to('cuda') for v in snapshot['logits']]
    extent=fitted_merge(z,x['records'],x['frame_ids'])
    result=decode(x['native_logits'],extent,x['frame_ids'],native_indices=x['native_indices'])
    return dict(indices=list(result['indices']),extent=list(extent),decode=result,
        offset_indices=[list(native_view_indices(v)) for v in z],
        promoted_offset_indices=[list(native_view_indices(v.float())) for v in z],
        fp32_offset_indices=[list(native_view_indices(v.to('cuda'))) for v in snapshot['fp32_logits']])


def capture(limit=None):
    import torch
    from scripts.run_decota_refine_v1 import configure
    from vg_tta.temporal_optimizer_probe_v1 import probe
    p=plan();configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();run_id=str(time.time_ns());done=[];new_count=0;failure=None
    heads={};counts=collections.Counter()
    try:
        for r in p['receipts']:
            key=r['key'];c,idx=key.split(':');f=OUT/'capture'/c/(idx+'.pt')
            if f.with_suffix('.json').exists():
                rr=read(f.with_suffix('.json'));assert rr['sha256']==sha(f);done.append(rr);continue
            if limit is not None and counts[c]>=limit:continue
            guard(start+p['GPU_budget_seconds']);t=time.perf_counter()
            if c not in heads:heads[c]=make_head(load(PRIOR/'temporal'/c/'source_head.pt'))
            x=load(r['path']);assert not x['GT_used']
            z=probe(heads[c],x['head_inputs'],x['config']['lr'])
            for step,snap in enumerate(z['path']):
                snap['prediction']=decode_output(snap,x)
                old=x['outputs'][f'steps{step}']
                assert snap['prediction']['indices']==old['indices'],(key,step,'indices')
                assert snap['loss']==old['audit']['audit']['final_loss'],(key,step,'loss')
                assert all(torch.equal(a,b) for a,b in zip(snap['logits'],old['offset_logits'])),(key,step,'logits')
            for name,snap in z['branches'].items():snap['prediction']=decode_output(snap,x)
            assert z['selected_best_step']==x['outputs']['loss_min']['selected_step']
            selected=z['path'][z['selected_best_step']]['prediction']['indices']
            assert selected==x['outputs']['loss_min']['indices']
            torch.cuda.synchronize();sec=time.perf_counter()-t
            z.update({k:x[k] for k in ('key','source','cohort','split','query_type','frame_ids','config')})
            z.update(prior_path=r['path'],prior_sha256=r['sha256'],seconds=sec,
                     exact_all_six_prefixes=True,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30)
            save(f,z);rr=dict(key=key,path=str(f),sha256=sha(f),seconds=sec)
            write(f.with_suffix('.json'),rr);done.append(rr);new_count+=1;counts[c]+=1
            status(OUT/'progress.json',dict(run='running',done=len(done),total=42,last=key,seconds=sec))
            print('E4B',len(done),42,key,round(sec,3),'best',z['selected_best_step'],'backtrack',z['selected_backtrack'],flush=True)
            del x,z
        plan()
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        elapsed=time.time()-start
        write(OUT/'leases'/(run_id+'.json'),dict(started=start,seconds=elapsed,new_queries=new_count,
            failure=failure,ended=time.time()))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    if len(done)==42:
        seconds=sum(read(f)['seconds'] for f in (OUT/'leases').glob('*.json'))
        write(OUT/'barrier.json',dict(receipts=done,queries=42,GT_used=False,
            lock_sha256=sha(OUT/'lock.json'),GPU_lease_seconds=seconds,created=time.time()))
        status(OUT/'progress.json',dict(run='completed',done=42,total=42))
    else:print('SMOKE_COMPLETED',len(done),flush=True)


def score_results():
    from scripts.run_stvg_fullscale_v1 import plan as source_plan,label_payload
    from scripts.score_stvg_fullscale_v1 import score
    from vg_tta.corrective_evidence_v1 import source_mean_ci
    p=plan();bar=read(OUT/'barrier.json');assert bar['lock_sha256']==sha(OUT/'lock.json')
    assert bar['queries']==len(bar['receipts'])==42
    labels=label_payload(source_plan());rows=[]
    for receipt in bar['receipts']:
        assert sha(receipt['path'])==receipt['sha256'];x=load(receipt['path']);old=load(x['prior_path'])
        assert x['prior_sha256']==sha(x['prior_path']) and not x['GT_used']
        choices={k:v for k,v in x['branches'].items()}
        choices.update(frozen=x['path'][0],step4=x['path'][4],
            best_iterate=x['path'][x['selected_best_step']],last_backtrack=x['branches'][x['selected_backtrack']])
        metrics={k:score(old['boxes'],labels[x['key']],x['frame_ids'],v['prediction']['indices'])[0] for k,v in choices.items()}
        assert len({m['sIoU'] for m in metrics.values()})==1
        process={name:{k:v for k,v in snap.get('process',{}).items() if k!='gradients'} for name,snap in choices.items()}
        rows.append({k:x[k] for k in ('key','cohort','source','split','query_type')}|dict(metrics=metrics,
            losses={k:v['loss'] for k,v in choices.items()},offset_losses={k:v['offset_losses'] for k,v in choices.items()},
            original_curve=[v['loss'] for v in x['path']],
            selected_best_step=x['selected_best_step'],selected_backtrack=x['selected_backtrack'],
            predictions={k:v['prediction'] for k,v in choices.items()},process=process))
    summary=dict(run='completed',measurement='pending_independent_audit',groups={},
        GPU_lease_seconds=bar['GPU_lease_seconds'],GT_used_only_offline=True,production_changed=False)
    for c in p['cohorts']:
        for split in ('reviewed_cases','source_reference'):
            rr=[r for r in rows if r['cohort']==c and r['split']==split];ss=[r['source'] for r in rr]
            methods={}
            for name in rr[0]['metrics']:
                vals={k:source_mean_ci([r['metrics'][name][k] for r in rr],ss,seed=p['seed'],bootstrap=p['bootstrap']) for k in ('vIoU_corrected','tIoU','sIoU','span')}
                delta={k:source_mean_ci([r['metrics'][name][k]-r['metrics']['base5'][k] for r in rr],ss,seed=p['seed'],bootstrap=p['bootstrap']) for k in ('vIoU_corrected','tIoU')}
                diff=[r['metrics'][name]['vIoU_corrected']-r['metrics']['base5']['vIoU_corrected'] for r in rr]
                methods[name]=dict(metrics=vals,delta=delta,improved=sum(d>.001 for d in diff),harmed=sum(d<-.001 for d in diff),neutral=sum(abs(d)<=.001 for d in diff))
            summary['groups'][c+'/'+split]=dict(queries=len(rr),sources=len(set(ss)),methods=methods,
                backtrack_choices=dict(collections.Counter(r['selected_backtrack'] for r in rr)),
                original_loss_rebound_queries=sum(r['losses']['base5']>r['losses']['step4']+1e-12 for r in rr))
    write(OUT/'rows.json',rows);write(OUT/'summary.json',summary)
    write(OUT/'complete.json',dict(summary_sha256=sha(OUT/'summary.json'),rows_sha256=sha(OUT/'rows.json'),created=time.time()))
    print('E4B_SCORED',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','capture','score']);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:
        if args.action=='prepare':print('PREPARED',len(plan()['receipts']))
        elif args.action=='capture':capture(args.limit)
        else:score_results()
    except Exception:
        write(OUT/'failures'/(str(time.time_ns())+'.json'),dict(action=args.action,error=traceback.format_exc(),created=time.time()))
        raise
