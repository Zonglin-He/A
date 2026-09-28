"""CPU sealed full-source-val scorer; explicit invocation after an epoch.

No target access, no inference, no GPU. Selection is the preregistered equal-
domain parent macro, with full saved per-query outcomes and adverse tails.
"""
import argparse,collections,json,math,sqlite3,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.score_desta3d_v2_reference_audit import score_tube_independently,binary_auc
from scripts.desta3d_source_fit_v1 import _tube_metric
OUT=ROOT/'artifacts/desta3d_v3/full_source_fit_v1'
ROSTER=ROOT/'artifacts/desta3d_v3/full_source_roster_v1'
METRICS=['vIoU','sIoU','tIoU'];DOMAINS=['Vid','HC1']

def summarize(records,arm):
    per=collections.defaultdict(list)
    for r in records:per[(r['domain'],r['parent'])].append(r['metrics'][arm])
    parents={d:{} for d in DOMAINS}
    for (d,p),rows in per.items():
        parents[d][p]={m:float(np.mean([r[m] for r in rows])) for m in METRICS}
    domains={d:{m:float(np.mean([v[m] for v in parents[d].values()])) for m in METRICS} for d in DOMAINS}
    return dict(parents=parents,domains=domains,equal_domain_parent_macro={m:float(np.mean([domains[d][m] for d in DOMAINS])) for m in METRICS})

def compare(a,b,reps=10000):
    # Stratified paired raw-parent resampling. Queries never become bootstrap
    # units and a large domain never overrides the declared domain weighting.
    rng=np.random.default_rng(20260928);out={}
    for m in METRICS:
        boot=np.zeros(reps);mean=0.;tails=[];signs=collections.Counter()
        for d in DOMAINS:
            keys=sorted(a['parents'][d]);assert keys==sorted(b['parents'][d]) and keys
            delta=np.asarray([a['parents'][d][k][m]-b['parents'][d][k][m] for k in keys])
            boot+=np.mean(delta[rng.integers(0,len(keys),size=(reps,len(keys)))],axis=1)/2
            mean+=float(delta.mean())/2
            if m=='vIoU':
                tails.extend(dict(domain=d,parent=k,delta_pp=float(v*100)) for k,v in zip(keys,delta) if v<-.05)
                signs.update('positive' if v>1e-12 else 'negative' if v< -1e-12 else 'zero' for v in delta)
        out[m]=dict(delta_pp=100*mean,CI95_pp=(100*np.quantile(boot,[.025,.975])).tolist())
        if m=='vIoU':out[m].update(parent_signs=dict(signs),parent_damage_gt5pp=tails)
    return out

def choose_epoch(reports):
    scores=[r['summary']['trained']['equal_domain_parent_macro']['vIoU'] for r in reports]
    best=max(range(len(scores)),key=lambda i:scores[i]);epochs=len(scores)
    stop=epochs>=5 or (epochs>=3 and epochs-1-best>=2)
    return dict(best_epoch=best,best_value=scores[best],completed_epochs=epochs,stop=stop,
        reason='maximum_complete_epochs' if epochs>=5 else 'source_val_patience' if stop else 'continue_complete_epoch',
        mean_gate_vs_Frozen=scores[best]>=reports[best]['summary']['Frozen']['equal_domain_parent_macro']['vIoU'])

def run(epoch):
    dest=OUT/'scores'/f'E{epoch}';assert not dest.exists()
    cfg=read(OUT/'CONFIG.json');seal=read(OUT/'seals'/f'E{epoch}.json')
    assert seal['lock_sha']==sha(OUT/'LOCK.json') and seal['epoch']==epoch
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(p)==h,p
    val=[r for r in read(ROSTER/'INPUTS.json') if r['split']=='validation']
    assert len(val)==seal['queries'] and len(seal['pins'])==2*len(val)
    assert seal['step']==(epoch+1)*cfg['source_counts']['updates_per_epoch']
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    pairs=[];expected=set()
    for i,r in enumerate(val):
        meta=json.loads(db.execute('SELECT row_json FROM examples WHERE key=?',(r['key'],)).fetchone()[0]);pair={}
        for arm,directory in [('Frozen','Frozen'),('trained',f'E{epoch}')]:
            p=OUT/'predictions'/directory/f'{i:05d}.pt';expected.add(str(p));assert sha(p)==seal['pins'][str(p)]
            pred=torch.load(p,map_location='cpu',weights_only=False)
            assert pred['key']==r['key'] and pred['source']==r['parent'] and pred['domain']==r['domain']
            assert pred['frame_ids']==meta['input']['frame_ids'] and pred['video_sha256']==meta['input']['video_sha256']
            assert pred['adapter_sha']==('frozen_gate0' if arm=='Frozen' else seal['adapter_sha']) and pred['validation_GT_used'] is False
            assert torch.isfinite(pred['boxes_cxcywh']).all() and torch.isfinite(pred['event_logits']).all()
            pair[arm]=pred
        assert pair['Frozen']['preprocess']==pair['trained']['preprocess']
        pairs.append(pair)
    assert expected==set(seal['pins'])
    save_once(dest/'PRE_SCORE_AUDIT.json',dict(time=time.time(),status='all_predictions_hash_identity_and_complete_epoch_verified',
        queries=len(val),predictions=2*len(val),actual_step=seal['step'],seal_sha=sha(OUT/'seals'/f'E{epoch}.json'),
        source_GT_exposure='source-label preparation already occurred; first metric access for this new epoch is after this audit',target_access=False))
    records=[];worst=0.
    try:
        for row,pair in zip(val,pairs):
            lab=json.loads(db.execute('SELECT labels_json FROM examples WHERE key=?',(row['key'],)).fetchone()[0])
            result=dict(key=row['key'],domain=row['domain'],parent=row['parent'],metrics={})
            for arm,pred in pair.items():
                scalar=score_tube_independently(pred,lab);tensor=_tube_metric(pred,lab)
                err=max(abs(scalar[m]-tensor[m]) for m in METRICS);worst=max(worst,err);assert err<=3e-6
                scalar['direct_event_AUROC']=binary_auc(lab['event_active'],pred['event_logits'].reshape(-1).tolist()) if arm=='trained' else None
                result['metrics'][arm]=scalar
            records.append(result)
        summary={arm:summarize(records,arm) for arm in ['Frozen','trained']}
        retention={m:dict(denominator=sum(r['metrics']['Frozen'][m]>.5 for r in records),
            retained=sum(r['metrics']['Frozen'][m]>.5 and r['metrics']['trained'][m]>.5 for r in records)) for m in ['vIoU','tIoU']}
        auc=collections.defaultdict(list)
        for r in records:
            v=r['metrics']['trained']['direct_event_AUROC']
            if v is not None:auc[(r['domain'],r['parent'])].append(v)
        domain_auc={d:[float(np.mean(v)) for (domain,p),v in auc.items() if domain==d] for d in DOMAINS}
        report=dict(epoch=epoch,queries=len(records),summary=summary,comparison=compare(summary['trained'],summary['Frozen']),
            native_good_query_retention=retention,format_failures={arm:sum(not r['metrics'][arm]['format_ok'] for r in records) for arm in summary},
            direct_event_AUROC=dict(equal_domain_parent_macro=float(np.mean([np.mean(domain_auc[d]) for d in DOMAINS])) if all(domain_auc.values()) else None,
                defined_queries=sum(r['metrics']['trained']['direct_event_AUROC'] is not None for r in records),total_queries=len(records),defined_parents=len(auc),
                denominator='each query has positive and negative event frames; query average within raw parent, then equal domain macro; not native endpoint AUROC'),
            geometry_scalar_tensor_max_abs=worst,metric_count=6*len(records),target_data=False,
            limitations='source development/early selection; historical exposure; descriptive paired-parent CI uncorrected; no target or native actuation qualification')
        save_once(dest/'PER_QUERY.json',records);save_once(dest/'REPORT.json',report)
        reports=[read(OUT/'scores'/f'E{i}'/'REPORT.json') for i in range(epoch)]+[report]
        decision=choose_epoch(reports);decision['selected_checkpoint']=str(OUT/'epoch_states'/f"E{decision['best_epoch']}.pt")
        decision['selected_checkpoint_sha']=sha(decision['selected_checkpoint']);save_once(dest/'DECISION.json',decision)
        save_once(dest/'COMPLETE.json',dict(pins={str(p):sha(p) for p in dest.iterdir()},target_data=False,GPU_seconds=0))
        print(json.dumps(dict(comparison=report['comparison'],decision=decision),ensure_ascii=False))
    except BaseException:
        save_once(dest/'FAILURE.json',dict(error=traceback.format_exc(),source_metric_access_begun=True));raise
    finally:db.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);run(p.parse_args().epoch)
