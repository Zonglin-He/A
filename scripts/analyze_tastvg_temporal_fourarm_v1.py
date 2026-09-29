"""Post-seal four-arm paired comparison; no experimental qualification gate."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_temporal_fourarm_v1 import OUT,OLD,verify,CONDS
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS
ARMS=['Frozen','Rerank','Hard','OPD']
COMPARISONS=[('Rerank','Frozen'),('Hard','Frozen'),('OPD','Frozen'),('Hard','Rerank'),('OPD','Rerank'),('OPD','Hard')]


def macro(rr,fn):
    d=collections.defaultdict(list)
    for r in rr:d[r['parent']].append(fn(r))
    return stats([np.mean(d[k]) for k in sorted(d)])


def summarize(rr):
    return dict(cells=len(rr),parents=len({r['parent'] for r in rr}),
        arms={a:{m:macro(rr,lambda r:r['arms'][a][m]) for m in METRICS} for a in ARMS},
        comparisons={a+' - '+b:{m:macro(rr,lambda r:r['arms'][a][m]-r['arms'][b][m]) for m in METRICS} for a,b in COMPARISONS},
        harms_gt5pp={a+' - '+b:{m:sum(r['arms'][a][m]-r['arms'][b][m]<-.05 for r in rr) for m in METRICS} for a,b in COMPARISONS},
        loss_decreased={a:sum(r['diagnostics'][a]['loss_decreased'] for r in rr) for a in ['Hard','OPD']},
        changed_interval={a:sum(r['changed_interval'][a] for r in rr) for a in ['Rerank','Hard','OPD']})


def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==256
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    labelpath=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json';keys={r['key'] for r in p['rows']};gt={}
    with labelpath.open('rb') as handle:
        for k,v in ijson.kvitems(handle,'',use_float=True):
            if k in keys:gt[k]=v
    assert set(gt)==keys
    write(OUT/'GT_EXPOSURE.json',dict(retained_queries=16,historically_exposed=True,GT_for_scoring_only=True,streamed_previous_container=True,container_sha256=sha(labelpath),prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),time=time.time()))
    rows=[];calls=0;reinsertion=0;nohit=0;critic_audits=0
    for r in p['rows']:
        clean=load(OUT/'predictions/clean'/f"{r['ordinal']:03}.pt")
        for cond in CONDS:
            x=load(OUT/'predictions'/cond/f"{r['ordinal']:03}.pt");c=load(OLD/'capture'/cond/f"{r['ordinal']:03}.pt");e=load(OUT/'c2'/cond/f"{r['ordinal']:03}.pt")
            values={};cvalues=[]
            for arm in ARMS:
                pred=x['arms'][arm]['prediction'];m,_=checked_score(pred['boxes'],gt[r['key']],r['frame_ids'],pred['indices']);values[arm]={k:m[k] for k in METRICS};calls+=1
                if arm in ['Hard','OPD']:
                    d=x['arms'][arm]['diagnostics'];assert d['gradient_steps']==1 and d['motion_only_exact'] and not d['backtracking']
                    assert d['realized_relative_step']==0 or abs(d['realized_relative_step']-.004)<1e-7
                    if 'reinsertion' in x['arms'][arm]:assert x['arms'][arm]['reinsertion']['full_pipeline_exact'];reinsertion+=1
            native=x['arms']['Frozen']['prediction'];assert torch.equal(native['boxes'],c['native']['boxes']);assert all(torch.equal(a,b) for a,b in zip(native['logits'],c['native']['logits']))
            scores=[]
            for (a,b),candidate in zip(e['candidate_intervals'],c['candidates']['temporal']):
                scores.append(max([max(0,min(b,d)-max(a,cc))/max(1e-12,max(b,d)-min(a,cc))*w for (cc,d),w in zip(e['proposals'],e['proposal_confidence'])] or [0.]))
                m,_=checked_score(native['boxes'],gt[r['key']],r['frame_ids'],candidate['indices']);cvalues.append({k:m[k] for k in METRICS});calls+=1
            assert np.allclose(scores,e['scores'],atol=1e-12,rtol=0);assert int(np.argmax(scores))==e['selected']==x['selected'];critic_audits+=1
            assert values['Frozen']==cvalues[0] and values['Rerank']==cvalues[x['selected']]
            same_clean={}
            if cond!='clean' and x['d_obs']==0:
                assert x['pixel_sha256']==clean['pixel_sha256'];nohit+=1
                for a in ARMS:
                    pred=x['arms'][a]['prediction'];base=clean['arms'][a]['prediction'];same_clean[a]=pred['indices']==base['indices'] and torch.equal(pred['boxes'],base['boxes'])
                    assert same_clean[a],('same_pixel_nondeterminism',r['ordinal'],cond,a)
            rows.append(dict(parent=r['ordinal'],condition=cond,d_obs=x['d_obs'],selected=x['selected'],candidate_count=len(cvalues),candidate_scores=scores,candidate_metrics=cvalues,oracle=int(np.argmax([z['tIoU'] for z in cvalues])),arms=values,
                diagnostics={a:x['arms'][a]['diagnostics'] for a in ['Hard','OPD']},changed_interval={a:x['arms'][a]['prediction']['indices']!=native['indices'] for a in ['Rerank','Hard','OPD']},nohit_exact_clean=same_clean))
    summary={name:summarize([r for r in rows if (r['condition']!='clean' if name=='transient' else r['condition']==name)]) for name in ['transient']+CONDS}
    corrupt=[r for r in rows if r['condition']!='clean'];clean={r['parent']:r for r in rows if r['condition']=='clean'}
    excess={a:{m:macro(corrupt,lambda r:(r['arms'][a][m]-r['arms']['Frozen'][m])-(clean[r['parent']]['arms'][a][m]-clean[r['parent']]['arms']['Frozen'][m])) for m in METRICS} for a in ['Rerank','Hard','OPD']}
    examples={}
    for comparison in [('OPD','Rerank'),('Rerank','Frozen')]:
        a,b=comparison;ordered=sorted(corrupt,key=lambda r:r['arms'][a]['vIoU_corrected']-r['arms'][b]['vIoU_corrected'])
        chosen=[]
        for sign,seq in [('worst',ordered),('best',ordered[::-1])]:
            seen=set()
            for r in seq:
                if r['parent'] in seen:continue
                seen.add(r['parent']);chosen.append(dict(kind=sign,parent=r['parent'],condition=r['condition'],arms=r['arms'],oracle_t=r['candidate_metrics'][r['oracle']]['tIoU'],selected=r['selected'],oracle=r['oracle']))
                if len(seen)==3:break
        examples[a+' - '+b]=chosen
    write(OUT/'analysis/ROWS.json',rows);write(OUT/'analysis/SUMMARY.json',summary);write(OUT/'analysis/EXCESS.json',excess);write(OUT/'analysis/EXAMPLES.json',examples)
    write(OUT/'AUDIT.json',dict(status='pass',cells=256,dual_metric_calls=calls,independent_critic_checks=critic_audits,full_reinsertion_checks=reinsertion,nohit_cells_exact_clean=nohit,all_motion_only=True,all_fixed_one_step=True,barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),time=time.time()))
    print('PRIMARY',summary['transient']);print('CLEAN',summary['clean'])

if __name__=='__main__':run()
