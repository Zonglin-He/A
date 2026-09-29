"""O1 post-seal utility, conditional on one locked development stream."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_sparse_online_capture_v1 import OUT,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS
ARMS=['Frozen','Budgeted Rerank','Online Slow-Fast','Full Rerank']
COMPARISONS=[('Online Slow-Fast','Budgeted Rerank'),('Budgeted Rerank','Frozen'),('Online Slow-Fast','Frozen'),('Full Rerank','Frozen'),('Online Slow-Fast','Full Rerank')]


def summarize(rows):
    return dict(n=len(rows),arms={a:{m:stats([r['arms'][a][m] for r in rows]) for m in METRICS} for a in ARMS},
        comparisons={a+' - '+b:{m:stats([r['arms'][a][m]-r['arms'][b][m] for r in rows]) for m in METRICS} for a,b in COMPARISONS},
        harms_gt5pp={a+' - '+b:{m:sum(r['arms'][a][m]-r['arms'][b][m]<-.05 for r in rows) for m in METRICS} for a,b in COMPARISONS})


def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert sha(OUT/'SEALED_SELECTIONS.json')==bar['selection_sha256'];assert read(OUT/'STATE_AUDIT.json')['status']=='pass'
    selected=read(OUT/'SEALED_SELECTIONS.json');labelpath=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json';keys={r['key'] for r in p['rows']};gt={}
    with labelpath.open('rb') as handle:
        for k,v in ijson.kvitems(handle,'',use_float=True):
            if k in keys:gt[k]=v
    assert len(gt)==32 and set(gt)==keys
    write(OUT/'GT_EXPOSURE.json',dict(queries=32,historically_exposed=True,scoring_only=True,container_sha256=sha(labelpath),prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),time=time.time()))
    rows=[];calls=0;score_checks=0
    for r,sel in zip(p['rows'],selected):
        pos=r['position'];assert sel['position']==pos;x=load(OUT/'capture'/f'{pos:03}.pt');z=load(OUT/'online'/f'{pos:03}.pt');phase='online' if r['expert'] else 'full';e=load(OUT/'teacher'/phase/f'{pos:03}.pt')
        cm=[]
        for candidate in x['candidates']:
            m,_=checked_score(x['native']['boxes'],gt[r['key']],r['frame_ids'],candidate['indices']);cm.append({k:m[k] for k in METRICS});calls+=1
        rs=[]
        for a,b in e['candidate_intervals']:rs.append(max([max(0,min(b,d)-max(a,c))/max(1e-12,max(b,d)-min(a,c))*s for (c,d),s in zip(e['proposals'],e['proposal_confidence'])] or [0.]))
        assert np.allclose(rs,e['scores'],rtol=0,atol=1e-12) and int(np.argmax(rs))==e['selected'];score_checks+=1
        arms={a:cm[sel['selected'][a]] for a in ARMS};assert len({arms[a]['sIoU'] for a in ARMS})==1
        if r['expert']:assert arms['Online Slow-Fast']==arms['Budgeted Rerank']==arms['Full Rerank']
        else:assert arms['Budgeted Rerank']==arms['Frozen']
        rows.append(dict(position=pos,parent=r['ordinal'],condition=r['condition'],expert=r['expert'],d_obs=x['d_obs'],selected=sel['selected'],candidate_metrics=cm,base_score=x['base_score'].tolist(),arrival_scores=z['arrival_scores'].tolist(),teacher_scores=rs,
            arms=arms,arrival_hash=z['arrival_hash'],after_hash=z['after_hash'],arrival_norm=float(z['arrival_state']['w'].norm()),after_norm=float(z['after_state']['w'].norm()),diagnostics=z['diagnostics']))
    groups=dict(nonexpert=[r for r in rows if not r['expert']],expert=[r for r in rows if r['expert']],all=rows);summaries={n:summarize(rr) for n,rr in groups.items()}
    progression=[]
    for end in range(4,33,4):
        prefix=[r for r in rows[:end] if not r['expert']];block=[r for r in rows[end-4:end] if not r['expert']]
        progression.append(dict(end_position=end,nonexpert_seen=len(prefix),cumulative_delta={m:float(np.mean([r['arms']['Online Slow-Fast'][m]-r['arms']['Budgeted Rerank'][m] for r in prefix])) for m in METRICS},block_delta={m:float(np.mean([r['arms']['Online Slow-Fast'][m]-r['arms']['Budgeted Rerank'][m] for r in block])) for m in METRICS},weight_norm=rows[end-1]['after_norm']))
    examples=sorted(groups['nonexpert'],key=lambda r:r['arms']['Online Slow-Fast']['vIoU_corrected']-r['arms']['Budgeted Rerank']['vIoU_corrected'])
    write(OUT/'analysis/ROWS.json',rows);write(OUT/'analysis/SUMMARY.json',summaries);write(OUT/'analysis/PROGRESSION.json',progression);write(OUT/'analysis/EXAMPLES.json',dict(worst=[r['position'] for r in examples[:3]],best=[r['position'] for r in examples[-3:][::-1]]))
    write(OUT/'UTILITY_AUDIT.json',dict(status='pass',parents=32,unique_sources=True,dual_metric_calls=calls,independent_critic_checks=score_checks,expert_position_identity=8,nonexpert_budgeted_native_identity=24,all_spatial_boxes_unchanged=True,conditional_single_stream_statistics=True,time=time.time()))
    print('PRIMARY',summaries['nonexpert']['comparisons']['Online Slow-Fast - Budgeted Rerank'])

if __name__=='__main__':run()
