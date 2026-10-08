"""Synthetic query/parent aggregation contracts; no research labels or model calls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.assemble_stvg_opd_table1_v1 import aggregate
from scripts.score_stvg_opd_p1_v1 import source_stats

def run():
    rows=[]
    # Three queries in parent 0, one in parent 1: query and parent macros differ.
    for source,queries in [(0,[(0,.1),(1,.2),(2,.3)]),(1,[(3,.9)])]:
        for query,value in queries:
            for order in range(3):
                after=value+.01*order
                rows.append(dict(source_id=source,query_ordinal=query,order='order'+str(order+1),
                    After_v=after,After_t=.4,After_s=.6,After_R30=float(after>.3),After_R50=float(after>.5),
                    delta_total_v=after-.2,delta_current_v=.015,delta_inherited_v=after-.215))
    stats=aggregate(rows);checks=0
    assert stats['queries']==4 and stats['parent_sources']==2 and stats['logical_arrivals']==12;checks+=3
    assert abs(stats['metrics']['After_v']['official_query_macro']-.385)<1e-12;checks+=1
    assert abs(stats['metrics']['After_v']['source_macro']-.56)<1e-12;checks+=1
    assert abs(stats['metrics']['delta_total_v']['source_macro']-.36)<1e-12;checks+=1
    assert abs(stats['metrics']['delta_total_v']['source_macro']-stats['metrics']['delta_current_v']['source_macro']-stats['metrics']['delta_inherited_v']['source_macro'])<1e-12;checks+=1
    for order in ['order1','order2','order3']:
        sub=aggregate([r for r in rows if r['order']==order])
        assert sub['queries']==4 and sub['logical_arrivals']==4;checks+=2
    reverse=aggregate(list(reversed(rows)))
    for field,value in stats['metrics'].items():
        assert abs(value['source_macro']-reverse['metrics'][field]['source_macro'])<1e-12;checks+=1
        assert np.max(np.abs(np.array(value['paired_parent_ci95'])-reverse['metrics'][field]['paired_parent_ci95']))<1e-12;checks+=1
    scorer=source_stats(rows,['After_v','delta_total_v'])
    assert abs(scorer['metrics']['After_v']['mean']-.56)<1e-12;checks+=1
    assert abs(scorer['metrics']['After_v']['query_macro']-.385)<1e-12;checks+=1
    subset=aggregate([r for r in rows if r['source_id']!=0])
    assert subset['queries']==1 and subset['parent_sources']==1;checks+=2
    assert abs(subset['metrics']['After_v']['source_macro']-.91)<1e-12;checks+=1
    return dict(status='pass',synthetic_contracts=checks,real_GPU_qualification=False,
        research_GT_or_predictions_read=False)

if __name__=='__main__':
    import json
    print(json.dumps(run()))
