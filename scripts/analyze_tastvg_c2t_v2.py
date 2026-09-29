"""Post-seal critic utility and parent-clustered pairwise calibration screen."""
import sys,collections,itertools,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_c2t_v2 import OUT,OLD,verify,CONDS
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS

def macro(rr,fn):
    d=collections.defaultdict(list)
    for r in rr:
        v=fn(r)
        if v is not None:d[r['parent']].append(v)
    return stats([np.mean(d[k]) for k in sorted(d)])

def run():
    p=verify();bar=read(OUT/'C2_BARRIER.json');assert len(bar['files'])==64
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    gt=read(OLD/'GT_SUBSET.json');rows=[];pairs=[];calls=0;cuts=bar['margin_terciles']
    c1={(r['ordinal'],r['condition']):r for r in read(OLD/'analysis/C1_ROWS.json')}
    for r in p['rows']:
        for cond in CONDS:
            x=load(OLD/'capture'/cond/f"{r['ordinal']:03}.pt");e=load(OUT/'c2'/cond/f"{r['ordinal']:03}.pt");cc=x['candidates']['temporal'];values=[]
            assert e['candidate_intervals']==[c['physical_interval'] for c in cc]
            # Independent scalar overlap recomputation; no vectorized bridge import.
            rs=[]
            for a,b in e['candidate_intervals']:
                rs.append(max([max(0,min(b,d)-max(a,c))/max(1e-12,max(b,d)-min(a,c))*s for (c,d),s in zip(e['proposals'],e['proposal_confidence'])] or [0.]))
            assert np.allclose(rs,e['scores'],rtol=0,atol=1e-12);assert int(np.argmax(rs))==e['selected']
            for c in cc:
                m,_=checked_score(x['native']['boxes'],gt[r['key']],r['frame_ids'],c['indices']);values.append({k:m[k] for k in METRICS});calls+=1
            chosen=e['selected'];oracle=int(np.argmax([m['tIoU'] for m in values]));native=values[0]
            arms=dict(Native=native,Expert=values[chosen],Oracle=values[oracle],Uniform={m:float(np.mean([v[m] for v in values])) for m in METRICS})
            row=dict(parent=r['ordinal'],condition=cond,candidate_count=len(cc),selected=chosen,oracle=oracle,candidate_scores=rs,candidate_metrics=values,arms=arms,
                     delta={a:{m:arms[a][m]-native[m] for m in METRICS} for a in ['Expert','Oracle','Uniform']},pair_accuracy={},pair_count={})
            assert abs(row['delta']['Oracle']['tIoU']-c1[(r['ordinal'],cond)]['gain_T'])<1e-10
            buckets=collections.defaultdict(list);ties=0
            for i,j in itertools.combinations(range(len(cc)),2):
                dg=values[i]['tIoU']-values[j]['tIoU'];de=rs[i]-rs[j];gap=abs(de)
                if abs(dg)<=1e-12:ties+=1;continue
                acc=.5 if gap<=1e-12 else float((dg>0)==(de>0));name='low' if gap<=cuts[0] else 'middle' if gap<=cuts[1] else 'high'
                buckets['all'].append(acc);buckets[name].append(acc)
                pairs.append(dict(parent=r['ordinal'],condition=cond,i=i,j=j,margin=gap,bin=name,accuracy=acc,GT_difference=dg,critic_difference=de))
            for name in ['all','low','middle','high']:
                row['pair_accuracy'][name]=float(np.mean(buckets[name])) if buckets[name] else None;row['pair_count'][name]=len(buckets[name])
            row['GT_tied_pairs_excluded']=ties;rows.append(row)
    summaries={}
    for name in CONDS+['corrupted_parent_macro']:
        rr=[r for r in rows if r['condition']!='clean'] if name=='corrupted_parent_macro' else [r for r in rows if r['condition']==name]
        summaries[name]=dict(arms={a:{m:macro(rr,lambda r:r['arms'][a][m]) for m in METRICS} for a in ['Native','Expert','Oracle','Uniform']},
            delta={a:{m:macro(rr,lambda r:r['delta'][a][m]) for m in METRICS} for a in ['Expert','Oracle','Uniform']},
            pair_accuracy={b:macro(rr,lambda r:r['pair_accuracy'][b]) for b in ['all','low','middle','high']},
            pair_counts={b:sum(r['pair_count'][b] for r in rr) for b in ['all','low','middle','high']},
            expert_minus_uniform={m:macro(rr,lambda r:r['arms']['Expert'][m]-r['arms']['Uniform'][m]) for m in METRICS},
            expert_harms_gt5pp={m:sum(r['delta']['Expert'][m]<-.05 for r in rr) for m in METRICS},expert_min_delta={m:min(r['delta']['Expert'][m] for r in rr) for m in METRICS},native_kept=sum(r['selected']==0 for r in rr),cells=len(rr))
    clean={r['parent']:r for r in rows if r['condition']=='clean'}
    corrupt=[r for r in rows if r['condition']!='clean']
    excess={m:macro(corrupt,lambda r:r['delta']['Expert'][m]-clean[r['parent']]['delta']['Expert'][m]) for m in METRICS}
    recovery=[]
    for r in corrupt:
        loss=clean[r['parent']]['arms']['Native']['tIoU']-r['arms']['Native']['tIoU']
        if loss>.05:recovery.append(dict(parent=r['parent'],condition=r['condition'],clean=clean[r['parent']]['arms']['Native']['tIoU'],native=r['arms']['Native']['tIoU'],expert=r['arms']['Expert']['tIoU'],oracle=r['arms']['Oracle']['tIoU'],restored_clean=r['arms']['Expert']['tIoU']>=clean[r['parent']]['arms']['Native']['tIoU']))
    primary=summaries['corrupted_parent_macro'];ci=primary['pair_accuracy']['high']['ci95']
    passed=bool(primary['delta']['Expert']['tIoU']['ci95'][0]>0 and ci and ci[0]>.5)
    write(OUT/'analysis/C2_ROWS.json',rows);write(OUT/'analysis/C2_PAIRS.json',pairs);write(OUT/'analysis/C2_SUMMARY.json',summaries);write(OUT/'analysis/C2_EXCESS.json',excess);write(OUT/'analysis/C2_RECOVERY.json',recovery)
    write(OUT/'C2_DECISION.json',dict(status='completed',measurement='valid_with_bridge_and_sampled_input_scope',OPD_resource_gate=passed,OPD='not_started',adaptation=False,critic='confidence weighted proposal overlap, fixed without GT tuning',margin_terciles=cuts,GT_for_ranking=False))
    write(OUT/'C2_AUDIT.json',dict(status='pass',cells=64,parents=16,pairs=len(pairs),dual_metric_calls=calls,scalar_critic_reconstruction=True,oracle_matches_C1=True,barrier_sha256=sha(OUT/'C2_BARRIER.json'),time=time.time()))
    print('C2 PRIMARY',primary,'EXCESS',excess,'PASS',passed)

if __name__=='__main__':run()
