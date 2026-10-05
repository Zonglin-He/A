"""Exact matched-budget episodic readout from sealed independent episode fits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import PUB,write
from scripts.decota_public_result_io_v1 import read
from scripts.score_decota_optimizer_posterior_v1 import stats

def derive(rows):
    key=lambda r:(r['dataset'],r['split'],r['condition'],r['order'],r['arrival'])
    episodes={key(r):r['v'] for r in rows if r['stream']=='episodic'};out=[]
    for r in rows:
        ref=episodes[key(r)] if r['expert'] else r['frozen_v']
        out.append({k:r[k] for k in ['dataset','split','condition','order','arrival','source_id','stream','expert']}|dict(online_v=r['v'],matched_episodic_v=ref,vs_matched_episodic_v=r['v']-ref))
    summary={}
    for ds in ['vidstg','hc2']:
        summary[ds]={}
        for sp in ['search','confirm']:
            summary[ds][sp]={}
            for stream in sorted({r['stream'] for r in out}):
                q=[r for r in out if r['dataset']==ds and r['split']==sp and r['stream']==stream]
                summary[ds][sp][stream]={}
                for name,test in [('corruption',lambda r:r['condition']!='clean'),('clean',lambda r:r['condition']=='clean'),('nonexpert_corrupt',lambda r:r['condition']!='clean' and not r['expert']),('expert_corrupt',lambda r:r['condition']!='clean' and r['expert'])]:
                    summary[ds][sp][stream][name]=stats([r for r in q if test(r)],['vs_matched_episodic_v'])
            for rate in [25,50]:
                q=[r for r in out if r['dataset']==ds and r['split']==sp and r['condition']!='clean' and r['stream'] in [f'seed{s}_budget{rate}' for s in range(5)]]
                summary[ds][sp][f'five_schedule_mean_{rate}']=stats(q,['vs_matched_episodic_v'])
    return out,summary

def run(folder):
    p=Path(folder);rows=read(p/'ROWS.json');out,summary=derive(rows)
    write(p/'MATCHED_PERSISTENCE_ROWS.json',out);write(p/'MATCHED_PERSISTENCE_SUMMARY.json',summary)
    write(p/'MATCHED_PERSISTENCE_POLICY.json',dict(reference='Same expert roster; episodic100 at expert positions and Frozen at nonexpert positions',new_forward_calls=0,new_backward_calls=0,scientific_prediction_change=False,full_budget_episodic_comparison_is_not_pure_persistence=True))
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else PUB/'online')
