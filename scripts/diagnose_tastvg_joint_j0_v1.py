"""Read-only J0 temporal failure attribution against already scored C3 candidates."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.score_tastvg_spatial_online_opd_s1_v1 import macro
OUT=ROOT/'artifacts/tastvg_joint_j0_v1';OLD=ROOT/'artifacts/tastvg_temporal_fourarm_v1/analysis/ROWS.json'


def run():
    j=read(OUT/'ROWS.json');old=read(OLD);conds=read(OUT/'LOCK.json')['conditions'];lookup={(r['parent'],r['condition']):r for r in old};rows=[];checked=0
    for x in j:
        r=lookup[x['parent'],x['condition']];row={k:x[k] for k in ['parent','condition','arrival','expert_scheduled']};row.update(selected=r['selected'],candidate_count=r['candidate_count'],candidate_scores=r['candidate_scores'],candidate_metrics=r['candidate_metrics'])
        for short,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:
            native=r['arms']['Frozen'][key];rerank=r['arms']['Rerank'][key];assert abs(native-x['frozen_'+short])<1e-12
            if x['expert_scheduled']:assert abs(rerank-x['fast_'+short])<1e-12;checked+=1
            row['native_'+short]=native;row['rerank_'+short]=rerank;row['historical_full_rerank_gain_'+short]=rerank-native
            row['historical_candidate_oracle_gain_'+short]=max(c[key] for c in r['candidate_metrics'])-native
        rows.append(row)
    sums={};sources=[]
    for group in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']=='clean')];sums[group]={}
        for subset in ['all','scheduled','unscheduled']:
            seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='scheduled')];keys=[k for k in rows[0] if k.endswith(('_s','_t','_v'))]
            sums[group][subset]=dict(cells=len(seq),sources=len({r['parent'] for r in seq}),metrics={k:macro(seq,k) for k in keys})
            if subset=='all':
                for parent in sorted({r['parent'] for r in seq}):
                    values=[r for r in seq if r['parent']==parent];sources.append(dict(parent=parent,group=group,expert_scheduled=values[0]['expert_scheduled'],**{k:float(np.mean([r[k] for r in values])) for k in keys}))
        # Exact partition: fixed quarter scheduled, three quarters unavailable.
        for short in ['s','t','v']:
            key='historical_full_rerank_gain_'+short;a=sums[group]['all']['metrics'][key]['mean'];b=sums[group]['scheduled']['metrics'][key]['mean'];c=sums[group]['unscheduled']['metrics'][key]['mean'];assert abs(a-(.25*b+.75*c))<1e-14
    write(OUT/'TEMPORAL_REFERENCE_ROWS.json',rows);write(OUT/'TEMPORAL_DIAGNOSIS.json',dict(status='completed_read_only',historical_rows_sha256=sha(OLD),matched_metric_checks=checked,groups=sums,sources=sources,new_inference=0,new_GT=0,schedule_changed=False,new_candidate_or_expert=False,scope='Same16source prior C3 full-availability result partitioned by locked J0 schedule. Posthoc explanatory reference, not a new J0 arm or a deployable schedule chosen with GT.'))
    print({g:{subset:{k:100*v['mean'] for k,v in z['metrics'].items() if 'gain' in k and k.endswith(('_t','_v'))} for subset,z in subsets.items()} for g,subsets in sums.items()})

if __name__=='__main__':run()
