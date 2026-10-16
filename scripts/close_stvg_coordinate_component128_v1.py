"""Finite study closing: saved pair arithmetic and measurement/public evidence only.

No inference, labels, tensor loading, fitting, or scoring is performed.
"""
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SEARCH=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1'
COMP=ROOT/'artifacts/stvg_component_ablation128_v1'
PUBLIC=ROOT/'results/stvg_coordinate_component128/2026-10-16'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    bindings={}
    def read(p):
        bindings[str(p.relative_to(ROOT))]=sha(p)
        return json.loads(p.read_text())
    history=read(SEARCH/'SEARCH_COMPLETION006.json')['history']
    assert len(history)==len(set(history))==42
    plans=[(SEARCH/'trials'/t,'POSTSEAL_SCORE_ROWS.json','POSTSEAL_PAIRED_CONTRASTS.json') for t in history]
    plans += [(COMP/'trials'/f'A{i:03}','POSTSEAL_ROWS.json','POSTSEAL_PAIRED.json') for i in range(16)]
    plans += [(COMP/'temporal_component/revision005','TEMPORAL_COMPONENT_SCORE_ROWS.json','TEMPORAL_COMPONENT_SCORE_PAIRED.json'),(COMP/'native_clock/revision002/trials/N000','POSTSEAL_ROWS.json','POSTSEAL_PAIRED.json')]
    pairs_checked=0;pair_source_values=0;max_error=0.;cost=[]
    for path,rows_name,pairs_name in plans:
        rows=read(path/rows_name)['rows'];pairs=read(path/pairs_name)['paired_contrasts']
        idx={(r['dataset'],r['arm'],r['case_id']):r for r in rows}
        assert len(idx)==len(rows)
        if rows_name.startswith('TEMPORAL'):
            root=read(path/'ROOT_TEMPORAL_COMPONENT_READBACK.json')
            assert root['status']=='pass' and root['count']==128
            assert read(path/'SCORE_CPU_COMPLETION.json')['status']=='complete'
        else:
            assert read(path/'CPU_COMPLETION.json')['status']=='complete'
            root=read(path/'ROOT_READBACK.json');assert root['status']=='pass' and root['count']==128
            barrier=read(path/'STUDENT_GLOBAL_BARRIER.json')
            assert barrier['status']=='complete' and barrier['count']==128
            assert root['barrier_sha256']==sha(path/'STUDENT_GLOBAL_BARRIER.json')
            global_=read(path/'GLOBAL_PREDICTION_BARRIER.json')
            assert global_['status']=='complete' and global_['GT_read'] is False
            cost.append(dict(trial=path.name,scope='recorded Student worker only; includes reuse where declared, excludes inherited Teacher/Temporal cost',**{k:barrier[k] for k in ['wall_seconds','actual_fit_invocations','mode_fit_invocations','mode_optimizer_steps','source_capture_calls','full_readout_calls','reused_fit_count'] if k in barrier}))
        for p in pairs:
            a,b=p['contrast'].split('_minus_',1)
            assert p.get('comparison_arm',a)==a and p.get('baseline_arm',b)==b
            ds,metric=p['dataset'],p['metric'];ids={r['case_id'] for r in rows if r['dataset']==ds and r['arm']==a}
            assert len(ids)==64
            values={};unknown=[]
            for cid in sorted(ids):
                av,bv=idx[(ds,a,cid)]['metrics'][metric],idx[(ds,b,cid)]['metrics'][metric]
                if av is None or bv is None:unknown.append(cid)
                else:values[cid]=(av-bv)*100
            assert p['n_pairs']==len(values) and p['unknown_count']==len(unknown)
            assert set(p['unknown_case_ids'])==set(unknown)
            assert p['per_source_delta_pp']==values
            tol=p.get('neutral_tolerance_pp',1e-10)
            for key,expected in [('positive_case_ids',{c for c,v in values.items() if v>tol}),('negative_case_ids',{c for c,v in values.items() if v<-tol}),('neutral_case_ids',{c for c,v in values.items() if abs(v)<=tol})]:
                assert set(p[key])==expected
            if values:
                vv=list(values.values())
                error=abs(math.fsum(vv)/len(vv)-p['mean_delta_pp']);assert error<1e-12;max_error=max(max_error,error)
                if 'median_delta_pp' in p:assert abs(statistics.median(vv)-p['median_delta_pp'])<1e-12
                if 'worst_delta_pp' in p:assert min(vv)==p['worst_delta_pp']
            else:assert p['mean_delta_pp'] is None
            pairs_checked+=1;pair_source_values+=len(values)
    assert pairs_checked==5450
    public=read(COMP/'FINAL_GITHUB_RECEIPT.json')
    assert public['status']=='pass' and public['count']==637
    for item in public['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    audit=read(PUBLIC/'ROOT_PUBLIC_SCALAR_AUDIT.json');assert audit['status']=='pass' and audit['rows_checked']==52864
    source=read(PUBLIC/'SOURCE_BINDINGS.json')
    for item in source['source_snapshot']:
        assert sha(ROOT/item['path'])==item['original_sha256']
        assert sha(PUBLIC/'source_snapshot'/item['path'])==item['public_snapshot_sha256']
    visual=read(PUBLIC/'ROOT_VISUAL_REVIEW.json');assert visual['status']=='pass' and len(visual['files'])==3
    for item in visual['files']:assert sha(PUBLIC/item['name'])==item['sha256']
    old=read(COMP/'analysis/PROGRESS_READBACK_1792106159.json')
    assert old['status']=='pass' and old['max_summary_error']<1e-12
    # Earlier actual saved-box/interval equivalence proof is preserved, not rerun here.
    assert old['independent_exact_prediction_checks']
    for filename in ['COMPLETED_SPATIAL_TEMPORAL_READBACK_1792110627.json','NATIVE_FINAL_READBACK_1792114820.json']:
        assert read(COMP/'analysis'/filename)['status']=='pass'
    report=dict(status='pass',scope='complete finite 128-development coordinate search and hierarchical deletion study; saved-pair arithmetic, existing root witnesses, actual visual and public evidence',count=128,coordinate_slots=42,spatial_trials=16,temporal_deletions=4,native_controls=1,measurement_blocks=60,pair_cells_checked=pairs_checked,pair_source_values_checked=pair_source_values,max_pair_mean_error=max_error,public_files_verified=637,source_snapshot_files_verified=317,actual_visual_files_verified=3,commit_sha=public['commit'],tree_sha=public['tree'],source_hashes=bindings,CPU_only=True,GT_read=False,new_GT_read=False,new_inference=False,new_fit=False,new_scoring=False,independent_full_backward_repeated=False,bootstrap_executed=False,independent_confirmation_complete=False,CURRENT_promoted=False,full_evaluation_started=False,combined_simplification_executed=False,reader_sha256=sha(Path(__file__)),time=time.time())
    (COMP/'ROOT_STUDY_READBACK.json').write_text(json.dumps(report,indent=2)+'\n')
    (COMP/'STUDENT_MEASUREMENT_COST_READBACK.json').write_text(json.dumps(dict(status='complete',scope='saved actual worker timings/call counters only, not matched speed benchmark or complete cold deployment latency',records=cost,inherited_expert_cost='unknown; not zero',CPU_only=True,GT_read=False),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}))


if __name__=='__main__':main()
