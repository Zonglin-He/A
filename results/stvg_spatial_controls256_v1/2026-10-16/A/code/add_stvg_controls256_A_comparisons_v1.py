"""Add exact same-cohort A scalar comparisons after seals; never reopen GT/model."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha
from scripts.score_stvg_spatial_controls256_v1 import aggregate,validate_barrier
CORE=('Full_E1','A','B','C','A_BT','B_BT')
def merge_rows(current,control):
    rows=copy.deepcopy(current);ids={r['case_id'] for r in rows};assert len(ids)==256
    old={(r['case_id'],r['arm']):r for r in rows};other={(r['case_id'],r['arm']):r for r in control}
    assert len(old)==len(rows) and len(other)==len(control) and {r['case_id'] for r in control}==ids
    for cid in sorted(ids):
        for arm in ('Frozen','Temporal','Full_E1','Full_E1_Spatial'):
            assert old[cid,arm]['dataset']==other[cid,arm]['dataset'] and old[cid,arm]['metrics']==other[cid,arm]['metrics'],(cid,arm)
        for arm in ('A','A_Spatial'):
            rec=other[cid,arm]
            if (cid,arm) in old:assert old[cid,arm]['metrics']==rec['metrics'] and old[cid,arm]['dataset']==rec['dataset']
            else:rows.append(copy.deepcopy(rec))
    return rows
def prepare(arm):
    assert arm in CORE
    if not(BASE/'trials/A/CPU_COMPLETION002.json').exists():
        assert arm=='Full_E1';return False
    for a in dict.fromkeys((arm,'A')):
        validate_barrier(a);b=BASE/'trials'/a;p=b/'POSTSEAL_RESULTS002.json';c=read(b/'CPU_COMPLETION002.json')
        assert c['status']=='complete' and c['count']==256 and c['GT_read'] and c['result_sha256']==sha(p)
    b=BASE/'trials'/arm;dest=b/'POSTSEAL_RESULTS_MATCHED_A003.json'
    d=read(b/'POSTSEAL_RESULTS002.json');control=read(BASE/'trials/A/POSTSEAL_RESULTS002.json')
    if dest.exists():
        c=read(b/'CPU_MATCHED_A003.json');assert c['result_sha256']==sha(dest)
        x=read(dest);assert x['matched_A_scalar_readback']['current_sha256']==sha(b/'POSTSEAL_RESULTS002.json') and x['matched_A_scalar_readback']['A_sha256']==sha(BASE/'trials/A/POSTSEAL_RESULTS002.json');return True
    d=copy.deepcopy(d);d['rows']=merge_rows(d['rows'],control['rows']);d['summary'],d['paired']=aggregate(d['rows'])
    d['matched_A_scalar_readback']=dict(status='pass',count=256,current_sha256=sha(b/'POSTSEAL_RESULTS002.json'),A_sha256=sha(BASE/'trials/A/POSTSEAL_RESULTS002.json'),A_prediction_sha256=sha(BASE/'trials/A/GLOBAL_PREDICTION_BARRIER.json'),common_support_metrics_only=True,all_shared_Frozen_Temporal_Full_E1_scalars_exact=True,new_model_calls=0,new_GT_reader_calls=0)
    # Arm-specific adopted support stays exactly as originally scored. It cannot
    # be merged across different reference sets as if it were matched support.
    put(dest,d);put(b/'CPU_MATCHED_A003.json',dict(status='complete',count=256,GT_read=False,CPU_only=True,result_sha256=sha(dest),original_cpu_sha256=sha(b/'CPU_COMPLETION002.json'),A_cpu_sha256=sha(BASE/'trials/A/CPU_COMPLETION002.json'),new_model_calls=0,new_GT_reader_calls=0))
    return True
def contracts():
    def row(ds,i,arm,value):return dict(case_id=ds+str(i),dataset=ds,arm=arm,metrics={'m_vIoU':value,'sparse':None if i%2 else value},GT_read=True)
    cur=[row(ds,i,a,.1) for ds in ('hc2','vidstg') for i in range(128) for a in ('Frozen','Temporal','Full_E1','Full_E1_Spatial','B','B_Spatial')]
    control=[row(ds,i,a,.2 if a in ('A','A_Spatial') else .1) for ds in ('hc2','vidstg') for i in range(128) for a in ('Frozen','Temporal','Full_E1','Full_E1_Spatial','A','A_Spatial')]
    rows=merge_rows(cur,control);s,p=aggregate(rows);assert len(rows)==2048
    for r in p:
        if r['comparison']=='B_minus_A' and r['metric']=='m_vIoU':assert abs(r['mean_delta_pp']+10)<1e-10 and r['negative']==128
    bad=copy.deepcopy(control);bad[0]['metrics']['m_vIoU']=.11
    try:merge_rows(cur,bad)
    except AssertionError:pass
    else:raise AssertionError('mismatched shared Frozen accepted')
    assert merge_rows(rows,control)==rows
    print('MATCHED_A_CONTRACTS_PASS_BOTH_DOMAINS_SIGNS_NULLS_MISMATCH_NO_MODEL_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else print(json.dumps(dict(matched_A_available=prepare(sys.argv[1]))))
