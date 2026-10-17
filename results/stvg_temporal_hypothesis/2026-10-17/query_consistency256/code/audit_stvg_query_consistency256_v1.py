"""Independent saved scalar, probability-target, identity and attribution audit."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_query_consistency256_v1 import BASE,B3,read,write,sha

def main():
    import numpy as np
    from scripts.report_stvg_temporal_hypothesis256_v1 import independent
    d=read(BASE/'RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'RESULTS.json');checks=independent(d['rows'],d['summary'],d['paired']);lock=read(BASE/'DIAGNOSIS_LOCK.json')
    for p,h in lock['pins'].items():assert sha(ROOT/p)==h,p
    older={(r['case_id'],r['arm']):r['metrics']for r in read(B3/'POSTSEAL_RESULTS.json')['rows']};oldchecks=0;captions=0;informative=0;targetchecks=0
    for row in d['rows']:
        k=(row['case_id'],row['arm'])
        if k in older:assert row['metrics']==older[k];oldchecks+=len(row['metrics'])
    for r in d['details']:
        p=np.asarray(r['prior_conditional']);q=np.asarray(r['target_conditional']);captions+=len(r['captions'])
        if r['informative']:
            ll=np.log(p)+np.asarray(r['rewards']);ee=np.exp(ll-ll.max());ref=ee/ee.sum();assert np.max(abs(ref-q))<1e-12;assert abs(r['requested_logodds_shift']-r['reward_gap'])<1e-12;informative+=1
        else:assert np.array_equal(p,q)and abs(r['Student_minus_control_v_pp'])<1e-12
        assert abs(r['probability_alt_shift_requested']-(q[1]-p[1]))<1e-12;targetchecks+=1
    assert informative==239 and captions==512 and targetchecks==256 and d['baseline_scalar_checks']==4608
    for s in d['diagnostics']:
        ds=s['dataset'];dd=[r for r in d['details']if r['dataset']==ds];assert len(dd)==128;point=next(p['mean_delta_pp']for p in d['paired']if p['dataset']==ds and p['comparison']=='B_MARGINAL_DECODE_minus_MARGINAL_NO_UPDATE'and p['metric']=='m_vIoU');assert abs(math.fsum(x['Student_minus_control_v_pp']for x in dd)/128-point)<1e-12;assert abs(math.fsum(z['total_signed_contribution_v_pp']for z in s['reward_geometry_strata'])-point)<1e-12
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=256,scalar_checks=checks,full_saved_baseline_scalar_checks=oldchecks,official_postseal_baseline_rechecks=d['baseline_scalar_checks'],Teacher_caption_records=512,informative_queries=239,independent_tilted_target_checks=targetchecks,max_FP32_candidate_probability_error=d['max_candidate_FP32_probability_error'],max_saved_objective_error=d['max_saved_objective_error'],all_source_contributions_and_strata_exact=True,new_models=0,new_Teacher_calls=0,new_fits=0,new_GPU_forwards=0,GT_only_posthoc_diagnostic=True,result_sha256=sha(BASE/'RESULTS.json')));print('QC_DIAG_ROOT_PASS',checks)
if __name__=='__main__':main()
