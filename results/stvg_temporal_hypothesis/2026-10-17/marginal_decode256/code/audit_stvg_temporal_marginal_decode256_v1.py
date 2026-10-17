"""Independent seven-arm scalar/root/control reconstruction, no new model or GT."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_marginal_decode256_v1 import BASE,PREV,read,write,sha,verify

def main():
    import numpy as np,torch
    from scripts.report_stvg_temporal_hypothesis256_v1 import independent
    verify();d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');assert d['baseline_scalar_checks']==9216;checks=independent(d['rows'],d['summary'],d['paired']);exact=[]
    for name in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','INPUT_BARRIER.json','NO_UPDATE_CONTROL_BARRIER.json','TEACHER_BARRIER.json','TEACHER_ROOT_READBACK.json','FIT_BARRIER.json']:
        assert (BASE/name).read_bytes()==(PREV/name).read_bytes();exact.append(name)
    pm={r['case_id']:r for r in read(BASE/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};fm={r['case_id']:r for r in read(BASE/'FIT_BARRIER.json')['files']};m={(r['case_id'],r['arm']):r['metrics']for r in d['rows']};diag=read(BASE/'DECODER_READBACK.json');zeros=0
    for r in diag['records']:
        cid=r['case_id'];p=pm[cid];assert sha(ROOT/p['path'])==p['sha256'];z=torch.load(ROOT/p['path'],map_location='cpu',weights_only=False);assert z['reused_B2_fit_sha256']==fm[cid]['sha256']==sha(ROOT/fm[cid]['path']);assert z['arms']['B_MARGINAL_DECODE']['interval']==r['adapted_marginal_interval']and z['arms']['MARGINAL_NO_UPDATE']['interval']==r['Source_marginal_interval'];assert np.array_equal(z['arms']['B_MARGINAL_DECODE']['boxes'],z['arms']['MARGINAL_NO_UPDATE']['boxes'])
        if r['no_information']:assert m[cid,'B_MARGINAL_DECODE']==m[cid,'MARGINAL_NO_UPDATE'];zeros+=1
    assert zeros==diag['no_information_exact_no_update']==17;assert diag['max_probability_error']<1e-12
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=256,scalar_checks=checks,baseline_scalar_checks=d['baseline_scalar_checks'],all_old_fits_hash_exact=256,exact_upstream_barriers=exact,independent_CDF_bins=diag['independent_CDF_bins'],max_probability_error=diag['max_probability_error'],zero_parameter_same_decoder_exact=17,changed_from_no_update=diag['changed_from_same_decoder_no_update'],actual_B2_full_head_output_reuse_bound=True,new_GPU_forwards=0,new_Teacher_calls=0,new_fits=0,all_signed_unknown_reconstructed=True,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=False));print('MARGINAL_DECODE_ROOT_PASS',checks)
if __name__=='__main__':main()
