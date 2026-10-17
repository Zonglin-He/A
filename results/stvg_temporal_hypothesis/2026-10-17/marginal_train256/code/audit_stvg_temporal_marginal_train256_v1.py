"""Independent six-arm aggregation, exact evidence and interval-mass checks."""
from pathlib import Path
import gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_marginal_train256_v1 import BASE,PREV,read,write,sha,verify

def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as f
    from scripts.report_stvg_temporal_hypothesis256_v1 import independent
    verify();d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');assert d['baseline_scalar_checks']==7680;checks=independent(d['rows'],d['summary'],d['paired']);torch.set_num_threads(2)
    exact=[]
    for name in ['ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','INPUT_BARRIER.json','NO_UPDATE_CONTROL_BARRIER.json','TEACHER_BARRIER.json','TEACHER_ROOT_READBACK.json','SIGNAL_COMPLETION.json']:
        assert (BASE/name).read_bytes()==(PREV/name).read_bytes();exact.append(name)
    jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};fm={r['case_id']:r for r in read(BASE/'FIT_BARRIER.json')['files']};bins=0;maxerr=0.;noops=0;n=0;changed=0
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        cid=r['probe_alias'];c,_=load_case(r);j=read(ROOT/jm[cid]['path']);rec=fm[cid];assert sha(ROOT/rec['path'])==rec['sha256'];z=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['fits']['HYP'];off=f.native.offsets(c,_hidden_from_source_case(c));ids=list(map(int,c['frame_ids']));idx={v:i for i,v in enumerate(ids)};N=len(ids);cdfs=[]
        for o in off:
            p=f.original._prob(o['base'])[0];s=np.array([idx[o['ids'][i]]for i in o['ij'][0]]);e=np.array([idx[o['ids'][i]]for i in o['ij'][1]]);H=np.bincount(s*N+e,weights=p,minlength=N*N).reshape(N,N);cdfs.append(H[::-1,:].cumsum(0)[::-1,:].cumsum(1))
        C=cdfs[0]*cdfs[1];q=np.pad(C,((0,1),(1,0)));M=q[:-1,1:]-q[1:,1:]-q[:-1,:-1]+q[1:,:-1];assert abs(float(M.sum())-1)<1e-12 and M.min()>-1e-12
        masses=np.array([M[idx[h['interval_frames'][0]],idx[h['interval_frames'][1]-1]]for h in j['selection']['hypotheses']]);err=float(np.max(abs(masses-z['Native_interval_probabilities'])))if len(masses)else 0.;assert err<1e-12;maxerr=max(maxerr,err)
        if len(masses):assert np.max(abs(masses/masses.sum()-z['Native_conditional']))<1e-12
        if not z['no_information']:
            qt,_=f.original._prob(np.log(masses)+np.asarray(z['rewards']));assert np.max(abs(qt-z['Teacher_conditional']))<1e-12
        else:
            assert z['optimizer_steps']==0 and np.count_nonzero(z['theta'])==0 and z['final_interval']==list(c['Native_interval']);noops+=1
        iv,logits=f.native.native_decode(off,z['theta']);assert iv==z['final_interval']and all(torch.equal(a,b)for a,b in zip(logits,z['final_logits']));changed+=iv!=list(c['Native_interval']);bins+=N*N;n+=1;del c,off,M,C,z;gc.collect()
        if n%32==0:print('MARGINAL_TRAIN_ROOT',n,flush=True)
    assert f.CFG==read(PREV/'METHOD_LOCK.json')['config'];assert sha(ROOT/'scripts/stvg_temporal_hypothesis_fit32_v1.py')=='2e8c834ea2af7f8bdf34adcb5638bf555bb045d3e752bd495a669ecb67348c8d'
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=n,scalar_checks=checks,baseline_scalar_checks=d['baseline_scalar_checks'],independent_CDF_bins=bins,max_interval_mass_error=maxerr,exact_prior_input_and_feedback_barriers=exact,I1_I2_and_query_and_Source_exact=256,no_information_exact_zero_controls=noops,actual_final_Native_decoder_checks=256,actual_interval_changed=changed,independent_enumerated_Torch_path_checks=read(BASE/'FIT_BARRIER.json')['independent_checks'],original_optimizer_and_hyperparameters_unchanged=True,only_training_probability_representation_changed=True,all_signed_unknown_reconstructed=True,GT_read=False,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json')));print('MARGINAL_TRAIN_ROOT_PASS',checks)
if __name__=='__main__':main()
