"""Independent CDF ranking and complete matched result reconstruction, CPU only."""
from pathlib import Path
import gc,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_marginal_i2_256_v1 import BASE,OLD,CORE,read,write,sha,verify
def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f
    from scripts.report_stvg_temporal_hypothesis256_v1 import independent
    verify();d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json');checks=independent(d['rows'],d['summary'],d['paired']);torch.set_num_threads(2);jm={r['case_id']:r for r in read(BASE/'INPUT_BARRIER.json')['jobs']};oldj={r['case_id']:r for r in read(OLD/'INPUT_BARRIER.json')['jobs']};bins=0;maxerr=0.;n=0
    unchanged={r['case_id']for r in read(BASE/'INPUT_CHANGE_READBACK.json')['records']if r['I2_identical_to_old']};of={r['case_id']:r for r in read(OLD/'FIT_BARRIER.json')['files']};nf={r['case_id']:r for r in read(BASE/'FIT_BARRIER.json')['files']};unchanged_checks=0
    for cid in unchanged:
        oldfit=torch.load(ROOT/of[cid]['path'],map_location='cpu',weights_only=False)['fits']['HYP'];newfit=torch.load(ROOT/nf[cid]['path'],map_location='cpu',weights_only=False)['fits']['HYP']
        assert np.array_equal(oldfit['theta'],newfit['theta'])and oldfit['final_interval']==newfit['final_interval']and oldfit['optimizer_steps']==newfit['optimizer_steps'];assert all(torch.equal(a,b)for a,b in zip(oldfit['final_logits'],newfit['final_logits']));unchanged_checks+=1
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        c,_=load_case(r);cid=r['probe_alias'];rec=jm[cid];assert sha(ROOT/rec['path'])==rec['sha256'];j=read(ROOT/rec['path']);o=read(ROOT/oldj[cid]['path']);assert j['selection']['hypotheses'][0]==o['selection']['hypotheses'][0];assert j['query_text']==o['query_text']and j['source_case_sha256']==o['source_case_sha256'];off=f.native.offsets(c,_hidden_from_source_case(c));ids=list(map(int,c['frame_ids']));idx={x:i for i,x in enumerate(ids)};N=len(ids);cdfs=[]
        for z in off:
            p=f.original._prob(z['base'])[0];s=np.array([idx[z['ids'][i]]for i in z['ij'][0]]);e=np.array([idx[z['ids'][i]]for i in z['ij'][1]]);H=np.bincount(s*N+e,weights=p,minlength=N*N).reshape(N,N);cdfs.append(H[::-1,:].cumsum(0)[::-1,:].cumsum(1))
        C=cdfs[0]*cdfs[1];q=np.pad(C,((0,1),(1,0)));M=q[:-1,1:]-q[1:,1:]-q[:-1,:-1]+q[1:,:-1];assert abs(float(M.sum())-1)<1e-12 and float(M.min())> -1e-12
        mode=j['selection']['marginal_no_update_interval'];value=float(M[idx[mode[0]],idx[mode[1]-1]]);err=abs(value-float(M.max()));assert err<1e-12;maxerr=max(maxerr,err);av=sorted(x['physical_frame_id']for x in read(CORE/'temporal/requests'/(cid+'.json'))['frame_map']);sample=j['selection']['hypotheses'][0]['sampled_physical_frame_ids'];best=None
        for flat in np.argsort(-M.ravel(),kind='stable'):
            a,b=divmod(int(flat),N)
            if a>=b or M[a,b]<=0:continue
            iv=[ids[a],ids[b]+1];sf=f.sampled(iv,av)
            if sf is not None and sf!=sample:best=float(M[a,b]);break
        if len(j['selection']['hypotheses'])==2:
            iv=j['selection']['hypotheses'][1]['interval_frames'];val=float(M[idx[iv[0]],idx[iv[1]-1]]);assert abs(val-best)<1e-12;maxerr=max(maxerr,abs(val-best));assert abs(val-j['selection']['I2_marginal_mass'])<1e-12
        else:assert best is None
        bins+=N*N;n+=1;del c,off,M,C;gc.collect()
        if n%32==0:print('MARGINAL_I2_INDEPENDENT',n,flush=True)
    write(BASE/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=n,scalar_checks=checks,baseline_scalar_checks=d['baseline_scalar_checks'],independent_CDF_bins=bins,max_ranking_mass_error=maxerr,I1_and_query_and_source_exact=256,unchanged_I2_exact_refit_controls=unchanged_checks,fitter_sha256=sha(ROOT/'scripts/stvg_temporal_hypothesis_fit32_v1.py'),only_I2_ranking_intervention=True,all_signed_unknown_reconstructed=True,GT_read=False,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json')));print('MARGINAL_I2_AUDIT_PASS',checks)
if __name__=='__main__':main()
