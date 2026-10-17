"""Independent CPU CDF pushforward and scalar reconstruction of sealed diagnostics."""
from pathlib import Path
import gc,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_temporal_b256_v1 import BASE,B,read,write,sha
def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f
    from scripts.report_stvg_temporal_hypothesis256_v1 import independent
    d=read(BASE/'RESULTS.json');p=read(BASE/'native_marginal_probe001/RESULTS.json')
    checks=independent(d['rows'],d['summary'],d['paired'])+independent(p['rows'],p['summary'],p['paired'])
    torch.set_num_threads(2);maxerr=0.;cases=0;bins=0;paired=0
    old={(r['case_id'],r['arm']):r['metrics']for r in d['rows']}
    for r in p['rows']:
        if r['arm']!='Native_marginal_MAP_CPU':assert r['metrics']==old[r['case_id'],r['arm']];paired+=len(r['metrics'])
    for r in read(B/'ROSTER_LOCK.json')['records']:
        c=load_case(r)[0];off=f.native.offsets(c,_hidden_from_source_case(c));ids=list(map(int,c['frame_ids']));idx={v:i for i,v in enumerate(ids)};n=len(ids);ps=[];ss=[];ee=[];cdfs=[]
        for o in off:
            prob=f.original._prob(o['base'])[0];s=np.array([idx[o['ids'][i]]for i in o['ij'][0]]);e=np.array([idx[o['ids'][i]]for i in o['ij'][1]])
            hist=np.bincount(s*n+e,weights=prob,minlength=n*n).reshape(n,n)
            cdfs.append(hist[::-1,:].cumsum(0)[::-1,:].cumsum(1));ps.append(prob);ss.append(s);ee.append(e)
        C=cdfs[0]*cdfs[1];pad=np.pad(C,((0,1),(1,0)));M=pad[:-1,1:]-pad[1:,1:]-pad[:-1,:-1]+pad[1:,:-1]
        outer=ps[0][:,None]*ps[1][None,:];keys=np.minimum(ss[0][:,None],ss[1][None,:])*n+np.maximum(ee[0][:,None],ee[1][None,:]);direct=np.bincount(keys.ravel(),weights=outer.ravel(),minlength=n*n).reshape(n,n)
        err=float(np.max(np.abs(M-direct)));assert err<1e-12 and abs(float(M.sum())-1)<1e-12 and float(M.min())> -1e-12;maxerr=max(maxerr,err)
        mode=int(direct.argmax());assert abs(float(M.ravel()[mode])-float(M.max()))<1e-12
        detail=next(z for z in p['details']if z['case_id']==r['probe_alias']);assert abs(float(M.ravel()[mode])-detail['mode_interval_mass'])<1e-12
        cases+=1;bins+=n*n;del c,off,outer,keys,M,direct;gc.collect()
        if cases%32==0:print('INDEPENDENT_CDF',cases,flush=True)
    for x in d['details']:assert abs(x['P_minus_F_vIoU_pp']+x['T2_minus_P_vIoU_pp']-x['T2_minus_F_vIoU_pp'])<1e-12
    write(BASE/'INDEPENDENT_READBACK.json',dict(status='pass',count=cases,scalar_checks=checks,baseline_scalars_exact=paired,independent_probability_bins=bins,max_CDF_vs_outer_product_absolute_error=maxerr,probability_sum_checks=256,mode_max_checks=256,additive_decomposition_checks=256,tiny_CDF_subtraction_roundoff_not_clamped=True,GT_read=False,new_models=0,new_fits=0,new_Teacher_calls=0,source_sha256=sha(Path(__file__)),diagnosis_sha256=sha(BASE/'RESULTS.json'),probe_sha256=sha(BASE/'native_marginal_probe001/RESULTS.json')))
    print('DIAGNOSIS_INDEPENDENT_PASS',cases,checks,flush=True)
if __name__=='__main__':main()
