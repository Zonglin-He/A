"""CPU-only presealed duration diagnostic, no parameter update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import read,write,sha,load,BASE,PUB,r1,c1,previous
import numpy as np
import torch
from scipy.stats import spearmanr
D=BASE/'duration';P=PUB/'duration'

def prepare():
    assert not (D/'UNLABELED_BARRIER.json').exists();rows=[];pins={};missing=[]
    def pin(p):pins[str(p.relative_to(ROOT))]=sha(p)
    pin(ROOT/'scripts/decota_duration_bias_v1.py');pin(ROOT/'protocols/decota_identity_commitment_v1.md')
    for ds in ['vidstg','hc2']:
        plan=read(r1.BASE/ds/'PLAN.json');pin(r1.BASE/ds/'PLAN.json')
        for split,sp in plan['splits'].items():
            for parent in sp['orders']['order1']:
                row=plan['rows'][parent]
                for cond in plan['conditions']:
                    ef=c1.POOL/ds/'experts/temporal'/cond/f"{row['pool_parent']:05}.json"
                    if not ef.exists():missing.append(dict(dataset=ds,split=split,source_id=parent,condition=cond,reason='no_previous_UVTG_cache'));continue
                    er=read(ef);ep=c1.POOL/ds/'experts'/er['cache'];assert sha(ep)==er['cache_sha256'] and not er['GT_read'];e=load(ep);assert not e['GT_read']
                    pp=np.asarray(e['proposals'],float).reshape(-1,2);assert np.isfinite(pp).all() and (pp[:,1]>pp[:,0]).all()
                    elog=float(np.median(np.log(pp[:,1]-pp[:,0])));pin(ef);pin(ep)
                    rcfile=c1.POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json";rc=read(rcfile);cf=c1.POOL/ds/rc['cache'];assert sha(cf)==rc['sha256']
                    within=load(cf);assert within['frame_ids']==row['frame_ids'] and within['pixel_sha256']==er['pixel_sha256'];pin(rcfile);pin(cf)
                    crossfile=previous.BASE/'cross_domain'/ds/'capture'/cond/f'{parent:05}.pt';crossrc=read(crossfile.with_suffix('.json'))
                    assert sha(crossfile)==crossrc['sha256'] and not crossrc['GT_read'];cross=load(crossfile);pin(crossfile);pin(crossfile.with_suffix('.json'))
                    assert cross['pixel_sha256']==within['pixel_sha256'] and cross['frame_ids']==within['frame_ids'] and not cross['GT_read']
                    for domain,x in [('same',within),('cross',cross)]:
                        iv=x['prediction']['physical_interval'];nlog=float(np.log(iv[1]-iv[0]));assert np.isfinite(nlog)
                        rows.append(dict(dataset=ds,split=split,source_id=parent,condition=cond,domain=domain,expert_log_duration=elog,native_log_duration=nlog,r_expert=elog-nlog,
                            raw_proposals=len(pp),pixel_sha256=er['pixel_sha256'],native_checkpoint_sha256=x['checkpoint_state_sha256'],GT_read=False))
    write(D/'UNLABELED_ROWS.json',rows);write(D/'MISSING.json',missing)
    write(D/'LOCK.json',dict(pins=pins,GT_read=False,rule='median all raw log durations, duplicates retained',two_orders_counted_once=True,
        label_inputs={str((c1.POOL/ds/f'GT_LABELS_{sp}.json').relative_to(ROOT)):sha(c1.POOL/ds/f'GT_LABELS_{sp}.json') for ds in ['vidstg','hc2'] for sp in ['search','confirm']}))
    write(D/'UNLABELED_BARRIER.json',dict(status='sealed',time=time.time(),rows=len(rows),unique_inputs=len(rows)//2,missing=len(missing),rows_sha256=sha(D/'UNLABELED_ROWS.json'),GT_read=False,new_GPU=0,new_expert=0,new_model_forwards=0))
    print('DURATION_UNLABELED_SEALED',len(rows),len(missing),flush=True)

def corr(a,b):
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])
def panel(rr):
    sources=sorted({r['source_id'] for r in rr});fields=['r_expert','r_GT','expert_log_duration','GT_log_duration','native_log_duration']
    a=np.array([[np.mean([r[f] for r in rr if r['source_id']==s]) for f in fields] for s in sources]);n=len(a)
    rng=np.random.default_rng(20261005);ix=rng.integers(n,size=(10000,n));b=a[ix]
    med=np.median(b[:,:,:2],axis=1)
    x,y=b[:,:,0],b[:,:,1];xc=x-x.mean(1,keepdims=True);yc=y-y.mean(1,keepdims=True);den=np.sqrt((xc*xc).sum(1)*(yc*yc).sum(1));cc=np.divide((xc*yc).sum(1),den,out=np.full(len(den),np.nan),where=den>1e-12)
    good=cc[np.isfinite(cc)];rho=None if len(a)<2 or np.std(a[:,0])<1e-12 or np.std(a[:,1])<1e-12 else float(spearmanr(a[:,0],a[:,1]).statistic)
    within=[]
    for s in sources:
        q=np.array([[r['r_expert'],r['r_GT']] for r in rr if r['source_id']==s]);within.extend(q-q.mean(0))
    within=np.asarray(within)
    return dict(inputs=len(rr),sources=n,source_ids=sources,source_averaged_rE_median=float(np.median(a[:,0])),source_averaged_rGT_median=float(np.median(a[:,1])),
        rE_median_ci95=np.quantile(med[:,0],[.025,.975]).tolist(),rGT_median_ci95=np.quantile(med[:,1],[.025,.975]).tolist(),
        aggregate_sign_agrees=bool(np.sign(np.median(a[:,0]))==np.sign(np.median(a[:,1]))),
        source_r_correlation=corr(a[:,0],a[:,1]),source_r_correlation_ci95=None if not len(good) else np.quantile(good,[.025,.975]).tolist(),finite_corr_bootstrap_draws=len(good),
        source_r_spearman=rho,source_expert_GT_log_duration_correlation=corr(a[:,2],a[:,3]),
        condition_within_source_centered_r_correlation=corr(within[:,0],within[:,1]),
        naive_input_r_correlation=corr([r['r_expert'] for r in rr],[r['r_GT'] for r in rr]),
        common_native_subtraction_can_inflate_r_correlation=True)

def score():
    assert read(D/'UNLABELED_BARRIER.json')['status']=='sealed';lock=read(D/'LOCK.json')
    for p,h in {**lock['pins'],**lock['label_inputs']}.items():assert sha(ROOT/p)==h,p
    write(D/'GT_EXPOSURE.json',dict(time=time.time(),unlabeled_barrier_sha256=sha(D/'UNLABELED_BARRIER.json'),GT_used_in_rule=False))
    rows=read(D/'UNLABELED_ROWS.json');labels={(ds,sp):read(c1.POOL/ds/f'GT_LABELS_{sp}.json') for ds in ['vidstg','hc2'] for sp in ['search','confirm']}
    out=[]
    for r in rows:
        span=labels[r['dataset'],r['split']][str(r['source_id'])]['span'];g=float(np.log(span[1]-span[0]));assert span[1]>span[0]
        out.append({k:v for k,v in r.items() if k not in ['pixel_sha256','native_checkpoint_sha256','GT_read']}|dict(GT_log_duration=g,r_GT=g-r['native_log_duration']))
    summary={}
    for ds in ['vidstg','hc2']:
        summary[ds]={}
        for sp in ['search','confirm']:
            summary[ds][sp]={}
            for dom in ['same','cross']:
                summary[ds][sp][dom]={}
                for condition,test in [('corruption',lambda r:r['condition']!='clean'),('clean',lambda r:r['condition']=='clean')]:
                    rr=[r for r in out if r['dataset']==ds and r['split']==sp and r['domain']==dom and test(r)];summary[ds][sp][dom][condition]=panel(rr)
    # A diagnostic gate, not permission to invent a calibration mechanism.
    checks={}
    for ds in ['vidstg','hc2']:
        z=summary[ds]['confirm']['cross']['corruption'];ci=z['source_r_correlation_ci95']
        checks[ds]=bool(z['aggregate_sign_agrees'] and ci is not None and ci[0]>0 and z['rE_median_ci95'][0]*z['rE_median_ci95'][1]>0 and z['rGT_median_ci95'][0]*z['rGT_median_ci95'][1]>0)
    write(P/'ROWS.json',out);write(P/'SUMMARY.json',summary);write(P/'MISSING.json',read(D/'MISSING.json'))
    write(P/'CONFIGURATION.json',dict(rule=lock['rule'],unique_inputs=len(out)//2,total_design_unique_inputs=576,historical_exposure=True,no_new_GPU=True,no_new_expert=True,duplicates_preserved=True,two_orders_counted_once=True))
    write(P/'DECISION.json',dict(cross_confirmation_diagnostic_positive=checks,slow_temporal_state_started=False,
        qualification=all(checks.values()),not_a_universal_temporal_impossibility_claim=True,common_subtraction_confound_disclosed=True))
    assert not torch.cuda.is_initialized();write(D/'COMPLETION.json',dict(status='completed_pending_root_review',time=time.time(),GPU_initialized=False,new_expert=0,rows=len(out)))
    print('DURATION_SCORE_DONE',len(out),checks,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
