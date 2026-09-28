"""CPU law-of-total-variance audit of sealed source moments; no new inference."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
BASE=ROOT/'artifacts/desta3d_v2/tta_v2'
SOURCE=BASE/'source_moments_B1_v1'
OUT=BASE/'view_alignment_signal_source_v1'

def main():
    torch.set_num_threads(2)
    seal=read(SOURCE/'COMPLETE.json')['pins'];moments=read(SOURCE/'MOMENTS.json')
    assert sha(SOURCE/'MOMENTS.json')==seal[str(SOURCE/'MOMENTS.json')]
    rows=[]
    for p in sorted((SOURCE/'queries').glob('*.pt')):
        assert sha(p)==seal[str(p)]
        rows.append(torch.load(p,map_location='cpu',weights_only=False))
    parents=sorted({r['source'] for r in rows});assert len(rows)==618 and len(parents)==95
    counts={p:sum(r['source']==p for r in rows) for p in parents}
    w=np.array([1/(len(parents)*counts[r['source']]) for r in rows]);assert abs(w.sum()-1)<1e-12
    result={}
    for old,new in [('referent','spatial'),('event','event')]:
        mu=np.array([r['moments'][old]['mean'].numpy() for r in rows],dtype=np.float64)
        second=np.array([r['moments'][old]['second_moment'].numpy() for r in rows],dtype=np.float64)
        raw_var=second-mu*mu;assert raw_var.min()>-1e-10
        var=np.maximum(raw_var,0);global_mu=w@mu
        within=w@var;between=w@((mu-global_mu)**2);total=w@second-global_mu**2
        expected=np.asarray(moments[new]['std'])**2
        assert np.max(np.abs(total-expected))<1e-12
        assert np.max(np.abs(total-within-between))<1e-12
        frac=between/total
        p_mu=np.array([mu[[r['source']==p for r in rows]].mean(0) for p in parents])
        between_parents=np.mean((p_mu-global_mu)**2,axis=0)
        between_queries_in_parents=between-between_parents
        assert between_queries_in_parents.min()>-1e-12
        result[new]={'global_mean':global_mu.tolist(),'total_variance':total.tolist(),
            'within_query_THW_variance':within.tolist(),'between_query_mean_variance':between.tolist(),
            'between_parent_mean_variance':between_parents.tolist(),
            'between_query_within_parent_variance':between_queries_in_parents.tolist(),
            'between_mean_fraction_quantiles':np.quantile(frac,[0,.25,.5,.75,1]).tolist(),
            'between_mean_fraction_total_trace':float(between.sum()/total.sum()),
            'source_global_std_to_within_query_rms_std_quantiles':np.quantile(np.sqrt(total/within),[0,.25,.5,.75,1]).tolist(),
            'query_weighted_mean_std':(w@np.sqrt(var)).tolist(),
            'total_variance_reconstruction_max_error':float(np.max(np.abs(total-within-between))),
            'registered_variance_max_error':float(np.max(np.abs(total-expected)))}
    # Saved observed target initialization losses: no GT re-read or inference.
    target=BASE/'target8_B1_identity_view_v1';target_seal=read(target/'ALL_PREDICTIONS_SEAL.json')['pins']
    contexts=[]
    for condition in ['clean','noise_medium','defocus_extreme']:
        for i,row in enumerate(read(target/'INPUTS.json')):
            p=target/'episodes'/condition/f'{i:02}'/'calibration_alignment_identity_view.pt'
            assert sha(p)==target_seal[str(p)]
            obj=torch.load(p,map_location='cpu',weights_only=False)
            contexts.append({'key':row['key'],'source':row['source'],'cohort':row['cohort'],'condition':condition,
                'initial_observed_alignment':obj['update']['history'][0]['terms_before']['alignment']})
    save_once(OUT/'MOMENT_ESTIMAND_AUDIT.json',{'status':'passed','time':time.time(),
        'queries':618,'parents':95,'aggregation':'equal parents, equal queries within parent; THW population moments within query',
        'quantile_levels':[0,.25,.5,.75,1],'branches':result,'target_observed_initial_alignment':contexts,
        'GT_read':False,'new_inference':False,'GPU_seconds':0,'weights_changed':False,
        'interpretation':'global variance includes variation of query means, whereas each episodic target std is within one query; this is an estimand difference, not proof of harmful bias or a mandate to replace statistics',
        'source_moments_preserved':sha(SOURCE/'MOMENTS.json')})
    for n,r in result.items():print(n,r['between_mean_fraction_quantiles'],r['between_mean_fraction_total_trace'],r['source_global_std_to_within_query_rms_std_quantiles'])
    for c in ['clean','noise_medium','defocus_extreme']:
        x=[r['initial_observed_alignment'] for r in contexts if r['condition']==c];print(c,min(x),np.median(x),max(x))

if __name__=='__main__':main()
