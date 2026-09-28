"""Independent NumPy algebra readback, no models, labels, or target inputs."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
BASE=ROOT/'artifacts/desta3d_v2/tta_v2'
OUT=BASE/'view_alignment_signal_source_v1'

def norm(x): return float(np.sqrt(np.dot(x,x)))
def cosine(a,b): return float(np.dot(a,b)/(norm(a)*norm(b))) if norm(a)*norm(b)>0 else None

def main():
    torch.set_num_threads(2)
    done=read(OUT/'COMPLETE.json');assert done['queries']==done['parents']==4 and done['signal_cases']==8
    for p,h in {**read(OUT/'LOCK.json')['pins'],**done['pins']}.items():assert sha(Path(p))==h,p
    assert len(done['pins'])==12
    moments=read(BASE/'source_moments_B1_v1/MOMENTS.json')
    rows=[];max_norm_error=0.;max_alignment_error=0.;max_sum_error=0.
    names=None
    for i,row in enumerate(read(OUT/'INPUTS.json')):
        identity=read(OUT/'queries'/f'{i:02}_IDENTITY.json');pair={}
        for view in ['observed','mild']:
            x=torch.load(OUT/'queries'/f'{i:02}_{view}.pt',map_location='cpu',weights_only=False)
            assert x['key']==row['key'] and x['source']==row['source'] and x['adapter_sha']==done['adapter_unchanged']
            assert x['optimizer_steps']==0 and x['parameters_unchanged'] and x['backbone_grad_free']
            assert not x['GT_read'] and not x['target_data_read']
            if names is None:names=x['ordered_names']
            assert names==x['ordered_names']
            v={n:g.numpy().astype(np.float64) for n,g in x['raw_gradients'].items()}
            assert all(g.shape==(66816,) and np.isfinite(g).all() for g in v.values())
            norms={n:norm(g) for n,g in v.items()}
            max_norm_error=max(max_norm_error,max(abs(norms[n]-x['gradient_norms'][n]) for n in v))
            summed=sum(v[n] for n in x['terms'])
            max_sum_error=max(max_sum_error,float(np.max(np.abs(summed-v['total']))))
            assert np.allclose(summed,v['total'],atol=1e-7,rtol=2e-4)
            align=[]
            for branch in ['event','spatial']:
                z=x['branch_channel_moments'][branch]
                mean=z['mean'].numpy().astype(np.float64);std=z['std'].numpy().astype(np.float64)
                align.append(np.mean((mean-np.array(moments[branch]['mean']))**2)+np.mean((std-np.array(moments[branch]['std']))**2))
            max_alignment_error=max(max_alignment_error,abs(float(np.mean(align))-x['terms']['alignment']))
            assert norm(v['parameter_anchor'])==0
            if view=='observed':
                assert all(abs(x['terms'][n])<1e-7 and norms[n]<1e-6 for n in ['latent','referent','event'])
                assert norms['alignment']>0
            consistency=sum(v[n] for n in ['latent','referent','event'])
            pair[view]={'key':row['key'],'source':row['source'],'view':view,'terms':x['terms'],
                'weighted_gradient_norms':norms,'consistency_sum_norm':norm(consistency),
                'alignment_to_consistency_norm_ratio':norm(v['alignment'])/max(norm(consistency),1e-30),
                'alignment_consistency_cosine':cosine(v['alignment'],consistency),
                'alignment_total_cosine':cosine(v['alignment'],v['total'])}
            pair[view+'_raw']=v
        rows.append({'key':row['key'],'source':row['source'],'identity':identity,
            'observed':pair['observed'],'mild':pair['mild'],
            'view_total_to_observed_total_cosine':cosine(pair['observed_raw']['total'],pair['mild_raw']['total']),
            'view_alignment_to_observed_alignment_cosine':cosine(pair['observed_raw']['alignment'],pair['mild_raw']['alignment']),
            'view_total_change_relative_L2':norm(pair['mild_raw']['total']-pair['observed_raw']['total'])/norm(pair['observed_raw']['total'])})
    assert max_norm_error<1e-10 and max_alignment_error<1e-7
    # All saved source-query moments, no fresh inference or GT: contextual scale.
    source=BASE/'source_moments_B1_v1';seal=read(source/'COMPLETE.json')['pins']
    source_values=[]
    for path in sorted((source/'queries').glob('*.pt')):
        assert sha(path)==seal[str(path)]
        q=torch.load(path,map_location='cpu',weights_only=False);errors=[]
        for old,new in [('referent','spatial'),('event','event')]:
            m=q['moments'][old]['mean'].numpy().astype(np.float64)
            second=q['moments'][old]['second_moment'].numpy().astype(np.float64)
            std=np.sqrt(np.maximum(second-m*m,1e-12))
            errors.append(np.mean((m-np.array(moments[new]['mean']))**2)+np.mean((std-np.array(moments[new]['std']))**2))
        source_values.append({'key':q['key'],'source':q['source'],'unweighted_alignment':float(np.mean(errors))})
    assert len(source_values)==618 and len({r['source'] for r in source_values})==95
    result={'status':'passed','time':time.time(),'queries':4,'parents':4,'signal_cases':8,
        'parameters':66816,'max_norm_error':max_norm_error,'max_alignment_scalar_error':max_alignment_error,
        'max_gradient_sum_error':max_sum_error,'rows':rows,'source_moment_context':{
            'queries':618,'parents':95,'quantile_levels':[0,.25,.5,.75,1],
            'per_query_alignment_quantiles':np.quantile([r['unweighted_alignment'] for r in source_values],[0,.25,.5,.75,1]).tolist(),
            'all_rows':source_values,'GT_read':False,'new_inference':False},
        'optimizer_steps':0,'GT_read':False,'target_inputs_read':False,
        'limits':'initial gradients on four source parents, no task supervision; no task harm or target population claim'}
    save_once(OUT/'ROOT_RAW_READBACK.json',result)
    print({k:v for k,v in result.items() if k not in ['rows','source_moment_context']})
    for r in rows:print(r['key'],r['mild']['alignment_to_consistency_norm_ratio'],r['mild']['alignment_consistency_cosine'],r['view_total_to_observed_total_cosine'],r['view_total_change_relative_L2'])

if __name__=='__main__':main()
