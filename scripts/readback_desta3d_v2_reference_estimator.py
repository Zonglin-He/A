"""NumPy raw-gradient and scalar audit of the source-only reference comparison."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_reference_estimator_probe import OUT,OLD,SOURCE,read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once

def norm(v):return float(np.sqrt(np.dot(v,v)))
def compare(a,b):
    return {'old_norm':norm(a),'new_norm':norm(b),'norm_ratio':norm(b)/norm(a) if norm(a) else None,
        'cosine':float(np.dot(a,b)/(norm(a)*norm(b))) if norm(a)*norm(b) else None,
        'relative_L2_change':norm(b-a)/norm(a) if norm(a) else None,
        'max_abs_change':float(np.max(np.abs(b-a)))}

def main():
    torch.set_num_threads(2)
    complete=read(OUT/'COMPLETE.json');cfg=read(OUT/'CONFIG.json')
    assert complete['signal_cases']==8 and complete['queries']==complete['parents']==4
    for p,h in {**read(OUT/'LOCK.json')['pins'],**complete['pins']}.items():assert sha(Path(p))==h,p
    assert len(complete['pins'])==12
    refs={'population':read(SOURCE/'MOMENTS.json'),'expected_query_std':read(OUT/'ALTERNATIVE_MOMENTS.json')}
    for branch in ['event','spatial']:assert refs['population'][branch]['mean']==refs['expected_query_std'][branch]['mean']
    state=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
    rows=[];norm_error=0.;scalar_error=0.;sum_error=0.;repeat_max=0.;control_exact=True
    for i,row in enumerate(read(OUT/'INPUTS.json')):
        identity=read(OUT/'queries'/f'{i:02}_IDENTITY.json')
        assert identity==read(OLD/'queries'/f'{i:02}_IDENTITY.json')
        cases={};vectors={}
        for arm in refs:
            x=torch.load(OUT/'queries'/f'{i:02}_{arm}.pt',map_location='cpu',weights_only=False)
            assert x['key']==row['key'] and x['source']==row['source'] and x['reference']==arm
            assert x['optimizer_steps']==0 and x['parameters_unchanged'] and x['backbone_grad_free']
            assert not x['GT_read'] and not x['target_data_read']
            assert x['adapter_sha']==complete['adapter_unchanged']==cfg['checkpoint']['adapter_sha256']
            v={n:g.numpy().astype(np.float64) for n,g in x['raw_gradients'].items()}
            assert all(g.shape==(66816,) and np.isfinite(g).all() for g in v.values())
            norm_error=max(norm_error,max(abs(norm(g)-x['gradient_norms'][n]) for n,g in v.items()))
            algebra=sum(v[n] for n in x['terms'])
            err=float(np.max(np.abs(algebra-v['total'])));sum_error=max(sum_error,err)
            assert np.allclose(algebra,v['total'],atol=1e-7,rtol=2e-4)
            scalars=[]
            for branch in ['spatial','event']:
                stats=x['branch_channel_moments'][branch]
                a=stats['mean'].numpy().astype(np.float64);b=stats['std'].numpy().astype(np.float64)
                scalars.append(np.mean((a-np.asarray(refs[arm][branch]['mean']))**2)+
                    np.mean((b-np.asarray(refs[arm][branch]['std']))**2))
            scalar_error=max(scalar_error,abs(float(np.mean(scalars))-x['terms']['alignment']))
            if arm=='population':
                control=torch.load(OLD/'queries'/f'{i:02}_mild.pt',map_location='cpu',weights_only=False)
                for name,g in x['raw_gradients'].items():
                    actual=float((g-control['raw_gradients'][name]).abs().max());repeat_max=max(repeat_max,actual)
                    control_exact=control_exact and torch.equal(g,control['raw_gradients'][name])
                    assert torch.allclose(g,control['raw_gradients'][name],**cfg['control_repeat_tolerance'])
            cases[arm]=x;vectors[arm]=v
        assert cases['population']['ordered_names']==cases['expected_query_std']['ordered_names']
        for term in ['latent','referent','event','parameter_anchor']:
            assert cases['population']['terms'][term]==cases['expected_query_std']['terms'][term]
            assert np.array_equal(vectors['population'][term],vectors['expected_query_std'][term])
        result={'key':row['key'],'source':row['source'],
            'terms':{n:x['terms'] for n,x in cases.items()},
            'alignment':compare(vectors['population']['alignment'],vectors['expected_query_std']['alignment']),
            'total':compare(vectors['population']['total'],vectors['expected_query_std']['total']),
            'per_tensor':{}}
        offset=0
        for name in cases['population']['ordered_names']:
            length=state[name].numel()
            result['per_tensor'][name]=compare(vectors['population']['total'][offset:offset+length],vectors['expected_query_std']['total'][offset:offset+length])
            offset+=length
        assert offset==66816
        rows.append(result)
    assert norm_error<1e-10 and scalar_error<1e-7
    rule=cfg['near_same_resource_rule']
    near_same=all(x['total']['cosine']>=rule['all_total_cosine_at_least'] and
        x['total']['relative_L2_change']<=rule['all_relative_L2_at_most'] for x in rows)
    report={'status':'passed','time':time.time(),'queries':4,'parents':4,'cases':8,'rows':rows,
        'control_repeat_all_exact':control_exact,'control_repeat_max_abs':repeat_max,
        'max_gradient_norm_error':norm_error,'max_alignment_scalar_error':scalar_error,'max_gradient_sum_error':sum_error,
        'unaffected_component_gradients_exact':True,'mean_reference_exact':True,
        'predeclared_resource_rule':rule,'near_same_initial_signal':near_same,
        'resource_decision':'no target trial; lower estimator-only hypothesis' if near_same else 'review sensitivity before any separately registered target trial',
        'optimizer_steps':0,'source_GT_read':False,'target_inputs_read':False,
        'limits':'four source parents at initial B1; no task-gradient, Adam trajectory or target efficacy inference'}
    save_once(OUT/'ROOT_RAW_READBACK.json',report)
    print({k:v for k,v in report.items() if k!='rows'})
    for x in rows:print(x['key'],x['alignment'],x['total'])

if __name__=='__main__':main()
