"""Independent NumPy endpoint/CE readback; no GPU, target input, new GT or optimizer."""
import json, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256
OUT=ROOT/'artifacts/desta3d_v2/tta_v2/source_cast_probe_v1'
OLD=ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2'


def arr(t): return t.detach().float().cpu().numpy().astype(np.float64)


def main():
    assert not (OUT/'ROOT_RAW_READBACK.json').exists()
    for p,h in read(OUT/'COMPLETE.json')['pins'].items(): assert sha(Path(p))==h,p
    for p,h in read(OUT/'LOCK.json')['pins'].items(): assert sha(Path(p))==h,p
    d=torch.load(OUT/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
    report=read(OUT/'REPORT.json');cast={};maxerr=0.
    for name,entry in d['differences'].items():
        delta=arr(entry['precast_delta']);s=entry['sparse_postcast'];idx=s['indices'].numpy().astype(np.int64)
        assert idx.ndim==1 and (np.diff(idx)>0).all() and idx.min()>=0 and idx.max()<len(delta)
        before,after=arr(s['before']),arr(s['after']);pd=after-before
        assert np.all(pd!=0)
        # Independent bit-level round-to-nearest-ties-to-even BF16 at every changed position.
        for pre,post in [(s['pre_before'],before),(s['pre_after'],after)]:
            u=pre.numpy().view(np.uint32)
            rounded=((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
            assert np.array_equal(rounded.astype(np.float64),post)
        assert np.array_equal((s['pre_after'].numpy()-s['pre_before'].numpy()).astype(np.float64),delta[idx])
        continuous_count=int(np.count_nonzero(delta));post_count=len(idx)
        norms={'continuous_L2':float(np.linalg.norm(delta)),'postcast_L2':float(np.linalg.norm(pd)),
               'continuous_maxabs':float(np.max(np.abs(delta))),'postcast_maxabs':float(np.max(np.abs(pd))),
               'postcast_dot_continuous':float(np.dot(pd,delta[idx]))}
        norms['postcast_over_continuous_L2']=norms['postcast_L2']/norms['continuous_L2']
        for k,v in norms.items():
            maxerr=max(maxerr,abs(v-entry['stats'][k]));assert np.isclose(v,entry['stats'][k],atol=1e-12,rtol=1e-12),(k,v)
        erased=continuous_count-int(np.count_nonzero(delta[idx]))
        assert erased==entry['stats']['continuous_changed_postcast_unchanged']
        assert continuous_count==entry['stats']['continuous_nonzero'] and post_count==entry['stats']['postcast_nonzero']
        err2=float(np.dot(delta,delta)+np.dot(pd,pd)-2*np.dot(pd,delta[idx]))
        cast[name]={'elements':len(delta),'continuous_nonzero':continuous_count,'postcast_nonzero':post_count,
            'continuous_changed_postcast_unchanged':erased,'erased_fraction_of_continuous_changed':erased/continuous_count,
            'postcast_fraction_of_all':post_count/len(delta),**norms,
            'pre_post_delta_cosine':float(np.dot(pd,delta[idx])/(np.linalg.norm(pd)*np.linalg.norm(delta))),
            'post_minus_pre_delta_L2':float(np.sqrt(max(0,err2))),
            'sparse_rounding_verified_independent_integer_bits':True}
    for k,v in d['task_support'].items(): assert tensor_sha256(v)==d['task_support_sha'][k]
    assert d['input_identity']==read(OLD/'episodes/00/INPUT_IDENTITY.json')
    assert tensor_sha256(d['task_support']['input_ids'])==d['input_identity']['teacher_forced_input_sha']
    token_rows=[];aggregate={};max_ce_algebra_error=0.;max_aggregation_error=0.
    for branch in ['event','spatial']:
        arrays={}
        for state in ['B1','saved_supervised3']:
            entry=d['states'][state]['branches'][branch];ev=entry['tokens'];chunks=ev['chunks']
            assert entry['support_sha']==d['task_support_sha']
            arrays[state]={k:np.concatenate([arr(c[k]) for c in chunks]) for k in ['cross_entropy','target_logit','logsumexp']}
            a=arrays[state];positions=ev['positions'].numpy();targets=ev['targets'].numpy();ntp=ev['ntp'].numpy()
            assert len(positions)==entry['info']['tokens'] and np.all(np.diff(positions)>0)
            assert np.array_equal(d['task_support']['labels'][0,positions].numpy(),targets)
            assert np.array_equal(ntp,positions<int(d['task_support']['ptd_prefix_lengths'][0]))
            if state=='B1': ref=(positions.copy(),targets.copy(),ntp.copy())
            else: assert all(np.array_equal(x,y) for x,y in zip(ref,(positions,targets,ntp)))
            error=float(np.max(np.abs(a['logsumexp']-a['target_logit']-a['cross_entropy'])))
            max_ce_algebra_error=max(max_ce_algebra_error,error);assert error<1e-5
            # Scalar NumPy means differ only in the original FP32 chunk summation rounding.
            err=abs(float(a['cross_entropy'].mean())-entry['info']['ce'])
            max_aggregation_error=max(max_aggregation_error,err);assert err<1e-6
            for group,mask in [('ntp',ntp),('mtp',~ntp)]:
                assert int(mask.sum())==entry['info'][group+'_count']
                assert abs(float(a['cross_entropy'][mask].mean())-entry['info'][group+'_loss'])<1e-6
        a,b=arrays['B1'],arrays['saved_supervised3'];positions,targets,ntp=ref
        changes={k:b[k]-a[k] for k in a}
        aggregate[branch]={'tokens':len(positions),
            'ce_B1':d['states']['B1']['branches'][branch]['info']['ce'],
            'ce_supervised3':d['states']['saved_supervised3']['branches'][branch]['info']['ce'],
            'CE_decreased_tokens':int(np.sum(changes['cross_entropy']<0)),
            'CE_increased_tokens':int(np.sum(changes['cross_entropy']>0)),
            'CE_equal_tokens':int(np.sum(changes['cross_entropy']==0)),
            'mean_target_logit_change':float(changes['target_logit'].mean()),
            'mean_logsumexp_change':float(changes['logsumexp'].mean()),
            'max_abs_target_logit_change':float(np.max(np.abs(changes['target_logit']))),
            'NTP_mean_CE_change':float(changes['cross_entropy'][ntp].mean()),
            'MTP_mean_CE_change':float(changes['cross_entropy'][~ntp].mean())}
        aggregate[branch]['CE_change']=aggregate[branch]['ce_supervised3']-aggregate[branch]['ce_B1']
        for i,pos in enumerate(positions):
            token_rows.append({'branch':branch,'position':int(pos),'target_id':int(targets[i]),'NTP':bool(ntp[i]),
                'B1':{k:float(v[i]) for k,v in a.items()},'supervised3':{k:float(v[i]) for k,v in b.items()},
                'change':{k:float(v[i]) for k,v in changes.items()}})
    cfg=read(OLD/'CONFIG.json')
    initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
    final=torch.load(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
    step=torch.load(OLD/'episodes/00/supervised/step1.pt',map_location='cpu',weights_only=False)
    names=step['ordered_names'];assert names==d['linearization']['ordered_names']
    delta=np.concatenate([(final[n].numpy()-initial[n].numpy()).reshape(-1).astype(np.float64) for n in names])
    g=(step['raw']['task_event'].numpy()+step['raw']['task_spatial'].numpy()).astype(np.float64)
    linear=float(np.dot(g,delta));assert abs(linear-d['linearization']['initial_task_dot_actual_three_step_delta'])<1e-15
    finite=sum(e['CE_change'] for e in aggregate.values())
    out={'time':time.time(),'status':'passed','source_queries':1,'source_parents':1,'optimizer_steps':0,
         'raw_sha':sha(OUT/'RAW_ENDPOINTS.pt'),'checker_sha':sha(Path(__file__)),
         'cast':cast,'CE':aggregate,'full_token_rows':len(token_rows),
         'max_cast_stat_absolute_error':maxerr,'max_per_token_FP64_LSE_minus_target_vs_saved_FP32_CE_error':max_ce_algebra_error,
         'max_numpy_mean_vs_original_chunk_CE_error':max_aggregation_error,
         'initial_gradient_dot_actual_three_step_parameter_delta':linear,'finite_total_CE_change':finite,
         'finite_over_linear_signed_ratio':finite/linear,
         'scope':'same first source training query, two saved exact states, original teacher-forced PTD path; no new native or target data',
         'limitation':'full endpoint cast equality was live checked and endpoints hashed; independent raw checks cover full delta and every changed postcast position. Cannot prove BF16 causality or task benefit.'}
    save_once(OUT/'ROOT_TOKEN_READBACK.json',{'tokens':token_rows})
    save_once(OUT/'ROOT_RAW_READBACK.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
