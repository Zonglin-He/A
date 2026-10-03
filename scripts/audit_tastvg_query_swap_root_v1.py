"""Private readback of immutable probes, intervention bindings and every scalar."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_tastvg_query_swap_v1 import BASE,PUBLIC,ATLAS,LATENT,PLAN,POOL,read,write,sha,checked,verify,prefix
checks=0;maximum=0.

def compare(a,b,tol=1e-11):
    global checks,maximum
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape
    err=float(np.max(abs(a-b))) if a.size else 0.;checks+=a.size;maximum=max(maximum,err)
    assert np.isfinite(err) and err<=tol*max(1,float(np.max(abs(b))) if b.size else 1),(err,tol)

def feature_values(h,ids,pairs,fps):
    out=[];f=np.array(ids);context=[]
    for i,j in pairs:
        left=[n for n in range(len(ids)) if ids[i]-fps<=ids[n]<ids[i]]
        right=[n for n in range(len(ids)) if ids[j]+1<=ids[n]<ids[j]+1+fps]
        l=np.mean(h[left],axis=0) if left else np.zeros(256)
        r=np.mean(h[right],axis=0) if right else np.zeros(256)
        out.append(np.r_[h[i],h[j],np.mean(h[i:j+1],axis=0),l,r,h[i]-l,h[j]-r])
        context.append(dict(left_frames=len(left),right_frames=len(right),inside_frames=j-i+1))
    width=ids[-1]+1-ids[0]
    geometry=np.array([[(ids[i]-ids[0])/width,(ids[j]+1-ids[0])/width,
                        (ids[j]+1-ids[i])/width] for i,j in pairs])
    return np.asarray(out),geometry,context

def audit():
    verify(True);tick=time.time();co=read(BASE/'COHORT.json');cfg=read(PUBLIC/'CONFIG.json')
    seal=read(BASE/'GLOBAL_READOUT_SEAL.json');assert not seal['GT_read'] and seal['cells']==288
    accepted=read(BASE/'SMOKE_ROOT_ACCEPTANCE.json');assert accepted['status']=='passed' and not accepted['GT_read']
    for binding in accepted['checks']:
        assert sha(BASE/binding['dataset']/'SMOKE.json')==binding['receipt_sha256']
        assert accepted['time']<read(BASE/binding['dataset']/'CAPTURE_BARRIER.json')['time']
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==cfg['production_method_sha256']
    count=changed=0;diagnostics={};probe_models=0
    for ds in ['vidstg','hc2']:
        rows=read(PLAN/ds/'PLAN.json')['rows'];donors=read(BASE/'PRIVATE_QUERY_DONORS.json')[ds]
        mapping=[r for r in read(PUBLIC/'QUERY_MAPPING.json') if r['dataset']==ds]
        ordered=sorted(range(48),key=lambda i:hashlib.sha256(
            f'query-swap-v1|{ds}|{rows[i]["source"]}'.encode()).hexdigest())
        def valid_offset(offset):
            for j,i in enumerate(ordered):
                d=ordered[(j+offset)%48]
                if rows[i]['source']==rows[d]['source'] or rows[i]['input']['video_sha256']==rows[d]['input']['video_sha256']:return False
                if ' '.join(rows[i]['input']['caption'].casefold().split())==' '.join(rows[d]['input']['caption'].casefold().split()):return False
            return True
        offset=mapping[0]['cyclic_offset'];assert valid_offset(offset)
        assert not any(valid_offset(i) for i in range(1,offset))
        for q in mapping:
            i=q['source_index'];d=ordered[(ordered.index(i)+offset)%48]
            assert d==q['donor_index']==donors[str(i)]['donor_index']
            assert donors[str(i)]['caption']==rows[d]['input']['caption']
            assert hashlib.sha256(rows[d]['input']['caption'].encode()).hexdigest()==q['swap_caption_sha256']
            assert hashlib.sha256(rows[i]['input']['caption'].encode()).hexdigest()==q['true_caption_sha256']
            subject=read(POOL/ds/'subjects'/f'{d:05}.json')
            assert donors[str(i)]['subject']==subject['parses']['subject']
        sm=read(BASE/ds/'SMOKE.json');assert sm['status']=='passed' and len(sm['rows'])==2
        for r in sm['rows']:
            for field in ['pixel_parity','hidden_bitwise_parity','feature_bitwise_parity',
                          'A_spatial_bitwise_parity','A_interval_parity','no_GT','no_update']:
                assert r[field]
            z=checked(BASE/ds/'smoke'/f'{r["cell"]}.pt');true=checked(LATENT/ds/'target_features'/f'{r["cell"]}.pt')
            assert torch.equal(z['hidden_true'],true['hidden'])
        f=ATLAS/ds/'FROZEN_ATLAS.pt';assert sha(f)==cfg['source_frozen_probes'][ds]
        probe=checked(f);probe_models+=len(probe['models'])
        old={r['cell']:r['predictions'] for r in checked(ATLAS/ds/'SEALED_READOUT.pt')}
        outputs={r['cell']:r for r in checked(BASE/ds/'SEALED_READOUT.pt')}
        prior={r['cell']:r for r in read(ROOT/f'results/tastvg_temporal_information_atlas/2026-10-03/{ds}/ROWS.json') if r['domain']=='target'}
        scored={r['cell']:r for r in read(PUBLIC/ds/'ROWS.json')}
        gt={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']}
        diff=[]
        for c in co['cells']:
            if c['dataset']!=ds:continue
            tr=checked(ROOT/c['feature']);sw=checked(BASE/ds/'swap_features'/f'{prefix(c)}.pt')
            for key in ['cell_key','frame_ids','candidate_indices','anchor_index','pixel_sha256',
                        'A_state_pre_sha256','A_state_post_sha256']:
                assert sw[key]==tr[key],key
            assert sw['donor_index']==c['query_swap_donor']==donors[str(c['parent'])]['donor_index']
            assert not sw['GT_read'] and not sw['candidate_support_changed'] and sw['parameter_updates']==0
            oldstate=load_checked_state(ROOT/c['state_payload']);assert oldstate['pre_sha']==tr['A_state_pre_sha256']
            from methods.decota_final_simplified_v1.tensors import state_hash
            assert state_hash(oldstate['pre_state'])==tr['A_state_pre_sha256']==c['pre_sha']
            assert state_hash(oldstate['post_state'])==tr['A_state_post_sha256']==c['post_sha']
            for arm,feature in [('true',tr),('swap',sw)]:
                h=feature['hidden'].double().numpy();assert np.isfinite(h).all()
                x,g,ctx=feature_values(h,feature['frame_ids'],feature['candidate_indices'],rows[c['parent']]['input']['fps'])
                compare(x,feature['x']);compare(g,feature['geometry']);assert ctx==feature['context']
                ids=np.array(feature['frame_ids']);position=((ids-ids[0])/(ids[-1]+1-ids[0]))[:,None]
                interval=np.array([[ids[i],ids[j]+1] for i,j in feature['candidate_indices']]);s,e=gt[c['split']][str(c['parent'])]['span']
                overlap=np.maximum(0,np.minimum(interval[:,1],e)-np.maximum(interval[:,0],s))
                length=interval[:,1]-interval[:,0]
                labels={'event':((ids>=s)&(ids<e)).astype(float),'precision':overlap/length,
                        'recall':overlap/(e-s),'tiou':overlap/(length+e-s-overlap)}
                for name,p in outputs[tr['cell_key']][arm].items():
                    if name.endswith('/logit'):continue
                    fam,view,task,control=name.split('/');m=probe['models'][name]
                    if fam=='frame':xx=h if view=='Hidden' else position
                    else:
                        bounds={'Endpoint':(0,512),'Inside':(512,768),'Context':(768,1280),
                                'Contrast':(1280,1792),'Full':(0,1792)}
                        xx=g if view=='Geometry' else x[:,slice(*bounds[view])]
                    linear=((xx-np.array(m['mean']))/np.array(m['std']))@np.array(m['weight'])+m['bias']
                    expected=expit(linear) if m['model']=='logistic' else linear
                    compare(p,expected)
                    if arm=='true':compare(p,old[tr['cell_key']][name],tol=0)
                    y=labels[task];error=np.asarray(p)-y
                    metrics=scored[tr['cell_key']]['metrics'][arm+'/'+name]
                    compare([metrics['y'],metrics['y2'],metrics['prediction'],metrics['mse'],metrics['mae']],
                            [sum(y)/len(y),sum(y*y)/len(y),sum(p)/len(p),sum(error**2)/len(y),sum(abs(error))/len(y)])
                    variance=np.var(y)
                    if variance>1e-12:compare(metrics['within_r2'],1-np.mean(error**2)/variance)
                    else:assert metrics['within_r2'] is None
                    if task=='event':
                        compare(metrics['logloss'],np.mean(np.logaddexp(0,linear)-y*linear))
                        if len(np.unique(y))==2:
                            compare([metrics['auc'],metrics['ap']],[roc_auc_score(y,p),average_precision_score(y,p)])
                        else:assert metrics['auc'] is metrics['ap'] is None
                    if arm=='true':
                        for field,val in metrics.items():
                            oldval=prior[tr['cell_key']]['metrics'][name][field]
                            if val is None:assert oldval is None
                            else:compare(val,oldval,tol=0)
            d=(sw['hidden'].double()-tr['hidden'].double()).numpy()
            changed+=int(np.any(d!=0));count+=1
            diff.append(dict(cell=tr['cell_key'],source_index=c['parent'],condition=c['condition'],
                hidden_RMS_change=float(np.sqrt(np.mean(d*d))),hidden_max_change=float(np.max(abs(d)))))
        diagnostics[ds]=diff
        barrier=read(BASE/ds/'CAPTURE_BARRIER.json')
        for f,h in barrier['files'].items():assert sha(BASE/f)==h
        assert barrier['time']<seal['time']
    from scripts.audit_tastvg_query_swap_public_v1 import audit as public_audit
    independent=public_audit(PUBLIC);assert independent['status']=='passed'
    assert count==288 and probe_models==136 and not torch.cuda.is_initialized()
    receipt=dict(status='passed',time=time.time(),cells=count,unchanged_frozen_models=probe_models,
        scalar_checks=checks,maximum_numeric_error=maximum,changed_hidden_cells=changed,
        true_readout_and_metric_bitwise_parity=True,fixed_original_candidate_support=True,
        original_GT_only=True,all_scores_before_GT_join=True,production_method_unchanged=True,
        independent_public_audit=independent,CPU_wall_seconds=time.time()-tick)
    write(BASE/'FINAL_ROOT_AUDIT.json',receipt);write(PUBLIC/'ROOT_AUDIT.json',receipt)
    write(PUBLIC/'FEATURE_CHANGE.json',diagnostics);write(PUBLIC/'PUBLIC_AUDIT.json',independent)
    print(json.dumps(receipt,indent=2))

def load_checked_state(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256']
    return torch.load(p,map_location='cpu',weights_only=False)

if __name__=='__main__':audit()
