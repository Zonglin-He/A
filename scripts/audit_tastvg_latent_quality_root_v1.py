"""Independent CPU readback of the source probe and immutable target replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys, json, time, hashlib, collections
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
PUB=ROOT/'results/tastvg_temporal_latent_quality/2026-10-03'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):Path(p).write_text(json.dumps(z,indent=2)+'\n')
def key(z):return '/'.join(str(z[k]) for k in ['dataset','split','condition','order','arrival'])

def audit():
    torch.set_num_threads(2);count=collections.Counter();errors=collections.defaultdict(float)
    cfg=read(PUB/'CONFIG.json');lock=read(BASE/'RUNTIME_LOCK.json');join=read(PUB/'LABEL_JOIN.json')
    seal=read(PUB/'GLOBAL_SCORE_SEAL.json');fitseal=read(BASE/'FIT_BARRIER.json')
    assert fitseal['target_GT_read'] is False and seal['target_GT_read'] is False
    assert fitseal['time']<seal['time']<join['time'] and join['target_GT_used_for_fit'] is False
    assert sha(BASE/'FIT_BARRIER.json')==seal['source_fit_barrier_sha256']
    assert sha(PUB/'SCORE_ROWS.json')==seal['score_rows_sha256']
    for f,h in {**lock['pins'],**lock['metadata'],**lock['source_annotation_pins']}.items():
        assert sha(ROOT/f)==h,f;count['immutable_file_hashes']+=1
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==cfg['production_method_sha256']
    inputs=read(BASE/'SOURCE_INPUTS.json');gt=read(BASE/'SOURCE_GT.json')
    scores={r['cell_key']:r for r in read(PUB/'SCORE_ROWS.json')}
    bindings=read(BASE/'TARGET_INPUT_BINDINGS.json')
    for f,h in bindings.items():assert sha(ROOT/f)==h;count['original_input_hashes']+=1

    def close(a,b,kind,atol=1e-9):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        err=float(np.max(np.abs(a-b))) if a.size else 0.
        assert err<atol,(kind,err);errors[kind]=max(errors[kind],err);count[kind+'_scalars']+=a.size

    def private(p):
        r=read(p.with_suffix('.json'));assert sha(p)==r['sha256'];count['payload_sha256']+=1
        assert r['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        return torch.load(p,map_location='cpu',weights_only=False)

    def assemble(z,fps):
        h=z['hidden'].double().numpy();ids=np.asarray(z['frame_ids']);xs=[];gs=[];contexts=[]
        pairs=[c['indices'] for c in z['candidates']] if 'candidates' in z else z['candidate_indices']
        assert h.shape==(len(ids),256) and np.isfinite(h).all()
        for i,j in pairs:
            assert 0<=i<j<len(ids)
            s,e=ids[i],ids[j]+1
            left=[k for k,v in enumerate(ids) if s-fps<=v<s]
            right=[k for k,v in enumerate(ids) if e<=v<e+fps]
            ml=sum((h[k] for k in left),np.zeros(256))/len(left) if left else np.zeros(256)
            mr=sum((h[k] for k in right),np.zeros(256))/len(right) if right else np.zeros(256)
            inside=sum(h[k] for k in range(i,j+1))/(j-i+1)
            xs.append(np.r_[h[i],h[j],inside,ml,mr,h[i]-ml,h[j]-mr])
            width=ids[-1]+1-ids[0];gs.append([(s-ids[0])/width,(e-ids[0])/width,(e-s)/width])
            contexts.append(dict(left_frames=len(left),right_frames=len(right),inside_frames=j-i+1))
        close(xs,z['x'],'independent_features');close(gs,z['geometry'],'independent_geometry')
        assert contexts==z['context'];count['context_counts']+=len(contexts)*3
        return np.asarray(xs),np.asarray(gs),pairs

    def tiou(candidates,label):
        vals=[]
        for c in candidates:
            s,e=c['physical_interval'];a,b=label;inter=max(0,min(e,b)-max(s,a))
            vals.append(inter/(e-s+b-a-inter))
        return np.asarray(vals)

    source_details={};source_export=[];contexts={}
    for ds in ['vidstg','hc2']:
        rr=inputs[ds];sourcebar=read(BASE/ds/'SOURCE_FEATURE_BARRIER.json')
        targetbar=read(BASE/ds/'TARGET_FEATURE_BARRIER.json');assert sourcebar['GT_read'] is False
        assert sourcebar['time']<fitseal['time']<targetbar['time']<seal['time']
        assert targetbar['model_restored'] and targetbar['target_GT_read'] is False
        targetrows=read(ROOT/'artifacts/tastvg_current_correction_views_v1'/ds/'PLAN.json')['rows']
        assert not {r['source'] for r in rr}&{r['source'] for r in targetrows}
        assert not {r['input']['video_sha256'] for r in rr}&{r['input']['video_sha256'] for r in targetrows}
        assert len({r['source'] for r in rr})==len(rr)
        train=[r['index'] for r in rr if r['split']=='train'];val=[r['index'] for r in rr if r['split']=='validation']
        assert {'train':len(train),'validation':len(val)}==cfg['source_counts'][ds]
        if ds=='hc2':
            chosen=set(sorted([r['source'] for r in rr],key=lambda s:hashlib.sha256(('temporal-latent-v1|'+s).encode()).hexdigest())[:16])
            assert chosen=={r['source'] for r in rr if r['split']=='validation'}
        # Read official training labels only, with an independent source/clip join.
        import ijson
        lookup={r['official_annotation_key']:r for r in rr} if ds=='hc2' else {
            (r['source'],r['input']['caption'].lower(),r['input']['start_frame'],r['input']['end_frame']):r for r in rr}
        matched=set();annotation=ROOT/'data/attribute_tta_official_release'/('vidstg' if ds=='vidstg' else 'hc-stvg2')/'annos/train.json'
        with annotation.open('rb') as f:
            for annotation_key,a in ijson.kvitems(f,'',use_float=True):
                if ds=='vidstg':
                    if a['qtype']!='declar':continue
                    r=lookup.get((a['vid'],a['sentence']['description'].lower(),a['used_segment']['begin_fid'],a['used_segment']['end_fid']))
                    if r is None:continue
                    interval=[a['ori_temp_gt']['begin_fid'],a['ori_temp_gt']['end_fid']+1]
                else:
                    r=lookup.get(annotation_key)
                    if r is None:continue
                    assert r['input']['caption'].lower()==a['English'].lower()
                    interval=[a['st_frame']-1,a['ed_frame']]
                    assert interval[1]-interval[0]==len(a['bbox'])
                assert interval==gt[ds][str(r['index'])];matched.add(r['index']);count['official_training_label_joins']+=1
        assert len(matched)==len(rr)
        xx={};gg={};yy={}
        for r in rr:
            p=BASE/ds/'source_features'/f'{r["index"]:04}.pt';z=private(p)
            assert z['GT_read'] is False and z['head_exact'] is True and z['split']==r['split']
            assert z['checkpoint_state_sha256']==cfg['checkpoint_state_sha256'][ds]
            assert z['source_video_sha256']==r['input']['video_sha256']
            assert sha(r['input']['video_path'])==r['input']['video_sha256']
            assert z['frame_ids']==r['frame_ids'] and len(z['candidates'])==32
            for c in z['candidates']:
                i,j=c['indices'];assert c['physical_interval']==[z['frame_ids'][i],z['frame_ids'][j]+1]
            x,g,_=assemble(z,r['input']['fps']);xx[r['index']]=x;gg[r['index']]=g
            yy[r['index']]=tiou(z['candidates'],gt[ds][str(r['index'])]);count['source_queries']+=1
            receipt=read(p.with_suffix('.json'));assert receipt['time']<sourcebar['time']
            assert sha(p.with_suffix('.json'))==sourcebar['files'][str(p.with_suffix('.json').relative_to(BASE))]
        model=private(BASE/ds/'FROZEN_PROBE.pt');summ=read(BASE/ds/'SOURCE_FIT_SUMMARY.json');source_details[ds]={}
        assert sha(BASE/ds/'FROZEN_PROBE.pt')==fitseal['probes'][ds]
        assert read((BASE/ds/'FROZEN_PROBE.pt').with_suffix('.json'))['time']<fitseal['time']
        for signal,feat in [('L',xx),('G',gg)]:
            tx=np.concatenate([feat[j] for j in train]);y=np.concatenate([yy[j] for j in train])
            vx=np.concatenate([feat[j] for j in val]);vy=np.concatenate([yy[j] for j in val]);m=model[signal]
            mean=tx.mean(0);std=tx.std(0);std[std<1e-8]=1.;bias=y.mean()
            close(mean,m['mean'],'training_standardization');close(std,m['std'],'training_standardization')
            close(bias,m['bias'],'training_intercept');norm=(tx-mean)/std
            a=norm.T@norm/len(y);b=norm.T@(y-bias)/len(y)
            # Solve the selected normal equation directly, independently of fit's eigensolver.
            w=np.linalg.solve(a+m['alpha']*np.eye(len(mean)),b)
            close(w,m['weight'],'direct_ridge_coefficients',1e-8)
            ev,u=np.linalg.eigh(a);basis=u.T@b;path=[]
            for alpha in cfg['alphas']:
                ww=u@(basis/(np.maximum(ev,0)+alpha));pred=(vx-mean)/std@ww+bias
                selected=[vy[j*32+int(np.argmax(pred[j*32:(j+1)*32]))] for j in range(len(val))]
                tr=norm@ww+bias;res=a@ww+alpha*ww-b
                item=dict(alpha=alpha,validation_top1_tIoU=float(np.mean(selected)),validation_MSE=float(np.mean((pred-vy)**2)),
                    training_MSE=float(np.mean((tr-y)**2)),normal_equation_max_error=float(np.max(np.abs(res))),prediction_min=float(pred.min()),prediction_max=float(pred.max()))
                for k,v in item.items():close(v,summ[signal]['path'][len(path)][k],'source_validation_path',1e-8)
                path.append(item)
            best=min(range(len(path)),key=lambda j:(-path[j]['validation_top1_tIoU'],path[j]['validation_MSE'],path[j]['alpha']))
            assert best==summ[signal]['selected_index'] and path[best]['alpha']==m['alpha']
            assert hashlib.sha256(np.asarray(m['weight'],dtype='<f8').tobytes()).hexdigest()==summ[signal]['coefficients_sha256']
            source_details[ds][signal]=dict(alpha=m['alpha'],independent_training_queries=len(train),independent_validation_queries=len(val),
                direct_solve_weight_error=float(np.max(np.abs(w-m['weight']))),normalization_training_only=True,validation_refit=False)
        for r in rr:
            z=private(BASE/ds/'source_features'/f'{r["index"]:04}.pt');ids=z['frame_ids'];width=ids[-1]+1-ids[0]
            ss={s:((feat[r['index']]-model[s]['mean'])/model[s]['std']@model[s]['weight']+model[s]['bias']).tolist() for s,feat in [('L',xx),('G',gg)]}
            source_export.append(dict(dataset=ds,source_id=r['index'],split=r['split'],
                source_id_sha256=hashlib.sha256(r['source'].encode()).hexdigest(),video_sha256=r['input']['video_sha256'],
                observed_frames=len(ids),candidate_intervals_normalized=[[(c['physical_interval'][0]-ids[0])/width,(c['physical_interval'][1]-ids[0])/width] for c in z['candidates']],
                candidate_t=yy[r['index']].tolist(),frozen_scores=ss,native_index=0,
                source_GT_used_for_labels=True,GT_coordinates_exported=False))
        cc=collections.Counter()
        for r in [r for r in read(BASE/'COHORT.json')['cells'] if r['dataset']==ds and r['scheduled']]:
            pre=f'{r["split"]}_{r["condition"]}_{r["order"]}_{r["arrival"]:05}'
            p=BASE/ds/'target_features'/f'{pre}.pt';z=private(p);e=scores[key(r)]
            assert z['GT_read'] is False and z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
            assert z['A_state_pre_sha256']==e['A_state_pre_sha256'] and z['A_state_post_sha256']==e['A_state_post_sha256']
            assert z['probe_sha256']==fitseal['probes'][ds]
            x,g,_=assemble(z,targetrows[r['parent']]['input']['fps'])
            for signal,feat in [('L',x),('G',g)]:
                m=model[signal];pred=(feat-m['mean'])/m['std']@m['weight']+m['bias']
                close(pred,e['scores'][signal],'frozen_target_scores')
                for n in [8,32]:
                    top=np.where(pred[:n].max()-pred[:n]<=1e-12)[0];anchor=e['anchor_index']
                    choice=int(top[0]) if len(top)==1 and pred[top[0]]>pred[anchor]+1e-12 else anchor
                    assert choice==e['choices'][signal+str(n)];count['target_decisions']+=1
            for c in z['context']:
                cc['candidates']+=1;cc['empty_left']+=c['left_frames']==0;cc['empty_right']+=c['right_frames']==0
            receipt=read(p.with_suffix('.json'));assert fitseal['time']<receipt['time']<targetbar['time']
            assert sha(p.with_suffix('.json'))==targetbar['files'][str(p.with_suffix('.json').relative_to(BASE))]
            count['target_expert_arrivals']+=1
        contexts[ds]=dict(cc)
        from scripts.audit_tastvg_large_evidence_public_v1 import independent_summary
        for split in ['search','confirm']:
            rows=read(PUB/split/ds/'ROWS.json');prior=read(ROOT/'results/tastvg_temporal_boundary_support/2026-10-03'/split/ds/'ROWS.json')
            old=read(ROOT/'results/tastvg_large_correction_evidence/2026-10-03'/split/ds/'ROWS.json')
            for z,r,l in zip(rows,prior,old):
                assert key(z)==key(r)==key(l)
                for f in ['A8_v','A8_t','A_state_pre_sha256','A_state_post_sha256']:assert z[f]==r[f]
                if r['expert_scheduled']:
                    assert z['candidate_v']==r['candidate_v'] and z['candidate_t']==r['candidate_t']
                    for arm in ['L8','L32','G8','G32']:
                        for s in ['N','U','S']:
                            for met in ['v','t']:close(z[arm+'_vs_'+s+arm[1:]+'_'+met],z[arm+'_'+met]-l[s+arm[1:]+'_'+met],'legacy_scalar_comparisons')
                        assert z[arm+'_destroyed_old_fast']==float(r['old_fast_gain']>1e-12 and z[arm+'_gain']< -1e-12)
                else:
                    for arm in ['L8','L32','G8','G32']:assert z[arm+'_v']==r['A8_v'] and z[arm+'_t']==r['A8_t']
                count['target_metric_rows']+=1
    assert count['source_queries']==190 and count['target_expert_arrivals']==288 and count['target_metric_rows']==1152
    assert count['target_decisions']==1152 and not torch.cuda.is_initialized()
    result=dict(status='pass',checks=dict(count),maximum_errors=dict(errors),source_probe_audits=source_details,
        contexts=contexts,source_supervision_only=True,target_annotation_files_opened=0,
        target_GT_used_for_fit=False,target_GT_join_after_global_score_seal=True,production_unchanged=True,
        CUDA_initialized=False,new_model_forwards=0,live_head_parity_scope='Per-offset bitwise checks performed during capture; root independently verifies saved features and scores without a new model replay.',time=time.time())
    write(BASE/'FINAL_ROOT_AUDIT.json',result);write(PUB/'ROOT_READBACK.json',result)
    write(PUB/'SOURCE_ROWS.json',source_export)
    (PUB/'SOURCE_FIT_SEAL.json').write_bytes((BASE/'FIT_BARRIER.json').read_bytes())
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':audit()
