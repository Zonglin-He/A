"""Independent physical/annotation audit and portable anonymous scalar audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,load
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary
from scripts.spatial_guided_temporal_import_v1 import core
BASE=ROOT/'artifacts/tastvg_spatial_guided_temporal_p0_v1';PUB=ROOT/'results/tastvg_spatial_guided_temporal_p0/2026-10-04'

def t_iou(a,g):
    intersection=max(0.,min(a[1],g[1])-max(a[0],g[0]));union=a[1]-a[0]+g[1]-g[0]-intersection
    return intersection/union

def public(directory):
    directory=Path(directory);tick=time.monotonic();count=collections.Counter();maximum=0.
    def close(a,b):
        nonlocal maximum
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        e=float(np.max(np.abs(a-b))) if a.size else 0;assert e<1e-11,e
        maximum=max(maximum,e);count['numeric_scalars']+=a.size
    cfg=read(directory/'CONFIG.json');rows=read(directory/'ROWS.json');summ=read(directory/'SUMMARY.json')
    pred=read(directory/'PREDICTIONS.json');completion=read(directory/'SCORE_COMPLETION.json');bar=read(directory/'GLOBAL_PREDICTION_BARRIER.json')
    dep=read(directory/'DEPLOYABLE_BARRIER.json');accept=read(directory/'SMOKE_ROOT_ACCEPTANCE.json')
    assert len(rows)==len(pred)==288 and len({r['cell_key'] for r in rows})==288 and sum(r['condition']!='clean' for r in rows)==240
    assert dep['time']<bar['time']<completion['time'] and not dep['GT_read'] and not bar['temporal_GT_scoring_started']
    assert cfg['parameter_updates']==cfg['backward_calls']==cfg['backbone_calls']==0 and cfg['spatial_A_fixed'] and not cfg['DTA_started']
    assert accept['status']=='pass' and cfg['context_ratio']==1.5
    for f,h in completion['hashes'].items():assert sha(directory/f)==h;count['public_seal_hashes']+=1
    for r,p in zip(rows,pred):
        for k,v in p.items():assert r[k]==v
        assert not r['Full_A_ROI_GT_used'] and r['GT_ROI_spatial_GT_used'] and not r['GT_ROI_temporal_span_used']
        assert r['temporal_GT_scored_after_global_seal']
        for arm in core.ARMS:
            assert 0<=r[arm+'_t']<=r[arm+'_oracle_t']+1e-12<=1+1e-12
            close(r[arm+'_success'],float(r[arm+'_t']>.5));close(r[arm+'_disjoint'],float(r[arm+'_t']==0))
            s=r['selections'][arm];assert 0<=s['index']<s['proposal_count'] and 0<=s['interval'][0]<s['interval'][1]<=1.000001
        for a,b in [('A_ROI','Full'),('GT_ROI','Full'),('GT_ROI','A_ROI')]:close(r[a+'_minus_'+b+'_t'],r[a+'_t']-r[b+'_t'])
    for ds in summ:
        for sp in summ[ds]:
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp]
            for panel,z in summ[ds][sp].items():
                q=[r for r in rr if panel=='all' or (panel=='clean')==(r['condition']=='clean')]
                check=summary(q,list(z['metrics']));assert (check['sources'],check['cells'])==(z['sources'],z['cells'])
                for f in z['metrics']:
                    for k in ['mean','ci95','cell_macro','source_positive','source_negative']:close(z['metrics'][f][k],check['metrics'][f][k])
                    for sid,v in check['metrics'][f]['source_values'].items():close(v,z['metrics'][f]['source_values'][sid])
                for arm,saved in z['tails'].items():
                    d=[r[arm+'_t']-r['Full_t'] for r in q]
                    want=dict(gain=sum(x>1e-12 for x in d),harm=sum(x< -1e-12 for x in d),severe_harm_gt5pp=sum(x<-.05 for x in d),
                        severe_harm_gt20pp=sum(x<-.2 for x in d),new_disjoint=sum(r['Full_t']>0 and r[arm+'_t']==0 for r in q),
                        success_destroyed=sum(r['Full_t']>.5 and r[arm+'_t']<=.5 for r in q),success_rescued=sum(r['Full_t']<=.5 and r[arm+'_t']>.5 for r in q))
                    assert saved==want;count['tail_counts']+=len(want)
                for o,saved in z['orders'].items():
                    check=summary([r for r in q if r['order']==o],list(z['metrics']))
                    assert check['cells']==saved['cells']
                    for f in z['metrics']:
                        close(saved['metrics'][f]['mean'],check['metrics'][f]['mean']);close(saved['metrics'][f]['ci95'],check['metrics'][f]['ci95'])
    dec=read(directory/'DECISION.json');passed={ds+'/'+sp:summ[ds][sp]['corrupt']['metrics']['A_ROI_minus_Full_t']['ci95'][0]>0 for ds in summ for sp in summ[ds]}
    assert dec['A_ROI_panel_pass']==passed and (dec['status']=='GO_TEACHER_EVIDENCE_ONLY')==all(passed.values())
    with (directory/'ROWS.csv').open() as f:table=list(csv.DictReader(f))
    assert len(table)==288
    for r,z in zip(rows,table):
        assert r['cell_key']==z['cell_key']
        for k in core.FIELDS:close(r[k],float(z[k]))
    return dict(status='pass',checks=dict(count),max_absolute_error=maximum,CPU_wall_seconds=time.monotonic()-tick,time=time.time(),
        scope='All anonymous scalar arithmetic, choice metadata, counts, paired-source/bootstrap/order intervals, tails, CSV and pre-score barriers; no private pixels/weights reconstructed')

def root():
    tick=time.monotonic();count=collections.Counter();maximum=0.
    def close(a,b,tol=1e-12):
        nonlocal maximum
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        e=float(np.max(np.abs(a-b))) if a.size else 0;assert e<=tol,(e,tol);maximum=max(maximum,e);count['numeric_scalars']+=a.size
    lock=read(BASE/'RUNTIME_LOCK.json')
    code=dict(lock['code'])
    for f in sorted((BASE/'revisions').glob('*.json')):code.update(read(f)['pin_overrides'])
    for f,h in {**code,**lock['assets'],**lock['inputs'],**lock['GT_inputs']}.items():assert sha(f if f.startswith('/') else ROOT/f)==h;count['input_hashes']+=1
    cells=read(BASE/'COHORT.json')['cells'];rows={r['cell_key']:r for r in read(PUB/'ROWS.json')};atlas=read(BASE/'GT_BOXES_INPUT.json')
    a_boxes=read(BASE/'A_BOXES_INPUT.json');full=read(BASE/'FULL_INPUT.json');meta=read(BASE/'GT_INPUT_RECEIPT.json')
    dep=read(BASE/'DEPLOYABLE_BARRIER.json');glob=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert dep['time']<meta['time']<glob['time'] and meta['box_atlas_sha256']==sha(BASE/'GT_BOXES_INPUT.json')
    assert sha(BASE/'DEPLOYABLE_BARRIER.json')==glob['deployable_barrier_sha256']
    assert sha(BASE/'GT_ROI_BARRIER.json')==glob['GT_ROI_barrier_sha256'] and sha(BASE/'GT_INPUT_RECEIPT.json')==glob['GT_input_receipt_sha256']
    for a in ['A_ROI','GT_ROI']:
        for f,h in read(BASE/(a+'_BARRIER.json'))['receipts'].items():assert sha(BASE/f)==h;count['pre_score_receipts']+=1
    plans={ds:read(ROOT/'artifacts/tastvg_current_correction_views_v1'/ds/'PLAN.json') for ds in ['vidstg','hc2']}
    labels={(ds,sp):read(ROOT/'artifacts/tastvg_extended_sensitivity_v3'/ds/f'GT_LABELS_{sp}.json') for ds in plans for sp in ['search','confirm']}
    names=lambda c:f'{c["dataset"]}_{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
    for c in cells:
        k='/'.join(str(c[f]) for f in ['dataset','split','condition','order','arrival']);r=rows[k];p=plans[c['dataset']]['rows'][c['parent']]
        old=load(ROOT/c['old_payload']);assert (old['pre_sha'],old['post_sha'],old['pixel_sha256'])==(c['pre_sha'],c['post_sha'],c['pixel_sha256'])
        b=old['slow']['boxes'].numpy().astype(float);w,h=p['input']['width'],p['input']['height']
        xy=np.column_stack(((b[:,0]-b[:,2]/2)*w,(b[:,1]-b[:,3]/2)*h,(b[:,0]+b[:,2]/2)*w,(b[:,1]+b[:,3]/2)*h));close(xy,a_boxes[k],0)
        label=labels[(c['dataset'],c['split'])][str(c['parent'])];truth=label['truth'];keys=sorted(map(int,truth));gt=np.array([truth[str(i)] for i in keys],float)
        interp=[]
        for fid in p['frame_ids']:
            if fid<=keys[0]:q=gt[0]
            elif fid>=keys[-1]:q=gt[-1]
            else:
                right=int(np.searchsorted(keys,fid));left=right-1;alpha=(fid-keys[left])/(keys[right]-keys[left]);q=gt[left]*(1-alpha)+gt[right]*alpha
            interp.append(q)
        close(interp,atlas['boxes'][k],1e-10);count['GT_spatial_boxes']+=len(interp)
        fc=load(ROOT/full[k]['full_cache']);supports=dict(Full=fc['proposals']);confs=dict(Full=fc['proposal_confidence'])
        selections=dict(Full=full[k]['Full'])
        for arm in ['A_ROI','GT_ROI']:
            receipt=read(BASE/arm/(names(c)+'.json'));assert sha(BASE/receipt['cache'])==receipt['cache_sha256'];z=load(BASE/receipt['cache'])
            supports[arm]=z['proposals'];confs[arm]=z['confidence'];selections[arm]=z['selection']
            assert z['duration']==fc['duration'] and z['picked']==fc['picked_observations']
            assert not z['GT_temporal_span_used'] and z['GT_spatial_used']==(arm=='GT_ROI')
            source=xy if arm=='A_ROI' else np.asarray(atlas['boxes'][k])
            for bbox,win in zip(source,z['windows']):
                if not np.isfinite(bbox).all() or np.any(bbox[2:]<=bbox[:2]):assert win['fallback_full'];continue
                n=max(2,int(np.ceil(1.5*max(bbox[2]-bbox[0],bbox[3]-bbox[1]))))
                x=int(np.floor((bbox[0]+bbox[2]-n)/2));y=int(np.floor((bbox[1]+bbox[3]-n)/2))
                assert win==dict(fallback_full=False,x=x,y=y,side=n,padding=[max(0,-x),max(0,-y),max(0,x+n-w),max(0,y+n-h)])
                count['crop_windows']+=1
            assert receipt['A_state_pre_sha256']==c['pre_sha'] and receipt['A_state_post_sha256']==c['post_sha']
        length=p['frame_ids'][-1]-p['frame_ids'][0]+1;origin=p['frame_ids'][0]
        for arm in core.ARMS:
            pp=supports[arm];cf=confs[arm];at=max(range(len(cf)),key=lambda j:cf[j]);assert at==r['selections'][arm]['index']==selections[arm]['index']
            close([(x-origin)/length for x in pp[at]],r['selections'][arm]['interval'],0)
            values=[t_iou(x,label['span']) for x in pp];close(values[at],r[arm+'_t']);close(max(values),r[arm+'_oracle_t'])
            count['raw_support_proposals']+=len(pp)
        # Confirm the old Full teacher is exactly R2's scored baseline.
        count['state_chains']+=1
    oldrows={(r['cell_key'],r['arm']):r for r in read(ROOT/'results/tastvg_dta_expert_r2/2026-10-04/TEACHER_SELECTION_SCORED.json')}
    for k,r in rows.items():close(r['Full_t'],oldrows[(k,'EDeploy')]['continuous_teacher_tIoU'],0)
    # Independently reconstruct actual crop pixels for one predetermined cell
    # per dataset/split, in both arms. No feature/model forward is performed.
    from vg_tta.exact_frame_decode_audit_v2 import decode as vd
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hd
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    import hashlib
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            c=next(c for c in cells if c['dataset']==ds and c['split']==sp and c['condition']=='occlusion_5')
            k='/'.join(str(c[f]) for f in ['dataset','split','condition','order','arrival']);p=plans[ds]['rows'][c['parent']]
            from vg_tta import exact_frame_decode_audit_v2 as decoder_binding
            decoder_binding.decode=hd if ds=='hc2' else vd
            frames,ids=(vd if ds=='vidstg' else hd)(p['input']);shifted,pixel,_=observation(p,c['condition'],frames)
            assert pixel==c['pixel_sha256']
            for arm in ['A_ROI','GT_ROI']:
                receipt=read(BASE/arm/(names(c)+'.json'));z=load(BASE/receipt['cache'])
                for i in sorted(set(z['picked'])):
                    win=z['windows'][i]
                    if win['fallback_full']:image=shifted[i]
                    else:
                        # Row-by-row reconstruction independently of core.crop.
                        left,up,right,down=win['padding']
                        padded=np.pad(shifted[i],((up,down),(left,right),(0,0)),mode='edge')
                        x,y=win['x']+left,win['y']+up
                        image=np.ascontiguousarray(padded[y:y+win['side'],x:x+win['side']])
                    assert hashlib.sha256(image.tobytes()).hexdigest()==z['crop_pixel_sha256'][str(i)]
                    count['independent_pixel_crops']+=1
                input_hash=core.digest(dict(caption=p['input']['caption'],ids=ids,fps=p['input']['fps'],duration=z['duration'],
                    windows=z['windows'],picked=z['picked'],image_hashes=z['crop_pixel_sha256'],transform='stock_PE336_FP16_batch16'))
                assert input_hash==z['feature_input_sha256']==receipt['feature_input_sha256']
    portable=public(PUB);write(PUB/'PUBLIC_AUDIT.json',portable)
    out=dict(status='pass',checks=dict(count),max_absolute_error=maximum,public_audit=portable,
        CPU_wall_seconds=time.monotonic()-tick,spatial_states_and_all_prior_inputs_unchanged=True,
        oracle_scope='GT coordinate extension, never evaluated as GT boxes outside support',time=time.time())
    write(PUB/'ROOT_AUDIT.json',out);print(out['status'],out['checks'],flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='root':root()
    else:print(public(sys.argv[2] if len(sys.argv)>2 else PUB))
