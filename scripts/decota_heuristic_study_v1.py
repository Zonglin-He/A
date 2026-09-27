"""GT-free prediction stages for the explicitly requested heuristic study.

All old methods/caches are read-only. This version fixes the stable temporal
configuration, compares fixed-budget spatial rules, and permits dev-only gate
retuning. Scoring and GT-anchor oracles live in a separate module.
"""
import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.tune_decota_refine_v1 import configure,imports,SPATIAL,anchors

OUT=ROOT/'artifacts/decota_heuristic_study_v1'
PREV=ROOT/'artifacts/decota_refine_tuning_v1'
BS=['tastvg','tubedetr'];GS=['hc_to_vid','vid_to_hc'];SPLITS=['development','evaluation']
SEEDS=[20260910,20260911,20260912]
SELECTORS=['current','uniform','no_new','no_evidence','full_uniform']+[f'random{s}' for s in SEEDS]
FAMILIES=['main','uniform','no_new','no_evidence','full_uniform','random',
          'text_noun','text_full','gate_none','gate_score','absolute','direct']


def digest(x):return hashlib.sha256(str(x).encode()).hexdigest()


def prepare():
    if (OUT/'lock.json').exists():return
    from scripts.decota_refine_protocol_v1 import verify_code
    old=read(PREV/'lock.json');protected=verify_code(read(ROOT/'artifacts/decota_refine_v1/lock.json'))
    p=dict(version='decota_heuristic_study_v1',created=time.time(),seed=SEEDS[0],
        previous_lock=str(PREV/'lock.json'),previous_sha256=sha(PREV/'lock.json'),
        temporal_configs=old['selected_start'],spatial_default=SPATIAL,
        heads=old['head_files'],expert_snapshot=old['expert_snapshot'],expert_sha256=old['expert_sha256'],
        dev_label_manifest=old['dev_label_manifest'],dev_label_sha256=old['dev_label_manifest_sha256'],
        evaluation_label_specs=old['evaluation_label_specs'],protected_pins=protected,
        retrospective=True,globally_untouched=False,corruption=False,GT_used_for_predictions=False,
        study='All six heuristic families plus deployment/matched 2x2 and boundary sensitivity',
        tuning='Two coordinate gate sweeps at fixed K=8, stable temporal config for every family. K sensitivity is separate, not a selected winner.',
        grids=dict(phrase_threshold=[.15,.25,.35,.5],distinct_margin=[0.,.025,.05,.1],nms_iou=[.2,.5,.8]),
        prior_grids=dict(fraction=[i/20 for i in range(1,21)],tau=[.5,.6,.7,.8,.9,.95,.99],ramp=[0.,.1,.3,1.,3.,10.,30.,100.]),
        budget_grid=[2,4,8,16],seeds=SEEDS,families=FAMILIES,rows={})
    for split in SPLITS:
        p['rows'][split]={}
        for g in GS:
            rows=[]
            for j,r in enumerate(old[split][g]):
                rr=copy.deepcopy(r);q=rr['input'];rr['native']={}
                for b in BS:
                    if split=='development':
                        bar=read(old['native_caches'][b][g]['path'])
                        receipt=next(x for x in bar['receipts'] if x['index']==q['index'])
                        f=Path(receipt['path']);assert sha(f)==receipt['sha256']
                    else:f=PREV/'evaluation'/b/g/'native'/f'{j:03d}.pt'
                    x=load(f);assert not x['GT_used'] and x['frame_ids']==q['frame_ids']
                    rr['native'][b]=dict(path=str(f),sha256=sha(f))
                    if b=='tastvg':
                        rr['evidence']=r['native_evidence'] if split=='development' else x['evidence']
                rr['ordinal']=j;rows.append(rr)
            assert len(rows)==len({r['input']['source'] for r in rows})
            p['rows'][split][g]=rows
    for g in GS:
        dev=p['rows']['development'][g];ev=p['rows']['evaluation'][g]
        for key in ['source','video_sha256']:
            assert not {r['input'][key] for r in dev}&{r['input'][key] for r in ev}
    p['pins']={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    write(OUT/'lock.json',p)
    print('PREPARED', {s:{g:len(p['rows'][s][g]) for g in GS} for s in SPLITS},flush=True)


def plan():
    p=read(OUT/'lock.json')
    assert sha(p['previous_lock'])==p['previous_sha256']
    for f,h in {**p['protected_pins'],**p['pins']}.items():assert sha(ROOT/f)==h,f
    return p


def tpath(b,g,split,j):return OUT/'temporal'/b/g/split/f'{j:03d}.pt'


def native_row(r,b):
    z=r['native'][b];assert sha(z['path'])==z['sha256'];return load(z['path'])


def fraction(ij,ids):return (ids[ij[1]]+1-ids[ij[0]])/(ids[-1]+1-ids[0])


def placements(z,ids,ij):
    import numpy as np
    wanted=ids[ij[1]]+1-ids[ij[0]]
    legal=[(i,j) for i in range(len(ids)) for j in range(i+1,len(ids)) if ids[j]+1-ids[i]==wanted]
    center=(ids[0]+ids[-1]+1)/2
    out={'center':min(legal,key=lambda a:(abs((ids[a[0]]+ids[a[1]]+1)/2-center),a))}
    for seed in SEEDS:out[f'place_random{seed}']=legal[int(np.random.default_rng(seed).integers(len(legal)))]
    return out,len(legal)


def donor_permutation(rows,seed):
    """Derangement minimizes total frame-count mismatch, independent of GT."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    sizes=np.array([len(r['input']['frame_ids']) for r in rows]);n=len(rows)
    cost=abs(sizes[:,None]-sizes[None,:]).astype(float)
    cost+=np.random.default_rng(seed).uniform(0,.01,(n,n));np.fill_diagonal(cost,1e12)
    i,j=linear_sum_assignment(cost);assert (i!=j).all()
    return dict(zip(i.tolist(),j.tolist()))


def temporal(b,g):
    import numpy as np,torch
    from scripts.run_decota_story_v1 import fitted_result
    from methods.decota_v1 import decode
    from vg_tta.native_coverage_calibration_v1 import native_at_length
    from vg_tta.metrics import interval_from_logits
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure();imports(b)
    h=p['heads'][b][g];assert sha(h['path'])==h['sha256'];head=load(h['path']).cuda().eval().requires_grad_(False)
    before=state_digest(head);cfg=p['temporal_configs'][b][g];receipts=[]
    if b=='tastvg':
        from vg_tta.tastvg_baseline_expansion import replay_temporal_head_native as replay
        from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    else:from methods.decota_v1._fullspan import replay_temporal_head as replay
    for split in SPLITS:
        rows=p['rows'][split][g];cache=[]
        for j,r in enumerate(rows):
            f=tpath(b,g,split,j);nf=f.with_name(f.stem+'_own.pt')
            if nf.exists():x=load(nf)
            else:
                n=native_row(r,b);ids=n['frame_ids'];base=dict(indices=n['predictions']['frozen']['indices'],logits=n['native_logits'],boxes=n['predictions']['frozen']['boxes'],ids=ids,views=n.get('native_views'))
                result=fitted_result(head,[v.cuda() for v in n['head_inputs']],base,b,cfg,ablations=j==0)
                ij=result['predictions']['decota']['indices'];pos,nc=placements(n['native_logits'],ids,ij)
                intervals={k:list(v['indices']) for k,v in result['predictions'].items()}
                intervals.update({k:list(v) for k,v in pos.items()})
                x=dict(intervals=intervals,state=result['fitted_head_state'],audits=result['audits'],
                    candidate_count=nc,GT_used=False,config=cfg,source=r['input']['source'])
                assert state_digest(head)==before;save(nf,x)
            cache.append(x)
        for j,r in enumerate(rows):
            f=tpath(b,g,split,j)
            if not f.exists():
                x=copy.deepcopy(cache[j]);n=native_row(r,b);ids=n['frame_ids'];x['donors']={}
                for seed in SEEDS:
                    k=donor_permutation(rows,seed)[j];d=cache[k];private=copy.deepcopy(head)
                    private.load_state_dict(d['state'])
                    with torch.no_grad():zs=[replay(private,v.cuda(),'cuda:0') for v in n['head_inputs']]
                    extent=fitted_merge(zs,n['native_views'],ids) if b=='tastvg' else interval_from_logits(zs[0])
                    x['intervals'][f'donor{seed}']=list(decode(n['native_logits'],extent,ids,native_indices=x['intervals']['frozen'])['indices'])
                    fr=fraction(d['intervals']['decota'],rows[k]['input']['frame_ids'])
                    x['intervals'][f'shuffle{seed}']=list(native_at_length(n['native_logits'],ids,fraction=fr)['indices'])
                    x['donors'][str(seed)]=dict(source=rows[k]['input']['source'],ordinal=k,frame_count_distance=abs(len(ids)-len(rows[k]['input']['frame_ids'])),fraction=fr)
                    del private,zs
                assert state_digest(head)==before;save(f,x)
            receipts.append(dict(split=split,ordinal=j,path=str(f),sha256=sha(f)))
            status(OUT/'progress_temporal.json',dict(backbone=b,group=g,split=split,done=j+1,total=len(rows),time=time.time()))
            print('TEMPORAL',b,g,split,j+1,len(rows),flush=True)
    f=OUT/'temporal'/b/g/'barrier.json'
    if not f.exists():write(f,dict(receipts=receipts,source_head_unchanged=True,GT_used=False))


def select_positions(ids,extent,native,evidence,k=8,mode='current'):
    import numpy as np
    from vg_tta.tg_spatial_tta_v1 import keyframes
    if k==0:return []
    if mode=='current':return keyframes(ids,extent,native,evidence,k)
    if mode=='no_new':return keyframes(ids,extent,extent,evidence,k)
    a,b=(0,len(ids)-1) if mode=='full_uniform' else extent
    k=min(k,b-a+1);ids=np.asarray(ids);ev=np.asarray(evidence)
    if mode.startswith('random'):return sorted(np.random.default_rng(int(mode[6:])).choice(np.arange(a,b+1),k,replace=False).tolist())
    edges=np.linspace(ids[a],ids[b]+1,k+1);out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        pool=[i for i in range(a,b+1) if lo<=ids[i]<hi]
        if mode=='no_evidence':
            added=[i for i in pool if not native[0]<=i<=native[1]];pool=added or pool
        if pool:out.append(min(pool,key=lambda i:(abs(ids[i]-(lo+hi)/2),i)))
    while len(out)<k:
        rest=[i for i in range(a,b+1) if i not in out]
        out.append(max(rest,key=lambda i:(min(abs(ids[i]-ids[j]) for j in out),-i)))
    return sorted(out)


def evidence(r,b,n):
    if b=='tastvg':return r['evidence']
    from methods.decota_refine_v1.predictor import tube_inclusion
    return tube_inclusion(n['native_logits'][0])


def required_positions(p,r,g,split):
    needed={v:set() for v in ['parsed','noun','full']};ids=r['input']['frame_ids']
    for b in BS:
        n=native_row(r,b);t=load(tpath(b,g,split,r['ordinal']));ev=evidence(r,b,n)
        ni=t['intervals']['frozen'];ij=t['intervals']['decota']
        for mode in SELECTORS:
            needed['parsed'].update(select_positions(ids,ij,ni,ev,8,mode))
        for k in p['budget_grid']:needed['parsed'].update(select_positions(ids,ij,ni,ev,k))
        needed['parsed'].update(select_positions(ids,ni,ni,ev,8))
        positions=select_positions(ids,ij,ni,ev,8)
        for text in ['noun','full']:needed[text].update(positions)
    return {k:sorted(v) for k,v in needed.items()}


def epath(g,split,j,text,pos):return OUT/'expert'/g/split/f'{j:03d}'/text/f'{pos:03d}.pt'


def text_spec(r,mode):
    """Full-caption control keeps exact caption and identical referent gating.

    If lemmatized referent is absent, use explicit entity index validation;
    do not silently replace it with an arbitrary first noun.
    """
    v=r['visual_query'];entity=v['entity']
    if not v['phrase']:return '',entity
    return (v['phrase'] if mode=='parsed' else entity if mode=='noun' else r['input']['caption'].lower().strip()),entity


def expert_run(g):
    import gc,torch
    from vg_tta.tg_spatial_tta_v1 import SpatialExpert
    from vg_tta.dense_expansion_data_v1 import decode_raw
    from vg_tta.foreground_runtime import state_digest
    p=plan();configure();bar=OUT/'expert'/g/'barrier.json'
    if bar.exists():return
    expert=SpatialExpert(p['expert_snapshot']);before=state_digest(expert.model);receipts=[];calls=0;reuse=0
    assert sha(Path(p['expert_snapshot'])/'model.safetensors')==p['expert_sha256']
    for split in SPLITS:
        for r in p['rows'][split][g]:
            j=r['ordinal'];q=r['input'];need=required_positions(p,r,g,split);raw=None;old={}
            if split=='development':
                f=PREV/'expert'/g/f"{q['index']:06d}.pt"
                x=load(f);assert x['frame_ids']==q['frame_ids'] and not x['GT_used'];old={z['position']:z for z in x['probes']}
            for mode,positions in need.items():
                phrase,entity=text_spec(r,mode)
                # An absent literal referent is a declared parser failure for
                # this control, not a different noun chosen with labels.
                if not phrase or entity not in phrase:continue
                for pos in positions:
                    f=epath(g,split,j,mode,pos)
                    if not f.exists():
                        z=None
                        if mode!='parsed':
                            pf=epath(g,split,j,'parsed',pos)
                            if pf.exists():
                                candidate=load(pf)
                                if candidate['text']==phrase.lower().strip()+'.':z=candidate
                        if mode=='parsed':
                            if pos in old:z=old[pos]
                            elif split=='evaluation':
                                ff=PREV/'evaluation_expert'/g/f"{q['index']:06d}"/f'{pos:03d}.pt'
                                if ff.exists():z=load(ff);assert z['video_sha256']==q['video_sha256'] and not z['GT_used']
                        if z is not None:
                            assert z['text']==phrase.lower().strip()+'.' and z['frame_id']==q['frame_ids'][pos]
                            z=copy.deepcopy(z);z['reused']=True;reuse+=1
                        else:
                            if raw is None:
                                assert sha(q['video_path'])==q['video_sha256'];raw,ids=decode_raw(q);assert ids==q['frame_ids']
                            z=expert(raw[pos],phrase,entity);z.update(position=pos,frame_id=q['frame_ids'][pos],reused=False);calls+=1
                        z.update(video_sha256=q['video_sha256'],GT_used=False,text_mode=mode);save(f,z)
                    receipts.append(dict(path=str(f),sha256=sha(f),ordinal=j,split=split,mode=mode,position=pos))
            del raw,old;gc.collect()
            status(OUT/'progress_expert.json',dict(group=g,split=split,done=j+1,total=len(p['rows'][split][g]),new_calls=calls,reuse=reuse,time=time.time()))
            print('EXPERT',g,split,j+1,len(p['rows'][split][g]),'new',calls,'reused',reuse,flush=True)
    assert state_digest(expert.model)==before
    write(bar,dict(receipts=receipts,GT_used=False,state_unchanged=True,new_calls_this_run=calls,reused_this_run=reuse))


def expert_probes(g,split,j,mode):
    return [load(f) for f in sorted((OUT/'expert'/g/split/f'{j:03d}'/mode).glob('*.pt'))]


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','temporal','expert']);ap.add_argument('--backbone',choices=BS);ap.add_argument('--group',choices=GS)
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='temporal':temporal(a.backbone,a.group)
    else:expert_run(a.group)
