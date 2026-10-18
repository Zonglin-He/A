"""Sealed-cache geometric/category/scope counterfactuals, not a new TTA method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,itertools,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_error_state_correction_v1 import encode,decode,correct
OLD=ROOT/'artifacts/stvg_error_state32_v1'
REVIEW=ROOT/'artifacts/stvg_dino_error_attribution32_v1'
D=ROOT/'artifacts/stvg_error_state_hc_diagnosis32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/hc_diagnosis32'
GROUPS=['WI','CP','MX','CI','UN']


def bindings():
    old=read(OLD/'PLAN_LOCK.json');r=read(REVIEW/'PROTOCOL_LOCK.json')
    roster=read(ROOT/'artifacts/stvg_joint_event_critic_v1/PLAN_LOCK.json')['target']
    assert [(x['alias'],x['dataset'],x['query_ordinal'],x['parent'])for x in old['records']]==[(x['alias'],x['dataset'],x['query_ordinal'],x['parent'])for x in roster]
    for f,h in r['inputs'].items():assert sha(ROOT/f)==h,f
    v=read(REVIEW/'VISUAL_REVIEW.json');panels=read(REVIEW/'PIXEL_PANEL_MANIFEST.json')['panels']
    assert len(v['rows'])==32 and v['model_assisted'] and v['independent_human_labels']==0
    for row in v['rows']:
        f=next(p for p in panels if p['source_alias']==row['source_alias'])
        assert sha(ROOT/f['path'])==f['sha256']==row['actual_panel_sha256']
    return old,{x['source_alias']:x for x in v['rows']}


def prepare():
    assert not(D/'PLAN_LOCK.json').exists();old,review=bindings()
    own=['scripts/run_stvg_error_state_hc_diagnosis32_v1.py','protocols/stvg_error_state_hc_diagnosis32_v1.md',
         'vg_tta/stvg_error_state_correction_v1.py','vg_tta/tastvg_oracle_event5_v1.py']
    inputs={}
    for r in old['records']:
        for folder in ['dino_full','oracle','oracle_sparse_inputs']:
            f=OLD/folder/(r['alias']+'.pt');load(f);inputs[str(f.relative_to(ROOT))]=sha(f)
    for f in [OLD/'PLAN_LOCK.json',OLD/'CPU_COMPLETION.json',ROOT/'results/stvg_error_state/2026-10-18/cached32/RESULTS.json',REVIEW/'VISUAL_REVIEW.json',REVIEW/'ROOT_PIXEL_BINDING.json',REVIEW/'PIXEL_PANEL_MANIFEST.json']:
        inputs[str(f.relative_to(ROOT))]=sha(f)
    put(D/'PLAN_LOCK.json',dict(status='locked_before_new_diagnostic_scores',count=32,records=old['records'],
        primary='HC16/Before/final_MBR',context='Vid16 and Source sensitivity',groups=GROUPS,
        uncertain_role_queries=[a for a,v in review.items()if v['semantic_role_unknown']],
        common_GT_repair_is_oracle_only=True,scope_masks=['global','planned_hull','admitted_nodes'],
        geometry_factors=['center','size'],no_automatic_selection_or_followup=True,
        CPU_seconds_cap=3600,new_GPU_models_experts_training_updates=0,created_epoch=time.time()))
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},inputs=inputs,
        registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))
    put(D/'REVIEW_BINDING.json',dict(status='pass',count=32,old_pixel_panels_SHA_bound=True,
        old_query_parent_cohort_exact=True,model_assisted_not_new_human_labels=True,
        new_pixels_read=False,uncertain_roles_overridden_to_UN=True))


def verify():
    p=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json')
    for f,h in {**lock['pins'],**lock['registries'],**lock['inputs']}.items():assert sha(ROOT/f)==h,f
    return p


def candidates():
    import torch
    torch.set_num_threads(4);p=verify();_,review=bindings();files=[];started=time.perf_counter()
    for r in p['records']:
        a=r['alias'];z=load(OLD/'dino_full'/(a+'.pt'));o=load(OLD/'oracle'/(a+'.pt'));rv=review[a];refs=o['references'];pos=refs['GT_common']['positions'];gt=refs['GT_common']['targets'];dn=refs['DINO_common']['targets'];assert pos==refs['DINO_common']['positions']
        original=[rv['slots'][z['planned_positions'].index(j)]for j in pos]
        cats=['UN'if rv['semantic_role_unknown']else c for c in original];present=[c for c in GROUPS if c in cats]
        assert set(cats)<=set(GROUPS)
        labels_full=['UN'if rv['semantic_role_unknown']else rv['slots'][z['planned_positions'].index(j)]for j in z['admitted_positions']]
        outputs={};coord={};anchor_records=[]
        for i,j in enumerate(pos):
            ez=encode(dn[[i]])[0]-encode(gt[[i]])[0]
            anchor_records.append(dict(slot=z['planned_positions'].index(j),original_code=original[i],category=cats[i],
                log_width_ratio=float(ez[2]),log_height_ratio=float(ez[3]),area_ratio=float(dn[i,2:].prod()/gt[i,2:].prod()),
                center_error_in_target_sizes=float(np.linalg.norm(ez[:2]/gt[i,2:])),
                reference_IoU=_iou_cx(dn[i],gt[i])))
        for base,b in z['bases'].items():
            out={'Base':b.copy()};meta={};co={}
            for mask in range(1<<len(present)):
                target=dn.copy();changed=[g for k,g in enumerate(present)if(mask>>k)&1]
                for i,g in enumerate(cats):
                    if g in changed:target[i]=gt[i]
                label='groups_'+str(mask);out[label]=correct(b,pos,target)['global_error'];meta[label]=dict(kind='group_repair',repaired=changed)
            for mask in range(4):
                target=dn.copy()
                if mask&1:target[:,:2]=gt[:,:2]
                if mask&2:target[:,2:]=gt[:,2:]
                label='geometry_'+str(mask);out[label]=correct(b,pos,target)['global_error'];meta[label]=dict(kind='geometry',GT_center=bool(mask&1),GT_size=bool(mask&2))
            for ref,pp,tt in [('DINO_full',z['reference_positions'],z['reference_targets']),('DINO_common',pos,dn),('GT_common',pos,gt)]:
                cc=correct(b,pp,tt);delta=cc['delta'];co[ref]=delta
                for scope in ['global','planned_hull','admitted_nodes']:
                    mask=np.ones(len(b),bool)if scope=='global'else(np.arange(len(b))>=min(z['planned_positions']))&(np.arange(len(b))<=max(z['planned_positions']))if scope=='planned_hull'else np.isin(np.arange(len(b)),z['admitted_positions'])
                    zz=encode(b);zz[mask]+=delta;bb=decode(zz)
                    if not pp:bb=b.copy()
                    assert np.array_equal(bb[~mask],b[~mask])or np.max(abs(bb[~mask]-b[~mask]),initial=0)<1e-15
                    label='scope_'+ref+'_'+scope;out[label]=bb;meta[label]=dict(kind='scope',reference=ref,scope=scope)
            category_delta={g:(encode(dn)[np.array(cats)==g]-encode(gt)[np.array(cats)==g]).sum(0)/len(pos)if pos else np.zeros(4)for g in GROUPS}
            noise=co['DINO_common']-co['GT_common'];assert np.max(abs(sum(category_delta.values())-noise))<1e-12
            co['category_delta_error']=category_delta;co['DINO_minus_GT_delta']=noise;coord[base]=co
            # Existing matched endpoints must be replayed without selecting new outputs.
            for name,ref in [('geometry_0','DINO_common'),('geometry_3','GT_common')]:
                assert np.max(abs(out[name]-o['corrections'][ref][base]['global_error']))<1e-14
            assert np.max(abs(out['scope_DINO_full_global']-z['corrections'][base]['global_error']))<1e-14
            outputs[base]=out
        files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],query_ordinal=r['query_ordinal'],
            frame_ids=z['frame_ids'],interval=z['intervals']['final_MBR'],planned_positions=z['planned_positions'],admitted_positions=z['admitted_positions'],
            common_positions=pos,common_categories=cats,full_categories=labels_full,groups_present=present,
            anchor_geometry=anchor_records,metadata=meta,outputs=outputs,delta_diagnostics=coord,
            inherited_old_visual_labels=True,new_visual_labels=False,new_dense_GT_used_for_candidates=False)))
        assert time.perf_counter()-started<3600
    assert not torch.cuda.is_initialized()
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,
        sparse_cached_GT_only_for_oracle_generation=True,new_dense_GT_read=False,new_models=0,
        CPU_wall_seconds=time.perf_counter()-started))


def _iou_cx(a,b):
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    return float(box_iou(np.r_[a[:2]-a[2:]/2,a[:2]+a[2:]/2],np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]))


def avg(v):
    x=[float(a)for a in v if a is not None];return float(np.mean(x))if x else None


def paired(values):
    v=np.array([x for x in values if x is not None],float)
    if not len(v):return dict(mean=None,CI95=None,parents=0)
    ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
    return dict(mean=float(v.mean()),CI95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),parents=len(v),
        positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),neutral=int((abs(v)<=1e-10).sum()),per_parent=v.tolist())


def shapley(game,n):
    if not n:return []
    out=[]
    for i in range(n):
        value=0.
        for m in range(1<<n):
            if m&(1<<i):continue
            k=m.bit_count();w=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            value+=w*(game[m|(1<<i)]-game[m])
        out.append(value)
    assert abs(sum(out)-game[(1<<n)-1]+game[0])<1e-10
    return out


def score():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    truth={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in truth}
    old=read(ROOT/'results/stvg_error_state/2026-10-18/cached32/RESULTS.json');ob={r['alias']:r for r in old['rows']};rows=[];checks=0;started=time.perf_counter()
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=truth[ds][0][q],truth[ds][1][q];ids=np.array(z['frame_ids']);planned=np.isin(np.arange(len(ids)),z['planned_positions']);known=np.array([int(i)in gt for i in ids]);unobs=known&~planned;inhull=(ids>=ids[min(z['planned_positions'])])&(ids<=ids[max(z['planned_positions'])]);inmbr=(ids>=z['interval'][0])&(ids<z['interval'][1]);scale=np.array([row['input']['width'],row['input']['height']]*2);gxy=np.array([gt[int(fid)]for fid in ids[known]])
        masks=dict(unobserved=unobs,unobserved_inside_hull=unobs&inhull,unobserved_outside_hull=unobs&~inhull,
            unobserved_inside_MBR=unobs&inmbr,observed=known&planned);values={};group_sh={};geometry_sh={};scope_parts={};reference_quality={g:[]for g in GROUPS}
        for a in z['anchor_geometry']:reference_quality[a['category']].append(a)
        for base,outputs in z['outputs'].items():
            values[base]={};all_ious={}
            for name,b in outputs.items():
                tube=DenseTube(b,row,gt,span,clip=ds=='hc2');met=tube.score(z['interval']);ind=official(b,row,gt,span,z['interval'],ds);assert all(abs(met[k]-ind[k])<2e-10 for k in ['v','t','s']);checks+=3
                xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*scale
                if ds=='hc2':xy=np.maximum(xy,0)
                ious=np.zeros(len(ids));ious[known]=box_iou(xy[known],gxy);all_ious[name]=ious
                support={s:dict(frames=int(mask.sum()),s=avg(ious[mask]))for s,mask in masks.items()}
                values[base][name]=dict(**met,supports=support)
                if name in ['geometry_0','geometry_3','scope_DINO_full_global']:
                    ref='DINO_common'if name=='geometry_0'else'GT_common'if name=='geometry_3'else'DINO_full'
                    orig=ob[r['alias']]['values'][base][ref]['GlobalError'];assert all(abs(orig['clocks']['final_MBR'][k]-met[k])<1e-12 for k in ['v','t','s']);assert abs(orig['supports']['unobserved_decoder']['s']-support['unobserved']['s'])<1e-12;checks+=4
            endpoints=['v','s','unobserved','unobserved_inside_MBR'];group_sh[base]={};geometry_sh[base]={}
            def endpoint(name,key):return values[base][name][key]if key in ['v','s']else values[base][name]['supports'][key]['s']
            for ep in endpoints:
                if endpoint('Base',ep)is None:
                    group_sh[base][ep]=None;geometry_sh[base][ep]=None;continue
                game={m:100*endpoint('groups_'+str(m),ep)for m in range(1<<len(z['groups_present']))};sp=shapley(game,len(z['groups_present']));gs={g:0. for g in GROUPS};gs.update(dict(zip(z['groups_present'],sp)));group_sh[base][ep]=gs
                geom=shapley({m:100*endpoint('geometry_'+str(m),ep)for m in range(4)},2);geometry_sh[base][ep]=dict(center=geom[0],size=geom[1]);assert abs(sum(gs.values())-sum(geom))<1e-9;checks+=1
            scope_parts[base]={}
            for ref in ['DINO_full','DINO_common','GT_common']:
                scoped={}
                for scope in ['global','planned_hull','admitted_nodes']:
                    name='scope_'+ref+'_'+scope;diff=all_ious[name]-all_ious['Base'];den=int(unobs.sum())
                    parts={side:float(100*diff[mask].sum()/den)if den else None for side,mask in [('inside_hull',unobs&inhull),('outside_hull',unobs&~inhull)]}
                    if den:assert abs(sum(parts.values())-100*(endpoint(name,'unobserved')-endpoint('Base','unobserved')))<1e-10;checks+=1
                    scoped[scope]=parts
                scope_parts[base][ref]=scoped
        counts={g:z['common_categories'].count(g)for g in GROUPS}
        rows.append(dict(alias=r['alias'],dataset=ds,common_reference_counts=counts,full_reference_counts={g:z['full_categories'].count(g)for g in GROUPS},
            semantic_role_unknown=r['alias']in p['uncertain_role_queries'],values=values,category_repair_shapley_pp=group_sh,
            center_size_repair_shapley_pp=geometry_sh,scope_signed_unobserved_contributions_pp=scope_parts,
            reference_geometry_by_category=reference_quality,old_visual_model_assisted_only=True,new_visual_labels=0))
        print('SCORED',r['alias'],flush=True);assert time.perf_counter()-started<3600
    summary={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];summary[ds]={}
        for base in ['Before','Source']:
            cat={};geo={};scope={}
            for ep in ['v','s','unobserved','unobserved_inside_MBR']:
                cat[ep]={g:dict(cohort_contribution=paired([r['category_repair_shapley_pp'][base][ep][g]if r['category_repair_shapley_pp'][base][ep]is not None else None for r in rr]),
                    reference_count=sum(r['common_reference_counts'][g]for r in rr),present_parents=sum(r['common_reference_counts'][g]>0 for r in rr))for g in GROUPS}
                geo[ep]={key:paired([r['center_size_repair_shapley_pp'][base][ep][key]if r['center_size_repair_shapley_pp'][base][ep]is not None else None for r in rr])for key in ['center','size']}
            for ref in ['DINO_full','DINO_common','GT_common']:
                scope[ref]={}
                for name in ['global','planned_hull','admitted_nodes']:
                    key='scope_'+ref+'_'+name;out={}
                    for ep in ['v','s','unobserved','unobserved_inside_MBR','unobserved_inside_hull','unobserved_outside_hull']:
                        vals=[]
                        for r in rr:
                            a=r['values'][base][key];b=r['values'][base]['Base'];av=a[ep]if ep in ['v','s']else a['supports'][ep]['s'];bv=b[ep]if ep in ['v','s']else b['supports'][ep]['s'];vals.append(None if av is None or bv is None else 100*(av-bv))
                        out[ep]=paired(vals)
                    out['signed_unobserved_parts']={side:paired([r['scope_signed_unobserved_contributions_pp'][base][ref][name][side]for r in rr])for side in ['inside_hull','outside_hull']};scope[ref][name]=out
            summary[ds][base]=dict(category_oracle_repair_shapley=cat,center_size_oracle_repair_shapley=geo,scope=scope,
                reference_geometry={g:{m:avg([a[m]for r in rr for a in r['reference_geometry_by_category'][g]])for m in ['reference_IoU','center_error_in_target_sizes','log_width_ratio','log_height_ratio','area_ratio']}for g in GROUPS})
    P.mkdir(parents=True,exist_ok=True);put(P/'RESULTS.json',dict(status='complete',count=32,primary='HC16/Before/final_MBR',groups=summary,rows=rows,
        model_assisted_old_labels_not_independent_human=True,unknown_roles_overridden=True,
        overlapping_estimands_not_three_additive_causes=True,no_deployment_or_TTA_scores=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,metric_endpoint_identity_and_Shapley_checks=checks,
        CPU_score_seconds=time.perf_counter()-started,result_sha256=sha(P/'RESULTS.json'),new_models_experts_training_updates=0))


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','candidates','score']);stage=a.parse_args().stage
    try:globals()[stage]()
    except BaseException as e:
        put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=stage,type=type(e).__name__,error=str(e),partial_preserved=True));raise
