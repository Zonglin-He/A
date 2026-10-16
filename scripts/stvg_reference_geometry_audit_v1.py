"""Finite post-Lite clock qualification and annotation-free geometry replay.

Historical teacher responses are never reused as corrected-clock responses.
RGB and intermediates remain private. This file neither fits nor reads labels.
"""
from pathlib import Path
import ast, collections, copy, gc, hashlib, inspect, json, math, os, sys, textwrap, time, types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read, write, sha
BASE=ROOT/'artifacts/stvg_spatial_reference_geometry_v1'
LITE=ROOT/'artifacts/stvg_lite_event_support_v1'
SEARCH=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1'

def put(path,value):
    path=Path(path);assert not path.exists(),str(path)
    write(path,value)

def clock_values(seconds,t,tps):
    import torch
    r=torch.arange(t).view(-1,1)
    old=(r*torch.as_tensor(seconds,dtype=r.dtype)*tps).long().flatten().tolist()
    new=(r*torch.as_tensor(seconds,dtype=torch.float64)*tps).long().flatten().tolist()
    return old,new

def clock_contracts():
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from scripts.stvg_qwen_uniform_clock002 import corrected_function
    fixed,proof=corrected_function(Qwen2_5_VLModel.get_rope_index)
    config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2,tokens_per_second=2),image_token_id=98,video_token_id=99,vision_start_token_id=97)
    obj=types.SimpleNamespace(config=config);n=0
    for seconds in (0.0666666666666664,.8,1.,1.9999999999999998,2.51,3.,3.44):
        for t in (2,3,4):
            ids=torch.tensor([[1,2,97]+[99]*(t*4)+[3,4]])
            kw=dict(input_ids=ids,video_grid_thw=torch.tensor([[t,4,4]]),second_per_grid_ts=[seconds],attention_mask=torch.ones_like(ids))
            a,_=Qwen2_5_VLModel.get_rope_index(obj,**kw);b,_=fixed(obj,**kw)
            old,new=clock_values(seconds,t,2)
            assert (a[0,0,3:3+t*4].reshape(t,4)[:,0]-3).tolist()==old
            assert (b[0,0,3:3+t*4].reshape(t,4)[:,0]-3).tolist()==new
            n+=1
    return dict(status='pass',count=n,actual_model_tokens_per_second=2,proof=proof,CPU_only=True,GT_read=False)

def setup():
    assert not BASE.exists();BASE.mkdir(parents=True)
    recipe=read(LITE/'FINAL_RECIPE_LOCK.json')
    paths=[Path(__file__),ROOT/'scripts/stvg_weightfree_spatial_v1.py',ROOT/'scripts/stvg_evidence_spatial_inputs005.py',ROOT/'scripts/stvg_evidence_spatial_inputs009.py',ROOT/'scripts/prepare_stvg_lite_reclocked_inputs002.py',ROOT/'scripts/run_stvg_evidence_teacher_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',ROOT/'scripts/stvg_table1_temporal_adapter_v1.py',LITE/'FINAL_RECIPE_LOCK.json',LITE/'complete/CPU_COMPLETION.json',LITE/'complete/POSTSEAL_ROWS.json',LITE/'trials/C_FULL/REFERENCE_LOCK.json',SEARCH/'ROSTER_LOCK.json']
    put(BASE/'RUNTIME.json',dict(status='locked',pins={str(p.relative_to(ROOT)):sha(p)for p in paths},GT_read=False))
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',scope='one clock audit; one balanced development geometry decomposition; one conditional geometry OR identity change; fixed-parameter T+S then frozen confirmation',
        development_count=12,domain_counts={'hc2':6,'vidstg':6},selection='two positive/two negative/two neutral Full-minus-Temporal sources per domain, prefer nonempty references, then parent SHA; already exposed development only',
        sign_epsilon=1e-8,maximum_geometry_variants=3,geometry_rules=['R0 original selected candidate largest-component bbox','R1 same candidate full GrabCut foreground bbox only with exact anchor replay','R2 original Selective Search rectangle only with exact full-pool replay; reconstructed not historically cached'],
        conditional_execution='geometry fitting only after identity/geometry decomposition; unknown identities remain unknown; no GT candidate selection or Oracle fitting',
        clock_policy='v1 legacy and v2 fractional-time isolated namespaces; no changed request inherits legacy response; retain historical clocks for historical decomposition and qualify corrected inputs before any new combined run',
        locked_recipe_sha256=sha(LITE/'FINAL_RECIPE_LOCK.json'),lr={'hc2':.0045,'vidstg':.027},new_models=False,hyperparameter_search=False,bootstrap=False,CURRENT_change=False,official_fullset=False,GT_read=False))
    rows=read(LITE/'complete/POSTSEAL_ROWS.json')['rows'];idx={(r['case_id'],r['arm']):r for r in rows}
    refs={r['case_id']:r for r in read(LITE/'trials/C_FULL/REFERENCE_LOCK.json')['records']}
    roster=read(SEARCH/'ROSTER_LOCK.json')['records'];selected=[]
    for ds in ('hc2','vidstg'):
        for sign in ('positive','negative','neutral'):
            eligible=[]
            for rec in roster:
                if rec['dataset']!=ds:continue
                cid=rec['probe_alias'];d=idx[cid,'Full']['metrics']['m_vIoU']-idx[cid,'Temporal']['metrics']['m_vIoU']
                label='positive'if d>1e-8 else'negative'if d< -1e-8 else'neutral'
                if label==sign:eligible.append((not bool(refs[cid]['anchors']),rec['parent_sha256'],rec['query_key_sha256'],rec))
            assert len(eligible)>=2,(ds,sign,len(eligible))
            for _,_,_,rec in sorted(eligible,key=lambda x:x[:3])[:2]:
                selected.append(dict(**rec,diagnostic_stratum=sign,already_exposed_development=True))
    assert len({r['parent_sha256']for r in selected})==12
    put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=12,records=selected,selection_score_source_sha256=sha(LITE/'complete/POSTSEAL_ROWS.json'),selection_GT_read=False,uses_saved_postseal_sign_strata=True,GT_read=False))
    put(BASE/'CLOCK_CONTRACTS.json',clock_contracts())
    print(json.dumps({'status':'locked','count':12,'cases':[(r['probe_alias'],r['dataset'],r['diagnostic_stratum'])for r in selected]}))

def verify():
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel

def find_responses(value):
    if isinstance(value,dict):
        if isinstance(value.get('teacher_responses'),list):
            yield from value['teacher_responses'];return
        for v in value.values():yield from find_responses(v)
    elif isinstance(value,list):
        for v in value:yield from find_responses(v)

def audit_clocks():
    import torch, numpy as np
    verify();torch.set_num_threads(2)
    tps=read(ROOT/'checkpoints/STVG-R1-7B/config.json')['vision_config']['tokens_per_second'];assert tps==2
    source=ROOT/'.conda/tubedetr/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py'
    records=[];payload_files={};fallbacks=[]
    for cohort,roster_path in [('development',SEARCH/'ROSTER_LOCK.json'),('confirmation',LITE/'confirmation/ROSTER_LOCK.json')]:
        for rec in read(roster_path)['records']:
            cid=rec['probe_alias']
            if cohort=='confirmation':p=LITE/'confirmation/temporal'/f'{cid}.pt'
            else:
                p=ROOT/rec['cached_temporal_path']if rec.get('cached_temporal_path')else SEARCH/'confirmation/temporal'/f'{cid}.pt'
            doc=torch.load(p,map_location='cpu',weights_only=False);payload_files[str(p.relative_to(ROOT))]=sha(p)
            responses=list(find_responses(doc))
            if not responses and cohort=='development':
                parent=ROOT/rec['cached_parent']if rec.get('cached_parent')else SEARCH/'confirmation'
                candidates=[parent/'temporal_teacher'/f'{cid}.pt',ROOT/'artifacts/stvg_weightfree_tuning_v1/confirmation003/temporal'/f'{cid}.pt']
                for tp in candidates:
                    if not tp.exists():continue
                    td=torch.load(tp,map_location='cpu',weights_only=False)
                    if doc.get('source_temporal_teacher_sha256')is not None:assert sha(tp)==doc['source_temporal_teacher_sha256']
                    responses=list(find_responses(td));payload_files[str(tp.relative_to(ROOT))]=sha(tp)
                    if responses:break
            if not responses:fallbacks.append(dict(cohort=cohort,case_id=cid,status='no_temporal_response_native_fallback_or_unavailable_metadata'))
            for r in responses:
                meta=r['metadata'];fps=float(meta['clip_fps']);fids=meta['sampled_physical_frame_ids'];t=math.ceil(len(fids)/2);sec=2/fps
                old,new=clock_values(sec,t,tps)
                records.append(dict(cohort=cohort,branch='Temporal',case_id=cid,candidate_id=r['candidate_id'],frame_count=len(fids),temporal_grid=t,seconds_per_grid=sec,legacy_time_positions=old,corrected_time_positions=new,affected=old!=new,legacy_collapsed=len(set(old))==1,GT_read=False))
            del doc;gc.collect()
        for clock,base in [('temporal',LITE/'reclocked_inputs'if cohort=='development'else LITE/'confirmation/temporal_inputs'),('native',LITE/'native_inputs'if cohort=='development'else LITE/'confirmation/native_inputs')]:
            jp=base/'TEACHER_JOB_LOCK.json';payload_files[str(jp.relative_to(ROOT))]=sha(jp)
            for job in read(jp)['records']:
                for vi,v in enumerate(job['videos']):
                    arr=np.load(ROOT/v['path'],mmap_mode='r',allow_pickle=False);n=arr.shape[0];t=math.ceil(n/2);sec=2/float(v['fps']);old,new=clock_values(sec,t,tps)
                    records.append(dict(cohort=cohort,branch='Spatial',clock=clock,case_id=job['case_id'],job_id=job['job_id'],view=vi,frame_count=n,temporal_grid=t,seconds_per_grid=sec,legacy_time_positions=old,corrected_time_positions=new,affected=old!=new,legacy_collapsed=len(set(old))==1,GT_read=False))
    summary=[]
    for cohort in ('development','confirmation'):
        for branch in ('Temporal','Spatial'):
            sub=[r for r in records if r['cohort']==cohort and r['branch']==branch]
            summary.append(dict(cohort=cohort,branch=branch,count=len(sub),affected=sum(r['affected']for r in sub),legacy_collapsed=sum(r['legacy_collapsed']for r in sub),affected_cases=len({r['case_id']for r in sub if r['affected']})))
    put(BASE/'CLOCK_INTERFACE_AUDIT.json',dict(status='complete',records=records,summary=summary,fallbacks=fallbacks,source_bindings=payload_files,installed_qwen_sha256=sha(source),model_config_sha256=sha(ROOT/'checkpoints/STVG-R1-7B/config.json'),contracts_sha256=sha(BASE/'CLOCK_CONTRACTS.json'),GT_read=False,model_loaded=False,CPU_only=True,
        interpretation='exact temporal-axis formula checked against installed real get_rope_index with actual 2Hz configuration; same positions are interface-equivalent, changed positions require fresh corrected-clock response; no historical task score invalidation or effect-size inference',
        scope='final Lite Temporal response metadata plus both temporal/native Spatial job locks across 128 development and 64 confirmation; grids inferred from original fixed 2-frame temporal patches and verified saved NPY frame counts'))
    print(json.dumps({'status':'complete','summary':summary,'fallbacks':len(fallbacks)}))

def replay_function():
    # Only preserve extra intermediates in the return value. Removing the two
    # capture statements and extra dict keywords reconstructs the exact AST.
    from scripts import stvg_weightfree_spatial_v1 as s
    src=textwrap.dedent(inspect.getsource(s.proposals));tree=ast.parse(src);original=ast.dump(tree,include_attributes=False)
    class Capture(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Tuple)and len(t.elts)==4 and isinstance(t.elts[0],ast.Name)and t.elts[0].id=='n'for t in node.targets):
                return [ast.parse('_audit_full_mask=m.copy()').body[0],node]
            return self.generic_visit(node)
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='dict'and any(k.arg=='proposal_rank'for k in node.keywords):
                for k,v in [('_audit_full_mask','_audit_full_mask'),('_audit_rect','list(r["rect"])')]:node.keywords.append(ast.keyword(arg=k,value=ast.parse(v,mode='eval').body))
            return node
    modified=ast.fix_missing_locations(Capture().visit(tree))
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name)and t.id=='_audit_full_mask'for t in node.targets):return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            self.generic_visit(node);node.keywords=[k for k in node.keywords if not(k.arg or'').startswith('_audit_')];return node
    assert ast.dump(Strip().visit(copy.deepcopy(modified)),include_attributes=False)==original
    ns=dict(s.__dict__);exec(compile(modified,'<geometry_intermediate_capture>','exec'),ns)
    return ns['proposals'],dict(status='pass',original_function_sha256=hashlib.sha256(src.encode()).hexdigest(),original_AST_recovered_exactly=True,changed_return_metadata_only=True)

def replay():
    # cv2 ABI must match the original proposal package, before importing numpy.
    sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'))
    import cv2,numpy as np
    from scripts import stvg_evidence_spatial_inputs005 as builder
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    verify();fn,proof=replay_function();cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    put(BASE/'REPLAY_AST_CONTRACT.json',proof)
    files=[];cases=read(BASE/'ROSTER_LOCK.json')['records'];start=time.perf_counter()
    refs={r['case_id']:r for r in read(LITE/'trials/C_FULL/REFERENCE_LOCK.json')['records']}
    for ci,rec in enumerate(cases,1):
        cid=rec['probe_alias'];case,cp=load_case(rec);inp=LITE/'reclocked_inputs';pp=inp/'new_crop/proposals'/f'{cid}.json';pool=read(pp)
        req=read(inp/'requests'/f'{cid}.json');assert pool['source_case_sha256']==case['source_case_sha256']
        assert sha(Path(req['video_path']))==req['canonical_video_sha256']
        centers={int(w['center_physical_frame_id'])for w in pool['windows']}
        fmap={int(r['local_frame_index']):int(r['physical_frame_id'])for r in req['frame_map']};rgb={};cap=cv2.VideoCapture(req['video_path']);i=0
        while len(rgb)<len(centers):
            ok,bgr=cap.read();assert ok
            fid=fmap[i]
            if fid in centers:rgb[fid]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            i+=1
        cap.release();wr=[]
        for w in pool['windows']:
            fid=w['center_physical_frame_id'];image=rgb[fid]
            assert hashlib.sha256(image.tobytes()).hexdigest()==w['rgb_sha256_by_physical_frame'][str(fid)]
            small=builder._resize_small(image,320);tic=time.perf_counter();new,stats=fn(small);elapsed=time.perf_counter()-tic
            mp=ROOT/w['mask_archive_path'];assert sha(mp)==w['mask_archive_sha256']
            with np.load(mp,allow_pickle=False)as archive:oldm=np.asarray(archive['masks']).copy()
            mid=w['physical_frame_ids'].index(fid);old=w['tracks'][1:];byrank={r['proposal_rank']:r for r in new};items=[]
            pool_exact=len(old)==len(new)
            for oi,tr in enumerate(old):
                cand=byrank.get(tr['proposal_rank']);anchor_exact=cand is not None and np.array_equal(oldm[oi,mid],cand['mask'])and tr['center_box']==cand['box']
                order_exact=oi<len(new)and new[oi]['proposal_rank']==tr['proposal_rank'];score_exact=cand is not None and cand['score']==tr['score'];pool_exact &=anchor_exact and order_exact and score_exact
                item=dict(candidate_id=tr['candidate_id'],proposal_rank=tr['proposal_rank'],historical_component_box=tr['center_box'],anchor_mask_and_box_exact=bool(anchor_exact),rank_order_exact=bool(order_exact),score_exact=bool(score_exact),replay_status='exact_anchor_intermediate_link'if anchor_exact else'unknown_historical_intermediate_not_exact')
                if cand is not None:
                    fh,fw=cand['mask'].shape;x,y,ww,hh=cand['_audit_rect'];full=cand['_audit_full_mask']
                    from scripts.stvg_weightfree_spatial_v1 import bbox
                    item.update(replayed_component_box=cand['box'],replayed_full_foreground_box=bbox(full),replayed_rectangle_box=[x/fw,y/fh,(x+ww)/fw,(y+hh)/fh],component_pixels=int(cand['mask'].sum()),full_foreground_pixels=int(full.sum()),component_count=int(cv2.connectedComponents(full.astype(np.uint8),8)[0]-1))
                    ap=BASE/'private'/cid/f'window{w["slot"]}_candidate{tr["candidate_id"]}.npz';ap.parent.mkdir(parents=True,exist_ok=True)
                    assert not ap.exists();np.savez_compressed(ap,component=cand['mask'].astype(np.uint8),full_foreground=full.astype(np.uint8));item['mask_path']=str(ap.relative_to(ROOT));item['mask_sha256']=sha(ap)
                items.append(item)
            sel=[a for a in refs[cid]['anchors']if a['physical_frame_id']==fid]
            wr.append(dict(slot=w['slot'],physical_frame_id=fid,source_grid_position=w['source_grid_position'],source_box=w['source_anchor_raw_box_xyxy'],selected_candidate_ids=[a['candidate_id']for a in sel],full_pool_exact=bool(pool_exact),raw_rect_historically_cached=False,reconstructed_rectangle_qualified=bool(pool_exact),candidates=items,proposal_counts=stats,proposal_seconds=elapsed,GT_read=False))
        dest=BASE/'replay'/f'{cid}.json';put(dest,dict(status='complete',case_id=cid,dataset=rec['dataset'],stratum=rec['diagnostic_stratum'],windows=wr,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),historical_proposal_sha256=sha(pp),canonical_video_sha256=req['canonical_video_sha256'],GT_read=False))
        files.append(dict(case_id=cid,path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
        write(BASE/'STATUS.json',dict(status='running',stage='CPU_intermediate_replay',done=ci,total=len(cases),GT_read=False),mutable=True)
        print('REPLAY',ci,len(cases),cid,sum(w['full_pool_exact']for w in wr),flush=True)
        del case,rgb;gc.collect()
    put(BASE/'GEOMETRY_PREDICTION_BARRIER.json',dict(status='complete',count=len(files),files=files,GT_read=False,CPU_only=True,wall_seconds=time.perf_counter()-start,roster_sha256=sha(BASE/'ROSTER_LOCK.json'),runtime_sha256=sha(BASE/'RUNTIME.json')))
    write(BASE/'STATUS.json',dict(status='pending_independent_geometry_readback',stage='postseal_diagnosis',done=len(files),total=len(cases),GT_read=False),mutable=True)

if __name__=='__main__':
    {'setup':setup,'clocks':audit_clocks,'replay':replay}[sys.argv[1]]()
