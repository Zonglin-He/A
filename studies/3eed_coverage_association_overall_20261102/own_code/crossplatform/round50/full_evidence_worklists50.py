"""UNEXECUTED R50 evidence worklist/index builder; no model or GT code.

Requires sealed full frames/text/prepared and a separate root preparation
grant. Old336 joins by exact path triple AND caption, never full sample ID.
Only an explicit grant permits decoding those joined RGB files for DINO's
historical original-view cache key. No raw XYZ or camera-metadata file reads.
"""
from pathlib import Path
import argparse, math, time, hashlib, re
import numpy as np
from full_evidence50 import (ROOT,sha,read,dump,digest,array_sha,numeric_binding,dino_binding,dino_case_binding,bound_file)

EXP=ROOT/'research/round50/experiment'
LEGACY_MANIFEST='research/round24/input/ALL336_PRIVATE.json'
VISUAL='crossplatform/round50/visual336_attempt001'
PREPARED='crossplatform/round50/prepared336_attempt001'
LEGACY_DINO_PROGRAMS=('crossplatform/round23/detect23.py','crossplatform/round24/detectM24.py',
                      'crossplatform/round20/detect.py')
LEGACY_DINO_ASSETS_RECEIPT='crossplatform/round20/assets/DINO_ASSET_RECEIPT.json'


def join_key(row):return tuple(row[k] for k in ('imagepath','pointpath','metapath','caption'))


def old_roots(row):
    prefix='crossplatform/round24' if row['stage']=='M' else 'crossplatform/round23'
    suffix='M64' if row['stage']=='M' else '272'
    return {name:prefix+'/'+name+suffix+'_attempt001' for name in ('text','evidence','dino')}


def crop_sizes(box,size_wh):
    """Literal visualZ.crops pad0/pad.2 int/clipping; returns (W,H)."""
    W,H=map(int,size_wh);x1,y1,x2,y2=map(float,box);w=x2-x1;h=y2-y1
    sizes=[]
    for pad in (0.,.2):
        a=max(0,int(x1-pad*w));b=max(0,int(y1-pad*h))
        c=min(W,int(x2+pad*w));d=min(H,int(y2+pad*h));assert c>a and d>b
        sizes.append([c-a,d-b])
    return sizes


def reference_size(box,size_wh):
    W,H=map(int,size_wh);x1,y1,x2,y2=map(float,box);dx=.1*(x2-x1);dy=.1*(y2-y1)
    a,b=max(0,int(x1-dx)),max(0,int(y1-dy));c,d=min(W,int(x2+dx)),min(H,int(y2+dy))
    assert c>a and d>b
    return [c-a,d-b]


def gray_sha(size_wh):
    remaining=3*math.prod(size_wh);h=hashlib.sha256();block=bytes([127])*65536
    while remaining:
        n=min(remaining,len(block));h.update(block[:n]);remaining-=n
    return h.hexdigest()


class Index:
    def __init__(self,recipe):self.recipe=recipe;self.entries={};self.values={};self.duplicates=0

    def add(self,kind,binding,file,value_path,value):
        key=digest(binding);entry={'kind':kind,'input_key':key,'model_recipe':self.recipe,
            'file':Path(file).relative_to(ROOT).as_posix(),'sha256':sha(file),'value_path':value_path}
        if key in self.entries:
            assert self.values[key]==value, ('Identical historical input has different outputs',key)
            self.duplicates+=1
        else:self.entries[key]=entry;self.values[key]=value
        return key

    def numeric(self,file,value_path,value,size_wh,is_gray=False):
        assert value['generated_tokens']==0 and 0<value['input_tokens']<=8192
        assert np.isfinite(value['log_odds_yes_no'])
        if is_gray:assert value['image_sha256']==gray_sha(size_wh), 'Historical gray pixels/shape mismatch'
        return self.add('numeric',numeric_binding(self.recipe,value['image_sha256'],size_wh,
            value['prompt_sha256']),file,value_path,value)

    def export(self):return [self.entries[k] for k in sorted(self.entries)]


def context(args):
    auth=read(args.authorization);pp=Path(args.plan)
    assert auth['state']=='ROOT_AUTHORIZED_R50_FULL_EVIDENCE_WORKLISTS'
    assert auth['plan_sha256']==sha(pp) and args.stage in auth['allowed_stages']
    plan=read(pp)
    assert plan['state']=='ROOT_LOCKED_R50_FULL_EVIDENCE_WORKLISTS'
    assert plan['n']==5912 and plan['selected_configuration']=='wideentry_oldW'
    assert plan['output_root']=='crossplatform/round50/full5912_attempt001'
    assert plan['no_GT_rawXYZ_or_model_execution'] is True
    assert plan['chunk_size'] in (128,256)
    for group in ('programs_sha256','inputs_sha256'):
        for rel,h in plan[group].items():
            assert 'FINAL384' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h,rel
    assert LEGACY_MANIFEST in plan['inputs_sha256']
    assert sha(EXP/'FULL_FRONTEND_PLAN_LOCK.json')==plan['frontend_plan_sha256']
    from full_frontend50 import context as parent_context
    parent,rows,_=parent_context()
    assert parent['output_root']==plan['output_root'] and parent['selected_configuration']=='wideentry_oldW'
    for group in ('sealed_roots','legacy_sealed_roots'):
        for rel,h in plan[group].items():assert sha(ROOT/rel/'SEALED.json')==h,rel
    if plan['allow_joined_legacy_RGB_decode']:
        assert auth['allow_joined_legacy_RGB_decode'] is True
    folder=ROOT/plan['output_root']/'worklists'/args.stage
    assert not folder.exists(), 'Preparation is deterministic and must use a distinct attempt if interrupted'
    folder.mkdir(parents=True)
    return plan,rows,folder


def load_inputs(plan,rows):
    """Sealed numeric/pool metadata only, no observations or raw scene load."""
    runroot=ROOT/plan['output_root'];out={}
    for row in rows:
        sid=row['id'];tp=runroot/'text/cases_PRIVATE'/(sid+'.json')
        cp=runroot/'prepared/cases_PRIVATE'/(sid+'.json')
        ap=runroot/'prepared/arrays_PRIVATE'/(sid+'.npz')
        fp=runroot/'frames/cases_PRIVATE'/(sid+'.json')
        mp=runroot/'frames/masks_PRIVATE'/(sid+'.npz')
        for p in (tp,cp,ap,fp,mp):bound_file(p,plan['sealed_roots'])
        graph=read(tp);case=read(cp);frame=read(fp)
        assert frame['image_sha256']==case['bindings']['image_sha256']
        assert frame['point_sha256']==case['bindings']['point_sha256']
        assert frame['mask_cache_sha256']==case['bindings']['mask_cache_sha256']
        assert sha(mp)==frame['mask_cache_sha256']
        with np.load(ap,allow_pickle=False) as a:
            old=a['old_pool'].tolist();wide=a['wide_pool'].tolist()
            det=a['detector_boxes'].copy();msha=a['mask_sha256'].tolist()
        with np.load(mp,allow_pickle=False) as a:H,W=map(int,a['original_hw'])
        assert case['mask_pool_PRIVATE']==old and case['wide_pool_PRIVATE']==wide
        assert set(old)<=set(wide) and len(set(wide))==len(wide)
        assert hashlib.sha256(graph['subject_query'].encode()).hexdigest()==case['actual_subject_query_sha256']
        assert len(graph['references'])<=4
        out[sid]={'graph':graph,'prepared':case,'frame':frame,'old_pool':old,'wide_pool':wide,
            'detector_boxes':det,'mask_sha256':msha,'size_wh':[W,H]}
    return out


def exact_legacy_index(plan,rows,inputs,stage,recipe):
    legacy=read(ROOT/LEGACY_MANIFEST);assert len(legacy)==336
    keyed={join_key(r):r for r in rows};assert len(keyed)==5912
    assert len({join_key(r) for r in legacy})==336
    index=Index(recipe);ledger=[];joined={};case_entries=[]
    for old in legacy:
        row=keyed.get(join_key(old))
        if row is None:
            ledger.append({'legacy_id_PRIVATE':old['id'],'status':'NOT_IN_LOCKED_FULL_TRIPLE_CAPTION'})
            continue
        sid=row['id'];joined[sid]=old;current=inputs[sid];roots=old_roots(old)
        oldtextpath=ROOT/roots['text']/'cases_PRIVATE'/(old['id']+'.json')
        visualpath=ROOT/VISUAL/'cases_PRIVATE'/(old['id']+'.json')
        oldapath=ROOT/PREPARED/'arrays_PRIVATE'/(old['id']+'.npz')
        for p in (oldtextpath,visualpath,oldapath):bound_file(p,plan['legacy_sealed_roots'])
        oldgraph=read(oldtextpath);visual=read(visualpath)
        with np.load(oldapath,allow_pickle=False) as a:
            olddet=a['detector_boxes'].copy();oldmsha=a['mask_sha256'].tolist();oldpool=a['old_pool'].tolist()
        assert visual['actual_subject_query_sha256']==hashlib.sha256(oldgraph['subject_query'].encode()).hexdigest()
        same_image=visual['image_sha256']==current['frame']['image_sha256']
        record={'id_PRIVATE':sid,'legacy_id_PRIVATE':old['id'],'join':'EXACT_PATH_TRIPLE_AND_CAPTION',
            'same_image_file_sha256':same_image,'target_numeric_entries':0,'reference_numeric_entries':0,
            'DINO_original_entries':0,'DINO_flip_raw_cache_forbidden':True}
        # A mismatch is a cache miss, never deletion of a full sample or label gate.
        if not same_image:
            record['status']='CACHE_MISS_IMAGE_HASH_CHANGED';ledger.append(record);continue
        if stage=='numeric':
            subject_same=current['graph']['subject_query']==oldgraph['subject_query']
            record['same_subject_query']=subject_same
            for masktext,views in visual['targets_PRIVATE'].items():
                m=int(masktext)
                if m>=len(current['mask_sha256']) or m not in current['wide_pool']:continue
                same=(oldmsha[m]==current['mask_sha256'][m]==visual['mask_sha256_by_index_PRIVATE'][masktext]
                      and np.array_equal(olddet[m],current['detector_boxes'][m]))
                if not same or not subject_same:continue
                sizes=crop_sizes(olddet[m],current['size_wh']);assert len(views)==2
                for view,v in enumerate(views):
                    assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
                    assert v['full']['prompt_sha256']==v['gray']['prompt_sha256']
                    for mode in ('full','gray'):
                        index.numeric(visualpath,['targets_PRIVATE',masktext,view,mode],v[mode],sizes[view],is_gray=mode=='gray')
                        record['target_numeric_entries']+=1
            evidencepath=ROOT/roots['evidence']/'cases_PRIVATE'/(old['id']+'.json')
            bound_file(evidencepath,plan['legacy_sealed_roots']);evidence=read(evidencepath)
            if current['old_pool']:
                for ref in current['graph']['references']:
                    candidates=[r for r in oldgraph['references'] if r['phrase']==ref['phrase'] and r['own_query']==ref['own_query']]
                    for oldref in candidates:
                        entity=oldref['entity']
                        for j,observation in enumerate(evidence['references'].get(entity,[])):
                            size=reference_size(observation['box_PRIVATE'],current['size_wh'])
                            assert observation['visual_evidence']==observation['full']['log_odds_yes_no']-observation['gray']['log_odds_yes_no']
                            for mode in ('full','gray'):
                                index.numeric(evidencepath,['references',entity,j,mode],observation[mode],size,is_gray=mode=='gray')
                                record['reference_numeric_entries']+=1
        else:
            dinopath=ROOT/roots['dino']/'cases_PRIVATE'/(old['id']+'.json')
            bound_file(dinopath,plan['legacy_sealed_roots']);detections=read(dinopath)
            if current['old_pool'] and current['graph']['references'] and plan['allow_joined_legacy_RGB_decode']:
                # No XYZ, scene rebuilding or metadata field parsing is needed.
                from PIL import Image
                assert sha(row['imagepath'])==current['frame']['image_sha256']
                rgb=np.array(Image.open(row['imagepath']).convert('RGB'))
                assert rgb.shape==tuple(reversed(current['size_wh']))+(3,)
                image_hash=array_sha(rgb)
                same_pool=current['old_pool']==oldpool;equal_graph=current['graph']==oldgraph
                record.update(same_old_pool=same_pool,entire_graph_equal=equal_graph)
                if same_pool and equal_graph:
                    expected_entities={r['entity'] for r in current['graph']['references']}
                    assert set(detections['objects'])==expected_entities
                    for rel in LEGACY_DINO_PROGRAMS:
                        assert rel in plan['programs_sha256'] and sha(ROOT/rel)==plan['programs_sha256'][rel]
                    receipt=ROOT/LEGACY_DINO_ASSETS_RECEIPT
                    assert LEGACY_DINO_ASSETS_RECEIPT in plan['inputs_sha256']
                    assert sha(receipt)==plan['inputs_sha256'][LEGACY_DINO_ASSETS_RECEIPT]
                    root=roots['dino'];assert root in plan['legacy_sealed_roots']
                    case_entries.append({'binding':dino_case_binding(recipe,sid,current['graph'],current['old_pool'],
                        current['prepared']['bindings'],image_hash,rgb.shape),
                        'file':dinopath.relative_to(ROOT).as_posix(),'sha256':sha(dinopath),
                        'legacy_dino_root':root,'legacy_dino_seal_sha256':plan['legacy_sealed_roots'][root],
                        'legacy_recipe_programs_sha256':{rel:plan['programs_sha256'][rel] for rel in LEGACY_DINO_PROGRAMS},
                        'legacy_assets_receipt_file':LEGACY_DINO_ASSETS_RECEIPT,
                        'legacy_assets_receipt_sha256':sha(receipt),
                        'objects_sha256':digest(detections['objects']),
                        'complete_original_already_unflipped_flip_and_stable_indices':True,
                        'flip_raw_cache_or_second_flip':False})
                    record['DINO_whole_case_exact_reuse']=True
                for ref in current['graph']['references']:
                    candidates=[r for r in oldgraph['references'] if r['own_query']==ref['own_query']]
                    for oldref in candidates:
                        entity=oldref['entity']
                        if entity not in detections['objects']:continue
                        value=detections['objects'][entity]['original']
                        query_hash=hashlib.sha256((ref['own_query'].casefold().strip(' .')+'.').encode()).hexdigest()
                        assert value['query_sha256']==query_hash
                        index.add('dino',dino_binding(recipe,image_hash,rgb.shape,query_hash),dinopath,
                            ['objects',entity,'original'],value)
                        record['DINO_original_entries']+=1
        record['status']='JOINED_INDEXED_OR_INPUT_EXACT_CACHE_MISS';ledger.append(record)
    assert len(joined)==plan['expected_legacy_join_count']==336
    return index,ledger,joined,case_entries


def dino_cases(plan,rows):
    """Read only root-pinned completed DINO case outputs, never select by score."""
    pp=ROOT/plan['completed_dino_plan_file'];assert sha(pp)==plan['completed_dino_plan_sha256']
    dplan=read(pp);result={};runroot=ROOT/plan['output_root'];extra_seals={}
    assert dplan['selected_configuration']=='wideentry_oldW'
    for job in dplan['chunks'].values():
        assert job['stage']=='dino'
        folder=runroot/job['output_subdir'];rel=folder.relative_to(ROOT).as_posix()
        assert rel in plan['completed_dino_seals_sha256']
        h=plan['completed_dino_seals_sha256'][rel];assert sha(folder/'SEALED.json')==h
        extra_seals[rel]=h;seal=read(folder/'SEALED.json')
        assert seal['state']=='R50_FULL_FROZEN_DINO_EVIDENCE_NO_GT'
        jobfile=ROOT/job['rows_file'];assert sha(jobfile)==job['rows_sha256']
        for row in read(jobfile):
            sid=row['id'];assert sid not in result
            path=folder/'cases_PRIVATE'/(sid+'.json')
            assert sha(path)==seal['files'][path.relative_to(folder).as_posix()]
            result[sid]={'file':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
    assert set(result)=={r['id'] for r in rows}
    return result,extra_seals


def build(plan,rows,folder,stage):
    start=time.perf_counter();inputs=load_inputs(plan,rows)
    recipe=digest(plan['model_recipes'][stage])
    cache,joinledger,joined,case_entries=exact_legacy_index(plan,rows,inputs,stage,recipe)
    cachepath=folder/'EXACT_REUSE_INDEX_PRIVATE.json';dump(cachepath,cache.export())
    dump(folder/'LEGACY_JOIN_LEDGER_PRIVATE.json',joinledger)
    casecachepath=folder/'DINO_CASE_REUSE_INDEX_PRIVATE.json'
    if stage=='dino':dump(casecachepath,case_entries)
    case_reuse_ids={entry['binding']['id_PRIVATE'] for entry in case_entries}
    dindex={};dseals={}
    if stage=='numeric':dindex,dseals=dino_cases(plan,rows)
    if dindex:dump(folder/'DINO_CASE_INDEX_PRIVATE.json',dindex)
    jobs={};casecounts=[];logical_total=0;physical_cap_total=0;whole_case_logical_total=0
    size=plan['chunk_size'];runroot=ROOT/plan['output_root']
    for number,offset in enumerate(range(0,len(rows),size)):
        rr=rows[offset:offset+size];key=f'{stage}_{number:04d}';path=folder/'chunks_PRIVATE'/(key+'.json')
        dump(path,rr);cap=0
        for row in rr:
            c=inputs[row['id']];graph=c['graph'];refs=graph['references'] if c['old_pool'] else []
            if stage=='dino':
                counts=[1+(not bool(re.search(r'\b(left|right|clockwise)\b',r['own_query'],re.I))) for r in refs]
                logical=sum(counts);targetlogical=0;reflogical=logical
            else:
                det=read(ROOT/dindex[row['id']]['file']);assert bool(det['objects'])==bool(refs)
                targetlogical=4*len(c['wide_pool'])
                reflogical=2*sum(len(det['objects'][r['entity']]['stable_indices'][:3]) for r in refs)
                logical=targetlogical+reflogical
            whole_case_logical=logical if stage=='dino' and row['id'] in case_reuse_ids else 0
            cap+=logical-whole_case_logical;logical_total+=logical;whole_case_logical_total+=whole_case_logical
            casecounts.append({'id_PRIVATE':row['id'],'chunk':key,'old_pool':len(c['old_pool']),
                'wide_pool':len(c['wide_pool']),'reference_entities_after_oldpool_gate':len(refs),
                'target_numeric_logical_calls':targetlogical,'reference_logical_calls':reflogical,
                'logical_call_upper_bound':logical,'sample_retained_even_zero_calls':True,
                'historical_whole_case_logical_uses':whole_case_logical,
                'physical_new_forward_upper_bound':logical-whole_case_logical,
                'legacy_exact_join':row['id'] in joined})
        physical_cap_total+=cap
        job={'stage':stage,'rows_file':path.relative_to(ROOT).as_posix(),'rows_sha256':sha(path),
             'max_new_calls':cap,'model_recipe':recipe,
             'output_subdir':('dino_chunks/' if stage=='dino' else 'evidence_chunks/')+key,
             'reuse_index_file':cachepath.relative_to(ROOT).as_posix(),'reuse_index_sha256':sha(cachepath)}
        if stage=='dino':job.update(case_reuse_index_file=casecachepath.relative_to(ROOT).as_posix(),
                                  case_reuse_index_sha256=sha(casecachepath))
        rate=plan.get('seconds_per_forward_estimate',{}).get(stage)
        if rate is not None:
            assert type(rate) in (int,float) and np.isfinite(rate) and rate>0
        job['estimated_seconds_using_locked_rate']=None if rate is None else 8+cap*rate
        job['estimated_seconds_are_not_runtime_measurements']=True
        if stage=='numeric':job.update(dino_case_index_file=(folder/'DINO_CASE_INDEX_PRIVATE.json').relative_to(ROOT).as_posix(),
                                      dino_case_index_sha256=sha(folder/'DINO_CASE_INDEX_PRIVATE.json'))
        jobs[key]=job
    draft={'state':'ROOT_LOCKED_R50_FULL_FROZEN_EVIDENCE_CHUNKS','selected_configuration':'wideentry_oldW',
        'no_GT_or_parameter_updates':True,'output_root':plan['output_root'],
        'frontend_plan_sha256':plan['frontend_plan_sha256'],
        'programs_sha256':plan['programs_sha256'],'inputs_sha256':dict(plan['inputs_sha256']),
        'sealed_roots':dict(plan['sealed_roots'],**dseals),'model_recipes':plan['model_recipes'],
        'model_assets':plan['model_assets'],'chunks':jobs,
        'draft_only_requires_root_review_and_separate_execution_authorization':True}
    for p in (cachepath,casecachepath,folder/'DINO_CASE_INDEX_PRIVATE.json'):
        if p.exists():draft['inputs_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
    dump(folder/'EVIDENCE_PLAN_DRAFT.json',draft)
    dump(folder/'CASE_BUDGETS_PRIVATE.json',casecounts)
    counts={'n':len(rows),'chunks':len(jobs),'chunk_size':size,'physical_call_safe_upper_bound':physical_cap_total,
        'logical_requests_fixed_before_model_execution':logical_total,'cache_unique_input_entries':len(cache.entries),
        'identical_historical_cache_duplicates':cache.duplicates,'legacy_exact_join_count':len(joined),
        'DINO_whole_case_exact_reuse_cases':len(case_entries),'DINO_whole_case_logical_uses':whole_case_logical_total,
        'per_chunk_model_initialization_seconds_estimate':8,'estimated_total_model_initialization_seconds':8*len(jobs),
        'seconds_per_forward_estimate':plan.get('seconds_per_forward_estimate',{}).get(stage),
        'estimated_seconds_at_physical_cap':None if rate is None else 8*len(jobs)+physical_cap_total*rate,
        'forward_seconds_not_claimed_from_initialization_estimate':True,
        'rawXYZ_reads':0,'metadata_GT_reads':0,'model_calls':0,'optimizer_steps':0,
        'seconds_worklist_preparation':time.perf_counter()-start}
    dump(folder/'COUNTS.json',counts)
    dump(folder/'SEALED.json',{'state':'R50_FULL_'+stage.upper()+'_WORKLIST_AND_EXACT_CACHE_INDEX_PREPARED_NO_MODELS',
        **counts,'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}})
    return counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True)
    ap.add_argument('--stage',choices=('dino','numeric'),required=True);args=ap.parse_args()
    plan,rows,folder=context(args)
    try:print('FULL_EVIDENCE_WORKLIST50',args.stage,build(plan,rows,folder,args.stage),flush=True)
    except BaseException as error:
        dump(folder/f'FAILED_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),
            'model_calls':0,'rawXYZ_or_metadata_GT_reads':False,'partial_attempt_preserved':True});raise


if __name__=='__main__':main()
