"""Root-locked 16-arm old336 cached G stage replay; never reads task GT/media.

Not executable until root creates experiment/STAGE_PLAN_LOCK.json and a
separate STAGE_EXECUTION_AUTHORIZATION.json. No imports load historical models.
"""
from pathlib import Path
import hashlib,json,datetime,collections
import numpy as np
import inventory49 as inv
from stages49 import replay,assert_bank_exact,readouts16,ARM_NAMES,mask_scene_lite

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'research/round49/experiment'
OUT=ROOT/'crossplatform/round49/stage336_attempt001'

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def historical_prediction(p):
    """Prediction containers are outside inventory49's sample-only adapter."""
    p=Path(p);folder=p.parent.parent;sp=folder/'SEALED.json';seal=read(sp)
    got=inv.sha(p);assert seal['files'][str(p.relative_to(folder))]==got
    inv.BINDINGS[str(sp.relative_to(ROOT))]=inv.sha(sp);inv.BINDINGS[str(p.relative_to(ROOT))]=got
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def check_lock():
    plan=read(EXP/'STAGE_PLAN_LOCK.json');auth=read(EXP/'STAGE_EXECUTION_AUTHORIZATION.json')
    assert auth['state']=='ROOT_AUTHORIZED_R49_CACHED_STAGE_PREDICTION'
    assert auth['stage_plan_sha256']==inv.sha(EXP/'STAGE_PLAN_LOCK.json')
    assert plan['arm_names']==ARM_NAMES and plan['n']==336 and plan['no_GT_no_media_no_models_no_TTA'] is True
    for rel,h in plan['programs_sha256'].items():assert inv.sha(ROOT/rel)==h,rel
    for rel,h in plan['inputs_sha256'].items():assert inv.sha(ROOT/rel)==h,rel
    inventory=ROOT/'research/round49/inventory_attempt001'
    recorded=read(inventory/'SEALED.json')
    for rel,h in recorded['files'].items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert inv.sha(inventory/rel)==h,rel
    for rel,h in read(inventory/'SOURCE_BINDINGS_PRIVATE.json').items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert inv.sha(ROOT/rel)==h,rel
    assert read(ROOT/'research/round49/inventory_attempt001/SUMMARY.json')['R24_exact_W_bridge_pass'] is True
    return plan

def mask_route(r,A,frame_index):
    if r['stage']=='A':
        case=A[r['id']];assert case['platform']==r['platform'] and case['scene_PRIVATE']==r['scene']
        return ROOT/case['mask_dir']/(case['frame_alias']+'.npz')
    matches=[(key,f) for key,f in frame_index.items() if all(f.get(k)==r[k] for k in ('imagepath','pointpath','metapath'))]
    assert len(matches)==1,('frame_path_join_not_unique',r['id'],len(matches))
    key,f=matches[0];assert f['platform']==r['platform']
    return ROOT/'crossplatform/round12/masks501_attempt001'/(key+'.npz')

def main():
    plan=check_lock();assert not OUT.exists();OUT.mkdir()
    for folder in ('samples_PRIVATE','cases_PRIVATE'):(OUT/folder).mkdir()
    rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rr)==336 and len({r['id'] for r in rr})==336
    split={r['id']:s for s in ('D208','V64','M64') for r in read(ROOT/'research/round24/input'/(s+'_PRIVATE.json'))}
    A={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
    frame_index=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
    counts=collections.Counter();rows=[]
    try:
        for i,r in enumerate(rr):
            rid=r['id'];suffix='M64' if r['stage']=='M' else '272';base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23')
            z=inv.cached(ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(rid+'.npz'))
            prep=inv.cached(base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            arr=inv.cached(base/('prepared'+suffix+'_attempt001')/'arrays_PRIVATE'/(rid+'.npz'))
            ev=inv.cached(base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            text=inv.cached(base/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            bank=inv.cached(base/('bank'+suffix+'_attempt001')/'samples_PRIVATE'/(rid+'.npz'))
            assert np.array_equal(z['boxes'],arr['source_boxes']) and np.array_equal(z['scores'],arr['source_scores'])
            maskpath=mask_route(r,A,frame_index);assert inv.sha(maskpath)==prep['bindings']['mask_cache_sha256']
            masks=inv.cached(maskpath);scene=mask_scene_lite(masks)
            stage=replay(z['scores'],arr['weights'],prep['mask_pool_PRIVATE'],ev,text,scene);assert_bank_exact(stage,bank)
            joint=inv.cached(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(rid+'.json'))
            ej=np.array([joint['targets_PRIVATE'][str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in prep['mask_pool_PRIVATE']])
            result=readouts16(z['scores'],stage,ej)
            oldjoint=inv.cached(ROOT/'crossplatform/round27/scores528_attempt001/samples_PRIVATE'/(rid+'.npz'))
            jindex=oldjoint['cue_names'].tolist().index('crop_scene_joint')
            assert np.array_equal(oldjoint['source'],z['scores']) and np.array_equal(oldjoint['boxes'],z['boxes'])
            assert np.array_equal(oldjoint['cues'][jindex],result['joint_rawcue']),('joint_rawcue_replay_not_exact',rid)
            # Historical current-G direct q and selected original box are a
            # measurement endpoint, not just a matching aggregate task score.
            oldfolder=ROOT/'crossplatform/round24'/('finalM64_attempt001' if split[rid]=='M64' else 'modal_'+split[rid]+'_attempt001')
            old=historical_prediction(oldfolder/'predictions_PRIVATE'/(rid+'.npz'))
            ix=old['arm_names'].tolist().index('Qwen_G_direct');oldq=int(old['query_indices'][ix])
            assert oldq==int(result['queries'][7]) and np.array_equal(old['boxes'][ix],z['boxes'][oldq]),('legacy_endpoint_not_exact',rid)
            q=result['queries'];np.savez_compressed(OUT/'samples_PRIVATE'/(rid+'.npz'),
                arm_names=np.array(result['names']),arm_scores=result['scores'],query_domains=result['domains'],queries=q,
                selected_boxes=z['boxes'][q],source_boxes=z['boxes'],source_scores=z['scores'],
                pool=np.array(prep['mask_pool_PRIVATE'],np.int64),weights=stage['W'],image_e=stage['image_e'],image_full=stage['image_full'],image_gray=stage['image_gray'],
                joint_e=ej,joint_rawcue=result['joint_rawcue'],joint_operator_vision=result['joint_legacy_operator_vision'],full_yes_operator_vision=result['full_yes_legacy_operator_vision'],
                mapped=stage['mapped'],vv=stage['vv'],rr=stage['rr'],mask_relations=stage['mask_relations'],
                vision=stage['vision'],position=stage['position'],relation=stage['relation'],vision_unclipped=stage['vision_unclipped'],
                position_unclipped=stage['position_unclipped'],relation_unclipped=stage['relation_unclipped'])
            case={'id_PRIVATE':rid,'split':split[rid],'platform':r['platform'],'scene_PRIVATE':r['scene'],
                'max_tie_sets_PRIVATE':result['max_tie_sets'],'empty_mapped_source_fallback':result['empty_mapped_source_fallback'],
                'anchor_v':stage['anchor_v'],'anchor_r':None if stage['anchor_r'] is None else stage['anchor_r'].tolist(),
                'subject_query_PRIVATE':text['subject_query'],
                'visual_text_source':{'path_PRIVATE':str((base/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json')).relative_to(ROOT)),
                    'field':'subject_query','string_sha256':hashlib.sha256(text['subject_query'].encode()).hexdigest(),
                    'scope':'subject_and_attributes_only_not_full_caption'},
                'position_text_sha256':hashlib.sha256(json.dumps(text['global_relations'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'reference_text_sha256':hashlib.sha256(json.dumps({'references':text['references'],'relations':text['reference_relations']},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'joint_text_source':{'field':'original_full_caption','string_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()},
                'global_relations_PRIVATE':text['global_relations'],
                'reference_relations_PRIVATE':text['reference_relations'],'references_PRIVATE':text['references'],
                'vision_clip_n':int((stage['vision']!=stage['vision_unclipped']).sum()),
                'position_clip_n':int((stage['position']!=stage['position_unclipped']).sum()),
                'relation_clip_n':int((stage['relation']!=stage['relation_unclipped']).sum()),
                'legacy_bank_all_components_exact':True,'legacy_historical_q_box_exact':True,'joint_cachedcue_exact':True,
                'mask_cache_path_PRIVATE':str(maskpath.relative_to(ROOT)),'mask_cache_sha256':inv.sha(maskpath),
                'no_GT_no_RGB_no_raw_XYZ_no_models':True}
            dump(OUT/'cases_PRIVATE'/(rid+'.json'),case);rows.append(case)
            counts['n']+=1;counts['legacy_bank_exact']+=1;counts['joint_cachedcue_exact']+=1
            if (i+1)%32==0:print('STAGE49',i+1,flush=True)
    except Exception as exc:
        dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':counts['n'],
                              'STOP_no_prediction_seal_no_scoring':True});raise
    dump(OUT/'COUNTS.json',dict(counts));dump(OUT/'INPUT_BINDINGS_PRIVATE.json',inv.BINDINGS)
    dump(OUT/'SEALED.json',{'state':'R49_ALL336_16_CACHED_STAGE_READOUTS_BEFORE_TASK_SCORING',
        'files':{p.relative_to(OUT).as_posix():inv.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        'n':336,'arms':ARM_NAMES,'GT_media_model_access':False,'plan_sha256':inv.sha(EXP/'STAGE_PLAN_LOCK.json'),
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('STAGE49_SEALED',dict(counts),flush=True)

if __name__=='__main__':main()
