"""R11 real reliability assay, weighted constraints and exact-coverage controls.
All labels are excluded. Seal every delivered arm, atomic application and score.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys,time,math
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path[:0]=[str(D),str(D.parent/'round10'),str(D.parent/'round7'),str(D.parent/'round7/experiment')]
from runner_helpers import sha,read,validate_manifest,eval_boxes,array_sha,to_json
from camera_projection import read_camera_allowlist,CameraSpec
from visibility_geometry import generate_geometry_targets,GeometryConfig
from rgb_support import make_rgb_support,RGBSupportConfig
from reliability import compute_reliability,ReliabilityConfig,select_exact_half
from weighted_geometry import generate_weighted_geometry_targets
ARMS=('source','unconditional_full','no_rgb_coordinate','weighted_constraints','uniform_weighted_constraints','reliable_50','hash_random_50','source_agreement_50','low_plane_residual_50','point_count_50','stability_only_50')
CONTROLS=('reliability','hash_random','source_agreement','low_plane_residual','point_count','stability_only')
FACES=tuple(f'{j}:{sign}' for j in range(3) for sign in (-1,1))

def empty_r(reason):
    return {'face_weights':dict.fromkeys(FACES,0.),'RGB_weight':0.,'RGB_weight_defined':False,
            'axis_reliability6':np.zeros(6),'axis_reliability_defined6':np.zeros(6,bool),
            'case_reliability':None,'active_face_keys':[],'association_status':'association_undefined',
            'features':{'not_computed_reason':reason,'faces':{}},'partition_hashes':{},
            'undefined_reasons':[reason],'control_scores':dict.fromkeys(CONTROLS,None)}

def face_defined(rel,key):
    if rel['association_status']=='association_conflicted':return True
    return rel['association_status']=='association_supported' and bool(rel['features'].get('faces',{}).get(key,{}).get('defined',False))

def main(a):
    a.plan=a.plan.resolve();a.output=a.output.resolve()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['stage']=='reliability_assay' and not plan['GT_read_permitted']
    source=Path(plan['source_dir']);s=read(source/'SEALED.json')
    assert s['state']=='sealed_R11_source_forward_and_input_observations_no_G_no_GT'
    assert sha(source/'SEALED.json')==plan['source_seal_sha256'] and s['n_samples']==320
    for n,h in s['files'].items():assert sha(source/n)==h,n
    for n,h in plan['program_sha256'].items():assert sha(ROOT/n)==h,n
    manifest=Path(plan['manifest']);assert sha(manifest)==s['manifest_sha256']==plan['manifest_sha256']
    rows=validate_manifest(read(manifest));assert len(rows)==320
    sp=read(source/'registered_plan.json');st={r['id']:r for r in read(source/'trace.json')}
    gc,rc,rlc=GeometryConfig(),RGBSupportConfig(),ReliabilityConfig()
    for key,cfg in [('geometry_config',gc),('rgb_config',rc),('reliability_config',rlc)]:assert dataclasses.asdict(cfg)==plan[key]
    import cv2,scipy
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'registered_plan.json').write_bytes(a.plan.read_bytes());(a.output/'manifest_input.json').write_bytes(manifest.read_bytes())
    start=time.perf_counter();records=[];camera_cache={};bridge=[]
    pools={p:{'face':[],'RGB':[]} for p in ('drone','quad')}
    for i,row in enumerate(rows):
        cp=source/'source_cache'/f'{row["id"]}.npz';op=source/'observations'/f'{row["id"]}.npz'
        with np.load(cp,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
        with np.load(op,allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
        pc=o['sampled_points_model'];ix=o['raw_indices'];boxes=c['boxes_raw6'][0];scores=c['text_scores'][0];q=int(c['query_index']);b0=boxes[q].copy()
        assert q==int(scores.argmax()) and boxes.shape==(256,6)
        assert c['point_sample_sha256'].tobytes().hex()==array_sha(pc)
        assert c['raw_indices_sha256'].tobytes().hex()==array_sha(ix)
        assert c['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()
        assert c['actual_model_text_sha256'].tobytes()==hashlib.sha256((row['caption']+sp['sentinel']).encode()).digest()
        assert sha(row['imagepath'])==st[row['id']]['image_sha256'] and sha(row['metapath'])==st[row['id']]['camera_container_sha256']
        p,first=np.unique(pc,axis=0,return_index=True);raw=ix[first]
        with Image.open(row['imagepath']) as im:rgb=np.array(im.convert('RGB'))
        camera_error=None
        try:
            if row['metapath'] not in camera_cache:camera_cache[row['metapath']]=read_camera_allowlist(row['metapath'])
            camera=CameraSpec.from_camera_fields(row['platform'],camera_cache[row['metapath']],waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared');camera.validated()
        except ValueError as error:camera=None;camera_error=str(error)
        if camera is None:
            seg={'mask2d':np.zeros(rgb.shape[:2],bool),'valid':False,'diagnostics':{'reason':'camera_contract:'+camera_error}}
            base={'arm_boxes_raw6':np.tile(b0,(4,1)),'acceptance_mask6':np.zeros((4,6),bool),'point_support_sha256':np.array(['0'*64]*4),'pre_constraint_geometry6':np.tile(b0,(4,1)),'arm_status':['source']+['camera_contract_fallback']*3,'diagnostics':{'reason':camera_error}}
        else:
            seg=make_rgb_support(rgb=rgb,points_fit=p,source_box6=b0,camera_spec=camera,config=rc)
            base=generate_geometry_targets(points_fit=p,raw_indices=raw,source_box6=b0,q0=q,camera_spec=camera,rgb_mask2d=seg['mask2d'],segmentation_valid=seg['valid'],config=gc)
        bb=np.asarray(base['arm_boxes_raw6'],np.float32);eligible=bool(np.any(bb[1]!=b0))
        assert np.array_equal(bb[0],b0)
        rel=compute_reliability(sample_id=row['id'],points=p,raw_indices=raw,rgb=rgb,source_box6=b0,camera_spec=camera,base_mask2d=seg['mask2d'],base_result=base,config=rlc) if eligible else empty_r('no_original_G_actual_change')
        params={'points_fit':p,'raw_indices':raw,'source_box6':b0,'q0':q,'camera_spec':camera,'rgb_mask2d':seg['mask2d'],'segmentation_valid':bool(seg['valid']),'config':gc}
        weighted=b0.copy();wd={'status':'source_no_candidate'}
        if eligible:
            ones=generate_weighted_geometry_targets(**params,face_weights=dict.fromkeys(FACES,1.),RGB_weight=1.)
            zeros=generate_weighted_geometry_targets(**params,face_weights=dict.fromkeys(FACES,0.),RGB_weight=0.)
            assert np.array_equal(np.asarray(ones['arm_boxes_raw6']),np.asarray(base['arm_boxes_raw6']))
            assert np.array_equal(ones['acceptance_mask6'],base['acceptance_mask6']) and np.array_equal(ones['point_support_sha256'],base['point_support_sha256'])
            assert np.array_equal(zeros['arm_boxes_raw6'][1],b0)
            bridge.append({'id':row['id'],'allone_base_bitexact':True,'allzero_source_bitexact':True})
            w=generate_weighted_geometry_targets(**params,face_weights=rel['face_weights'],RGB_weight=rel['RGB_weight'])
            weighted=np.asarray(w['arm_boxes_raw6'][1],np.float32);wd=w['diagnostics']
            assert np.array_equal(w['acceptance_mask6'],base['acceptance_mask6']) and np.array_equal(w['point_support_sha256'],base['point_support_sha256'])
        for key in rel['active_face_keys']:
            if face_defined(rel,key):pools[row['platform']]['face'].append(float(rel['face_weights'][key]))
        if eligible and rel['RGB_weight_defined']:pools[row['platform']]['RGB'].append(float(rel['RGB_weight']))
        rec={'row':row,'boxes':boxes,'scores':scores,'q':q,'b0':b0,'base_boxes':bb,'base':base,'mask':seg['mask2d'],'seg':seg['diagnostics'],'r':rel,'eligible':eligible,'weighted':weighted,'weighted_diagnostics':wd,
             'point_sha':c['point_sample_sha256'],'index_sha':c['raw_indices_sha256'],'cache_sha':sha(cp),'image_sha':sha(row['imagepath']),'unique_XYZ':len(p),'unique_raw':len(np.unique(ix)),
             'work':params if eligible else None}
        records.append(rec)
        if i%20==0 or i+1==len(rows):print(json.dumps({'stage':'base_and_reliability','completed':i+1,'n':len(rows),'base_actual_changed':sum(x['eligible'] for x in records),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    means={p:{k:{'n_defined_active':len(v),'mean':math.fsum(v)/len(v) if v else None,'defined_zero_count':sum(x==0 for x in v)} for k,v in pool.items()} for p,pool in pools.items()}
    units=[{'id':x['row']['id'],'platform':x['row']['platform'],'eligible':x['eligible'],'scores':x['r']['control_scores']} for x in records]
    selections={k:select_exact_half(units,k,config=rlc) for k in CONTROLS}
    selected={k:set(v['selected_ids']) for k,v in selections.items()}
    control_lock={'uniform_means_perplatform':means,'uniform_pool':'effective-defined active constraints; conflictdefinedzero included; no inactive/undefined zeros','uniform_application':'same correspondence and undefined mask as weighted; outside-originalC source bydesign','selection':selections,'hash_random_reference':{'salt':'R11_random_reference_v1','seed':0,'draws':1000,'literal':'SHA256(R11_random_reference_v1|0| + str(draw) + | + id)'},'transductive_without_GT':True}
    (a.output/'COHORT_CONTROL_LOCK.json').write_text(json.dumps(to_json(control_lock),indent=2,allow_nan=False)+'\n')
    (a.output/'ACTUAL_WEIGHT_BRIDGES.json').write_text(json.dumps(bridge,indent=2)+'\n')
    traces=[];files={}
    for i,x in enumerate(records):
        row=x['row'];rel=x['r'];b0=x['b0'];uniform=b0.copy();ud={'status':'source_no_candidate'}
        if x['eligible']:
            pool=means[row['platform']];fp=pool['face']['mean'];ri=pool['RGB']['mean'];supported=rel['association_status']=='association_supported'
            fw={key:float(fp) if supported and fp is not None and face_defined(rel,key) else 0. for key in FACES}
            iw=float(ri) if supported and ri is not None and rel['RGB_weight_defined'] else 0.
            u=generate_weighted_geometry_targets(**x['work'],face_weights=fw,RGB_weight=iw);uniform=np.asarray(u['arm_boxes_raw6'][1],np.float32);ud=u['diagnostics']
        gates=np.array([row['id'] in selected[k] for k in CONTROLS],bool)
        rawboxes=np.stack([b0,x['base_boxes'][1],x['base_boxes'][2],x['weighted'],uniform]+[x['base_boxes'][1] if keep else b0 for keep in gates]).astype(np.float32)
        atomic=np.tile(b0,(6,1));atomic[np.arange(6),np.arange(6)]=x['base_boxes'][1]
        active=x['base_boxes'][1]!=b0
        vals=[rel['control_scores'].get(k) for k in CONTROLS]
        z={'sample_id':np.array(row['id']),'source_boxes_raw6':x['boxes'],'source_text_scores256':x['scores'],'q_source':np.array(x['q'],np.int64),
           'arm_names':np.array(ARMS),'arm_query_index':np.full(11,x['q'],np.int64),'arm_boxes_raw6':rawboxes,'arm_boxes_eval6':eval_boxes(rawboxes),
           'arm_status':np.array(['source',x['base']['arm_status'][1],x['base']['arm_status'][2],x['weighted_diagnostics'].get('solver',{}).get('full_rgb',{}).get('status','source_weight_gate'),ud.get('solver',{}).get('full_rgb',{}).get('status','source_weight_gate')]+['retained_original_G' if b else 'source_coverage_fallback' for b in gates]),
           'atomic_boxes_raw6':atomic,'atomic_boxes_eval6':eval_boxes(atomic),'atomic_active6':active,'base_acceptance_mask6':np.asarray(x['base']['acceptance_mask6'][1],bool),
           'coordinate_reliability6':np.asarray(rel['axis_reliability6'],float),'coordinate_defined6':np.asarray(rel['axis_reliability_defined6'],bool),
           'RGB_reliability':np.array(rel['RGB_weight'],float),'RGB_defined':np.array(rel['RGB_weight_defined'],bool),
           'case_reliability':np.array(0. if rel['case_reliability'] is None else rel['case_reliability'],float),'case_defined':np.array(rel['case_reliability'] is not None,bool),
           'control_score_names':np.array(CONTROLS),'control_scores':np.array([0. if v is None else v for v in vals],float),'control_defined':np.array([v is not None for v in vals],bool),'gate_selection':gates,
           'association_status':np.array(rel['association_status']),'face_keys':np.array(FACES),'face_weights':np.array([rel['face_weights'][k] for k in FACES],float),
           'point_support_sha256':np.asarray(x['base']['point_support_sha256']),
           'point_sample_sha256':x['point_sha'],'raw_indices_sha256':x['index_sha'],'source_cache_sha256':np.frombuffer(bytes.fromhex(x['cache_sha']),np.uint8),
           'image_file_sha256':np.frombuffer(bytes.fromhex(x['image_sha']),np.uint8),'RGB_mask2d_private':x['mask']}
        assert rawboxes.shape==(11,6) and np.isfinite(rawboxes).all() and atomic.shape==(6,6)
        assert np.array_equal(rawboxes[1],x['base_boxes'][1]) and np.array_equal(rawboxes[2],x['base_boxes'][2])
        name=row['id']+'.npz';np.savez_compressed(a.output/name,**z);files[name]=sha(a.output/name)
        tr={'id':row['id'],'platform':row['platform'],'scene':row['scene'],'q0':x['q'],'base_actual_change':x['eligible'],'geometry_unique_XYZ':x['unique_XYZ'],'unique_raw_indices':x['unique_raw'],
            'RGB':x['seg'],'base_geometry':x['base'],'reliability':rel,'weighted_solver':x['weighted_diagnostics'],'uniform_solver':ud,'uniform_means':means[row['platform']],
            'gates':dict(zip(CONTROLS,gates.tolist())),'source_cache_sha256':x['cache_sha'],'atomic_is_coordinate_application_not_face_causal_effect':True}
        tr['base_geometry']={k:v for k,v in tr['base_geometry'].items() if k not in ('arm_boxes_raw6','pre_constraint_geometry6')}
        traces.append(to_json(tr))
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    for n in ['trace.json','COHORT_CONTROL_LOCK.json','ACTUAL_WEIGHT_BRIDGES.json','registered_plan.json','manifest_input.json']:files[n]=sha(a.output/n)
    seal={'state':'sealed_R11_reliability11_and_atomic6_before_GT','n_samples':320,'n_arms':11,'atomic_arms_per_sample':6,'source_seal_sha256':sha(source/'SEALED.json'),'source_checkpoint_sha256':s['source_checkpoint_sha256'],'source_model_state_fingerprint':s['source_model_state_fingerprint'],
          'manifest_sha256':sha(manifest),'plan_sha256':sha(a.plan),'program_sha256':plan['program_sha256'],'program_sha256_base':'workspace_root','all_queries_fixed_original_q0':True,'all_reliability_and_rank_weights_sealed_before_GT':True,'neural_parameter_updates_executed':False,
          'uniform_means_perplatform':means,'same_coverage_counts':{k:v['by_platform'] for k,v in selections.items()},'actual_allone_allzero_bridge_cases':len(bridge),'strict_FP32_flags':s['strict_FP32_flags'],'elapsed_seconds':time.perf_counter()-start,
          'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print('R11_RELIABILITY_SEALED',sha(a.output/'SEALED.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
