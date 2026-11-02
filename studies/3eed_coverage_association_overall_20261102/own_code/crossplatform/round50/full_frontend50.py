"""UNEXECUTED full5912 frontend/source/segmentation recipe draft.

Import does not open any full manifest/media/predictions. Even register/source
require a positive locked DEV gate and a separate root full-cohort grant. The
closed FINAL registration/results are never an input. Existing R8 source
arrays are reused; no new grounder, source training or source features claimed.
"""
from pathlib import Path
import argparse,collections,hashlib,json,math,re,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round50/experiment'
COHORT='research/round8/experiment/full5912_gpu_inputs.json'
SOURCE='crossplatform/round8/full5912_attempt001'
MODEL_RECIPES={
 'mask':{'reference':'crossplatform/round12/capture_masks.py + crossplatform/round46/frames46.py',
    'family':'maskrcnn_resnet50_fpn_v2','score_threshold':.5,'mask_threshold':.5,
    'min_size':800,'max_size':1333,'detections_per_image':100,'dtype':'FP32',
    'source_caption_boxes_or_GT_inputs':False,'native_label_and_order_unchanged':True},
 'text':{'reference':'crossplatform/round23/parse23.py::parse + round20/textZ.py::pipeline',
    'processors':'tokenize,mwt,pos,lemma,depparse','device':'CPU','download':False,
    'new_subject_parser_or_VLM_roles':False,'reference_entity_cap':4},
 'references':{'reference':'crossplatform/round23/detect23.py + round46/oldg46.py::detect_fresh',
    'original_oldpool_nonempty_gate':True,'DINO_threshold':.25,'NMS':.5,'top_after_NMS':10,
    'flip_stability_IoU':.5,'disable_flip_if_query_contains':['left','right','clockwise'],
    'max_original_and_flip_forwards_per_reference':2},
 'numeric':{'reference':'round20/visualZ.py::Evidence,crops + round50/visual50.py ledger/runtime',
    'actual_subject_attributes_only':True,'two_views':['tight_pad0_masked_gray127','context_pad0.2_unmasked'],
    'cue':'mean(two per-view full_minus_gray YesNo logodds)','max_stable_reference_boxes':3,
    'reference_numeric_gate':'original_oldpool_nonempty_only','teacher_frozen':True,'generated_tokens':0},
 'readout':{'reference':'crossplatform/round50/pool_weights50.py','lambda_VPR':[2,2,2],
    'no_new_box_or_parameter_updates':True,'shared_DEV_selected_k_only':True}}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def triple(row):return tuple(row[k] for k in ('imagepath','pointpath','metapath'))
def chunked(rows,n):return [rows[i:i+n] for i in range(0,len(rows),n)]

def context():
 planpath=EXP/'FULL_FRONTEND_PLAN_LOCK.json';auth=read(EXP/'FULL_FRONTEND_EXECUTION_AUTHORIZATION.json')
 assert auth['state']=='ROOT_AUTHORIZED_R50_LOCKED_FULL5912_FRONTEND'
 assert auth['plan_sha256']==sha(planpath)
 plan=read(planpath)
 assert plan['manifest']==COHORT and plan['source_cache']==SOURCE and plan['n']==5912
 assert (ROOT/plan['output_root']).resolve().is_relative_to((ROOT/'crossplatform/round50').resolve())
 assert plan['no_protected_FINAL_registration_or_results_input'] is True
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'round46/final' not in rel and 'FINAL384' not in rel and sha(ROOT/rel)==h
 # This gate is checked BEFORE full manifest decoding or any full sample I/O.
 gatepath=ROOT/plan['DEV_gate_receipt'];assert sha(gatepath)==plan['DEV_gate_sha256'];gate=read(gatepath)
 assert gate['state']=='PASS_R50_LOCKED_DEV_GATE_FOR_FULL'
 assert gate['selected_k'] in (3,10) and gate['selected_k']==plan['selected_k']
 assert gate['selected_configuration']==plan['selected_configuration']
 assert gate['balanced_gain_vs_oldG_pp']>0 and gate['balanced_gain_vs_source_pp']>0
 assert gate['all_seven_DEV_configurations_reported'] is True
 assert plan['manifest'] in plan['inputs_sha256'] and (SOURCE+'/SEALED.json') in plan['inputs_sha256']
 rows=read(ROOT/plan['manifest'])
 assert len(rows)==5912 and len({r['id'] for r in rows})==5912 and len({triple(r) for r in rows})==5912
 assert collections.Counter(r['platform'] for r in rows)=={'drone':2984,'quad':2928}
 keys={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
 assert all(set(r)==keys for r in rows)
 seal=read(ROOT/SOURCE/'SEALED.json');assert seal['n_samples']==5912 and seal['manifest_sha256']==sha(ROOT/COHORT)
 return plan,rows,seal

def source_arrays(row,source_seal):
 """Read only R8 frozen source fields, never the old R8 G/TTA arm outputs."""
 p=ROOT/SOURCE/(row['id']+'.npz');assert sha(p)==source_seal['files'][p.name]
 with np.load(p,allow_pickle=False) as z:
  assert str(z['sample_id'])==row['id'];boxes=z['source_boxes_raw6'].copy();scores=z['source_text_scores256'].copy()
  image_sha=z['image_file_sha256'].tobytes().hex()
 assert boxes.shape==(256,6) and scores.shape==(256,) and boxes.dtype==scores.dtype==np.float32
 assert np.isfinite(boxes).all() and np.isfinite(scores).all()
 return {'boxes':boxes,'scores':scores,'image_sha256':image_sha,'R8_file_sha256':sha(p),
         'query_or_text_features_available':False,'new_grounder_forwards':0}

def source_input_hashes():
 """After context only: streaming source input SHA fields, not old method data.

 The tokenizer traverses the trace bytes but drops all old arm results and
 geometries instead of materializing their objects.
 """
 import ijson
 seal=read(ROOT/SOURCE/'SEALED.json');p=ROOT/SOURCE/'trace.jsonl'
 assert sha(p)==seal['files']['trace.jsonl']
 allowed={'id','raw_point_file_sha256','image_file_sha256','camera_metadata_file_sha256'};out={};row=None
 with p.open('rb') as f:
  for prefix,event,value in ijson.parse(f,multiple_values=True):
   if prefix=='' and event=='start_map':row={}
   elif prefix=='' and event=='end_map':
    assert set(row)==allowed and row['id'] not in out;out[row['id']]=row;row=None
   elif row is not None and prefix in allowed and event=='string':row[prefix]=value
 assert len(out)==5912
 return out

def known_frame_registry():
 """Four registered metadata/mask inventories; no model/media/GT read here."""
 sources=[('R12',read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json'),ROOT/'crossplatform/round12/masks501_attempt001'),
          ('A64',read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json'),ROOT/'crossplatform/round15/masks64_attempt001')]
 for tag,roundno in [('P64',26),('Q64',27)]:
  prefix=tag[0];rs=read(ROOT/f'research/round{roundno}/input/{tag}_PRIVATE.json')
  index=read(ROOT/f'research/round{roundno}/input/{prefix}_FRAME_INDEX_PRIVATE.json')
  route={r['frame_alias']:dict(index[r['frame_alias']],**{k:r[k] for k in ('imagepath','pointpath','metapath')}) for r in rs}
  sources.append((tag,route,ROOT/f'crossplatform/round{roundno}/masks{tag}_attempt001'))
 out={}
 for tag,index,folder in sources:
  s=read(folder/'SEALED.json')
  for alias,fr in index.items():
   p=folder/(alias+'.npz');assert sha(p)==s['files'][p.name]
   key=triple(fr);assert key not in out,'Registered mask/camera cache pathtriples expected unique'
   fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
   if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
   out[key]={'cache_family':tag,'mask_path_PRIVATE':p.relative_to(ROOT).as_posix(),'mask_sha256':sha(p),
     'image_sha256':fr['image_sha256'],'point_sha256':fr['point_sha256'],'metadata_sha256':fr['metadata_sha256'],
     'camera_fields_PRIVATE':fields,'mask_seal_sha256':sha(folder/'SEALED.json')}
 assert len(out)==693
 return out

def legacy_category_without_features(caption,graph,lexicon):
 """R23 category expression only: gamma0 means token_delta is never consumed.

 Do not invent missing R8 token/query features or change source probabilities.
 Do not replace this with the R48 head parser or corrected COCO IDs.
 """
 end=graph['base_text'].get('subject',{}).get('span',[0,len(caption)])[1]
 match=re.search(r'\b(?:'+'|'.join(lexicon)+r')\b',caption[:end],re.I)
 persons=['pedestrian','person','people','man','woman','child','boy','girl','walker','worker']
 return 'person' if match and match.group().lower() in persons else match.group().lower() if match else graph['category'].lower()

def legacy_entry(scene,source,raw_geometry,category,original_choose):
 """Same R23 entry incl historical coarse-ID bug and original <=2 pairs.

 original_choose is the pinned round20/prepareZ.py::choose pure function,
 obtained by the audited worker adapter rather than a new proposal algorithm.
 """
 from pool_weights50 import legacy_weights,taxonomy_flags
 M=len(scene['objects']);allm=np.arange(M,dtype=np.int64)
 w=legacy_weights(allm,raw_geometry['point_support_count'],raw_geometry['point_support_fraction'],raw_geometry['projection_iou'],source['boxes'])
 quality=w['quality'];valid=w['valid'];s=source['scores'];q0=int(s.argmax())
 flags=taxonomy_flags(category,np.array([o['predicted_class_id'] for o in scene['objects']],np.int64))
 target=[m for m in range(M) if valid[:,m].any() and flags['historical_taxonomy_compatible'][m]]
 target=sorted(target,key=lambda m:(-float(np.max(s*quality[:,m])),m))[:6]
 source_mask=int(np.argmax(quality[q0])) if M and quality[q0].max()>0 else None
 pairs=original_choose(scene,source,raw_geometry);pool=list(target)
 if source_mask is not None:pool.append(source_mask)
 for pair in pairs['pairs']:pool+=pair['mask_indices_PRIVATE']
 return {'old_pool':np.array(sorted(set(pool)),np.int64),'legacy_pairs':pairs,'source_mask':source_mask,
         'taxonomy_flags':flags,'token_delta_not_constructed_because_literal_gamma_zero':True}

def register(plan,rows,source_seal):
 """After positive gate only, emit bounded stage jobs before model stages."""
 folder=ROOT/plan['output_root']/'registry';assert not folder.exists();folder.mkdir(parents=True)
 reuse=known_frame_registry();source_inputs=source_input_hashes();source_jobs=[];new_frames=[];reused=[]
 for row in rows:
  h=source_inputs[row['id']];prior=reuse.get(triple(row));source_jobs.append(row)
  if prior:
   assert (prior['image_sha256'],prior['point_sha256'],prior['metadata_sha256'])==(h['image_file_sha256'],h['raw_point_file_sha256'],h['camera_metadata_file_sha256'])
   reused.append({'id_PRIVATE':row['id'],'reuse_PRIVATE':prior})
  else:new_frames.append(row)
 assert len(reused)==693 and len(new_frames)==5219
 dump(folder/'FULL_ROWS_PRIVATE.json',rows);dump(folder/'SOURCE_INPUT_SHA_PRIVATE.json',source_inputs)
 dump(folder/'REUSED693_FRAME_BINDINGS_PRIVATE.json',reused);dump(folder/'MISSING5219_FRAMES_PRIVATE.json',new_frames)
 jobs=[]
 for stage,rr,size,resource in [('source',source_jobs,256,'cpu'),('fresh_masks',new_frames,128,'gpu'),('text_graph',rows,256,'cpu'),('geometry',rows,128,'cpu')]:
  for j,chunk in enumerate(chunked(rr,size)):
   name=f'{stage}_{j:04d}';dump(folder/'chunks_PRIVATE'/(name+'.json'),chunk)
   jobs.append({'stage':stage,'chunk':name,'n':len(chunk),'resource':resource,
       'new_mask_forward_upper_bound':len(chunk) if stage=='fresh_masks' else 0,
       'grounder_forward_count':0,'new_Qwen_calls':0,'GT_access':False})
 dump(folder/'STAGE_JOBS_PRIVATE.json',jobs);dump(folder/'MODEL_RECIPES.json',MODEL_RECIPES)
 dump(folder/'BUDGET_CAPACITY.json',{'n':5912,'source_forward_count':0,'mask_reuse':693,'new_mask_forward_upper_bound':5219,
   'DINO_capacity_upper_bound':5912*8,'Qwen_numeric_capacity_upper_bound':5912*(4*100+24),
   'capacity_not_an_execution_budget':True,'exact_reference_and_numeric_worklists_required_after_geometry':True,
   'old_and_wide_targets_scored_once_for_all_five_readouts':True,'no_source_features_or_TTA_ready_claim':True})
 dump(folder/'SEALED.json',{'state':'R50_LOCKED_FULL5912_INPUT_ROUTE_AND_SEGMENT_PLAN_NO_MODEL_CALLS',
    'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    'n':5912,'source_seal_sha256':sha(ROOT/SOURCE/'SEALED.json'),'plan_sha256':sha(EXP/'FULL_FRONTEND_PLAN_LOCK.json'),
    'no_GT_or_protected_FINAL_registration_or_result_access':True,'model_calls':0})

def source_chunk(plan,source_seal,chunk):
 assert re.fullmatch(r'source_[0-9]{4}',chunk)
 reg=ROOT/plan['output_root']/'registry';rs=read(reg/'chunks_PRIVATE'/(chunk+'.json'))
 seal=read(reg/'SEALED.json');rel='chunks_PRIVATE/'+chunk+'.json';assert sha(reg/rel)==seal['files'][rel]
 inputpath=reg/'SOURCE_INPUT_SHA_PRIVATE.json';assert sha(inputpath)==seal['files'][inputpath.name];inputs=read(inputpath)
 folder=ROOT/plan['output_root']/'source_chunks'/chunk;assert not folder.exists();folder.mkdir(parents=True);(folder/'samples_PRIVATE').mkdir()
 for row in rs:
  s=source_arrays(row,source_seal);assert s['image_sha256']==inputs[row['id']]['image_file_sha256']
  np.savez_compressed(folder/'samples_PRIVATE'/(row['id']+'.npz'),boxes=s['boxes'],scores=s['scores'])
 dump(folder/'SEALED.json',{'state':'R50_R8_SOURCE_BOX_SCORE_ADAPTER_BITPRESERVED_NO_NEW_FORWARD','n':len(rs),
    'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    'source_seal_sha256':sha(ROOT/SOURCE/'SEALED.json'),'source_features_available':False,'new_grounder_forwards':0})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['register','source'],required=True);ap.add_argument('--chunk');args=ap.parse_args()
 plan,rows,source_seal=context()
 if args.stage=='register':register(plan,rows,source_seal)
 else:
  assert args.chunk;source_chunk(plan,source_seal,args.chunk)

if __name__=='__main__':main()
