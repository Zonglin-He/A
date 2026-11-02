"""Pre-GT R51 cached relation-fidelity repair; no new support/model/media.

Read only a completed safe prediction and its same raw-safe input bundle.
Restore R23 weighted/(1+total), leaving fixed references/colors/P/source intact.
Historical safe attempt and its code remain unchanged. No retry or scoring.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence51_fidelity import object_geometry,raw_relations,propagate_measured

MANIFEST='research/round24/input/ALL336_PRIVATE.json'
FACTORS=[.5,1.,2.]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def npz(p,fields=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if fields is None else fields)}
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_CACHED_RELATION_FIDELITY_REPAIR_NO_GT' and auth['plan_sha256']==sha(args.plan)
 assert plan['state']=='ROOT_LOCKED_R51_CACHED_RELATION_FIDELITY_REPAIR' and plan['n']==336
 assert plan['manifest']==MANIFEST and plan['appearance_factor_values']==FACTORS
 assert plan['no_GT_new_model_media_or_refsupport_recompute'] is True
 required={MANIFEST,plan['safe_root']+'/SEALED.json',plan['prepared_root']+'/SEALED.json'}
 assert set(plan['inputs_sha256'])==required
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 for f in ('crossplatform/round51/safe_relation_fidelity51.py','crossplatform/round51/evidence51_fidelity.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 safe=path(plan['safe_root']);prepared=path(plan['prepared_root']);out=path(plan['output_root'])
 assert all(p.parent==ROOT/'crossplatform/round51' for p in (safe,prepared,out)) and not out.exists()
 ss=read(safe/'SEALED.json');ps=read(prepared/'SEALED.json')
 assert ss['state']=='R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT' and ss['n']==ps['n']==336
 assert ss['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert ps['external_vision_outputs_or_mask_derived_cache_read'] is False
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 out.mkdir(parents=True)
 for d in ('samples_PRIVATE','cases_PRIVATE','references_PRIVATE'):(out/d).mkdir()
 bindings=[]
 def bound(root,seal,rel,role):
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
  p=root/rel;assert sha(p)==seal['files'][rel]
  bindings.append({'role':role,'path_PRIVATE':str(p),'sha256':sha(p)})
  return p
 counts={'n':0,'P_V_and_noR_exact_bridges':0,'changed_R_vectors':0,'changed_full_queries':0,
         'global_relations':0,'global_non_camera_image':0,'global_predicate_counts':{}}
 start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];op=bound(safe,ss,'samples_PRIVATE/'+sid+'.npz','ORIGINAL_SAFE_SCORES')
   old=npz(op);pc=read(bound(prepared,ps,'cases_PRIVATE/'+sid+'.json','RAW_SAFE_CASE'))
   attrs=read(bound(prepared,ps,'attributes_PRIVATE/'+sid+'.json','PURE_GRAPH_AND_ATTRIBUTES'))
   pa=npz(bound(prepared,ps,'arrays_PRIVATE/'+sid+'.npz','RAW_SAFE_TARGET_GEOMETRY'),
          ['instance_rect_xyxy','instance_crop_mask_pixel_offsets','instance_crop_mask_pixel_xy','source_boxes','source_scores'])
   assert exact(pa['source_boxes'],old['source_boxes']) and exact(pa['source_scores'],old['source_scores'])
   graph=attrs['text_graph_unchanged'];objects=object_geometry(pa,pc['image_size_wh'])
   for rel in graph.get('global_relations',[]):
    counts['global_relations']+=1;counts['global_non_camera_image']+=rel.get('frame')!='camera_image'
    key=rel['predicate'];counts['global_predicate_counts'][key]=counts['global_predicate_counts'].get(key,0)+1
   rp=bound(safe,ss,'references_PRIVATE/'+sid+'.json','FIXED_SOURCE_TOP3_FACTOR_REFERENCES')
   references=read(rp);assert references['appearance_factor_values']==FACTORS
   Fs=[];Cs=[];PRs=[];Ks=[];QKs=[];anchors=[];masses=[];NR=[]
   for fi,c in enumerate(FACTORS):
    rr={key:value['hypotheses'] for key,value in references['references_by_factor'][fi].items()}
    pr,known=raw_relations(graph,objects,rr,pc['image_size_wh'])
    assert exact(pr[:,0],old['raw_PR'][fi,:,0]) and exact(known,old['raw_PR_known'][fi])
    F,C,info=propagate_measured(old['source_scores'],old['query_instance_conditional_weights'],c*old['instance_attribute_e'],pr,old['instance_attribute_known'],known)
    assert exact(C[:2],old['components'][fi,:2])
    npr=pr.copy();npr[:,1]=0.;nk=known.copy();nk[:,1]=False
    nr,_,_=propagate_measured(old['source_scores'],old['query_instance_conditional_weights'],c*old['instance_attribute_e'],npr,old['instance_attribute_known'],nk)
    assert exact(nr,old['no_relation_scores'][fi])
    counts['changed_R_vectors']+=not exact(C[2],old['components'][fi,2])
    Fs.append(F);Cs.append(C);PRs.append(pr);Ks.append(known);QKs.append(info['query_known']);anchors.append(info['anchors']);masses.append(info['query_known_weight_mass']);NR.append(nr)
   new=dict(old);new['scores']=np.stack([old['scores'][0]]+Fs);new['queries']=new['scores'].argmax(1).astype(np.int64)
   new['selected_original_boxes']=old['source_boxes'][new['queries']]
   new['components']=np.stack(Cs);new['raw_PR']=np.stack(PRs);new['raw_PR_known']=np.stack(Ks)
   new['cue_query_known']=np.stack(QKs);new['cue_anchors']=np.stack(anchors);new['cue_query_known_weight_mass']=np.stack(masses)
   new['no_relation_scores']=np.stack(NR)
   assert exact(new['no_relation_scores'],old['no_relation_scores'])
   counts['changed_full_queries']+=int(np.sum(new['queries'][1:]!=old['queries'][1:]))
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**new)
   for extension in ('.json','.npz'):
    src=bound(safe,ss,'references_PRIVATE/'+sid+extension,'UNCHANGED_REFERENCE_DISTRIBUTION')
    shutil.copyfile(src,out/'references_PRIVATE'/(sid+extension))
   case=read(bound(safe,ss,'cases_PRIVATE/'+sid+'.json','ORIGINAL_SAFE_CASE'))
   case.update(samples_sha256=sha(out/'samples_PRIVATE'/(sid+'.npz')),
       reference_arrays_sha256=sha(out/'references_PRIVATE'/(sid+'.npz')),
       relation_fidelity_repair='literal_R23_weighted/(1+total)_float64_rectangle_IoU',
       previous_safe_sample_sha256=sha(op),source_P_V_noR_unchanged_bytes=True)
   dump(out/'cases_PRIVATE'/(sid+'.json'),case)
   counts['n']+=1;counts['P_V_and_noR_exact_bridges']+=1
   if counts['n']%32==0:print('R51_RELATION_FIDELITY',counts['n'],336,flush=True)
  wp=bound(safe,ss,'REFERENCE_WORK_ITEMS_PRIVATE.json','UNCHANGED_SOURCE_TOP3_DIAGNOSTIC_WORKLIST')
  shutil.copyfile(wp,out/'REFERENCE_WORK_ITEMS_PRIVATE.json')
  assert sha(wp)==sha(out/'REFERENCE_WORK_ITEMS_PRIVATE.json')
  dump(out/'READ_BINDINGS_PRIVATE.json',{'files':bindings,'no_original_media_or_refsupport_recompute':True})
  dump(out/'COUNTS.json',counts)
  dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'grounder_forwards':0,'external_model_forwards':0,'optimizer_steps':0,'reference_point_support_recomputations':0})
  dump(out/'SEALED.json',{'state':'R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT','n':336,
      'plan_sha256':sha(args.plan),'previous_safe_seal_sha256':sha(safe/'SEALED.json'),
      'relation_operator':'literal_R23_weighted/(1+total)','reference_top3_and_distribution_unchanged_bytes':True,
      'external_vision_outputs_or_mask_derived_caches_read':False,'task_GT_read':False,
      'original_source_boxes_unchanged':True,'appearance_factor_values':FACTORS,
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as e:
  dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':counts['n'],'automatic_retry':False});raise

if __name__=='__main__':main()
