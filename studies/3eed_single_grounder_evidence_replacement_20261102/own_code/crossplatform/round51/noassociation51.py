"""R51 association-removal ablation, raw-safe CPU only; no run on import.

RGB is retained. Subject boxes directly collect all visible point pixels;
references keep source-only top3 QUERY slots, deliberately no instance dedup.
This removes instance binding AND duplicate suppression; it is not no-RGB.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from attributes51 import color_support,COLOR_RGB_PROTOTYPES,MINIMUM_COLOR_PIXELS,COLOR_BETA_PSEUDOCOUNT,COLOR_LOGODDS_CLIP
from evidence51_fidelity import attr_evidence,raw_relations,propagate_measured

FACTORS=[.5,1.,2.]
ARMS=['no_instance_association_c05','no_instance_association_c1','no_instance_association_c2']
MANIFESTS={336:'research/round24/input/ALL336_PRIVATE.json',5912:'research/round8/experiment/full5912_gpu_inputs.json'}

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
def npz(p,fields):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}

def query_observation(box,model_visible_xyz,pixels,W,H,index,domain):
 inside=(np.abs(model_visible_xyz.astype(np.float64)-box[:3])<=box[3:]/2).all(1) if (box[3:]>0).all() else np.zeros(len(pixels),bool)
 selected=np.flatnonzero(inside);xy=pixels[selected]
 rect=np.r_[xy.min(0),xy.max(0)+1] if len(xy) else np.zeros(4,np.int64)
 obj={'visible':bool(len(xy)),'xy':(xy.mean(0)+.5)/np.array([W,H]) if len(xy) else None,
      'rect':rect.astype(float),'border':bool(len(xy) and ((xy[:,0]==0)|(xy[:,1]==0)|(xy[:,0]==W-1)|(xy[:,1]==H-1)).any()),
      'domain':domain,'instance_index':index}
 return selected,obj

def pixel_color_record(rgb_pixels,scene_color,nscene,index):
 n=len(rgb_pixels);known=n>=MINIMUM_COLOR_PIXELS;local=color_support(rgb_pixels)
 e={name:None for name in COLOR_RGB_PROTOTYPES}
 if known and nscene>=MINIMUM_COLOR_PIXELS:
  for name in e:
   p=(n*local[name]+COLOR_BETA_PSEUDOCOUNT)/(n+2*COLOR_BETA_PSEUDOCOUNT)
   b=(nscene*scene_color[name]+COLOR_BETA_PSEUDOCOUNT)/(nscene+2*COLOR_BETA_PSEUDOCOUNT)
   e[name]=float(np.clip(np.log(p/(1-p))-np.log(b/(1-b)),-COLOR_LOGODDS_CLIP,COLOR_LOGODDS_CLIP))
 return {'instance_id':index,'n_unique_visible_rgb_pixels':n,'color_status':'MEASURED_DIRECT_QUERY_VISIBLE_SUPPORT' if known else 'UNKNOWN_insufficient_visible_pixels',
         'color_soft_support':local if known else {k:None for k in COLOR_RGB_PROTOTYPES},
         'color_logodds_relative_scene_visible':e,'color_evidence_reliability':float(n/(n+MINIMUM_COLOR_PIXELS)) if known else 0.}

def source_query_distribution(boxes,z,usable):
 valid=(boxes[:,3:]>0).all(1)&np.isfinite(z)
 if not usable:valid[:]=False
 mass=np.zeros(256,float)
 if not valid.any():return [],mass,1.,0.
 mx=max(0.,float(z[valid].max()));mass[valid]=np.exp(np.clip(z[valid]-mx,-700,0));u=float(np.exp(-mx));den=u+mass.sum();mass/=den;u/=den
 keep=sorted(np.flatnonzero(valid),key=lambda q:(-float(mass[q]),int(q)))[:3]
 outside=float(max(0.,1-u-sum(mass[q] for q in keep)))
 return keep,mass,u,outside

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_NO_ASSOCIATION_RAW_SAFE_ABLATION_NO_GT' and auth['plan_sha256']==sha(args.plan)
 assert plan['state']=='ROOT_LOCKED_R51_NO_ASSOCIATION_RAW_SAFE_ABLATION' and plan['n'] in MANIFESTS
 assert plan['manifest']==MANIFESTS[plan['n']] and plan['appearance_factor_values']==FACTORS
 assert plan['retain_RGB_and_remove_instance_binding_and_duplicate_suppression'] is True
 expected={plan['manifest'],plan['prepared_root']+'/SEALED.json',plan['reference_root']+'/SEALED.json'}
 assert set(plan['inputs_sha256'])==expected
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 for f in ('crossplatform/round51/noassociation51.py','crossplatform/round51/attributes51.py','crossplatform/round51/evidence51_fidelity.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 prep=path(plan['prepared_root']);refroot=path(plan['reference_root']);out=path(plan['output_root'])
 assert all(p.parent==ROOT/'crossplatform/round51' for p in (prep,refroot,out)) and not out.exists()
 ps=read(prep/'SEALED.json');rs=read(refroot/'SEALED.json');assert ps['n']==plan['n']
 assert ps['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT' and ps['engineering_limit'] is None
 assert ps['external_vision_outputs_or_mask_derived_cache_read'] is False
 if plan['n']==336:assert rs['state']=='R51_REFERENCE_GRAPH_HEADSPAN_READOUT_REPAIR_NO_GT' and rs['n_subject_rows']==336
 else:assert rs['state']=='R51_FULL5912_SELECTED_METHOD_REFERENCE_ONLY_HEADSPAN_CAPTURE_NO_GT' and rs['n']==5912
 rows=read(ROOT/plan['manifest']);assert len(rows)==plan['n'] and len({r['id'] for r in rows})==plan['n']
 out.mkdir(parents=True)
 for d in ('samples_PRIVATE','cases_PRIVATE','references_PRIVATE'):(out/d).mkdir()
 ledger=out/'INPUT_LEDGER_PRIVATE.jsonl';bindings=[]
 def bind(p,role,h,sid=None):
  p=Path(p);actual=sha(p);record={'role':role,'path_PRIVATE':str(p),'sha256':actual,'id_PRIVATE':sid}
  with ledger.open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
  bindings.append(record);assert actual==h,role
  return p
 def member(root,seal,rel,role,sid):return bind(root/rel,role,seal['files'][rel],sid)
 start=time.perf_counter();counts={'n':0,'no_visible_subject_queries':0,'color_unknown_subject_queries':0,'references':0,'all_query_top3':0}
 try:
  for row in rows:
   sid=row['id'];cp=member(prep,ps,'cases_PRIVATE/'+sid+'.json','RAW_SAFE_BINDINGS',sid);case=read(cp)
   aa=read(member(prep,ps,'attributes_PRIVATE/'+sid+'.json','PURE_GRAPH_SCENE_COLOR',sid));graph=aa['text_graph_unchanged']
   a=npz(member(prep,ps,'arrays_PRIVATE/'+sid+'.npz','RAW_SAFE_VISIBLE_POINTS_SOURCE_ONLY',sid),
         ['source_boxes','source_scores','visible_raw_point_indices','visible_pixels_xy'])
   bind(row['imagepath'],'ORIGINAL_RGB',case['raw_bindings']['image_sha256'],sid)
   bind(row['pointpath'],'ORIGINAL_RAW_XYZ',case['raw_bindings']['point_sha256'],sid)
   with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
   pointpath=Path(row['pointpath']);raw=np.load(pointpath,allow_pickle=False) if pointpath.suffix=='.npy' else np.fromfile(pointpath,np.float32).reshape(-1,4)
   raw=np.ascontiguousarray(raw[:,:3],np.float32);visible=raw[a['visible_raw_point_indices']].copy()
   if row['platform']=='drone':visible[:,2]+=np.float32(1.8)
   else:assert row['platform']=='quad'
   W,H=case['image_size_wh'];assert rgb.shape[:2]==(H,W)
   xy=a['visible_pixels_xy'];scene=aa['scene_visible_color_soft_support'];ns=len(xy)
   source=a['source_scores'];boxes=a['source_boxes'];objects=[];color=[];support_counts=[]
   for q,box in enumerate(boxes):
    part,obj=query_observation(box,visible,xy,W,H,q,'subject_query_box_no_dedup')
    objects.append(obj);support_counts.append(len(part));px=xy[part]
    color.append(pixel_color_record(rgb[px[:,1],px[:,0]],scene,ns,q))
   pseudoattrs={'entity_attribute_requests':aa['entity_attribute_requests'],'instances':color}
   e,ek=attr_evidence(pseudoattrs,'T')
   rc=read(member(refroot,rs,'cases_PRIVATE/'+sid+'.json','SAME_MODEL_REFERENCE_INPUTS',sid))
   assert rc['graph_sha256']==case['pure_text_graph_sha256']
   refdata={};ref_details={}
   for ref in graph['references']:
    entity=ref['entity'];route=rc['references'][entity]
    rp=member(refroot,rs,route['file'],'SAME_MODEL_REFERENCE_BOXES_LOGODDS',sid);assert sha(rp)==route['sha256']
    z=npz(rp,['boxes','logodds']);keep,mass,u,other=source_query_distribution(z['boxes'],z['logodds'],route['mask']['reference_usable'])
    cache=[]
    for q in keep:
     part,obj=query_observation(z['boxes'][q],visible,xy,W,H,q,'reference_query_box_no_dedup');px=xy[part]
     col=pixel_color_record(rgb[px[:,1],px[:,0]],scene,ns,q)
     av,ak=attr_evidence({'entity_attribute_requests':aa['entity_attribute_requests'],'instances':[col]},entity)
     cache.append((q,obj,float(av[0]),bool(ak[0])))
    refdata[entity]=(cache,mass,u,other)
    ref_details[entity]={'source_top3_queries_PRIVATE':list(map(int,keep)),'source_probabilities':mass.tolist(),
        'source_unknown_probability':u,'source_truncated_probability':other,
        'duplicate_query_confidence_intentionally_not_removed':True,'hypotheses_by_factor':[]}
   Fs=[];PR=[];PK=[];C=[];QK=[];QA=[];QM=[];refs_by_factor=[]
   identity=np.eye(256,dtype=float)
   for factor in FACTORS:
    references={}
    for entity,(cache,mass,u,other) in refdata.items():
     adjusted=np.array([mass[q]*np.exp(np.clip(factor*av,-8,8)) if ak else mass[q] for q,obj,av,ak in cache])
     den=u+other+float(adjusted.sum());hs=[]
     for (q,obj,av,ak),p in zip(cache,adjusted/den):
      hs.append({'instance_domain':'reference_query_box_no_dedup','instance_index':int(q),
       'xy_norm2':obj['xy'].tolist() if obj['xy'] is not None else None,'rect':obj['rect'].tolist(),
       'width_norm':float((obj['rect'][2]-obj['rect'][0])/W),
       'probability':float(p),'source_probability':float(mass[q]),'uncertainty_mass_explicit':True})
     references[entity]=hs
     ref_details[entity]['hypotheses_by_factor'].append({'hypotheses':hs,'unknown_probability':float((u+other)/den)})
    pr,pk=raw_relations(graph,objects,references,(W,H))
    F,cues,info=propagate_measured(source,identity,factor*e,pr,ek,pk)
    Fs.append(F);PR.append(pr);PK.append(pk);C.append(cues);QK.append(info['query_known']);QA.append(info['anchors']);QM.append(info['query_known_weight_mass'])
    refs_by_factor.append(references)
   scores=np.stack(Fs);queries=scores.argmax(1).astype(np.int64)
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS),scores=scores,queries=queries,
       selected_original_boxes=boxes[queries],source_boxes=boxes,source_scores=source,appearance_factor_values=np.array(FACTORS),
       query_attribute_e=e,query_attribute_known=ek,query_visible_support_counts=np.array(support_counts),
       raw_PR=np.stack(PR),raw_PR_known=np.stack(PK),components=np.stack(C),
       cue_query_known=np.stack(QK),cue_anchors=np.stack(QA),cue_query_known_weight_mass=np.stack(QM))
   dump(out/'references_PRIVATE'/(sid+'.json'),ref_details)
   statuses={name:{'unknown':not bool(QK[i].any()),'fallback':not bool(QK[i].any()),'no_parameter_update':True} for i,name in enumerate(ARMS)}
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'platform':row['platform'],'arm_statuses':statuses,
       'source_boxes_unchanged':True,'RGB_retained':True,'instance_clustering_ownership_and_dedup_removed':True,
       'same_raw_safe_visibility_used':True,'no_query_identity_match_across_subject_and_reference':True,
       'external_vision_outputs_or_mask_derived_caches_read':False})
   counts['n']+=1;counts['no_visible_subject_queries']+=int(np.sum(np.array(support_counts)==0));counts['color_unknown_subject_queries']+=int((~ek).sum())
   counts['references']+=len(refdata);counts['all_query_top3']+=sum(len(v[0]) for v in refdata.values())
   if counts['n']%16==0:print('R51_NO_ASSOCIATION',counts['n'],plan['n'],flush=True)
  dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':bindings})
  dump(out/'COUNTS.json',counts)
  dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'new_grounder_forwards':0,'external_model_forwards':0,'optimizer_steps':0})
  dump(out/'SEALED.json',{'state':'R51_RAW_SAFE_NO_INSTANCE_ASSOCIATION_ABLATION_COMPLETE_NO_GT',
      'n':counts['n'],'plan_sha256':sha(args.plan),'appearance_factor_values':FACTORS,'source_boxes_unchanged':True,
      'RGB_retained_instance_binding_and_duplicate_suppression_removed':True,
      'external_vision_outputs_or_mask_derived_caches_read':False,'GT_read':False,
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':counts['n'],'automatic_retry':False});raise

if __name__=='__main__':main()
