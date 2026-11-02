"""R51 sole carrier-repair independent audit; NO grouped_G import or GT/NN.

Recomputes every medoid/onehot/visibility/score. Hash-min one case per platform
reopens original RGB/XYZ for actual medoid color and reference-support replay.
Fixed HSV/Lab primitives are AST-extracted from pinned attributes51, not the
repair producer; no data, captions or model responses are executable code.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','grouped_replace_all_c05','grouped_no_relation_c05']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,label):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label

def color_kernel():
 p=ROOT/'crossplatform/round51/attributes51.py';tree=ast.parse(p.read_text());names={'rgb_to_hsv','rgb_to_lab','color_support'};nodes=[];ns={'np':np}
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='COLOR_RGB_PROTOTYPES':ns['COLOR_RGB_PROTOTYPES']=ast.literal_eval(n.value)
  if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
 assert len(nodes)==3 and 'COLOR_RGB_PROTOTYPES' in ns
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
 return ns['color_support'],ns['COLOR_RGB_PROTOTYPES']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_GROUPED_G_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['n']==336 and plan['appearance_factor']==.5 and plan['no_GT_NN_or_new_method'] is True and plan['rgb_recheck_per_platform']==1
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():assert sha(path(rel))==h,('pin',rel)
 for rel in ('research/round51/audit51.py','crossplatform/round51/attributes51.py'):assert rel in plan['programs_sha256']
 helperpath=ROOT/'research/round51/audit51.py';spec=importlib.util.spec_from_file_location('R51_INDEPENDENT_NUMERIC_PRIMITIVES',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 roots={k:path(plan[k]) for k in ('grouped_root','prepared_root','proposal_group_root','reference_root','canonical_source_root')};seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 for k,v in roots.items():assert plan['inputs_sha256'][(v/'SEALED.json').relative_to(ROOT).as_posix()]==sha(v/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['grouped_root']/'SEALED.json') and seals['grouped_root']['state']=='R51_UNIQUE_GROUP_MEDOID_G_REPAIR_PREDICTIONS_NO_GT'
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 rows=read(path(plan['manifest']));assert plan['manifest']=='research/round24/input/ALL336_PRIVATE.json' and len(rows)==336
 key=lambda r:hashlib.sha256((plan['rgb_recheck_salt']+'|'+json.dumps([r[k] for k in ('imagepath','pointpath','metapath','caption','platform','scene')],ensure_ascii=False,separators=(',',':'))).encode()).hexdigest()
 selected=[min([r for r in rows if r['platform']==pl],key=key) for pl in ('drone','quad')];chosen={r['id'] for r in selected}
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 dump(out/'RGB_HASH2_SELECTED_BEFORE_MEASUREMENT_PRIVATE.json',[{'id_PRIVATE':r['id'],'platform':r['platform'],'selection_hash':key(r)} for r in selected])
 colorkernel,prototypes=color_kernel();done=[];color_rows=[];refchecks=0;start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];a=load(bound('prepared_root','arrays_PRIVATE/'+sid+'.npz'));pc=read(bound('prepared_root','cases_PRIVATE/'+sid+'.json'))
   ga=load(bound('proposal_group_root','arrays_PRIVATE/'+sid+'.npz'));gi=read(bound('proposal_group_root','groups_PRIVATE/'+sid+'.json'))
   p=load(bound('grouped_root','samples_PRIVATE/'+sid+'.npz'));carrier=load(bound('grouped_root','carriers_PRIVATE/'+sid+'.npz'));attrs=read(bound('grouped_root','attributes_PRIVATE/'+sid+'.json'))
   with np.load(bound('canonical_source_root','samples_PRIVATE/'+sid+'.npz'),allow_pickle=False) as z:boxes=z['boxes'].copy();source=z['scores'].copy()
   for z in (a,ga,p):exact(z['source_boxes'],boxes,'canonical boxes')
   exact(p['source_scores'],source,'canonical source p');assert p['arm_names'].tolist()==ARMS
   offsets=ga['support_offsets'];flat=ga['support_indices'];exact(offsets,a['query_support_point_offsets'],'whole bag offsets');exact(flat,a['query_support_point_indices'],'whole bag indices')
   bags=[set(map(int,flat[offsets[q]:offsets[q+1]])) for q in range(256)];means=np.full(256,np.nan);medoids=[];positions=[]
   for group in gi['groups']:
    members=sorted(group['members']);values=[]
    for q in members:
     jj=[len(bags[q]&bags[r])/(len(bags[q])+len(bags[r])-len(bags[q]&bags[r])) for r in members]
     mean=float(np.mean(np.asarray(jj,np.float64)));means[q]=mean;values.append(mean)
    best=max(values);medoids.append(min(q for q,v in zip(members,values) if v==best))
   exact(np.asarray(medoids,np.int64),carrier['group_medoid_queries'],'independent mean-point-J medoid');exact(means,carrier['member_mean_point_jaccard'],'all member J means')
   qid=np.asarray(gi['query_group_ids'],np.int64);assert np.array_equal(qid,ga['query_group_ids']);expected=np.zeros((256,len(medoids)));good=qid>=0;expected[np.flatnonzero(good),qid[good]]=1.
   exact(expected,carrier['query_instance_conditional_weights'],'onehot not source/label weighted');exact(expected,p['query_instance_conditional_weights'],'sample onehot')
   for m,q in enumerate(medoids):
    ids=np.array(sorted(bags[q]),np.int64);lo,hi=carrier['instance_point_offsets'][m:m+2];exact(ids,carrier['instance_point_indices'][lo:hi],'actual member bag not union')
    pos=np.flatnonzero(np.isin(a['visible_raw_point_indices'],ids,assume_unique=True));positions.append(pos);px=a['visible_pixels_xy'][pos]
    lo,hi=carrier['instance_crop_mask_pixel_offsets'][m:m+2];exact(px,carrier['instance_crop_mask_pixel_xy'][lo:hi],'actual medoid visible pixel intersection')
    lo,hi=carrier['instance_visible_point_offsets'][m:m+2];exact(a['visible_raw_point_indices'][pos],carrier['instance_visible_point_indices'][lo:hi],'actual medoid visible point IDs')
    rect=np.r_[px.min(0),px.max(0)+1] if len(px) else np.zeros(4,np.int64);exact(rect,carrier['instance_rect_xyxy'][m],'actual support rectangle')
    center=a['visible_uv'][pos].mean(0) if len(pos) else np.full(2,np.nan);exact(center,carrier['instance_visible_centroid_uv'][m],'observed centroid, not object center')
   e,ek=helper.attributes(attrs,'T');exact(e,p['instance_attribute_e'],'measured color evidence');assert np.array_equal(ek,p['instance_attribute_known'])
   refs=read(bound('grouped_root','references_PRIVATE/'+sid+'.json'))['references_by_factor'][0];rz=load(bound('grouped_root','references_PRIVATE/'+sid+'.npz'));W,H=pc['image_size_wh']
   raw=None
   if sid in chosen:
    assert sha(row['imagepath'])==pc['raw_bindings']['image_sha256'] and sha(row['pointpath'])==pc['raw_bindings']['point_sha256']
    with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
    xy=a['visible_pixels_xy'];scene=colorkernel(rgb[xy[:,1],xy[:,0]]);assert scene==attrs['scene_visible_color_soft_support']
    for m,pos in enumerate(positions):
     px=xy[pos];n=len(px);ns=len(xy);kernel=colorkernel(rgb[px[:,1],px[:,0]]);rec=attrs['instances'][m];assert rec['n_unique_visible_rgb_pixels']==n
     corrected={c:None for c in prototypes}
     if n>=8 and ns>=8:
      for c in corrected:
       local=(n*kernel[c]+.5)/(n+1);background=(ns*scene[c]+.5)/(ns+1);corrected[c]=float(np.clip(np.log(local/(1-local))-np.log(background/(1-background)),-4,4))
     assert rec['color_logodds_relative_scene_visible']==corrected and rec['color_evidence_reliability']==(float(n/(n+8)) if n>=8 else 0.)
     assert rec['color_soft_support']==(kernel if n>=8 else {c:None for c in prototypes})
    pt=Path(row['pointpath']);raw=np.load(pt,allow_pickle=False) if pt.suffix=='.npy' else np.fromfile(pt,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32)
    if row['platform']=='drone':raw[:,2]+=np.float32(1.8)
    color_rows.append({'platform':row['platform'],'groups':len(medoids),'actual_RGB_medoid_color_exact':True,'shared_candidate_pixels_retained_not_exclusive_owner':True})
   refcase=read(bound('reference_root','cases_PRIVATE/'+sid+'.json'))
   for entity,r in refs.items():
    z=rz[entity+'_source_logits'];finite=np.isfinite(z);mass=np.zeros(len(z));u=1.;other=0.;top=[]
    if finite.any():
     mx=max(0.,float(z[finite].max()));mass[finite]=np.exp(np.clip(z[finite]-mx,-700,0));un=np.exp(-mx);den=un+mass.sum();mass/=den;u=float(un/den);top=sorted(np.flatnonzero(finite),key=lambda j:(-float(mass[j]),int(j)))[:3];other=float(max(0.,1-u-sum(mass[j] for j in top)))
    exact(mass,rz[entity+'_source_probability'],'source all-group unknown0 distribution')
    dist=r['distribution'];assert dist['source_top3_instance_ids']==[int(j) for j in top] and dist['source_unknown_probability']==u and dist['source_truncated_probability']==other
    av,ak=helper.attributes(attrs,entity);adjusted=np.array([mass[j]*np.exp(np.clip(.5*av[j],-8,8)) if ak[j] else mass[j] for j in top]);den=u+other+float(adjusted.sum());adjusted/=den
    assert [h['instance_index'] for h in r['hypotheses']]==list(top) and [h['probability'] for h in r['hypotheses']]==[float(x) for x in adjusted]
    assert dist['retained_probability']==float(adjusted.sum()) and dist['unknown_probability']==float((u+other)/den)
    if raw is not None:
     route=refcase['references'][entity];rr=load(bound('reference_root',route['file']));wanted=np.full(len(medoids),-np.inf)
     if route['mask']['reference_usable']:
      for m,q in enumerate(medoids):
       ids=np.asarray(sorted(bags[q]),np.int64);points=raw[ids];inside=(np.abs(points[None]-rr['boxes'][:,None,:3])<=rr['boxes'][:,None,3:]/2).all(-1);cnt=inside.sum(1);ok=(cnt>=3)&(cnt/len(ids)>=.15)&(rr['boxes'][:,3:]>0).all(1)
       if ok.any():wanted[m]=rr['logodds'][ok].max()
     exact(wanted,z,'hash2 rawXYZ independent reference binding')
    refchecks+=1
   pr,pk=helper.spatial(carrier,attrs['text_graph_unchanged'],refs,W,H);exact(pr,p['raw_PR'][0],'all typed spatial relation');assert np.array_equal(pk,p['raw_PR_known'][0])
   F,C,K,A,M=helper.measured(source,expected,.5*e,pr,ek,pk);exact(F,p['scores'][1],'all grouped F');exact(C,p['components'][0],'all PVR');assert np.array_equal(K,p['cue_query_known'][0]);exact(A,p['cue_anchors'][0],'anchors');exact(M,p['cue_query_known_weight_mass'][0],'knownmass')
   pr[:,1]=0.;pk[:,1]=False;nr,NC,*_=helper.measured(source,expected,.5*e,pr,ek,pk);exact(nr,p['scores'][2],'noR F');exact(NC,p['no_relation_components'][0],'noR cues')
   assert p['scores'].shape==(3,256) and np.array_equal(p['queries'],p['scores'].argmax(1));exact(p['selected_original_boxes'],boxes[p['queries']],'selected originals')
   case=read(bound('grouped_root','cases_PRIVATE/'+sid+'.json'));assert case['external_vision_outputs_or_mask_derived_caches_read'] is False
   for name,availability in ((ARMS[1],K),(ARMS[2],np.vstack([K[:2],np.zeros_like(K[2:])]))):
    unavailable=not bool(availability.any());assert case['arm_statuses'][name]=={'unknown':unavailable,'fallback':unavailable,'no_parameter_update':True}
   done.append(sid)
  assert len(done)==336 and len(color_rows)==2
  dump(out/'HASH2_RGB_XYZ_RECHECK_SUMMARY.json',color_rows)
  result={'state':'PASS_R51_INDEPENDENT_GROUP_MEDOID_G_NUMERIC_AND_HASH2_RGB_XYZ_AUDIT_NO_GT','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(roots['grouped_root']/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'all_medoid_point_J_onehot_visiblepixel_and_full_noR_256_scores_verified':True,'reference_distributions_verified':refchecks,'raw_RGB_XYZ_hash_selected_rechecks':2,'fixed_HSV_Lab_primitive_reused_from_pinned_original_not_repair_producer':True,'calibration_physical_truth_not_established':True,'no_GT_NN_or_producer_import':True,'program_sha256':sha(__file__),'plan_sha256':sha(args.plan),'seconds':time.perf_counter()-start}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(json.dumps(result),flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry_GT_NN':True});raise
if __name__=='__main__':main()
