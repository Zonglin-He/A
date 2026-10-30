"""Root post-seal official/class-assisted and text-only frozen candidate scorer.

Consumes sealed GT-free all-class candidate scores, then loads GT in a separate
CPU-only process. No predictions or model state are modified.
"""
from pathlib import Path
import argparse,json,hashlib,sys,math,time,collections
import numpy as np
R=Path(__file__).resolve().parent;B=R.parents[2];P=B/'crossplatform';E=P/'round5/baseline_engineering'
sys.path.insert(0,str(B/'tools'));from batch_yaw_iou import many_iou

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rot(y):
 c,s=math.cos(y),math.sin(y);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def target(raw,platform,suffix):
 b=np.asarray(raw[:7],np.float64).copy()
 if platform=='drone':b[2]+=1.8
 if platform=='waymo':
  angle=math.radians([0,-45,45,-90,90][suffix]);b[:3]=rot(angle)@b[:3];b[6]+=angle
 # Match actual official dataset tensor before stable float64 geometry.
 return b.astype(np.float32).astype(np.float64)
def native_check(s,top):
 assert s.shape==(256,) and top.shape==(10,) and np.isfinite(s).all()
 assert len(set(top.tolist()))==10 and (top>=0).all() and (top<256).all()
 assert np.all(s[top[:-1]]>=s[top[1:]])
 mask=np.ones(256,bool);mask[top]=False;assert s[mask].max()<=s[top[-1]]
def metrics(values,scores,top):
 native_check(scores,top);a={'query':int(top[0]),'iou':float(values[top[0]]),'oracle_iou':float(values.max()),'oracle_query_GT_assisted':int(values.argmax()),'topk':{}}
 for k in (1,5,10,256):
  v=float(np.max(values[top[:k]])) if k<=10 else float(values.max());a['topk'][str(k)]={'max_iou_GT_assisted_if_k_gt_1':v,'acc25':bool(v>.25),'acc50':bool(v>.5)}
 a['rank_recoverable50']=bool(a['iou']<=.5 and a['oracle_iou']>.5);a['candidate_coverage_failure50']=a['oracle_iou']<=.5
 a['top_score_tied_count']=int(np.sum(scores==scores[top[0]]));return a

def aggregate(rows):
 summaries=[]
 for platform in ['waymo','drone','quad']:
  pp=[x for x in rows if x['platform']==platform]
  for population in ['official_eligible','all_raw']:
   rr=[x for x in pp if x['official_eligible']] if population=='official_eligible' else pp
   names=sorted(set(k for x in rr for k in x['arms']))
   for name in names:
    if name.startswith('official_') and population=='all_raw':continue
    selected=[(x,x['arms'][name]) for x in rr if name in x['arms']]
    if not selected:continue
    scenes=sorted({x['scene'] for x,a in selected});values=[a['iou'] for x,a in selected];sv=[np.mean([a['iou'] for x,a in selected if x['scene']==s]) for s in scenes]
    d={'platform':platform,'population':population,'arm':name,'n':len(selected),'scene_clusters':len(scenes),'mean_iou':float(np.mean(values)),'equal_scene_mean_iou':float(np.mean(sv)),'mean_candidate_oracle_iou_GT_assisted':float(np.mean([a['oracle_iou'] for x,a in selected])),'topk':{str(k):{f'acc{t}':sum(a['topk'][str(k)][f'acc{t}'] for x,a in selected) for t in (25,50)} for k in (1,5,10,256)},'rank_recoverable50':sum(a['rank_recoverable50'] for x,a in selected),'candidate_coverage_failure50':sum(a['candidate_coverage_failure50'] for x,a in selected)}
    summaries.append(d)
 return summaries

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();began=time.perf_counter()
 release=read(a.release);assert release['all_predictions_sealed_before_numeric_GT']
 sealpath=a.cache/'SEALED.json'
 if not sealpath.exists():sealpath=a.cache/'sealed.json'
 seal=read(sealpath);assert release['prediction_seal_sha256']==sha(sealpath)
 for name,h in seal['files'].items():assert sha(a.cache/name)==h,(name,'seal mismatch')
 assert release['manifest_sha256']==sha(a.manifest)
 prep=read(E/'PREPARATION_RECEIPT.json');assert prep['scoring_index_sha256']==sha(E/'SCORING_INDEX_PRIVATE.json')
 allidx={x['id']:x for x in read(E/'SCORING_INDEX_PRIVATE.json')};inputs=read(a.manifest);assert len({x['id'] for x in inputs})==len(inputs)
 a.out.mkdir(exist_ok=False);lock={'source_seal_sha256':sha(sealpath),'manifest_sha256':sha(a.manifest),'release_sha256':sha(a.release),'scorer_sha256':sha(Path(__file__)),'batch_iou_sha256':sha(R/'batch_yaw_iou.py'),'all_sealed_files_verified_before_numeric_GT':True,'numeric_dtype':'actual GT float32 official buffer then float64 polygon geometry','scope':'union checkpoint B1 deterministic full VAL baseline; label-assisted official span vs allraw pure text; no adaptation'};(a.out/'SCORER_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
 rows=[];invalid=collections.Counter();raw_ious={}
 for num,s in enumerate(inputs):
  ix=allidx[s['id']];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256'];obj=read(mp)['ground_info'][ix['object_index']];cls=obj['class'].lower();gt=target(obj['bbox_3d'],s['platform'],s['waymo_view_suffix']);row={'id':s['id'],'platform':s['platform'],'scene':s['scene'],'frame':s['frame'],'object_index':s['object_index'],'arms':{},'noun_fallback_identity':s['noun_fallback_identity']}
  base=None
  for condition in ['full','noun']:
   fp=a.cache/(s['id']+'_'+condition+'.npz')
   with np.load(fp,allow_pickle=False) as z:
    hnames=z['head_names'].tolist();classes=z['class_names'].tolist();c=classes.index(cls);boxes=z['boxes'].astype(float);raw=z['boxes_raw'];expected=raw.copy();expected[...,3:6]=np.maximum(expected[...,3:6],np.float32(1e-6));assert np.array_equal(expected,z['boxes'])
    for k in ['point_sample_sha256','raw_indices_sha256']:
     val=z[k].copy()
     if condition=='full':row[k]=val.tobytes().hex()
     else:assert row[k]==val.tobytes().hex()
    valid=bool(z['class_synonym_found'][c]);mapped=bool(z['class_positive_map_nonzero'][c])
    if condition=='full':row['official_eligible']=valid;row['official_map_nonzero']=mapped
    values={}
    for h in [hnames.index('last_'),hnames.index('proposal_')]:values[h]=many_iou(np.c_[boxes[h],np.zeros(256)],gt)
    for mode in ['bbf','bbs']:
     for head in ['last_','proposal_']:
      h=hnames.index(head)
      if condition=='full' and valid:
       name=f'official_full_{head}{mode}';row['arms'][name]=metrics(values[h],z[mode+'_scores_by_class'][h,c],z[mode+'_native_top10_by_class'][h,c])
      name=f'text_{condition}_{head}{mode}';row['arms'][name]=metrics(values[h],z['firstnoun_'+mode][h],z['firstnoun_'+mode+'_native_top10'][h])
    if condition=='full' and valid:
     legacy=[]
     for h in range(len(hnames)):
      q=int(z['bbf_native_top10_by_class'][h,c,0]);v=float(many_iou(np.r_[boxes[h,q],0][None],gt)[0]);legacy.append(v)
     row['official_legacy_7head_bbf_mean_iou']=float(np.mean(legacy));row['legacy_7head_top1_iou']=legacy
    if condition=='full':
     fullvalues=values[hnames.index('last_')];base=boxes[hnames.index('last_')].copy();order=z['bbf_native_top10_by_class'][hnames.index('last_'),c] if valid else z['firstnoun_bbf_native_top10'][hnames.index('last_')];q=int(order[0]);selected=np.r_[base[q],0];inter={}
     for field,sl in [('center',slice(0,3)),('size',slice(3,6)),('yaw',slice(6,7))]:
      b=selected.copy();b[sl]=gt[sl];inter[field]=float(many_iou(b[None],gt)[0])
     oracle_yaw=np.c_[base,np.full(256,gt[6])];yawvalues=many_iou(oracle_yaw,gt)
     row['postseal_geometry_diagnostic']={'reference_readout':'official_full_last_bbf' if valid else 'text_full_last_bbf','selected_center_error_m':float(np.linalg.norm(selected[:3]-gt[:3])),'selected_size_mean_abs_logratio':float(np.abs(np.log(selected[3:6]/gt[3:6])).mean()),'selected_single_GT_field_iou':inter,'all256_GTyaw_oracle_iou':float(yawvalues.max()),'not_additive_not_causal_not_unlabelled_method':True}
     # Keep every candidate target IoU private for independent audit / figures.
     raw_ious['full_last']=fullvalues;raw_ious['full_last_GTyaw']=yawvalues
    else:raw_ious['noun_last']=values[hnames.index('last_')]
    invalid[condition+'_raw_nonpositive_candidates']+=int(np.any(raw[...,3:6]<=0,axis=-1).sum());invalid[condition+'_official_clamped_candidates']+=int(np.any(raw[...,3:6]<1e-6,axis=-1).sum())
   # source cache closed before next condition
  np.savez_compressed(a.out/(s['id']+'_IOU_PRIVATE.npz'),**raw_ious);rows.append(row)
  if (num+1)%100==0 or num==0:print(json.dumps({'scored':num+1,'total':len(inputs),'seconds':time.perf_counter()-began}),flush=True)
 result={'n':len(rows),'prediction_seal_sha256':sha(sealpath),'scorer_lock_sha256':sha(a.out/'SCORER_LOCK.json'),'scope':lock['scope'],'eligibility':{p:{'raw':sum(x['platform']==p for x in rows),'eligible':sum(x['platform']==p and x['official_eligible'] for x in rows),'eligible_empty_map':sum(x['platform']==p and x['official_eligible'] and not x['official_map_nonzero'] for x in rows)} for p in ['waymo','drone','quad']},'invalid_counts':dict(invalid),'summaries':aggregate(rows),'rows':rows,'seconds':time.perf_counter()-began}
 (a.out/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.out/'SEALED.json').write_text(json.dumps({'files':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()},'n':len(rows)},indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','summaries')},indent=2))
if __name__=='__main__':main()
