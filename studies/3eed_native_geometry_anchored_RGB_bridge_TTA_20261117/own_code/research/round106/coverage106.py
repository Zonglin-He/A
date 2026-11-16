"""Sensor-only incremental LiDAR support; no model forward, caption or task GT."""
from pathlib import Path
import sys,json,hashlib,datetime,collections,numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';sys.path.insert(0,str(R/'research/round105'));import geometry105 as geo
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,z):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def stats(x):
 a=np.array(x,float);return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None}
def voxel(x):return {tuple(v) for v in np.floor(np.asarray(x)/.1).astype(np.int64)}
def boxcounts(x,boxes):
 tree=cKDTree(x);out=[]
 for b in boxes:
  rad=np.linalg.norm(np.maximum(b[3:6],1e-6))/2;ix=tree.query_ball_point(b[:3],rad);pts=x[ix];out.append(int((np.abs(pts-b[:3])<=np.maximum(b[3:6],1e-6)/2).all(1).sum()))
 return np.array(out)
def main():
 pairs=rd(R/'research/round105/FULL_PAIRS_PRIVATE.json');frames={};rows={};lock={'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'sensor_incremental_geometry_prerequisite','fixed_pairs_sha256':sha(R/'research/round105/FULL_PAIRS_PRIVATE.json'),'registration_seal_sha256':sha(R/'crossplatform/round105/registration_full_PRIVATE/SEALED.json'),'sample_budget':16384,'voxel_m':.1,'new_surface_shell_m':[.1,.25],'equal_budget_control':'independent random sample of SAME original A full cloud','original_A256_boxes':'existing frozen original source caches only for region coverage; never target GT','no_new_model_or_GT':True,'no_external_experts':True,'protected_media_access':False};assert not (H/'experiment/COVERAGE_LOCK.json').exists();dump(H/'experiment/COVERAGE_LOCK.json',lock)
 for pl in ['waymo','drone','quad']:
  rr=rd(R/f'research/round99/input/TARGET_{pl.upper()}128_PRIVATE.json')
  for r in rr:rows.setdefault(str(Path(r['metapath']).resolve()),[]).append(r)
 @__import__('functools').lru_cache(maxsize=6)
 def cloud(framekey):
  f=frames[framekey];assert f['eligible_for_extra_media'];p=geo.points(f);p=p[np.isfinite(p).all(1)];return np.asarray(p,np.float64)
 records=[]
 for i,pair in enumerate(pairs):
  reg=rd(R/'crossplatform/round105/registration_full_PRIVATE'/(pair['pair_PRIVATE']+'.json'))
  if not reg['geometry_cycle_supported']:continue
  A,C=pair['A'],pair['B'];frames[A['frame_PRIVATE']]=A;frames[C['frame_PRIVATE']]=C;pa=cloud(A['frame_PRIVATE']);pc=cloud(C['frame_PRIVATE']);assert sha(A['pointpath'])==reg['A_sensor']['point_sha256'] and sha(C['pointpath'])==reg['B_sensor']['point_sha256'];T=np.array(reg['forward']['T_raw_B_to_A']);pb=geo.transform_points(pc,T);nearest=cKDTree(pa).query(pb)[0];seed=int(hashlib.sha256(('R106coverage|'+pair['pair_PRIVATE']).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);s1=pa[rng.choice(len(pa),16384,replace=len(pa)<16384)];s2=pa[rng.choice(len(pa),16384,replace=len(pa)<16384)];sb=pb[rng.choice(len(pb),16384,replace=len(pb)<16384)];base=voxel(s1);vb=voxel(sb);v2=voxel(s2);static=pb[nearest<=.25];rec={'pair_PRIVATE':pair['pair_PRIVATE'],'platform':A['platform'],'rawA_points':len(pa),'rawB_points':len(pb),'fraction_B_within_25cm_of_A':float((nearest<=.25).mean()),'fraction_B_new_10to25cm_surface_shell':float(((nearest>.1)&(nearest<=.25)).mean()),'B_new_voxel_fraction_over_sampleA':len(vb-base)/max(len(vb),1),'sameA_random_new_voxel_fraction':len(v2-base)/max(len(v2),1),'sampleA_voxels':len(base),'sampleB_voxels':len(vb),'pointcount_equal':True,'candidate_sets_PRIVATE':[]}
  for r in rows[A['metapath']]:
   for src in ['waymo','drone','quad']:
    if src==A['platform']:continue
    f=np.load(R/'crossplatform/round100'/(src+'_to_'+A['platform'])/'samples_PRIVATE'/(r['id']+'.npz'));boxes=f['source_boxes'].astype(float);cam=geo.camera(A);M=geo.raw_to_model_transform(cam);aa=geo.transform_points(s1,M);bb=geo.transform_points(sb,M);cc=geo.transform_points(s2,M);ca=boxcounts(aa,boxes);cb=boxcounts(bb,boxes);ccn=boxcounts(cc,boxes);sup=boxcounts(geo.transform_points(static,M),boxes) if len(static) else np.zeros(256,int);rec['candidate_sets_PRIVATE'].append({'source':src,'row_PRIVATE':r['id'],'A_nonempty_candidates':int((ca>0).sum()),'B_nonempty_candidates':int((cb>0).sum()),'sameA_second_nonempty_candidates':int((ccn>0).sum()),'B_support_where_A_sample_empty':int(((ca==0)&(cb>0)).sum()),'sameA_support_where_A_sample_empty':int(((ca==0)&(ccn>0)).sum()),'B_static_raw_support_candidates':int((sup>0).sum()),'fraction_candidates_B_increased_count':float((cb>ca).mean()),'fraction_candidates_sameA_increased_count':float((ccn>ca).mean())})
  records.append(rec)
  if len(records)%32==0:print('COVERAGE106',len(records),flush=True)
 out=[]
 for pl in ['waymo','drone','quad']:
  rr=[x for x in records if x['platform']==pl];cs=[c for x in rr for c in x['candidate_sets_PRIVATE']];out.append({'platform':pl,'pairs':len(rr),'candidate_sets':len(cs),'fraction_B_within_25cm_of_A':stats([x['fraction_B_within_25cm_of_A'] for x in rr]),'fraction_B_new_10to25cm_surface_shell':stats([x['fraction_B_new_10to25cm_surface_shell'] for x in rr]),'B_new_voxel_fraction_over_sampleA':stats([x['B_new_voxel_fraction_over_sampleA'] for x in rr]),'sameA_random_new_voxel_fraction':stats([x['sameA_random_new_voxel_fraction'] for x in rr]),**{k:stats([x[k] for x in cs]) for k in ['B_support_where_A_sample_empty','sameA_support_where_A_sample_empty','fraction_candidates_B_increased_count','fraction_candidates_sameA_increased_count']}})
 dump(B/'coverage_PRIVATE/ROWS_PRIVATE.json',records);dump(H/'COVERAGE_SUMMARY.json',{'state':'SENSOR_INCREMENTAL_SUPPORT_MEASURED_NOT_TASK_EFFICACY','groups':out,'no_new_model_or_GT':True,'FINAL_access':False,'all_geometrically_supported_pairs_retained':len(records)==243});print('COVERAGE106_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
