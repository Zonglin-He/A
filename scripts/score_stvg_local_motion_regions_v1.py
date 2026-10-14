"""Postseal geometry-only comparison of the fixed local motion candidate recipe."""
import json,hashlib,sys,statistics,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'artifacts/stvg_local_motion_regions_v1';O=R/'artifacts/stvg_short_event_tracklets_v1';M=R/'artifacts/stvg_matched_support_reference_v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,z):
 with p.open('x')as f:json.dump(z,f,indent=2,allow_nan=False);f.write('\n')
def independent_iou(a,b):
 l=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));v=l*h
 return v/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v,1e-15)
def main():
 import numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 bar=read(B/'CPU_GENERATOR_BARRIER.json');assert bar['status']=='complete'and bar['window_count']==32
 rt=read(B/'RUNTIME.json');assert sha(B/'RUNTIME.json')==bar['runtime_sha256']
 for p,h in rt['pins'].items():assert sha(R/p)==h,p
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']and(R/f['path']).stat().st_size==f['bytes']
 roster=read(M/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_source_case(R/r['source_case_path'])for r in roster};truth,bindings=_truths_after_seal(cases)
 refs=read(M/'REFERENCE_LOCK.json');primary={(r['case_id'],a['slot'])for r in refs['records']for a in r['references']['source']['anchors']};assert len(primary)==19
 oldreport={(r['case_id'],r['slot']):r for r in read(O/'ROOT_POSTSEAL_REPORT.json')['window_pool_teacher']}
 windows=[];geometry=[];maxerr=0
 for cid,c in cases.items():
  new=read(B/'cases'/f'{cid}.json');old=read(O/'tracklets'/f'{cid}.json')
  for w in new['windows']:
   slot=w['window_slot'];fid=w['center_physical_frame_id'];tr=truth[cid]['truth'].get(fid);gt=None if tr is None else np.asarray(tr,float)/[c['width'],c['height'],c['width'],c['height']]
   oldcs=[x for x in old['tracklets']if x['frames'][2]['physical_frame_id']==fid];sourceold=next(x for x in oldcs if x['candidate_id']==1)['frames'][2]['box_xyxy_norm'];assert sourceold==w['candidates'][0]['frames'][2]['box_xyxy_norm']
   pools={}
   for typ,cs in [('old',oldcs),('new',w['candidates'])]:
    vv=[]
    for cc in cs:
     f=cc['frames'][2];box=f['box_xyxy_norm'];valid=f['valid'];assert f['physical_frame_id']==fid
     val=None
     if gt is not None and valid:
      val=float(box_iou(np.asarray(box)[None],gt[None])[0]);err=abs(val-independent_iou(box,gt));maxerr=max(maxerr,err);assert err<1e-10
     geometry.append(dict(case_id=cid,dataset=c['dataset'],slot=slot,pool=typ,candidate_id=cc['candidate_id'],Source=cc['candidate_id']==1,center_valid=valid,supported_frame_count=sum(f['valid']for f in cc['frames']),IoU=val,unknown_reason='no_center_GT'if gt is None else'no_valid_center_box'if not valid else None))
     if val is not None:vv.append((cc['candidate_id'],val))
    pools[typ]=vv
   old_common=oldreport[(cid,slot)]['common_observation_available'];known=gt is not None;assert known==old_common
   so=next((v for k,v in pools['old']if k==1),None);sn=next((v for k,v in pools['new']if k==1),None);assert so==sn
   oo=max((v for k,v in pools['old']),default=None);nn=max((v for k,v in pools['new']),default=None)
   if known:assert abs(oo-oldreport[(cid,slot)]['oracle_IoU'])<1e-10
   windows.append(dict(case_id=cid,dataset=c['dataset'],slot=slot,primary_support=(cid,slot)in primary,center_GT_available=known,old_candidate_count=len(oldcs),new_candidate_count=len(w['candidates']),new_nonSource_count=w['new_region_count'],source_IoU=so,old_oracle_IoU=oo,new_oracle_IoU=nn,old_delta_pool_pp=None if oo is None else 100*(oo-so),new_delta_pool_pp=None if nn is None else 100*(nn-sn),new_minus_old_oracle_pp=None if nn is None else 100*(nn-oo),new_nonSource_supported_frames=sum(sum(f['valid']for f in cc['frames'])for cc in w['candidates'][1:]),unknown_reason=None if known else'center_has_no_GT'))
 groups=[]
 for support in ['primary19','all_known23']:
  for domain in ['all','hc2','vidstg']:
   a=[w for w in windows if w['center_GT_available']and(support!='primary19'or w['primary_support'])and(domain=='all'or w['dataset']==domain)]
   groups.append(dict(support=support,dataset=domain,count=len(a),old_mean_delta_pool_pp=statistics.mean(w['old_delta_pool_pp']for w in a),new_mean_delta_pool_pp=statistics.mean(w['new_delta_pool_pp']for w in a),mean_new_minus_old_oracle_pp=statistics.mean(w['new_minus_old_oracle_pp']for w in a),old_positive_pool_count=sum(w['old_delta_pool_pp']>1e-8 for w in a),new_positive_pool_count=sum(w['new_delta_pool_pp']>1e-8 for w in a),new_better_count=sum(w['new_minus_old_oracle_pp']>1e-8 for w in a),new_worse_count=sum(w['new_minus_old_oracle_pp']< -1e-8 for w in a),equal_count=sum(abs(w['new_minus_old_oracle_pp'])<=1e-8 for w in a)))
 out=dict(status='pass',scope='postseal center geometry comparison; not semantic identity or full tube efficacy',GT_read=True,GT_used_for_generation=False,CPU_only=True,GPU_used=False,teacher_calls=0,actual_OPD_fits=0,barrier_sha256=sha(B/'CPU_GENERATOR_BARRIER.json'),runtime_sha256=sha(B/'RUNTIME.json'),matched_reference_lock_sha256=sha(M/'REFERENCE_LOCK.json'),GT_bindings_sha256=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest(),window_count=len(windows),unknown_window_count=sum(not w['center_GT_available']for w in windows),candidate_count=len(geometry),new_nonSource_count=sum(w['new_nonSource_count']for w in windows),max_independent_IoU_error=maxerr,groups=groups,windows=windows,candidate_geometry=geometry,time=time.time())
 save(B/'ROOT_POSTSEAL_GEOMETRY.json',out);print(json.dumps({k:out[k]for k in ['groups','new_nonSource_count','candidate_count','unknown_window_count','max_independent_IoU_error']}))
if __name__=='__main__':main()
