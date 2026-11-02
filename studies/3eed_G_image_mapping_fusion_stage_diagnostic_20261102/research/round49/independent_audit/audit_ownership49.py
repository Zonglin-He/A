"""Independent count/ownership audit on SEALED frame arrays; no new projection/GT."""
from pathlib import Path
import ast,json,hashlib,time,collections
import ijson
import numpy as np
R=Path(__file__).resolve().parents[3];H=Path(__file__).resolve().parent;P=R/'crossplatform/round49/ownership336_attempt001';EXP=R/'research/round49/experiment'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def identity_rows():
 wanted={'id','platform','stage','scene','pointpath','metapath','imagepath'};rows=[];row=None
 with (R/'research/round24/input/ALL336_PRIVATE.json').open('rb') as f:
  for prefix,event,value in ijson.parse(f):
   if (prefix,event)==('item','start_map'):row={}
   elif (prefix,event)==('item','end_map'):assert set(row)==wanted;rows.append(row);row=None
   elif row is not None and prefix.startswith('item.') and prefix[5:] in wanted and event=='string':row[prefix[5:]]=value
 assert len(rows)==336 and len({r['id'] for r in rows})==336;return rows

def count_reference(points,owner,boxes,M):
 assert points.dtype==boxes.dtype==np.float32 and points.shape[1:]==(3,) and owner.shape==(len(points),);assert (owner>=-2).all() and (owner[owner>=0]<M).all();n=len(boxes);counts=np.zeros((n,M),np.int64);u=np.zeros(n,np.int64);a=u.copy();total=u.copy();invalid=(boxes[:,3:]<=0).any(1)
 for lo in range(0,n,16):
  hi=min(n,lo+16);inside=(np.abs(points[:,None,:]-boxes[None,lo:hi,:3])<=boxes[None,lo:hi,3:]/2).all(axis=2);inside[:,invalid[lo:hi]]=False;total[lo:hi]=inside.sum(axis=0)
  for code in range(-2,M):
   qcount=inside[owner==code].sum(axis=0)
   if code==-2:a[lo:hi]=qcount
   elif code==-1:u[lo:hi]=qcount
   else:counts[lo:hi,code]=qcount
 assert np.array_equal(counts.sum(1)+u+a,total);return counts,u,a,total,invalid,np.where(invalid,'UNKNOWN_nonpositive_source_size','DEFINED_yaw0_visible_support')

def main():
 start=time.perf_counter();lock=read(H/'OWNERSHIP_INDEPENDENT_CODE_LOCK.json');assert lock['program_sha256']==sha(__file__) and lock['ownership_seal_sha256']==sha(P/'SEALED.json');plan=read(EXP/'OWNERSHIP_PLAN_LOCK.json');s=read(P/'SEALED.json');assert s['plan_sha256']==sha(EXP/'OWNERSHIP_PLAN_LOCK.json') and s['n']==336
 for n,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(R/n)==h,n
 for n,h in s['files'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts and sha(P/n)==h,n
 # Bind bytes of all original inputs, but never decode raw pointcloud or RGB.
 binds=read(P/'BINDINGS_PRIVATE.json')
 for n,h in binds.items():assert sha(Path(n) if Path(n).is_absolute() else R/n)==h,n
 A={x['id']:x for x in read(R/'research/round17/input/A_CASES_PRIVATE.json')};AF=read(R/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(R/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');oldbind=read(R/'research/round49/inventory_attempt001/SOURCE_BINDINGS_PRIVATE.json');rows=identity_rows();cache={};flags=[];totalcount=0;unknown=0
 for i,r in enumerate(rows):
  sid=r['id'];case=read(P/'cases_PRIVATE'/(sid+'.json'));sample=load(P/'samples_PRIVATE'/(sid+'.npz'));key=case['frame_key_PRIVATE'];fp=P/'frames_PRIVATE'/(key+'.npz')
  if r['stage']=='A':
   ac=A[sid];fr=AF[ac['frame_alias']];mp=R/ac['mask_dir']/(ac['frame_alias']+'.npz')
  else:
   found=[(k,f) for k,f in RF.items() if tuple(f[k2] for k2 in ['imagepath','pointpath','metapath'])==tuple(r[k2] for k2 in ['imagepath','pointpath','metapath'])];assert len(found)==1;alias,fr=found[0];mp=R/'crossplatform/round12/masks501_attempt001'/(alias+'.npz')
  assert all(fr[k]==r[k] for k in ['platform','imagepath','pointpath','metapath']);assert sha(mp)==case['frame_mask_cache_sha256'];assert fr['point_sha256']==case['raw_XYZ_sha256']==binds[r['pointpath']];camera={k:fr[k] for k in ['image_intrinsic','image_extrinsic']}
  if fr['camera_distortion'] is not None:camera['camera_distortion']=fr['camera_distortion']
  data={'platform':r['platform'],'point_sha256':case['raw_XYZ_sha256'],'mask_sha256':case['frame_mask_cache_sha256'],'camera_fields':camera,'image_sha256':fr['image_sha256']};assert hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()==key
  if key not in cache:
   z=load(fp);ins=load(mp);mask=ins['masks_bool'];height,width=map(int,ins['original_hw']);points=z['visible_points_model'];owner=z['point_owner'];uv=z['visible_uv'];assert points.dtype==np.float32 and points.shape==(len(owner),3) and uv.shape==(len(owner),2);assert np.isfinite(points).all() and np.isfinite(uv).all() and np.isfinite(z['depth_camera']).all() and (z['depth_camera']>.1).all();assert ((uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)).all();assert len(np.unique(points,axis=0))==len(points) and len(np.unique(z['physical_raw_ids']))==len(points);assert np.array_equal(z['image_size_wh'],[width,height]);pixel=np.floor(uv).astype(np.int64);membership=mask[:,pixel[:,1],pixel[:,0]];num=membership.sum(axis=0);expected=np.full(len(owner),-1,np.int64);expected[num>1]=-2
   if len(mask):one=num==1;expected[one]=membership[:,one].argmax(axis=0)
   assert np.array_equal(expected,owner);mh=np.array([hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in mask]);assert np.array_equal(mh,z['mask_sha256']) and np.array_equal(z['native_instance_id'],ins['instance_indices'].astype(np.int64));cache[key]=z
  z=cache[key];sourcepath=R/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(sid+'.npz');assert sha(sourcepath)==case['source_cache_sha256']==oldbind[sourcepath.relative_to(R).as_posix()]
  with np.load(sourcepath,allow_pickle=False) as src:boxes=src['boxes'].copy()
  assert np.array_equal(boxes,sample['source_boxes']) and boxes.dtype==sample['source_boxes'].dtype==np.float32;assert np.array_equal(sample['mask_sha256'],z['mask_sha256']) and np.array_equal(sample['native_instance_id'],z['native_instance_id']);assert case['visible_point_n']==len(z['point_owner']) and case['independent_mask_n']==len(z['mask_sha256']);expected=count_reference(z['visible_points_model'],z['point_owner'],boxes,len(z['mask_sha256']))
  for a,k in zip(expected,['mask_counts','unassigned','ambiguous','allvisible','nonpositive_source_size','geometry_status']):assert np.array_equal(a,sample[k]) and a.dtype==sample[k].dtype,(sid,k)
  unknown+=int(expected[4].sum());totalcount+=expected[0].size;flags.append({'row':'O'+str(i+1).zfill(4),'platform':r['platform'],'source_query_count':256,'independent_mask_count':len(z['mask_sha256']),'visible_points':len(z['point_owner']),'query_mask_count_cells_exact':expected[0].size,'UNKNOWN_nonpositive_size_queries':int(expected[4].sum()),'unassigned_inside_query_point_sum':int(expected[1].sum()),'ambiguous_inside_query_point_sum':int(expected[2].sum()),'total_inside_query_point_sum':int(expected[3].sum()),'owner_from_saved_UV_and_original_masks_exact':True,'all256_partition_counts_exact':True,'not_distinct_point_or_object_count_due_query_overlap':True,'no_calibration_or_physical_identity_certification':True})
 # Pure software control exercises the nonpositive-size UNKNOWN branch even if
 # actual frozen boxes are all positive. No input/annotation/model is involved.
 tr=ast.parse((R/'crossplatform/round49/ownership49.py').read_text());node=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='ownership_counts');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-count-helper-synthetic>','exec'),ns);pts=np.array([[0,0,0],[.2,0,0],[-.2,0,0],[2,0,0]],np.float32);own=np.array([-2,-1,0,1],np.int64);bb=np.tile(np.array([0,0,0,2,2,2],np.float32),(256,1));bb[1,3]=0;bb[2]=[2,0,0,1,1,1];ref=count_reference(pts,own,bb,2);actual=ns['ownership_counts'](pts,own,bb,2)
 for a,k in zip(ref,['mask_counts','unassigned','ambiguous','allvisible','nonpositive_source_size','geometry_status']):assert np.array_equal(a,actual[k])
 assert actual['geometry_status'][1]=='UNKNOWN_nonpositive_source_size' and actual['allvisible'][1]==0 and actual['unassigned'][0]==actual['ambiguous'][0]==actual['mask_counts'][0,0]==1
 dump(H/'ALL336_ANONYMOUS_OWNERSHIP_AUDIT_SCALARS.json',flags);result={'state':'PASS_R49_INDEPENDENT_SEALED_POINT_OWNERSHIP_COUNT_AUDIT_NO_GT','n':336,'frames_audited':len(cache),'source_queries':336*256,'query_mask_count_cells_exact':totalcount,'all_source_boxes_maskhash_frame_camera_raw_input_bytebindings_verified':True,'raw_input_bytes_only_hashed_not_decoded_or_projected':True,'saved_UV_to_mask_ownership_independently_recomputed':True,'source_box_counts_recomputed_by_blocked_numpy':True,'all_unassigned_ambiguous_foreground_partitions_exact':True,'nonpositive_actual_source_queries_UNKNOWN':unknown,'synthetic_nonpositive_unknown_and_threeway_partition_PASS':True,'no_RGB_GT_blind_annotations_method_scores_model_or_FINAL_access':True,'not_independent_calibration_visibility_or_worldID_truth':True,'ownership_seal_sha256':sha(P/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(H/'OWNERSHIP_AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
