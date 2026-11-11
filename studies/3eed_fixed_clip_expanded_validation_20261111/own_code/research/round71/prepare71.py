"""Only fields affecting R70 projection-only G; no adapter/seed data needed."""
from common71 import *
import sys,time,argparse
from PIL import Image
sys.path.insert(0,str(ROOT/'research/round70'));from geometry70 import groups,rects,appearance,CameraSpec,read_camera_allowlist

def prepare(r,z):
 raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy();x=raw.copy()
 if r['platform']=='drone':x[:,2]+=np.float32(1.8)
 with Image.open(r['imagepath']) as im:wh=im.size
 spec=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');gs,_=groups(z['boxes'].astype(float),x.astype(float));rep=np.array([g[np.argmax(z['scores'][g])] for g in gs]);mapping=np.empty(256,np.int64)
 for k,g in enumerate(gs):mapping[g]=k
 rr,frac,ok=rects(z['boxes'][rep].astype(float),spec,wh);rproj=frac*ok;subject,status=appearance(r['caption']);return {'boxes':z['boxes'],'scores':z['scores'],'group_representatives':rep,'query_group':mapping,'rects':rr.astype(np.float32),'rect_valid':ok,'project_reliability':rproj},{'subject_PRIVATE':subject,'subject_parse_status':status,'group_count':len(gs),'projected_groups':int(ok.sum()),'missing_distortion_assumed_zero':spec.distortion is None}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();p=read(H/'experiment/DESIGN_LOCK.json');pin(p);out=H/'engineering_prepare4_attempt001' if args.engineering else ROOT/p['prepared_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();start=time.perf_counter();rr=rows();co=read(ROOT/p['cohort_map']);ss=read(ROOT/p['source_root']/'SEALED.json');bs=read(ROOT/p['baseline_root']/'SEALED.json');nproj=ngroups=0
 if args.engineering:rr=[min([r for r in rr if co[r['id']]==c and r['platform']==pl],key=lambda r:sha(ROOT/p['source_root']/'samples_PRIVATE'/(r['id']+'.npz'))) for c in ['VAL1280','TRAIN1024'] for pl in ['drone','quad']]
 for i,r in enumerate(rr):
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(ROOT/p['source_root']/rel)==ss['files'][rel];z=prefix(r);b=load(ROOT/p['baseline_root']/rel,['source_boxes','source_scores']);assert sha(ROOT/p['baseline_root']/rel)==bs['files'][rel];assert np.array_equal(z['boxes'],b['source_boxes']) and np.array_equal(z['scores'],b['source_scores']);a,rec=prepare(r,z);rec.update(image_sha256=sha(r['imagepath']),point_sha256=sha(r['pointpath']),source_capture_sha256=sha(ROOT/p['source_root']/rel),cohort=co[r['id']],source_candidate_consistency_verified=True);np.savez_compressed(out/rel,**a);dump(out/'cases_PRIVATE'/(r['id']+'.json'),rec);nproj+=rec['projected_groups'];ngroups+=rec['group_count']
  if(i+1)%64==0:print('PREP71',i+1,len(rr),nproj,flush=True)
 dump(out/'COUNTS.json',{'n':len(rr),'groups':ngroups,'projected_groups':nproj,'seconds':time.perf_counter()-start,'no_GT_or_new_source_inference':True});seal(out,'R71_FIXED_R70_PROJECT_ONLY_REGION_PREPARATION',n=len(rr));print(read(out/'COUNTS.json'))
