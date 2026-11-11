"""Prepare paired real RGB/point observations, never labels or vision caches."""
from common70 import *
from geometry70 import prepare
import time
p=read(H/'experiment/DESIGN_LOCK.json');pin(p);out=ROOT/p['prepared_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();s66=read(ROOT/p['prefix_root']/'SEALED.json');s53=read(ROOT/p['latent_root']/'SEALED.json');start=time.perf_counter();stats=[]
for i,r in enumerate(rows()):
 rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(ROOT/p['prefix_root']/rel)==s66['files'][rel] and sha(ROOT/p['latent_root']/rel)==s53['files'][rel];z=prefix(r);lat=load(ROOT/p['latent_root']/rel,['fp2_xyz','fp2_features','seed_raw_indices','boxes','scores']);assert np.array_equal(z['xyz'][0],lat['fp2_xyz']) and np.array_equal(z['vis0'][0],lat['fp2_features'].T) and np.array_equal(z['boxes'],lat['boxes']) and np.array_equal(z['scores'],lat['scores']);raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3];mapped=raw[lat['seed_raw_indices']].copy()
 if r['platform']=='drone':mapped[:,2]+=np.float32(1.8)
 assert np.array_equal(mapped,lat['fp2_xyz']);a,rec=prepare(r,z);rec.update(image_sha256=sha(r['imagepath']),point_sha256=sha(r['pointpath']),source_prefix_sha256=sha(ROOT/p['prefix_root']/rel),source_latent_sha256=sha(ROOT/p['latent_root']/rel),composed_seed_point_pixel_provenance_verified=True);np.savez_compressed(out/rel,**a);dump(out/'cases_PRIVATE'/(r['id']+'.json'),rec);stats.append({k:rec[k] for k in ['group_count','supported_visual_groups','visible_seeds','fit_seeds','held_seeds','B_eligible']})
 if(i+1)%32==0:print('PREP70',i+1,336,flush=True)
dump(out/'COUNTS.json',{'n':336,'point_seed_coordinate_checks':336,'prefix_latent_box_score_checks':336,'groups':sum(x['group_count'] for x in stats),'supported_visual_groups':sum(x['supported_visual_groups'] for x in stats),'B_eligible':sum(x['B_eligible'] for x in stats),'seconds':time.perf_counter()-start});seal(out,'R70_RAW_RGB_CALIBRATION_SOURCE_POINT_FEATURE_PAIRS_NO_GT',n=336);print(read(out/'COUNTS.json'))
