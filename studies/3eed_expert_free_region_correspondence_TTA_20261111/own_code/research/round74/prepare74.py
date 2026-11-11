from common74 import *
from region74 import select_regions,negative_masks,derangement
import time
def main():
 p=read(H/'experiment/PREPARE_PLAN_LOCK.json');pin(p);out=ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();start=time.perf_counter();tot=rgbactive=geoactive=0
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=ab(r);raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy()
  if r['platform']=='drone':raw[:,2]+=np.float32(1.8)
  for v in 'AB':assert z[v]['point_indices'].min()>=0 and z[v]['point_indices'].max()<len(raw)
  # Reuse only the geometry grouping field, never detector/VLM feature caches.
  group=load(ROOT/'crossplatform/round71/prepared2304_attempt001'/rel,['query_group'])['query_group'];boxes,reps,PA,PB,common=select_regions(z,raw.astype(float),group);graph=load(ROOT/'crossplatform/round72/evidence2640_attempt001/graphs_PRIVATE'/(sid+'.npz'));rgb,geom,visible=negative_masks(raw.astype(float),boxes,graph);old=load(ROOT/'crossplatform/round69/evidence2304_attempt001'/rel,['B_W','B_fixed_mass','base']);np.savez_compressed(out/rel,region_boxes=boxes,region_representatives=reps,A_pool=PA,B_pool=PB,shared_AB_raw_counts=common,RGB_negatives=rgb,geometry_negatives=geom,visible_point_counts=visible,shuffle=derangement(len(boxes),'R74_Bkeys|'+sid),**old);rec={'id_PRIVATE':sid,'regions':len(boxes),'RGB_directed_negative_pairs':int(rgb.sum()),'noRGB_directed_negative_pairs':int(geom.sum()),'RGB_active':bool(rgb.any()),'noRGB_active':bool(geom.any()),'raw_point_sha256':sha(r['pointpath']),'graph_sha256':sha(ROOT/'crossplatform/round72/evidence2640_attempt001/graphs_PRIVATE'/(sid+'.npz'))};dump(out/'cases_PRIVATE'/(sid+'.json'),rec);tot+=len(boxes);rgbactive+=int(rgb.any());geoactive+=int(geom.any())
 dump(out/'COUNTS.json',{'n':64,'regions':tot,'RGB_active_inputs':rgbactive,'noRGB_active_inputs':geoactive,'seconds':time.perf_counter()-start,'C_view_read':False,'task_labels_or_expert_features_read':False});seal(out,'R74_FIXED_AB_REGION_CORRESPONDENCE_NO_C_NO_GT_NO_EXPERT',n=64);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
