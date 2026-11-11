"""Fresh source pre-fusion prefixes with official Waymo camera-view convention."""
from common76 import *
import torch,time,argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
from interface66 import capture,replay
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rr=source_rows('TRAIN512')+source_rows('VAL128');rr=[next(r for r in rr if r['suffix']==s) for s in range(5)] if a.engineering else rr;out=H/'engineering_capture_attempt001' if a.engineering else BASE/'source_prefix640_attempt001';assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);from utils.transform_waymo import transform_to_front_view;start=time.perf_counter();calls=0
 try:
  for i,r in enumerate(rr):
   pts,ix,_=sample_points(r);pts,_=transform_to_front_view(pts,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pts=np.ascontiguousarray(pts,np.float32);phrase=' '.join(r['caption'].replace(',', ' ,').split());z,_=capture(m,pts,ix,phrase);calls+=1;old=load(ROOT/'crossplatform/round25/source_features_attempt002/samples_PRIVATE'/(r['id']+'.npz'),['h','query','tokens','boxes'])
   for k in old:assert np.array_equal(z[k],old[k]),('original_source_feature_bridge',k)
   if a.engineering:
    y=replay(m,z)
    for k in ['h','query','tokens']:assert np.array_equal(y[k],z[k]),('unfused_suffix_replay',k)
   np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**z)
   if(i+1)%64==0:print('CAPTURE76',i+1,len(rr),flush=True)
  assert fingerprint(m)==fp;cost={'n':len(rr),'new_full_source_forwards':calls,'source_unchanged':True,'all_source_boxes_queries_text_and_hidden_bitexact_to_corrected_R25':True,'all5_Waymo_views_in_engineering':a.engineering,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COUNTS.json',cost);seal(out,'PASS_R76_SOURCE_PREFIXES_OFFICIAL_COORDINATES_ORIGINAL_SOURCE_BRIDGE',n=len(rr));print(cost)
 except BaseException as e:dump(out/'FAILED.json',{'type':type(e).__name__,'error':str(e),'completed_full_forwards':calls});raise
if __name__=='__main__':main()
