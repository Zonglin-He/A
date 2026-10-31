"""Input-only real candidate-pair inventory. No VLM or task GT reads."""
import sys,time
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round13')]
from input_io import load_case
from scene_observation import build_scene_observation
from pair_geometry import prepare
def main():
 p=check_plan();inp=EXP.parent/'input';rows=read(inp/'A64_PRIVATE.json');cases={c['id']:c for c in read(inp/'A_CASES_PRIVATE.json')};idx=read(inp/'A_FRAME_INDEX_PRIVATE.json');out=ROOT/'crossplatform/round17/pairs64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();start=time.perf_counter();counts={'pairs':0,'role_calls':0,'pair_calls':0}
 for r in rows:
  c=cases[r['id']];frame,src,binding=load_case(r,c,idx);h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=binding['image_sha256']);prepared=prepare(scene,src['boxes_raw6'][0],r['id']);prepared.update(id_PRIVATE=r['id'],alias=c['case_alias'],platform=r['platform'],subset=c['subset'],bindings=binding)
  masks=sorted({m for pair in prepared['pairs'] for m in pair['mask_indices']});counts['pairs']+=len(prepared['pairs']);counts['role_calls']+=len(masks);counts['pair_calls']+=2*len(prepared['pairs']);dump(out/'cases_PRIVATE'/(r['id']+'.json'),prepared)
 dump(out/'COUNTS.json',counts);seal(out,'R17_real_pairs_prepared_before_models_labels',n=64,counts=counts,seconds=time.perf_counter()-start,programs=p['programs']);print('PAIRS_PREPARED',counts,flush=True)
if __name__=='__main__':main()
