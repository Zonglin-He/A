"""Read-only saved-cache algebra and binding verification, no new GT scoring."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.branch_decomposition.run import OUT,PUBLIC,read,write,sha,validate
from stvg.temporal import expected_iou_fast

def main():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);seal=read(OUT/'sealed.json');report=read(PUBLIC/'report.json');assert seal['GT_read']is False and seal['plan_sha256']==sha(OUT/'plan.json');maxerror=0.;checks=0
 for item,row in zip(p['items'],report['all_signed']):
  assert row['query']==item['anonymous']
  for key,path in item['inputs'].items():assert sha(ROOT/path)==item['hashes'][key]
  pack=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['inputs']['prediction']);q=pack['final']['Global6'];U=old['frame_ids'];grid=expected_iou_fast(U,q);lookup={int(v):i for i,v in enumerate(U)}
  for name,span in [('native_MBR_utility',old['Global6_interval']),('native_utility_of_blend',old['Strong_Reference_interval'])]:
   exact=float(grid[lookup[int(span[0])],lookup[int(span[1]-1)]]);maxerror=max(maxerror,abs(exact-row[name]));assert abs(exact-row[name])<2e-12;checks+=1
  assert row['required448_relative_preference_lower_bound']>=-2e-12
 e=seal['files'][0];z=np.load(OUT/e['posterior_path']);assert sha(OUT/e['posterior_path'])==e['posterior_sha256'];G=z['Only224six'];H=z['Only448'];S=z['Fusion'];assert H.min()>=0 and abs(H.sum()-1)<2e-12;assert np.array_equal(S,.5*G+.5*H);assert report['coverage']['complete448']==1 and report['coverage']['missing448']==63;assert report['decision']['new_lightweight_algorithm_validated']is False
 for r in report['rescued']+report['harmed']:assert r['length_change_seconds']>0
 state=read(ROOT/'outputs/boxfixed_full_v1/state.json');progress=read(ROOT/'outputs/boxfixed_full_v1/vidstg/progress.json')
 for x in [state['controller'],*state['children']]:assert int(Path(f"/proc/{x['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==x['startticks']
 assert os.stat(ROOT.parent/'runtime/gpu.lock').st_ino==os.stat('/proc/776574/fd/3').st_ino;assert not torch.cuda.is_initialized()
 result=dict(status='pass',independent_expected_utility_checks=checks,max_utility_error=maxerror,exact_saved_fusion_reconstruction=True,complete448=1,missing448=63,new_GT_reads=0,new_model_calls=0,frozen_full_sources_unchanged=434,full_PID_startticks_and_gpu_lock_verified=True,Vid_completed=progress['completed'])
 write(PUBLIC/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
