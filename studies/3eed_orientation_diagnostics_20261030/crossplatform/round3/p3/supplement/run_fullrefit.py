"""Posthoc full-box capacity supplement, no GT/model/GPU reads or old writes."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent;BASE=R.parent;CROSS=BASE.parents[1];OLD=CROSS/'round2'
sys.path.insert(0,str(BASE))
from geometry_baselines import DEFAULT,source7,crop_source_aabb,full_refit_from_crop

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,list):return [plain(x) for x in v]
 return v

def main():
 path=R/'PLAN_LOCK.json';plan=json.loads(path.read_text());assert sha(path)==path.with_suffix('.sha256').read_text().split()[0]
 assert sha(BASE/'RUN_LOCK.json')==plan['original_run_lock_sha256'];original=json.loads((BASE/'RUN_LOCK.json').read_text())
 for p,h in {**original['source_hashes'],**plan['source_hashes']}.items():assert sha(p)==h,p
 base_seal=json.loads((BASE/'predictions/SEALED.json').read_text());assert sha(BASE/'predictions/SEALED.json')==plan['original_prediction_seal_sha256']
 traces=json.loads((BASE/'predictions/trace.json').read_text());assert sha(BASE/'predictions/trace.json')==base_seal['trace_sha256'];base_trace={r['id']:r for r in traces}
 out=R/'predictions';out.mkdir(exist_ok=False);records=[];count=0
 for cohort in ['dev','confirm']:
  mp=OLD/'inputs'/f'{cohort}_manifest.json';assert sha(mp)==original['input_manifests'][cohort];samples=json.loads(mp.read_text());source=OLD/f'{cohort}_source';assert sha(source/'sealed.json')==original['source_seals'][cohort];source_seal=json.loads((source/'sealed.json').read_text())
  for sample in samples:
   tic=time.perf_counter();sid=sample['id'];name=sid+'.npz';assert sha(source/name)==source_seal['files'][name];assert sha(BASE/'predictions'/name)==base_seal['files'][name]
   point=Path(sample['point_path']);assert sha(point)==sample['point_sha256'];p=np.load(point,allow_pickle=False)
   with np.load(source/name,allow_pickle=False) as z:
    scores=z['semantic_scores'];assert np.isfinite(scores).all();query=int(scores.argmax());s=source7(z['boxes'][query])
   assert query==base_trace[sid]['source_query']
   with np.load(BASE/'predictions'/name,allow_pickle=False) as z:assert np.array_equal(s,z['source_yaw0'])
   crop,cd=crop_source_aabb(p,s);assert cd['crop_sha256_float64_xyz']==base_trace[sid]['diagnostics']['crop']['crop_sha256_float64_xyz'];assert len(crop)==base_trace[sid]['diagnostics']['crop']['crop_points']
   arrays={};diags={}
   for arm,method in plan['arm_methods'].items():
    try:b,d=full_refit_from_crop(crop,s,method,refit_z=True,config=DEFAULT)
    except ValueError as e:
     # Explicit undefined-shape degeneracies only; never swallow general errors.
     if str(e) not in ['isotropic_or_zero_PCA_covariance','degenerate convex hull']:raise
     b=s.copy();d={'capability_group':'classical_full_refit','updated':False,'reason':str(e),'fallback':'exactsource for undefined geometry'}
    assert b.shape==(7,) and np.isfinite(b).all();arrays[arm]=b;diags[arm]=d
   np.savez_compressed(out/name,**arrays);records.append(plain({'id':sid,'cohort_origin':cohort,'platform':sample['platform'],'scene':sample['scene'],'source_query':query,'crop_sha256_float64_xyz':cd['crop_sha256_float64_xyz'],'crop_points':len(crop),'same_source_query_and_crop_asoriginal':True,'diagnostics':diags,'seconds':time.perf_counter()-tic}));count+=1
 (out/'trace.json').write_text(json.dumps(records,indent=2));assert count==54
 seal={'state':'sealed','n':54,'arms':list(plan['arm_methods']),'files':{p.name:sha(p) for p in out.glob('*.npz')},'trace_sha256':sha(out/'trace.json'),'plan_sha256':sha(path),'original_run_lock_sha256':sha(BASE/'RUN_LOCK.json'),'original_prediction_seal_sha256':sha(BASE/'predictions/SEALED.json'),'capability_group':'classical_full_box_refit_centerXYZ_extentXYZ_yaw','same_source_selected_query_and_crop':True,'orientation_only':False,'scope':'posthoc54historicalunion; supplementplannedafterparentold54scorescompleted; notnewconfirmationorTTA','GT_metadata_targetscores_read':False,'models_GPU_training_used':False,'old_files_modified':False,'code_hashes':plan['source_hashes']}
 assert len(seal['files'])==54;(out/'SEALED.json').write_text(json.dumps(seal,indent=2));print('SEALED54_FULLREFIT',sha(out/'SEALED.json'),flush=True)
if __name__=='__main__':main()
