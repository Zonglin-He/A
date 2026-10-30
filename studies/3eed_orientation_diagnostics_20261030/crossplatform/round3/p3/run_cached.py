"""54 historical cachedframes, GT-free predictions; no scorer/meta/GT imports."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from geometry_baselines import *
R=Path(__file__).resolve().parent;P=R.parents[1];Q=P/'round2'
MASTER=R.parent/'PLAN_DIAGNOSTIC_LOCK.json'
MASTER_SHA='999d422a1c624b521d59e3793f0b817170e560916eb3b141bf7024ed1394e59e'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,list):return [plain(v) for v in x]
 return x

def main():
 lock=R/'RUN_LOCK.json';reg=json.loads(lock.read_text());assert sha(lock)==lock.with_suffix('.sha256').read_text().split()[0];assert sha(MASTER)==MASTER_SHA==reg['master_sha256']
 for f,h in reg['source_hashes'].items():assert sha(f)==h,f
 oldplan=json.loads((Q/'PLAN_LOCK.json').read_text());out=R/'predictions';out.mkdir(exist_ok=False);trace=[];manifest=[]
 for cohort in ['dev','confirm']:
  mp=Q/'inputs'/f'{cohort}_manifest.json';assert sha(mp)==oldplan['input_manifest_sha256'][cohort]==reg['input_manifests'][cohort]
  samples=json.loads(mp.read_text());source=Q/f'{cohort}_source';ss=json.loads((source/'sealed.json').read_text());legacy=Q/'v2'/f'{cohort}_predictions';ls=json.loads((legacy/'sealed.json').read_text());assert ss['n']==ls['n']==len(samples);assert sha(source/'sealed.json')==reg['source_seals'][cohort];assert sha(legacy/'sealed.json')==reg['legacy_seals'][cohort]
  for sample in samples:
   start=time.perf_counter();name=sample['id']+'.npz';assert sha(source/name)==ss['files'][name];assert sha(legacy/name)==ls['files'][name]
   pp=Path(sample['point_path']);assert sha(pp)==sample['point_sha256'];points=np.load(pp,allow_pickle=False)
   with np.load(source/name,allow_pickle=False) as z:top=int(z['semantic_scores'].argmax());src=source7(z['boxes'][top]);assert src[6]==0
   with np.load(legacy/name,allow_pickle=False) as z:oldsource=z['source'].copy();oldpca=z['pca'].copy();oldjoint=z['roof_constrained3'].copy()
   assert np.array_equal(src,oldsource) and np.array_equal(src[:6],oldpca[:6])
   arrays,diag=compare_orientation(points,src,selected_query=top);arrays={'source_yaw0':src.copy(),'legacy_PCA_reference':oldpca,**arrays,'legacy_joint_adapter3_reference':oldjoint}
   assert list(arrays)==reg['arms']
   for arm,b in arrays.items():
    assert np.isfinite(b).all()
    if arm!='legacy_joint_adapter3_reference':assert np.array_equal(b[:6],src[:6]),arm
   np.savez_compressed(out/name,**arrays)
   record={'id':sample['id'],'cohort_origin':cohort,'platform':sample['platform'],'scene':sample['scene'],'source_query':top,'diagnostics':diag,'seconds':time.perf_counter()-start,'orientation_only_source_center_size_bitexact':True,'legacy_joint_reference_capability':'oldselectedquery/centerpossiblydifferent, notorientationonly','source_cache_sha256':ss['files'][name],'old_prediction_sha256':ls['files'][name]};trace.append(plain(record));manifest.append({k:sample[k] for k in ['id','platform','scene','seed','point_path','point_sha256']});print(json.dumps({'id':sample['id'],'n_crop':diag['crop']['crop_points'],'seconds':record['seconds']}),flush=True)
 (out/'trace.json').write_text(json.dumps(trace,indent=2));(out/'input_index.json').write_text(json.dumps(manifest,indent=2));files={p.name:sha(p) for p in out.glob('*.npz')};assert len(files)==54
 seal={'state':'sealed','n':54,'scope':'posthoc54historicaldev18+confirm36 union; notnewconfirmation orsourceonlydomainshift','arms':reg['arms'],'capability_groups':reg['capability_groups'],'files':files,'trace_sha256':sha(out/'trace.json'),'input_index_sha256':sha(out/'input_index.json'),'run_lock_sha256':sha(lock),'master_sha256':sha(MASTER),'source_hashes':reg['source_hashes'],'numeric_GT_read':False,'meta_or_target_score_files_read':False,'models_or_GPU_used':False,'legacy_source_or_predictions_modified':False}
 (out/'SEALED.json').write_text(json.dumps(seal,indent=2));(out/'sealed.json').write_bytes((out/'SEALED.json').read_bytes());print('SEALED54',sha(out/'SEALED.json'),flush=True)
if __name__=='__main__':main()
