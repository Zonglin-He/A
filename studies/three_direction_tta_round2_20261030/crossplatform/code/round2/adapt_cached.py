"""CPU-only adaptation/controls; never imports a scorer or opens metadata."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch

R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/'public_export/code'));sys.path.insert(0,str(P/'replication36'))
from common import refine
from reference_surface import refine as surface_refine
from surface_teacher import build_teacher
from residual_adapter import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating)):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,list):return [plain(v) for v in x]
    return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();torch.set_num_threads(4)
    planpath=R/'PLAN_LOCK.json';plan=json.loads(planpath.read_text());assert sha(planpath)==(R/'PLAN_LOCK.sha256').read_text().strip()
    for name,h in plan['round2_code_hashes'].items():assert sha(R/name)==h,name
    for name,h in plan['source_hashes'].items():assert sha(P/name)==h,name
    mp=R/'inputs'/(a.cohort+'_manifest.json');assert sha(mp)==plan['input_manifest_sha256'][a.cohort];samples=json.loads(mp.read_text());source=R/(a.cohort+'_source');seal=json.loads((source/'sealed.json').read_text());assert seal['n']==len(samples)
    out=R/(a.cohort+'_predictions');out.mkdir(exist_ok=False);(out/'registered_plan.json').write_bytes(planpath.read_bytes());traces=[]
    for sample in samples:
        start=time.perf_counter();name=sample['id']+'.npz';assert sha(source/name)==seal['files'][name];z=np.load(source/name,allow_pickle=False);pp=Path(sample['point_path']);assert sha(pp)==sample['point_sha256'];pc=np.load(pp,allow_pickle=False);top=int(z['semantic_scores'].argmax());src=z['boxes'][top];s7,p7,pdiag=refine(pc,src);sf,sdiag=surface_refine(pc,s7,p7)
        teacher=build_teacher(pc,z['boxes'],z['semantic_scores'],sample['seed']);fit_end=time.perf_counter();adapt,tr=run(z['query_features'],z['boxes'],z['semantic_scores'],teacher);control,ct=run(z['query_features'],z['boxes'],z['semantic_scores'],teacher,control=True)
        assert np.array_equal(adapt[0]['box7'],s7) and adapt[0]['query']==top
        assert all(np.array_equal(v['box7'],s7) and v['query']==top for v in control.values()),'source-target control drift'
        arrays={'source':s7,'pca':p7,'surface':sf,'direct_latent':teacher['direct_box7'],'adapter0':adapt[0]['box7'],'adapter1':adapt[1]['box7'],'adapter3':adapt[3]['box7'],'source_target_control3':control[3]['box7']}
        assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(out/name,**arrays)
        record={'id':sample['id'],'platform':sample['platform'],'source_query':top,'direct_latent_query':teacher['direct_query'],'adapter_queries':{str(k):v['query'] for k,v in adapt.items()},'source_pca_teacher_seconds':fit_end-start,'adapter_and_control_seconds':time.perf_counter()-fit_end,'pca':pdiag,'surface':sdiag,'teacher':teacher,'adapter_trace':tr,'control_trace':ct,'adapter0_source_bitexact':True,'source_target_control3_bitexact':True};traces.append(plain(record));print(json.dumps({'id':sample['id'],'accepted_candidates':int(teacher['accepted'].sum()),'seconds':time.perf_counter()-start}),flush=True)
    (out/'trace.json').write_text(json.dumps(traces,indent=2));(out/'sealed.json').write_text(json.dumps({'n':len(samples),'arms':list(arrays),'files':{p.name:sha(p) for p in out.glob('*.npz')},'trace_sha256':sha(out/'trace.json'),'plan_sha256':sha(planpath),'source_seal_sha256':sha(source/'sealed.json'),'numeric_GT_read':False},indent=2));print('PREDICTIONS_SEALED',len(samples),flush=True)
if __name__=='__main__':main()
