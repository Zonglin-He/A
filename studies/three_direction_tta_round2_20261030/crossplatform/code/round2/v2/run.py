"""Finite CPU teacher×solver matrix, GT-free cached inputs only."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
import torch
R=Path(__file__).resolve().parent;Q=R.parent;P=Q.parent
sys.path[:0]=[str(R),str(Q),str(P/'replication36'),str(P/'public_export/code')]
import surface_teacher as old_teacher
import roof_teacher
from adapter_solver import run as adapt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,list):return [plain(v) for v in x]
    return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();torch.set_num_threads(4)
    pp=R/'PLAN_LOCK.json';plan=json.loads(pp.read_text());assert sha(pp)==(R/'PLAN_LOCK.sha256').read_text().strip()
    for n,h in plan['source_hashes'].items():assert sha(Q/n)==h,n
    for n,h in plan['code_hashes'].items():assert sha(R/n)==h,n
    if a.cohort=='confirm':assert (R/'CONFIRMATION_RELEASE.json').is_file(),'Final-method freeze required'
    manifest=Q/'inputs'/(a.cohort+'_manifest.json');assert sha(manifest)==plan['input_manifest_sha256'][a.cohort];samples=json.loads(manifest.read_text());source=Q/(a.cohort+'_source');source_seal=json.loads((source/'sealed.json').read_text());assert source_seal['n']==len(samples)
    out=R/(a.cohort+'_predictions');out.mkdir(exist_ok=False);(out/'registered_plan.json').write_bytes(pp.read_bytes());traces=[]
    for sample in samples:
        name=sample['id']+'.npz';assert sha(source/name)==source_seal['files'][name];z=np.load(source/name,allow_pickle=False);pcpath=Path(sample['point_path']);assert sha(pcpath)==sample['point_sha256'];pc=np.load(pcpath,allow_pickle=False);scores=z['semantic_scores'];top=int(scores.argmax());source7=np.r_[z['boxes'][top],np.float32(0)].astype(np.float32)
        if a.cohort=='dev':
            priorseal=json.loads((Q/'dev_predictions/sealed.json').read_text());assert sha(Q/'dev_predictions'/name)==priorseal['files'][name];prior=np.load(Q/'dev_predictions'/name,allow_pickle=False);pca,surface=prior['pca'],prior['surface']
        else:
            from common import refine
            from reference_surface import refine as surface_refine
            _,pca,_=refine(pc,z['boxes'][top]);surface,_=surface_refine(pc,source7,pca)
        start=time.perf_counter();teachers={'old':old_teacher.build_teacher(pc,z['boxes'],scores,sample['seed']),'roof':roof_teacher.build_teacher(pc,z['boxes'],scores,sample['seed'])};geometry_seconds=time.perf_counter()-start
        if a.cohort=='dev':assert np.array_equal(teachers['old']['direct_box7'],prior['direct_latent'])
        # Old accepted feedback must retain priority exactly in the extension.
        accepted=teachers['old']['accepted'];assert np.array_equal(teachers['roof']['target_boxes7'][accepted],teachers['old']['target_boxes7'][accepted])
        arrays={'source':source7,'pca':pca,'surface':surface,'direct_old':teachers['old']['direct_box7'],'direct_roof':teachers['roof']['direct_box7']};students={};adapter_seconds=0.
        for teacher_name,teacher in teachers.items():
            for solver in ['adam','constrained']:
                st=time.perf_counter();outputs,trace=adapt(z['query_features'],z['boxes'],scores,teacher,solver);adapter_seconds+=time.perf_counter()-st;assert np.array_equal(outputs[0]['box7'],source7)
                for step in [1,3]:arrays[f'{teacher_name}_{solver}{step}']=outputs[step]['box7']
                students[teacher_name+'_'+solver]={'trace':trace,'queries':{str(k):v['query'] for k,v in outputs.items()},'chosen_accepted':{str(k):v['chosen_accepted'] for k,v in outputs.items()}}
        control,ct=adapt(z['query_features'],z['boxes'],scores,teachers['old'],'constrained',control=True);assert all(np.array_equal(v['box7'],source7) for v in control.values());arrays['adapter0']=source7.copy();arrays['source_target_control3']=control[3]['box7'];assert list(arrays)==plan['arms'];assert all(np.isfinite(x).all() for x in arrays.values());np.savez_compressed(out/name,**arrays)
        record={'id':sample['id'],'platform':sample['platform'],'source_query':top,'teachers':teachers,'students':students,'source_control_trace':ct,'geometry_seconds_two_teachers':geometry_seconds,'CPU_adapter_seconds_four_cells':adapter_seconds,'source_bridge_and_control_bitexact':True};traces.append(plain(record));print(json.dumps({'id':sample['id'],'old_accepted':int(teachers['old']['accepted'].sum()),'roof_total_accepted':int(teachers['roof']['accepted'].sum()),'geometry_seconds':geometry_seconds,'adapter_seconds':adapter_seconds}),flush=True)
    (out/'trace.json').write_text(json.dumps(traces,indent=2));seal={'n':len(samples),'arms':plan['arms'],'files':{p.name:sha(p) for p in out.glob('*.npz')},'trace_sha256':sha(out/'trace.json'),'plan_sha256':sha(pp),'source_seal_sha256':sha(source/'sealed.json'),'numeric_GT_read':False};(out/'sealed.json').write_text(json.dumps(seal,indent=2));print('V2_PREDICTIONS_SEALED',len(samples),flush=True)
if __name__=='__main__':main()
