"""CPU-only decomposition from existing probability caches; no network calls."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.temporal import final_mbr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
OUT=ROOT/'outputs/branch_decomposition_v1';PUBLIC=ROOT/'results/branch_decomposition_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def utility(ids,q,span):
 ids=np.asarray(ids);i,j=np.nonzero(q);s=ids[i];e=ids[j]+1;a,b=span;intersection=np.maximum(0,np.minimum(b,e)-np.maximum(a,s));return float(q[i,j]@(intersection/((b-a)+(e-s)-intersection)))
def validate(p):
 for f,h in p['sources'].items():assert sha(ROOT/f)==h
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert len(formal['sources'])==434
 for f,h in formal['sources'].items():assert sha(ROOT/f)==h
 assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256']
def prepare():
 assert not(OUT/'plan.json').exists();old=read(ROOT/'outputs/tube_fusion_hc64/lock.json');items=[]
 for key in old['ids']:
  inputs={name:str(path)for name,path in dict(G=Path('outputs/tube_fusion_hc64/posteriors')/f'{key}.pt',pack=Path('outputs/interval_pool/development/posteriors')/f'{key}.pt',capture=Path('outputs/tube_temporal_hc64/captures')/f'{key}.pt',prediction=Path('outputs/tube_fusion_hc64/predictions')/f'{key}.npz',old_student=Path('outputs/layout_reward_dev/student_cache')/f'{key}.pt').items()};items.append(dict(id=key,anonymous='H'+key[2:],inputs=inputs,hashes={name:sha(ROOT/path)for name,path in inputs.items()}))
 p=dict(schema='cached_branch_decomposition_v1',items=items,smoke='outputs/checks/boxfixed_tube/post_strong_capture.pt',smoke_sha256=sha(ROOT/'outputs/checks/boxfixed_tube/post_strong_capture.pt'),old_report_sha256=sha(ROOT/'results/tube_fusion_hc64/report.json'),labels=dict(path='data/evaluation/development_labels.jsonl',sha256=sha(ROOT/'data/evaluation/development_labels.jsonl')),sources={str(f.relative_to(ROOT)):sha(f)for f in Path(__file__).parent.glob('*.py')},formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),GT_read=False,protocol=dict(coverage='Complete224 six-layer posteriors for64. Full Strong posterior found for one engineering case only. Legacy student caches and official predictions store finalStrong interval but no distribution. Do not infer missing63 H posteriors from a decision.',one_case='Bind Source/E/frame IDs/8positions/224 hidden bitwise. Recover qH=2qStrong-qG, assert nonnegative normalized, reconstruct blend and all saved intervals exactly. Decode originalMBR and unchanged spatialDP for448-only,224six,blend before GT.',full_panel='Reaggregate already sealed64 decisions. Calculate native posterior uncertainty, view disagreement, and lower bound on448 relative expected-IoU preference needed for fusion-selected interval; bounds are not reconstructed448 scores.',forbidden='No new RGB,encoder,decoder,head,expert,GPU,training or parameter scan. Do not use actionness or any auxiliary head. No production/fullrun changes.',design='Use observed failure types to specify a bounded generic lightweight alternative; do not declare it effective or execute a newmodel without missing branch evidence.'))
 validate(p);write(OUT/'plan.json',p)
@torch.no_grad()
def execute():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);assert not(OUT/'sealed.json').exists();assert sha(ROOT/p['smoke'])==p['smoke_sha256'];smoke=torch.load(ROOT/p['smoke'],map_location='cpu',weights_only=False);rows=[];artifacts=[];matching=[];maxblend=0.
 for item in p['items']:
  for name,path in item['inputs'].items():assert sha(ROOT/path)==item['hashes'][name]
  g=torch.load(ROOT/item['inputs']['G'],map_location='cpu',weights_only=False);pack=torch.load(ROOT/item['inputs']['pack'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);student=torch.load(ROOT/item['inputs']['old_student'],map_location='cpu',weights_only=False);old=np.load(ROOT/item['inputs']['prediction']);ids=old['frame_ids'];qG=np.asarray(g['final']['Global6']);base=old['Global6_interval'];blend=old['Strong_Reference_interval'];assert final_mbr(ids,qG)['interval']==base.tolist();nz=qG[qG>0];row=dict(query=item['anonymous'],complete224six=True,legacy_student_has_posterior='posterior'in student,old_prediction_has_posterior=any('posterior'in k or 'logits'in k for k in old.files),native_entropy=float(-(nz*np.log(nz)).sum()),normalized_entropy=float(-(nz*np.log(nz)).sum()/np.log(len(ids)*(len(ids)-1)/2)),native_MBR_utility=utility(ids,qG,base),native_utility_of_blend=utility(ids,qG,blend),blend_changed=bool(np.any(base!=blend)))
  row['required448_relative_preference_lower_bound']=row['native_MBR_utility']-row['native_utility_of_blend']
  candidates=np.asarray(pack['candidate_intervals']);starts=candidates[:,0];ends=candidates[:,1];D=float(ids[-1]-ids[0]+1);row['candidate_start_spread_fraction']=float(np.std(starts)/D);row['candidate_end_spread_fraction']=float(np.std(ends)/D);rows.append(row)
  if np.array_equal(smoke['Source'],cap['Source'])and np.array_equal(smoke['frame_ids'],cap['frame_ids']):
   for k in ['E','planned_positions']:assert np.array_equal(smoke[k],cap[k])
   assert all(torch.equal(a['hidden'],b['hidden'])for a,b in zip(smoke['temporal_cache']['offsets'],cap['temporal_cache']['offsets']));qS=np.asarray(smoke['posterior']['Strong']);qH=2*qS-qG;assert qH.min()>=0 and abs(qH.sum()-1)<2e-12;assert np.array_equal(qS,.5*qH+.5*qG);assert final_mbr(ids,qS)['interval']==blend.tolist();assert np.array_equal(smoke['intervals']['Strong'],blend)
   values=dict(frame_ids=ids,positions8=old['positions8'],FixedSpatial=old['Strong_Reference'],Source=old['Source_Native224'],Source_interval=old['Source_Native224_interval']);geom=restore_geometry(cap['geometry']);probabilities=dict(Only448=qH,Only224six=qG,Fusion=qS)
   for name,q in probabilities.items():
    span=final_mbr(ids,q)['interval'];track=evidence_dp(geom.for_interval(span))['final'].numpy();values[name]=track;values[name+'_interval']=span
   assert np.array_equal(values['Only224six'],old['Global6'])and np.array_equal(values['Fusion'],old['Strong_Reference']);dest=OUT/'predictions'/f"{item['anonymous']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**values);pp=OUT/'posteriors'/f"{item['anonymous']}.npz";pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,ids=ids,**probabilities);artifacts.append(dict(query=item['anonymous'],path=str(dest.relative_to(OUT)),sha256=sha(dest),posterior_path=str(pp.relative_to(OUT)),posterior_sha256=sha(pp)));matching.append(item['anonymous'])
 assert matching==['H001'];assert all(not r['legacy_student_has_posterior']and not r['old_prediction_has_posterior']for r in rows);validate(p);assert not torch.cuda.is_initialized();write(OUT/'label_free_audit.json',dict(rows=rows,GT_read=False));write(OUT/'sealed.json',dict(plan_sha256=sha(OUT/'plan.json'),audit_sha256=sha(OUT/'label_free_audit.json'),GT_read=False,complete448_samples=1,complete224_samples=64,files=artifacts,network_head_GPU_calls=0,matching=matching));print('sealed:64 cached decisions; one complete branch decomposition',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','execute']);x=a.parse_args();globals()[x.action]()
