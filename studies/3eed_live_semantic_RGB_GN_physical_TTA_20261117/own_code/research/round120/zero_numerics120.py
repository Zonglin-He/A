"""Post-task numerical boundary audit; not a new selected method or update."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round120';B=R/'crossplatform/round120';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round118'));import core118 as C

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);mat=rd(B/'MATRIX_SEALED.json');assert rd(H/'SCALAR_AUDIT.json')['n']==2304;rows=[];n=aliases=0
 for src in ['waymo','drone','quad']:
  with np.load(B/(src+'_SOURCE_READOUT_PRIVATE.npz')) as z:W={k[6:]:torch.from_numpy(z[k]).cuda() for k in z.files if k.startswith('match.')}
  model=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64)).cuda().eval().requires_grad_(False);model.load_state_dict(W)
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;D=B/'formal'/co/name;scored={r['id_PRIVATE']:r for r in rd(H/f'scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json')}
    for r in C.rows(co,tgt):
     sid=r['id'];rec=rd(D/'records_PRIVATE'/(sid+'.json'));n+=1
     if rec['protected_alias_fallback']:aliases+=1;continue
     N,_,_=C.components(co,name,sid);a=N['fusedABC'];ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));old=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(old['arm_names'],old['scores']));actualold=po['context_zero_F']
     with torch.no_grad():
      raw=model(torch.as_tensor(a['h'],device='cuda')[None])[0];q=F.normalize(raw+torch.as_tensor(ff['zero_SA_late'],device='cuda'),dim=-1);u=torch.as_tensor(a['u'],device='cuda');m=torch.as_tensor(a['mask'],device='cuda').bool();s=q@u.T/.07;p=(s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1)).double();p=(p+torch.as_tensor(ff['zero_SF_r'],device='cuda').double()).softmax(0).cpu().numpy()
     # Independently reproduce the exact original canonical CPU baseline.
     cpuW={k:v.cpu() for k,v in W.items()};rawcpu=C.project(a['h'],cpuW);cpu=C.read(rawcpu,a['u'],a['mask'],ff['zero_SA_late'],ff['zero_SF_r']);er=float(abs(cpu-actualold).max());assert er<1e-12,er
     qc,qg,qt=int(actualold.argmax()),int(p.argmax()),int(po['depth_trust_live_F'].argmax());vs={v['query_PRIVATE']:v['IoU'] for v in scored[sid]['arms'].values()};order=np.argsort(-actualold);margin=float(actualold[order[0]]-actualold[order[1]])
     rows.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'zero_CPU_query_PRIVATE':qc,'zero_GPU_query_PRIVATE':qg,'updated_query_PRIVATE':qt,'zero_backend_Top1_differs':qc!=qg,'update_changes_CPU_zero_Top1':qc!=qt,'update_changes_GPU_zero_Top1':qg!=qt,'zero_CPU_margin':margin,'zero_CPU_GPU_max_probability_error':float(abs(p-actualold).max()),'canonical_CPU_replay_error':er,'zero_CPU_IoU_reused':vs.get(qc),'zero_GPU_IoU_reused':vs.get(qg),'updated_IoU_reused':vs.get(qt),'new_task_GT_access':False});ff.close();old.close()
 O=H/'zero_numerics_attempt001';O.mkdir();(O/'ALL_ROWS_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');summary=[]
 for co in ['DEV','EXP']:
  cr=[r for r in rows if r['cohort']==co];changed=[r for r in cr if r['zero_backend_Top1_differs']];summary.append({'cohort':co,'n_nonalias':len(cr),'zero_backend_Top1_disagreements':len(changed),'updates_change_CPU_baseline':sum(r['update_changes_CPU_zero_Top1'] for r in cr),'updates_change_actual_GPU_zero':sum(r['update_changes_GPU_zero_Top1'] for r in cr),'max_zero_CPU_GPU_probability_error':max(r['zero_CPU_GPU_max_probability_error'] for r in cr),'max_zero_CPU_replay_error':max(r['canonical_CPU_replay_error'] for r in cr),'all_disagreements_within_numeric_probability_bound':all(r['zero_CPU_margin']<=2*r['zero_CPU_GPU_max_probability_error']+1e-12 for r in changed),'new_task_GT_access':False})
 assert n==2304 and aliases==154;(H/'ZERO_NUMERICS_SUMMARY.json').write_text(json.dumps({'postscore_diagnostic':True,'does_not_change_primary_or_predictions':True,'n':n,'aliases':aliases,'cohorts':summary,'FINAL_access':False},indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
