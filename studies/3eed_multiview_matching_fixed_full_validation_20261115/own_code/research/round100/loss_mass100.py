"""Read-only initial-loss geometry on all sealed R98 inputs; no new model or selector."""
from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round100';B=R/'crossplatform/round98';D=H/'loss_mass_diagnostic';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 D.mkdir(exist_ok=False);out=[];summary=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;ss=rd(R/'research/round98/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));part=[]
   for row in ss:
    sid=row['id_PRIVATE'];ev=np.load(B/name/'evidence_PRIVATE'/(sid+'.npz'));z=np.load(B/name/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(z['arm_names'],z['scores']));p=pp['prepared_frozen'];q=ev['target'];e=ev['delta'];energy=e*e;total=max(float(energy.sum()),1e-30);order=np.argsort(-p);union=np.union1d(order[:10],np.argsort(-q)[:10]);grad=q-p;cos=float(np.dot(e,grad)/max(np.linalg.norm(e)*np.linalg.norm(grad),1e-30));cumulative=np.cumsum(p[order]);a={'source':src,'target':tgt,'id_PRIVATE':sid,'source_top1_probability':float(p.max()),'source_entropy_normalized':float(-(p*np.log(np.maximum(p,1e-300))).sum()/np.log(256)),'queries_for_80percent_probability':int(np.searchsorted(cumulative,.8)+1),'top10_probability_mass':float(p[order[:10]].sum()),'top10_initial_L2_error_mass':float(energy[order[:10]].sum()/total),'top10_union_initial_L2_error_mass':float(energy[union].sum()/total),'L2_vs_KL_initial_score_gradient_cosine':cos,'initial_uniform_L2':float(energy.mean()/2),'G_direct_IoU_change':row['arms']['binding_direct']['IoU']-row['arms']['prepared_frozen']['IoU']};part.append(a);out.append(a)
   for label,rows in [('all',part),('G_good',[r for r in part if r['G_direct_IoU_change']>1e-12]),('G_bad',[r for r in part if r['G_direct_IoU_change']< -1e-12]),('G_same',[r for r in part if abs(r['G_direct_IoU_change'])<=1e-12])]:
    if not rows:continue
    keys=[k for k,v in rows[0].items() if isinstance(v,(int,float))];summary.append({'source':src,'target':tgt,'stratum':label,'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}})
   print('LOSS_MASS100',name,len(part),flush=True)
 assert len(out)==18620;dump(D/'ALL_ROWS_PRIVATE.json',out);dump(D/'SUMMARY_ROWS.json',summary);allrows=[r for r in summary if r['stratum']=='all'];mean={k:float(np.mean([r[k] for r in allrows])) for k in allrows[0] if isinstance(allrows[0][k],(int,float))};result={'state':'DESCRIPTIVE_OLD_SEALED_LOSS_MASS_DIAGNOSTIC','n':len(out),'equal_six_means':mean,'no_new_GT_no_updates_no_hyperparameter_selection':True,'FINAL_access':False};dump(D/'SUMMARY.json',result);print(result)
if __name__=='__main__':main()
