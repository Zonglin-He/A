"""Postseal no-GT lower bound from unmapped-logit conservation; no new predictions."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round104';B=R/'crossplatform/round104';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==768;rows=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt
   for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    sid=r['id']
    with np.load(B/name/'features_PRIVATE'/(sid+'.npz')) as f:known=f['0_known'];assert all(np.array_equal(known,f[v+'_known']) for v in ['1','2','C'])
    with np.load(R/'crossplatform/round98'/name/'evidence_PRIVATE'/(sid+'.npz')) as e:d=e['delta']
    du=d[~known];lower=float(np.square(du-du.mean()).sum()/512) if len(du) else 0.;rec=rd(B/name/'records_PRIVATE'/(sid+'.json'))['late'];old=rd(R/'crossplatform/round100'/name/'records_PRIVATE'/(sid+'.json'))['joint3'];assert rec['final_error']>=lower-1e-8;rows.append({'source':src,'target':tgt,'row_PRIVATE':sid,'mapped_fraction':float(known.mean()),'unmapped_logit_L2_lower_bound':lower,'final_RGB_L2':rec['final_error'],'final_query_L2':old['final_error'],'RGB_excess_above_free_mapped_bound':rec['final_error']-lower,'bound_fraction_of_initial_error':lower/max(rec['initial_error'],1e-30),'RGB_parameter_relative_L2':rec['parameter_L2']})
 groups=[]
 for label,rr in [('all',rows)]+[(src+'_to_'+tgt,[r for r in rows if r['source']==src and r['target']==tgt]) for src in ['waymo','drone','quad'] for tgt in ['waymo','drone','quad'] if src!=tgt]:groups.append({'group':label,'n':len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['mapped_fraction','unmapped_logit_L2_lower_bound','final_RGB_L2','final_query_L2','RGB_excess_above_free_mapped_bound','bound_fraction_of_initial_error']}})
 D=H/'scope_bound';D.mkdir(exist_ok=False);(D/'ALL_ROWS_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(D/'SUMMARY.json').write_text(json.dumps({'state':'POSTSEAL_NO_GT_UNMAPPED_LOGIT_L2_CAPACITY_BOUND','n':len(rows),'groups':groups,'no_new_method_predictions':True,'GT_used':False,'bound_scope':'perview centered score L2, free scores on mapped queries, fixed raw scores on unmapped queries; NOT a task-IoU bound'},indent=2)+'\n');print(groups[0])
if __name__=='__main__':main()
