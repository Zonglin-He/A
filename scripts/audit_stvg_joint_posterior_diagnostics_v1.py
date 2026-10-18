"""Post-seal representation and observation diagnostics; no optimization."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_v1 import BASE,OLD,read,put,load,condition,decode_state

def main():
 import torch,numpy as np
 from scipy.stats import rankdata
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
 assert read(BASE/'PREDICTION_BARRIER.json')['count']==32
 p=read(BASE/'PLAN_LOCK.json');targets={d:truths(d,dict(parents=[r['query_ordinal']for r in p['target']if r['dataset']==d]))for d in ['hc2','vidstg']};rows=[]
 for r in p['target']:
  z=load(BASE/'features/target'/(r['alias']+'.pt'));pr=load(BASE/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];gt=targets[ds][0][r['query_ordinal']];span=targets[ds][1][r['query_ordinal']];row=dict(input=z['input'],frame_ids=z['frame_ids']);result={}
  for index,arm in [(0,'Source'),(1,'Direct')]:
   zz=dict(z);zz['boxes']=[z['boxes'][index]];state=condition(zz,'cpu')['native'][0];decoded=decode_state(state,zz)
   assert decoded['interval']==z['native_interval'],(r['alias'],'interval encoder/decoder roundtrip')
   raw=official(z['boxes'][index].numpy(),row,gt,span,z['native_interval'],ds);compressed=official(decoded['boxes'],row,gt,span,z['native_interval'],ds)
   result[arm+'_32knot_readout']=compressed;result[arm+'_32knot_delta']={k:compressed[k]-raw[k]for k in raw}
  posterior=pr['arms']['joint_diffusion'];result['posterior_time_Source_space']=official(z['boxes'][0].numpy(),row,gt,span,posterior['interval'],ds);result['posterior_space_Source_time']=official(posterior['boxes'],row,gt,span,z['native_interval'],ds)
  observations=[]
  for i,a in enumerate(z['expert']['anchors']['single4']):
   fid=z['frame_ids'][a['position']];known=fid in gt
   if known:
    b=np.array(a['box']);xy=np.concatenate((b[:2]-b[2:]/2,b[:2]+b[2:]/2))*np.array([z['input']['width'],z['input']['height'],z['input']['width'],z['input']['height']]);iou=float(box_iou(xy,np.array(gt[fid])))
   else:iou=None
   observations.append(dict(anchor=i+1,prior=float(pr['association_prior'][i]),precise_GT_available=known,GT_geometry_IoU=iou,geometric_inlier=None if iou is None else iou>=.5,unknown_reason=None if known else'outside labeled event; not an inferred wrong identity'))
  rows.append(dict(alias=r['alias'],dataset=ds,readouts=result,observations=observations))
 means={};assoc={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];means[ds]={a:{k:float(np.mean([r['readouts'][a][k]for r in rr]))for k in rr[0]['readouts'][a]}for a in rr[0]['readouts']}
  obs=[a for r in rr for a in r['observations']];known=[a for a in obs if a['precise_GT_available']];scores=np.array([a['prior']for a in known]);labels=np.array([a['geometric_inlier']for a in known]);n1=int(labels.sum());n0=len(known)-n1
  auc=float((rankdata(scores)[labels].sum()-n1*(n1+1)/2)/(n1*n0))if n1 and n0 else None
  assoc[ds]=dict(total=len(obs),known=len(known),unknown=len(obs)-len(known),geometric_inliers=n1,geometric_noninliers=n0,AUROC=auc,Brier=float(np.mean((scores-labels)**2)),threshold05_accuracy=float(np.mean((scores>=.5)==labels)),labels='geometric compatibility not visual identity')
 put(BASE/'POSTSEAL_DIAGNOSTICS.json',dict(status='complete',count=32,rows=rows,means=means,target_association_geometry=assoc,scope='CPU diagnostic readouts only; no new fit, no method/threshold selection',CURRENT_unchanged=True))
 print(json.dumps(dict(means=means,association=assoc),indent=2))
if __name__=='__main__':main()
