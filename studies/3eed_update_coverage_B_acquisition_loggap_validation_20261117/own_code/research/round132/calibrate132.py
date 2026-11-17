"""One source-only AB/calibrated log-ratio threshold, no target outputs/labels."""
from common132 import *
import numpy as np,collections
def main():
 basecheck();assert rd(H132/'SOURCE_NATIVE_AUDIT.json')['status']=='PASS_SOURCE_CPU_READOUT_REGISTERED_SUPPORT';seal=rd(B132/'SOURCE_PREDICTIONS_SEALED.json');reg=rd(H132/'SOURCE_CALIBRATION_INPUTS_PRIVATE.json');gaps=[];rows=[];sourcefiles={}
 for src,h in seal['seals'].items():
  D=B132/'source_calibration_PRIVATE'/src;assert sha(D/'SEALED.json')==h
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
 for e in reg:
  entry=e['entry'];src=entry['source'];sid=entry['row']['id'];D=B132/'source_calibration_PRIVATE'/src;f=np.load(D/(sid+'.npz'));rec=rd(D/(sid+'.json'));S=f['S'];ass=f['assignment'];groups=[np.flatnonzero(ass==i) for i in range(int(ass.max()+1))];eligible=False;rr={'source':src,'id_PRIVATE':sid,'B_available':rec['B_available'],'S_groups':int(S.sum()),'view_gaps':[]}
  if rec['B_available'] and S.sum()>=2:
   label=entry['files']['label'];assert sha(R132/label['path'])==label['sha256'];sourcefiles[label['path']]=label['sha256']
   with np.load(R132/label['path']) as lab:y=lab['ious']
   good=np.array([y[g].max()>.25 for g in groups])[S]
   if good.any():
    eligible=True
    for v in ['A','B']:
     p=np.array([f['p'+v][g].mean() for g in groups])[S];logs=np.log(np.maximum(p,1e-300));gap=float(logs.max()-logs[good].max());gaps.append(gap);rr['view_gaps'].append(gap)
  rr['eligible']=eligible;rows.append(rr)
 if not gaps:
  dump(H132/'SOURCE_LOG_CALIBRATION.json',{'status':'BLOCKED_NO_SOURCE_AB_GOOD_SUPPORTED_GROUP','source_inputs':len(reg),'no_target_fallback':True});raise RuntimeError('no sourceAB calibration gap; stop log branch')
 kappa=float(np.quantile(gaps,.9));assert np.isfinite(kappa) and kappa>=0
 out={'status':'SOURCE_ONLY_AB_LOGRATIO_KAPPA_LOCKED','kappa':kappa,'quantile':.9,'rule':'pooled per-view source-own A/B logmax-minus-best-Acc25-supported-group gap90thpercentile; union over views unchanged','minimum_support_points':4,'minimum_registered_fraction':.5,'source_inputs':len(reg),'source_eligible_inputs':sum(r['eligible'] for r in rows),'source_view_gaps':len(gaps),'eligible_by_source':dict(collections.Counter(r['source'] for r in rows if r['eligible'])),'source_prediction_seal_sha256':sha(B132/'SOURCE_PREDICTIONS_SEALED.json'),'source_files':sourcefiles,'source_A_B_R131_content_overlap':0,'target_predictions_or_labels_read':False,'source_training':False,'FINAL_access':False,'utc':utc()};dump(H132/'SOURCE_LOG_CALIBRATION.json',out);dump(H132/'SOURCE_CALIBRATION_ROWS_PRIVATE.json',rows);print({k:v for k,v in out.items() if k!='source_files'})
if __name__=='__main__':main()
