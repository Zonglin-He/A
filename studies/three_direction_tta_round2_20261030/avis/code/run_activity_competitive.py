"""CPU latent adaptation with frozen external evidence, no annotation access."""
import json,pathlib,hashlib,sys,shutil,time
import numpy as np,torch
from activity_adapter import evidence,visibility,optimize,apply_gates,direct
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main(planpath,output):
 torch.set_num_threads(4);planpath=pathlib.Path(planpath);plan=json.loads(planpath.read_text());assert sha(planpath)==planpath.with_suffix('.json.sha256').read_text().split()[0]
 for file,h in plan['source_hashes'].items():assert sha(file)==h,file
 manifest=pathlib.Path(plan['manifest']);assert sha(manifest)==plan['manifest_sha256'];spec=json.loads(manifest.read_text());sourcefile=pathlib.Path(plan['source_predictions']);earlyfile=pathlib.Path(plan['old_latent_predictions'])
 assert sha(sourcefile)==plan['source_prediction_sha256'];assert sha(earlyfile)==plan['old_latent_prediction_sha256'];pred=json.loads(sourcefile.read_text());features=pathlib.Path(plan['features']);featseal=json.loads((features/'seal.json').read_text());assert featseal['state']=='sealed' and featseal['manifest_sha256']==sha(manifest) and featseal['feature_plan_sha256']==plan['feature_plan_sha256']
 out=pathlib.Path(output);out.mkdir(parents=True,exist_ok=False);rows={a:[] for a in plan['arms']};progress=[];start=time.perf_counter();torchseed=20261030
 for v in spec['videos']:
  name=v['video_id'];assert sha(features/f'{name}.npy')==next(x['cosine_sha256'] for x in featseal['rows'] if x['video_id']==name)
  q=evidence(np.load(features/f'{name}.npy'));valid=visibility(pred,name,v['length']);item={'video_id':name,'evidence_not_calibrated_probability':True,'q_min':float(q.min()),'q_max':float(q.max()),'arms':{}}
  bridge,stats=apply_gates(pred,name,np.ones_like(q,bool));original=[x for x in pred if x['video_id']==name];assert bridge==original,'zero-step activity/source bridge failed';item['zero_step_source_exact']=True
  for arm,smooth in [('direct_audio',False),('direct_audio_smooth',True),('direct_analytic_prior',False)]:
   tic=time.perf_counter();gates=(q.astype('float64') >= (.5-np.sqrt(.03))) if arm=='direct_analytic_prior' else direct(q,smooth);r,c=apply_gates(pred,name,gates);rows[arm]+=r;item['arms'][arm]={'seconds':time.perf_counter()-tic,**c}
  for arm,temporal in [('activity_tta',True),('activity_no_temporal',False)]:
   tic=time.perf_counter();gates,detail=optimize(q,valid,temporal,steps=plan['activity']['steps'],lr=plan['activity']['lr']);r,c=apply_gates(pred,name,gates);rows[arm]+=r;item['arms'][arm]={'seconds':time.perf_counter()-tic,**detail,**c}
  progress.append(item);(out/'progress.json').write_text(json.dumps(progress,indent=2));print(name,'done',flush=True)
 for arm in plan['arms']:
  d=out/arm;d.mkdir();p=d/'predictions.json'
  if arm=='source':shutil.copyfile(sourcefile,p)
  elif arm=='early_latent_v2':shutil.copyfile(earlyfile,p)
  else:p.write_text(json.dumps(rows[arm]))
  (d/'seal.json').write_text(json.dumps({'state':'sealed','plan_sha256':sha(planpath),'prediction_sha256':sha(p),'manifest_sha256':sha(manifest),'GT_access':False},indent=2))
 protocol={'state':'sealed','plan_sha256':sha(planpath),'manifest_sha256':sha(manifest),'args':{'plan':str(planpath)},'arms':{a:sha(out/a/'predictions.json') for a in plan['arms']},'teacher_feature_seal_sha256':sha(features/'seal.json'),'actual_adaptation':'per-video per-class per-frame activity residual, pretrained networkweights untouched','GT_access':False,'wall_seconds':time.perf_counter()-start}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('SEALED',flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
