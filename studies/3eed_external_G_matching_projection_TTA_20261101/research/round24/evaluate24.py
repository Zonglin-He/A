"""CPU-only post-seal scoring and exposed development selection."""
import sys,argparse,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'research/round5/audit')]
from common24 import *
from independent_core_v2 import transform_gt,candidate_ious

def score(folder,rr):
 verify(folder);index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};out=[]
 for r in rr:
  ix=index[r['id']];path=ix['meta_path'];assert sha(path)==ix['metadata_container_sha256']
  if path not in containers:containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));z=load_npz(folder/'predictions_PRIVATE'/(r['id']+'.npz'));names=z['arm_names'].tolist();q=z['query_indices'];b=z['boxes'].copy();assert np.array_equal(b,feature(r)['boxes'][q]);b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0];out.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'q':dict(zip(names,map(int,q))),'ious':dict(zip(names,map(float,v)))})
 return out

def summary(rr,ref='source'):
 out={}
 for platform in ['drone','quad']:
  r=[x for x in rr if x['platform']==platform];out[platform]={}
  for name in r[0]['q']:
   v=np.array([x['ious'][name] for x in r]);d=v-np.array([x['ious'][ref] for x in r]);scenes=sorted({x['scene_PRIVATE'] for x in r});means=[float(np.mean([x['ious'][name] for x in r if x['scene_PRIVATE']==scene])) for scene in scenes]
   out[platform][name]={'n':len(r),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'query_changed':sum(x['q'][name]!=x['q'][ref] for x in r),'scene_clusters':len(scenes),'equal_scene_mIoU_percent':float(np.mean(means)*100)}
 return out

def select():
 folder=BASE/'screen_attempt001';r=score(folder/'V64_predictions',rows('V64'));s=summary(r);dump(EXP/'V64_LAYER_ROWS_PRIVATE.json',r);dump(EXP/'V64_LAYER_SUMMARY.json',s);ranking=[];spec=read(EXP/'FIT_SPEC_LOCK.json')
 for layer in spec['layers']:
  options=[]
  for k,lr in enumerate(spec['LR'][layer]):
   for st in spec['checkpoints']:
    n=f'{layer}_lr{k}_s{st}';options.append({'name':n,'layer':layer,'lr':lr,'steps':st,'balanced_mIoU':float(np.mean([s[p][n]['mIoU_percent'] for p in ['drone','quad']]))})
  options.sort(key=lambda x:(-x['balanced_mIoU'],x['steps'],x['lr']));ranking.append(options[0])
 ranking.sort(key=lambda x:(-x['balanced_mIoU'],spec['layers'].index(x['layer'])));selection={'best':ranking[0],'per_layer':ranking,'selection_uses_exposed_V64_GT':True,'M64_GT_or_G_used_for_selection':False,'V64_prediction_seal_sha256':sha(folder/'V64_predictions/SEALED.json'),'fit_spec_sha256':sha(EXP/'FIT_SPEC_LOCK.json')};dump(EXP/'LAYER_SELECTED_LOCK.json',selection)
 d=score(folder/'D208_predictions',rows('D208'));dump(EXP/'D208_LAYER_ROWS_PRIVATE.json',d);dump(EXP/'D208_LAYER_SUMMARY.json',summary(d));print('SELECTED_LAYERS',ranking)
if __name__=='__main__':select()
