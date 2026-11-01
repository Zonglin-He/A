"""Development-only lambda choice, sealed held-out predictions, then CPU scoring."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round27'),str(ROOT/'research/round5/audit')]
from common27 import *
from independent_core_v2 import candidate_ious,transform_gt
INDEX=None;CONTAINERS={}
def ious(r,b):
 global INDEX
 if INDEX is None:INDEX={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')}
 ix=INDEX[r['id']];path=ix['meta_path']
 if path not in CONTAINERS:assert sha(path)==ix['metadata_container_sha256'];CONTAINERS[path]=read(path)
 gt=transform_gt(CONTAINERS[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=b.copy();b[:,3:]=np.maximum(b[:,3:],1e-6);return candidate_ious(b,gt)[0]
def grid():
 verify(BASE/'scores528_attempt001');out=BASE/'dev208grid_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();spec=[]
 names=read(BASE/'scores528_attempt001/VARIANTS.json');plan=read(EXP/'PLAN_LOCK.json')
 for n in names:
  if n=='legacy_complex':ll=[2]
  else:ll=plan['lambdas']
  for l in ll:spec.append({'name':n,'lambda':l})
 dump(out/'SPEC.json',spec)
 for r in rows('D208'):
  z=load_npz(BASE/'scores528_attempt001/samples_PRIVATE'/(r['id']+'.npz'));names=z['cue_names'].tolist();q=np.array([np.argmax(z['source']+s['lambda']*z['cues'][names.index(s['name'])]) for s in spec]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),queries=q,boxes=z['boxes'][q])
 seal(out,'R27_dev_grid_predictions_before_current_dev_GT',n=208)
def select():
 out=BASE/'dev208grid_attempt001';verify(out);spec=read(out/'SPEC.json');vals=[]
 for r in rows('D208'):
  z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));vals.append(ious(r,z['boxes']))
 vals=np.stack(vals);summary=[];selected={}
 for j,s in enumerate(spec):
  pmean={p:float(vals[[r['platform']==p for r in rows('D208')],j].mean()*100) for p in ['drone','quad']};summary.append(dict(s,by_platform=pmean,balanced_mIoU_percent=np.mean(list(pmean.values()))))
 for n in {s['name'] for s in spec}:
  best=sorted([s for s in summary if s['name']==n],key=lambda s:(-s['balanced_mIoU_percent'],s['lambda']))[0];selected[n]=best['lambda']
 dump(EXP/'DEV_GRID_RESULTS.json',summary);dump(EXP/'SELECTED_LAMBDAS_LOCK.json',{'lambda_by_variant':selected,'selection':'one shared scalar per variant, exposedD208 only','dev_prediction_seal_sha256':sha(out/'SEALED.json'),'selection_precedes_Q64_current_GT':True})

 dump(EXP/'PRACTICAL_FAMILY_LOCK.json',{'name':'crop_scene_joint','lambda':selected['crop_scene_joint'],'primary_fixed_before_outputs':True,'no_family_selection_on_Q64':True})
 print('SELECTED27',selected)

def predict():
 sel=read(EXP/'SELECTED_LAMBDAS_LOCK.json');out=BASE/'predictions528_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();strict=['scene_base','scene_hires','scene_repeat','crop_scene_joint','R26_scene']
 for r in rows():
  z=load_npz(BASE/'scores528_attempt001/samples_PRIVATE'/(r['id']+'.npz'));names=z['cue_names'].tolist();scores={'source':z['source']}
  for n,l in sel['lambda_by_variant'].items():scores[n]=z['source']+l*z['cues'][names.index(n)]
  for n in strict:scores[n+'_fixed2']=z['source']+2*z['cues'][names.index(n)]
  nn=list(scores);q=np.array([int(scores[n].argmax()) for n in nn]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(nn),queries=q,boxes=z['boxes'][q],scores=np.stack([scores[n] for n in nn]))
 seal(out,'R27_all528_predictions_sealed_before_Q64_GT',n=528,lambda_lock_sha256=sha(EXP/'SELECTED_LAMBDAS_LOCK.json'),target_optimizer_steps=0,source_and_boxes_frozen=True)
 dump(EXP/'Q64_PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(out/'SEALED.json'),'lambda_lock_sha256':sha(EXP/'SELECTED_LAMBDAS_LOCK.json'),'practical_family_lock_sha256':sha(EXP/'PRACTICAL_FAMILY_LOCK.json'),'newQ64_current_labels_not_read':True})
def score():
 out=BASE/'predictions528_attempt001';verify(out);assert sha(EXP/'SELECTED_LAMBDAS_LOCK.json')==read(EXP/'Q64_PRELABEL_LOCK.json')['lambda_lock_sha256'];rr=[]
 for r in rows():
  z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z['boxes'],feature(r)['boxes'][z['queries']]);vv=ious(r,z['boxes']);nn=z['arm_names'].tolist();rr.append({'alias':r['alias'],'platform':r['platform'],'stage':r['stage'],'scene_PRIVATE':r['scene'],'q':dict(zip(nn,map(int,z['queries']))),'ious':dict(zip(nn,map(float,vv)))})
 dump(EXP/'ALL528_ROWS_PRIVATE.json',rr)
 for phase,rs in [('D208',rr[:208]),('REPLAY256',rr[208:464]),('OLD400',rr[:400]),('OLD464',rr[:464]),('Q64',rr[464:])]:
  summary={}
  for p in ['drone','quad']:
   pr=[r for r in rs if r['platform']==p];summary[p]={}
   for n in pr[0]['q']:
    v=np.array([x['ious'][n] for x in pr]);d=v-np.array([x['ious']['source'] for x in pr]);summary[p][n]={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'query_changes':sum(x['q'][n]!=x['q']['source'] for x in pr),'scene_clusters':len({x['scene_PRIVATE'] for x in pr})}
  dump(EXP/(phase+'_SUMMARY.json'),summary)
 print('SCORE27_COMPLETE')
if __name__=='__main__':{'grid':grid,'select':select,'predict':predict,'score':score}[sys.argv[1]]()
