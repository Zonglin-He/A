"""Separate sealed CPU scoring; current labels never feed baseline execution."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round18'),str(ROOT/'research/round5/audit')]
from common import EXP,read,dump,sha,verify,seal,check_plan
from independent_core_v2 import transform_gt,candidate_ious
INDEX=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json'
def summarize(rows):
 out={}
 for split in sorted({r['split'] for r in rows}):
  out[split]={}
  for platform in ('drone','quad'):
   rr=[r for r in rows if r['split']==split and r['platform']==platform];names=list(rr[0]['ious']);cells={}
   for arm in names:
    v=np.array([r['ious'][arm] for r in rr]);base=np.array([r['ious']['source'] for r in rr]);d=v-base;sc=collections.defaultdict(list)
    for r,x in zip(rr,d):sc[r['scene_PRIVATE']].append(float(x))
    cells[arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_pp':float(d.mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'box_changed':sum(r['box_changed'][arm] for r in rr),'scene_clusters':len(sc),'equal_scene_gain_pp':float(np.mean([np.mean(x) for x in sc.values()])*100),'signed_deltas':d.tolist()}
   out[split][platform]=cells
 return out
def main():
 check_plan();lock=read(EXP/'SCORE_PROGRAM_LOCK.json');assert lock['scorer_sha256']==sha(__file__)
 for name,h in lock['dependencies'].items():assert sha(ROOT/name)==h
 formal=ROOT/'crossplatform/round18/formal96_attempt001';online=ROOT/'crossplatform/round18/reg_online32_attempt001';verify(formal);verify(online)
 out=EXP/'scored96_attempt001';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'formal_seal_sha256':sha(formal/'SEALED.json'),'online_seal_sha256':sha(online/'SEALED.json'),'scorer_sha256':sha(__file__),'metric_dependency_sha256':lock['dependencies'],'index_sha256':sha(INDEX),'predictions_sealed_before_current_GT':True})
 index={r['id']:r for r in read(INDEX)};inputs=read(EXP.parent/'input/ALL96_PRIVATE.json');containers={};rows=[];onrows=[]
 for i,r in enumerate(inputs):
  with np.load(formal/'predictions_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();assert str(z['sample_id'])==r['id'];pool=z['source_pool'].copy();assert np.array_equal(boxes[0],pool[int(z['source_scores'].argmax())])
  ix=index[r['id']];assert ix['platform']==r['platform'] and ix['scene']==r['scene'];p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
  if p not in containers:containers[p]=read(p)
  target=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));raw=boxes.copy();boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6));values=candidate_ious(boxes,target)[0];split='A64' if i<64 else 'B32';row={'alias':f'{split}_{r["platform"]}_{i:03d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'split':split,'ious':dict(zip(names,map(float,values))),'box_changed':{n:bool(not np.array_equal(b,raw[0])) for n,b in zip(names,raw)},'raw_nonpositive_size_count':{n:int((b[3:]<=0).sum()) for n,b in zip(names,raw)}};rows.append(row)
  if split=='B32':
   with np.load(online/'predictions_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:onames=['source']+z['arm_names'].tolist();ob=np.concatenate([raw[:1],z['arm_boxes_raw6']],0)
   orb=ob.copy();ob[:,3:]=np.maximum(ob[:,3:],np.float32(1e-6));v=candidate_ious(ob,target)[0];onrows.append(dict(row,split='B32_online_supplement',ious=dict(zip(onames,map(float,v))),box_changed={n:bool(not np.array_equal(b,orb[0])) for n,b in zip(onames,orb)}))
 summary=summarize(rows);on=summarize(onrows);dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',summary);dump(out/'ONLINE_ROWS_PRIVATE.json',onrows);dump(out/'ONLINE_SUMMARY.json',on);seal(out,'R18_current_task_scored_after_all_prediction_seals',n=96,online_n=32,all_negatives_retained=True,source_or_hyperparameters_not_selected_by_current_GT=True);print('R18_SCORED_96',flush=True)
if __name__=='__main__':main()
