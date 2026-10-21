"""Offline only: do sealed mask/transport rewards prefer partial bounding boxes?"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.layout_trial.worker import checked_plan,records
from stvg.metrics import box_iou
from stvg.metrics_base import xyxy
from tools.layout_trial.diagnostics import mean
PUBLIC=ROOT/'results/tubedetr_reward_extent';OUT=ROOT/'outputs/tubedetr_reward_extent';PLAN=PUBLIC/'plan.json'

def run():
    p=json.loads(PLAN.read_text());assert sha(Path(__file__))==p['code_sha256']
    for folder,h in p['parent_final_seals'].items():
        path=ROOT/folder/'final_seal.json';assert sha(path)==h;s=json.loads(path.read_text());assert s['status']=='complete'
        for name,v in s['files'].items():assert sha(ROOT/name)==v,name
    seals={}
    for folder,name in [('tubedetr_layout_trial','diagnostic_seal.json'),('tubedetr_mask_reward_qualification','reward_seal.json'),('tubedetr_correspondence','sealed.json')]:
        path=ROOT/'outputs'/folder/name;assert sha(path)==p['reference_seals'][folder];s=json.loads(path.read_text());assert s['status']=='complete' and s['queries']==128
        for e in s['files'].values():assert sha(ROOT/'outputs'/folder/e['path'])==e['sha256']
        seals[folder]=s
    old=checked_plan(ROOT/'results/tubedetr_layout_trial/plan.json');rows=records(old);lp=ROOT/old['labels'];assert sha(lp)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    names=['Geometry','Layout3','MaskDice','MaskContrast','TransportDice'];details=[]
    for index,row in enumerate(rows):
        truth=labels[row['id']];values={n:[]for n in names};teacher_delta=[]
        paths=[ROOT/'outputs'/folder/seals[folder]['files'][row['id']]['path']for folder in seals]
        with np.load(paths[0])as base,np.load(paths[1])as mask,np.load(paths[2])as trans:
            assert np.array_equal(base['observed_frame_ids'],mask['observed_frame_ids'])and np.array_equal(base['observed_frame_ids'],trans['observed_frame_ids'])
            rewards=dict(Geometry=base['geometry_reward'],Layout3=base['layout3_reward'],MaskDice=mask['MaskDice'],MaskContrast=mask['MaskContrast'],TransportDice=trans['TransportDice'])
            for i,fid in enumerate(base['observed_frame_ids']):
                if str(int(fid))not in truth['boxes']:continue
                gt=np.array(truth['boxes'][str(int(fid))]);pred=xyxy(base['boxes'][i],row['metadata']['width'],row['metadata']['height']);iou=box_iou(pred,gt);teacher_delta.append(float(iou[32]-iou[33]));c=pred[:32]
                inter=np.maximum(np.minimum(c[:,2:],gt[2:])-np.maximum(c[:,:2],gt[:2]),0).prod(-1);gtarea=np.prod(gt[2:]-gt[:2]);area=np.maximum(c[:,2:]-c[:,:2],0).prod(-1)
                logratio=np.log(area/gtarea);coverage=inter/gtarea;precision=inter/area
                metrics=dict(log_area_ratio=logratio,absolute_log_area_ratio=abs(logratio),area_smaller_than_GT=(area<gtarea).astype(float),GT_rectangle_coverage=coverage,box_precision=precision,IoU=iou[:32],IoU_regret=iou[:32].max()-iou[:32])
                for n in names:
                    r=rewards[n][i,:32];tie=np.isclose(r,r.max(),rtol=1e-10,atol=1e-12)
                    values[n].append({k:float(v[tie].mean())for k,v in metrics.items()})
        keys=['log_area_ratio','absolute_log_area_ratio','area_smaller_than_GT','GT_rectangle_coverage','box_precision','IoU','IoU_regret']
        details.append(dict(id=row['id'],eligible_frames=len(teacher_delta),teacher_minus_source=mean(teacher_delta),rewards={n:{k:mean([v[k]for v in values[n]])for k in keys}for n in names}))
        write(OUT/'status.json',dict(status='offline_diagnosis',complete=index+1,total=128,no_GPU=True))
    keys=list(details[0]['rewards']['Geometry'])
    def aggregate(subset):return {n:{k:mean([r['rewards'][n][k]for r in subset])for k in keys}for n in names}
    result=dict(status='complete_offline_diagnosis',queries=128,eligible_queries=sum(r['eligible_frames']>0 for r in details),rewards=aggregate(details),groups={},selection='sealed reward top1, ties averaged, no GT-selected prediction',GT_only_after_all_reward_seals=True,no_new_reward_or_training=True,posthoc_exploratory_not_causal=True,aggregation='mean eligible frames per query then mean eligible queries',correspondence_caveat='Previous .80846 is GT-rectangle containment of matched endpoints, not pixel/object identity truth; boxes may include background')
    for group,fn in dict(teacher_not_better=lambda r:r['teacher_minus_source']is not None and r['teacher_minus_source']<=0,teacher_better=lambda r:r['teacher_minus_source']is not None and r['teacher_minus_source']>0,no_GT_observation=lambda r:r['eligible_frames']==0).items():
        subset=[r for r in details if fn(r)];result['groups'][group]=dict(queries=len(subset),rewards=aggregate(subset))
    for folder in ['tubedetr_mask_reward_qualification','tubedetr_correspondence']:
        prior=json.loads((ROOT/'results'/folder/'summary.json').read_text())
        for n,v in prior['rewards'].items():assert abs(result['rewards'][n]['IoU_regret']-v['top1_IoU_regret'])<1e-12
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    text='# 奖励选择的框范围与STVG矩形目标\n\n完整奖励封存后的独立CPU分析，固定Vid128/同32候选，不生成新预测或奖励。GT框覆盖率不是像素前景完整性；log面积偏移、覆盖/精度须一起解释，不能仅按面积判定损伤。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n即使mask完美，区域Dice与矩形IoU也不等价。例如GT外接矩形面积1、两个相距很远的可见区域总面积.2，完整框Dice为.4/1.2=.3333；只框其中面积.1的区域可得.2/.3=.6667，尽管矩形IoU只有.1。这是反例，不是本数据的发生率或因果证明。\n'
    (PUBLIC/'report.md').write_text(text)
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},publication_pending=True,offline_only=True))
    write(OUT/'status.json',dict(status='complete',queries=128,no_GPU=True))

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(OUT/'failure.json',dict(error=type(e).__name__,message=str(e)));raise
