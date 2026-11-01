"""One fixed development grid, frozen selection, then sealed new-input scoring."""
import sys,argparse,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round23'),str(ROOT/'research/round5/audit')]
from common23 import *
from rank23_v2 import soft_score,prior_score
from independent_core_v2 import transform_gt,candidate_ious
ARMS=['source','EDA_raw','main_attr','R21_no_ref','selected_soft_full','soft_no_ref','soft_no_token','soft_no_vision','soft_legacy_pool','soft_no_prior']
def best(score,source):return int(np.lexsort((np.arange(len(score)),-source,-score))[0])
def bank(r):
    with np.load(BASE/'bank272_attempt001/samples_PRIVATE'/(r['id']+'.npz')) as p:e={k:p[k].copy() for k in p.files}
    e['legacy']=read(BASE/'bank272_attempt001/cases_PRIVATE'/(r['id']+'.json'))['legacy_PRIVATE'];return e
def controls(z,e,cfg):
    component=e['component'];s=soft_score(z,e,cfg);out={'source':int(np.argmax(z['scores'])),'EDA_raw':best(component[:,:4].sum(1)-component[:,4],z['scores']),'main_attr':best(z['scores']+.2*component[:,1],z['scores']),'R21_no_ref':int(e['R21_no_ref']),'selected_soft_full':best(s,z['scores'])}
    for arm,kw in [('soft_no_ref',{'relation':False}),('soft_no_token',{'token':False}),('soft_no_vision',{'vision':False}),('soft_legacy_pool',{'restricted':True})]:out[arm]=best(soft_score(z,e,cfg,**kw),z['scores'])
    out['soft_no_prior']=best(s-prior_score(z,cfg),z['scores']);return out
def summarize(rows,reference):
    out={}
    for p in ['drone','quad']:
        rr=[r for r in rows if r['platform']==p];scenes=sorted({r['scene_PRIVATE'] for r in rr});ix=np.array([scenes.index(r['scene_PRIVATE']) for r in rr]);weights=np.eye(len(scenes),dtype=int)[np.random.default_rng(2023).integers(0,len(scenes),(10000,len(scenes)))].sum(1);counts=np.bincount(ix);out[p]={}
        for arm in rr[0]['ious']:
            v=np.array([r['ious'][arm] for r in rr]);base=np.array([r['ious'][reference] for r in rr]);delta=v-base;bs=weights@np.bincount(ix,weights=delta,minlength=len(scenes))/(weights@counts)*100;out[p][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_reference_pp':float(delta.mean()*100),'gains_vs_reference':int((delta>1e-10).sum()),'harms_vs_reference':int((delta< -1e-10).sum()),'changed_vs_reference':sum(r['q'][arm]!=r['q'][reference] for r in rr),'scene_clusters':len(scenes),'equal_scene_mIoU_percent':float(np.mean([v[ix==i].mean() for i in range(len(scenes))])*100),'scene_cluster95_gain_pp':list(map(float,np.quantile(bs,[.025,.975])))}
    return out
def write_predictions(out,rows,predict):
    out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
    for r in rows:
        z=feature(r);e=bank(r);selected=predict(z,e);ix=np.array(list(selected.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(selected)),query_indices=ix,boxes=z['boxes'][ix])
    seal(out,'R23_direct_prediction_sealed_before_current_task_scoring',n=len(rows),optimizer_steps=0)
def score(out,rows):
    verify(out);index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};result=[];containers={}
    for r in rows:
        ix=index[r['id']];path=ix['meta_path'];assert sha(path)==ix['metadata_container_sha256']
        if path not in containers:containers[path]=read(path)
        gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'))
        with np.load(out/'predictions_PRIVATE'/(r['id']+'.npz')) as p:names=p['arm_names'].tolist();q=p['query_indices'];boxes=p['boxes'].copy()
        assert np.array_equal(boxes,feature(r)['boxes'][q]);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);ious=candidate_ious(boxes,gt)[0];result.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'q':dict(zip(names,map(int,q))),'ious':dict(zip(names,map(float,ious)))})
    return result
def development():
    lock('EVALUATE_v2_LOCK.json');verify(BASE/'bank272_attempt001');rows=read(EXP.parent/'input/D208_PRIVATE.json');configs=read(EXP/'G_SPEC_v2_LOCK.json')['weights'];out=BASE/'dev_grid208_attempt002'
    def predict(z,e):
        c=e['component'];s={'source':int(np.argmax(z['scores'])),'R21_no_ref':int(e['R21_no_ref']),'EDA_raw':best(c[:,:4].sum(1)-c[:,4],z['scores']),'main_attr':best(z['scores']+.2*c[:,1],z['scores'])}
        for i,cfg in enumerate(configs):s[f'config_{i:02d}']=best(soft_score(z,e,cfg),z['scores'])
        return s
    write_predictions(out,rows,predict);scored=score(out,rows);summary=summarize(scored,'R21_no_ref');ranking=[]
    for i,c in enumerate(configs):
        arm=f'config_{i:02d}';values=[summary[p][arm] for p in ['drone','quad']];ranking.append({'config_index':i,'configuration':c,'balanced_mIoU':float(np.mean([v['mIoU_percent'] for v in values])),'meanAcc25plus50':float(np.mean([v['Acc25_percent']+v['Acc50_percent'] for v in values]))})
    ranking=sorted(ranking,key=lambda x:(-x['balanced_mIoU'],-x['meanAcc25plus50'],sum(v for v in x['configuration'].values() if isinstance(v,(int,float))),x['config_index']));selected=ranking[0];dump(EXP/'DEV_GRID_v2_ROWS_PRIVATE.json',scored);dump(EXP/'DEV_GRID_v2_SUMMARY.json',summary);dump(EXP/'DEV_GRID_v2_RANKING.json',ranking);dump(EXP/'SELECTED_v2_LOCK.json',{'selected':selected,'shared_across_platforms':True,'selected_by_exposed_D208_labels':True,'selection_before_K64_task_labels':True,'no_zero_harm_requirement':True,'grid_prediction_seal_sha256':sha(out/'SEALED.json')});print('SELECTED23',selected)
def confirm():
    lock('EVALUATE_v2_LOCK.json');selection=read(EXP/'SELECTED_v2_LOCK.json');cfg=selection['selected']['configuration'];out=BASE/'confirm64_attempt001';rows=read(EXP.parent/'input/K64_PRIVATE.json');write_predictions(out,rows,lambda z,e:controls(z,e,cfg));dump(EXP/'CONFIRM_PRELABEL_LOCK.json',{'selection_sha256':sha(EXP/'SELECTED_v2_LOCK.json'),'prediction_seal_sha256':sha(out/'SEALED.json'),'current_K64_no_GT_before_predictions':True,'historical_VAL_exposure':True});scored=score(out,rows);summary=summarize(scored,'R21_no_ref');dump(EXP/'K64_ROWS_PRIVATE.json',scored);dump(EXP/'K64_SUMMARY.json',summary)
    # Same selected weights on development, with matched controls, retained separately.
    drows=read(EXP.parent/'input/D208_PRIVATE.json');dout=BASE/'dev_selected208_attempt002';write_predictions(dout,drows,lambda z,e:controls(z,e,cfg));ds=score(dout,drows);dump(EXP/'D208_ROWS_PRIVATE.json',ds);dump(EXP/'D208_SUMMARY.json',summarize(ds,'R21_no_ref'));print('CONFIRM23',{p:{a:round(v['mIoU_percent'],6) for a,v in vv.items()} for p,vv in summary.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','confirm']);a=p.parse_args();development() if a.stage=='development' else confirm()
