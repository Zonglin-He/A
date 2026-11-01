"""Freeze CLIP control and layer choices on exposed development, seal M64,
then independent CPU score. No fresh task label affects prediction/selection.
"""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'research/round24')]
from common24 import *
from fit24 import write_predictions
from evaluate24 import score,summary

def evidence(r):
 folder=BASE/'bankM64_attempt001' if r['stage']=='M' else ROOT/'crossplatform/round23/bank272_attempt001'
 return load_npz(folder/'samples_PRIVATE'/(r['id']+'.npz'))
def modality(r,scales=(5,10,20,40)):
 z=feature(r);e=evidence(r);b=load_npz(BASE/'clip336_attempt001/banks_PRIVATE'/(r['id']+'.npz'));qs={'source':int(z['scores'].argmax()),'Qwen_G_direct':int((z['scores']+2*e['vision']+2*e['position']+2*e['relation']).argmax()),'no_Qwen_G_direct':int((z['scores']+2*b['no_qwen_position']+2*b['no_qwen_relation']).argmax())}
 for scale in scales:qs[f'CLIP_G_scale{scale}']=int((z['scores']+2*b[f'clip{scale}_vision']+2*b[f'clip{scale}_position']+2*b[f'clip{scale}_relation']).argmax())
 return qs

def main():
 # All feature/evidence caches are already sealed; reject later mutation.
 for name in ['clip336_attempt001','closureM64_attempt001','memory_attempt001','memory_selected_scoresM64_attempt001','bankM64_attempt001']:verify(BASE/name)
 rr=rows('D208');q=[modality(r) for r in rr];p=BASE/'modal_D208_attempt001';write_predictions(p,rr,{k:np.array([x[k] for x in q]) for k in q[0]});d=score(p,rr);s=summary(d);dump(EXP/'D208_MODAL_ROWS_PRIVATE.json',d);dump(EXP/'D208_MODAL_SUMMARY.json',s);ranking=[]
 for scale in [5,10,20,40]:n=f'CLIP_G_scale{scale}';ranking.append({'scale':scale,'name':n,'balanced_mIoU':float(np.mean([s[p][n]['mIoU_percent'] for p in ['drone','quad']]))})
 ranking.sort(key=lambda x:(-x['balanced_mIoU'],x['scale']));chosen=ranking[0];dump(EXP/'CLIP_SELECTED_LOCK.json',{'selected':chosen,'all_development_trials':ranking,'D208_current_labels_used_to_select':True,'M64_labels_not_used':True,'prediction_seal_sha256':sha(p/'SEALED.json'),'interpretation':'CLIP replacement, not no-VLM; noQwen retains masks+GroundingDINO+geometry'})
 v=rows('V64');qq=[modality(r,[chosen['scale']]) for r in v];vp=BASE/'modal_V64_attempt001';write_predictions(vp,v,{k:np.array([x[k] for x in qq]) for k in qq[0]});vs=score(vp,v);dump(EXP/'V64_MODAL_ROWS_PRIVATE.json',vs);dump(EXP/'V64_MODAL_SUMMARY.json',summary(vs))
 layer=read(EXP/'LAYER_SELECTED_LOCK.json')['best']['layer'];memsel=read(EXP/'MEMORY_SELECTED_LOCK.json');rr=rows('M64');out=BASE/'finalM64_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 for r in rr:
  z=feature(r);q=modality(r,[chosen['scale']]);q['CLIP_G_direct']=q.pop(f'CLIP_G_scale{chosen["scale"]}');c=load_npz(BASE/'closureM64_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));names=c['arm_names'].tolist()
  for name in names:
   if name!='source':q[name+'_transfer']=int(c['query_indices'][names.index(name)])
  memory=load_npz(BASE/'memory_selected_scoresM64_attempt001/samples_PRIVATE'/(r['id']+'.npz'))['scores'];q['same_A_cache_transfer']=int(memory.argmax());e=evidence(r);extra=2*e['vision']+2*e['position']+2*e['relation'];q['selected_TTA_plus_current_G']=int((c[layer+'_scores']+extra).argmax());q['same_A_cache_plus_current_G']=int((memory+extra).argmax());ix=np.array(list(q.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(q)),query_indices=ix,boxes=z['boxes'][ix])
 seal(out,'R24_all_M64_predictions_sealed_before_M64_task_scoring',n=64,all_box_coordinates_frozen=True,CLIP_selection_sha256=sha(EXP/'CLIP_SELECTED_LOCK.json'),layer_selection_sha256=sha(EXP/'LAYER_SELECTED_LOCK.json'),memory_selection_sha256=sha(EXP/'MEMORY_SELECTED_LOCK.json'),M64_not_used_for_parameter_fit=True)
 dump(EXP/'M64_PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(out/'SEALED.json'),'selection_sha256':sha(EXP/'LAYER_SELECTED_LOCK.json'),'historical_full_VAL_exposure':True,'new_RGB_XYZ_content_disjoint':True,'no_current_M_GT_before_seal':True,'no_M_current_G_used_to_fit':True})
 r=score(out,rr);dump(EXP/'M64_ROWS_PRIVATE.json',r);s=summary(r);dump(EXP/'M64_SUMMARY.json',s);print('M64_FINAL',{p:{n:round(v['mIoU_percent'],6) for n,v in d.items()} for p,d in s.items()});print('CLIP_SELECTED',chosen)
if __name__=='__main__':main()
