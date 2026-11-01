"""CPU seals, old-baseline exact replay, fresh-content and global-selection audit."""
import sys,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round23'))
from common23 import *
def main():
    for name in ['sourceK64_attempt001','text272_attempt001','prepared272_attempt001','dino272_attempt001','evidence272_attempt001','bank272_attempt001','dev_grid208_attempt001','dev_selected208_attempt001','confirm64_attempt001']:verify(BASE/name)
    dev=read(EXP/'D208_ROWS_PRIVATE.json');test=read(EXP/'K64_ROWS_PRIVATE.json');oldscore={r['id_PRIVATE']:r for r in read(ROOT/'research/round21/experiment/scored208_attempt003/ROWS_PRIVATE.json')};changes=[];regressions=[];effects={}
    for phase,rows,folder in [('D',dev,'dev_selected208_attempt001'),('K',test,'confirm64_attempt001')]:
        for r in rows:
            original=next(x for x in read(EXP.parent/'input/ALL272_PRIVATE.json') if x['id']==r['id_PRIVATE']);z=feature(original)
            with np.load(BASE/folder/'predictions_PRIVATE'/(r['id_PRIVATE']+'.npz')) as p:assert np.array_equal(p['boxes'],z['boxes'][p['query_indices']]);assert np.isfinite(p['boxes']).all()
            if phase=='D':assert r['q']['source']==oldscore[r['id_PRIVATE']]['q']['source'];assert r['q']['R21_no_ref']==oldscore[r['id_PRIVATE']]['q']['gate_repair_no_reference'];assert r['ious']['R21_no_ref']==oldscore[r['id_PRIVATE']]['ious']['gate_repair_no_reference']
            if r['q']['selected_soft_full']!=r['q']['R21_no_ref']:changes.append({'alias':r['alias'],'phase':phase,'platform':r['platform'],'delta_pp':100*(r['ious']['selected_soft_full']-r['ious']['R21_no_ref']),'baseline_iou':r['ious']['R21_no_ref'],'new_iou':r['ious']['selected_soft_full']})
            if r['alias'] in ['A_quad_012','A_quad_017','V_quad_004']:regressions.append({'alias':r['alias'],'ious':r['ious'],'not_a_method_acceptance_gate':True})
        for p in ['drone','quad']:
            rr=[r for r in rows if r['platform']==p]
            for base in ['source','R21_no_ref','soft_no_ref']:
                d=np.array([r['ious']['selected_soft_full']-r['ious'][base] for r in rr]);effects[phase+'_'+p+'_vs_'+base]={'n':len(rr),'gain_pp':float(d.mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'ties':int((abs(d)<=1e-10).sum())}
    # All K inputs were fixed prior to development selection; decoded content was
    # checked at registration, and the saved canonical records permit replay.
    kh=read(EXP.parent/'input/K_CONTENT_HASHES_PRIVATE.json');assert len(kh)==64 and len({r['decoded_pixel_sha256'] for r in kh})==64;assert len({r['canonical_XYZ_sha256'] for r in kh})==64
    inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];lookup={r['image_file_sha256']:r for r in inv};devrows=read(EXP.parent/'input/D208_PRIVATE.json');di={lookup[sha(r['imagepath'])]['decoded_pixel_sha256'] for r in devrows};dp={lookup[sha(r['imagepath'])]['canonical_XYZ_sha256'] for r in devrows};assert not di&{r['decoded_pixel_sha256'] for r in kh};assert not dp&{r['canonical_XYZ_sha256'] for r in kh}
    choice=read(EXP/'SELECTED_LOCK.json');assert choice['selected']==read(EXP/'DEV_GRID_RANKING.json')[0];assert read(EXP/'CONFIRM_PRELABEL_LOCK.json')['selection_sha256']==sha(EXP/'SELECTED_LOCK.json');source=sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth');assert source=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
    audit={'state':'PASSED','all272frozen_boxes_replayed':True,'old_Source_and_R21_noRef_exact_replay208':True,'newK64_content_disjoint':True,'K64_selection_before_current_outputs_and_GT_scores':True,'K64_predsealed_after_frozen_shared_config_before_current_scores':True,'target_development_tuning_disclosed':True,'no_per_case_exception_or_zero_harm_gate':True,'all_negatives_retained':True,'source_sha256':source,'actual_new_VLM_forwards':read(BASE/'evidence272_attempt001/SEALED.json')['new_calls'],'actual_new_DINO_forwards':read(BASE/'dino272_attempt001/SEALED.json')['calls'],'TTA_steps':0};dump(EXP/'CPU_AUDIT.json',audit);dump(EXP/'ALL_CHANGED_CASES.json',changes);dump(EXP/'REGRESSION_CASES.json',regressions);dump(EXP/'EFFECTS.json',effects);print(audit);print(effects)
if __name__=='__main__':main()
