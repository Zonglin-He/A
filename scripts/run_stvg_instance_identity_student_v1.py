"""Single-source matched no-label identity/update worker; old artifacts immutable."""
import sys,os,copy,collections,time,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha,verify,guard,row

def same_anchors(a,b,torch,weights=False):
    if len(a)!=len(b):return False
    return all(x['position']==y['position'] and x['physical_frame_id']==y['physical_frame_id']
        and torch.equal(x['box'],y['box']) and (not weights or x['weight']==y['weight']) for x,y in zip(a,b))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();verify();sys.addaudithook(guard);rec=row(a.case)
    import torch,numpy as np
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for,frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward,transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_native_instance_spatial_fit_v1 import fit_spatial_instance
    from scripts.stvg_instance_validated_fit_v1 import fit_validated
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    def catalog(record):
        path=ROOT/record['source_case_path'];assert sha(path)==record['source_case_sha256']
        return load_source_case(path),{}
    h.catalog=catalog
    install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,_,case,cat,raw,frames,ids,records,batch,base,native,expected,pixel,capture_s=h._capture_source(rec,None,None,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
    oldpath=OLD/'student'/(a.case+'.pt');assert sha(oldpath)==read(oldpath.with_suffix('.receipt.json'))['sha256']
    old=torch.load(oldpath,map_location='cpu',weights_only=False)
    assert old['GT_read'] is False and old['frame_ids']==ids and old['pixel_sha256']==pixel and old['source_model_state_sha256']==expected
    boxes=h._xyxy(native['boxes']);assert np.array_equal(np.asarray(old['arms']['F']['boxes']),boxes)
    ref=next(x for x in read(BASE/'REFERENCE_LOCK.json')['records'] if x['case_id']==a.case)
    assert ref['source_case_sha256']==rec['source_case_sha256']
    rawref=ref['raw_decoupled'];supref=ref['presence_supported']
    rawanchors=h._anchors(rawref,ids,torch);supanchors=h._anchors(supref,ids,torch)
    # Presence fraction controls loss contribution; raw adjacent mask IoU is not used.
    assert len(supanchors)==len(supref['anchors'])
    for anchor,r in zip(supanchors,supref['anchors']):
        assert anchor['position']==r['position'] and anchor['physical_frame_id']==r['physical_frame_id']
        anchor['weight']=float(r['reliability']['presence_fraction'])
        assert 0<anchor['weight']<=1
    oldanchors=h._anchors(old['reference_record'],ids,torch)
    arms={k:copy.deepcopy(old['arms'][k]) for k in ['F','T']}
    arms['strict_iou']=copy.deepcopy(old['arms']['iou']);arms['T_strict_iou']=copy.deepcopy(old['arms']['T_iou'])
    fits={};outs={};receipts={};cfg=core.load_original_opd_config(rec['dataset']);cfg['steps']=5
    initial={k:v.detach().clone() for k,v in base.initial.items()}
    for name,anchors,gate in [('raw_iou',rawanchors,None),('supported_iou',supanchors,None),('loss_weighted_iou',supanchors,False),('validated_loss_weighted_iou',supanchors,True)]:
        if not anchors:
            arms[name]=copy.deepcopy(arms['F']);arms['T_'+name]=copy.deepcopy(arms['T'])
            receipts[name]=dict(status='exact_no_update',optimizer_steps=0,proposed_optimizer_steps=0,final_redecode_calls=0);continue
        if gate is None and same_anchors(anchors,oldanchors,torch):
            arms[name]=copy.deepcopy(old['arms']['iou']);arms['T_'+name]=copy.deepcopy(old['arms']['T_iou'])
            receipts[name]=dict(status='exact_prior_fit_reuse',source_sha256=sha(oldpath),optimizer_steps=0,proposed_optimizer_steps=0,final_redecode_calls=0);continue
        if name=='supported_iou' and same_anchors(anchors,rawanchors,torch):
            arms[name]=copy.deepcopy(arms['raw_iou']);arms['T_'+name]=copy.deepcopy(arms['T_raw_iou'])
            receipts[name]=dict(status='exact_current_arm_reuse',source_arm='raw_iou',optimizer_steps=0,proposed_optimizer_steps=0,final_redecode_calls=0);continue
        started=time.perf_counter();partial=BASE/'partial'/a.case/name;partial.mkdir(parents=True,exist_ok=True)
        def checkpoint(payload):
            dst=partial/('step'+str(payload['next_step'])+'.pt');assert not dst.exists();torch.save(payload,dst)
        args=dict(base=base,initial={k:v.clone() for k,v in initial.items()},anchors=anchors,frame_ids=ids,key='caption_query_fixed_space|'+a.case,config=cfg,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),checkpoint_callback=checkpoint)
        fit=fit_spatial_instance(**args,algorithm='iou') if gate is None else fit_validated(**args,validate=gate)
        out=h._readout(model,batch,ids,records,fit,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject)
        assert h._same_logits(out['logits'],native['logits']) and state_hash(model.state_dict())==expected
        assert torch.equal(out['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu())
        b=h._xyxy(out['boxes']);arms[name]=h._arm(b,arms['F']['interval']);arms['T_'+name]=h._arm(b,arms['T']['interval'])
        fits[name]=fit;outs[name]=out
        receipts[name]=dict(status='actual_final_redecode',optimizer_steps=fit['optimizer_steps'],proposed_optimizer_steps=fit.get('proposed_optimizer_steps',fit['optimizer_steps']),final_redecode_calls=1,seconds=time.perf_counter()-started)
    result=dict(case_id=a.case,dataset=rec['dataset'],source_case_sha256=rec['source_case_sha256'],frame_ids=ids,pixel_sha256=pixel,source_model_state_sha256=expected,initial_state=core._copy_cpu(initial),source_boxes=base.zero['boxes'].detach().cpu(),source_restored=True,
        arms=arms,fits=fits,readouts=outs,receipts=receipts,reference_record=ref,reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),configuration=cfg,source_capture_seconds=capture_s,GT_read=False,online_carry=False,
        actual_fit_count=len(fits),optimizer_steps=sum(x['optimizer_steps'] for x in receipts.values()),proposed_optimizer_steps=sum(x['proposed_optimizer_steps'] for x in receipts.values()),actual_final_redecode_calls=len(outs))
    dst=BASE/'student'/(a.case+'.pt');dst.parent.mkdir(exist_ok=True);assert not dst.exists();torch.save(result,dst)
    write(dst.with_suffix('.receipt.json'),dict(status='sealed',path=str(dst.relative_to(ROOT)),sha256=sha(dst),bytes=dst.stat().st_size,actual_fit_count=len(fits),optimizer_steps=result['optimizer_steps'],proposed_optimizer_steps=result['proposed_optimizer_steps'],final_redecode_calls=len(outs),GT_read=False,time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'STUDENT_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
