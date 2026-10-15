#!/usr/bin/env python3
"""Run the sealed 32-source mu=.25 LR-x2 Soft EP comparison; no GT access."""
from __future__ import annotations
import argparse, collections, copy, gc, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_recovery_common_v1 import read,write,sha,guard,load_case,save_pt,PARENT
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
LR1=ROOT/'artifacts/stvg_soft_ep_opd_v1/mu025'

def verify_runtime():
    d=read(BASE/'RUNTIME.json')
    assert d.get('status')=='locked' and d.get('GT_read') is False
    for rel,digest in {**d['pins'],**d['protected']}.items(): assert sha(ROOT/rel)==digest,rel
    return d

def _parent_filemap(path):
    doc=read(path); assert doc.get('status')=='complete' and doc.get('count')==32 and doc.get('GT_read') is False
    out={x['path']:x for x in doc['files']}; assert len(out)==32
    return out

def _fit_cache(base,cid,arm,fit_fn):
    d=BASE/'partial'/cid/arm
    if d.exists(): raise FileExistsError(f'refusing stale/partial fit directory: {d}')
    d.mkdir(parents=True)
    def checkpoint(payload): save_pt(d/f"step{int(payload['next_step'])}.pt",payload)
    return fit_fn(checkpoint)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['development'])
    args=ap.parse_args()
    import numpy as np
    import torch
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source_helper
    from scripts.run_stvg_table1_student004 import stable_fit_key
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
    from scripts.stvg_reference_recovery_fit_v1 import locked_config
    from scripts.stvg_soft_ep_fit_v1 import fit as fit_soft
    from scripts.stvg_weightfree_coordinate_fit_v1 import fit_validated

    verify_runtime(); sys.addaudithook(guard); install_clean_loader(); torch.set_num_threads(2)
    torch.manual_seed(20261015); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    design=read(BASE/'DESIGN_LOCK.json')
    assert design['status']=='locked' and design.get('GT_read') is False
    assert design['response_settings']['mu']==0.25
    refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
    roster=read(BASE/'ROSTER_LOCK.json')['records']; ids=[r['probe_alias'] for r in roster]
    assert len(ids)==32 and len(set(ids))==32 and set(ids)==set(refs)
    parent_files=_parent_filemap(PARENT/'STUDENT_GLOBAL_BARRIER.json')
    lr1_files=_parent_filemap(LR1/'STUDENT_GLOBAL_BARRIER.json')
    expected_files={str((BASE/'student'/f'{cid}.pt').relative_to(ROOT)) for cid in ids}
    if (BASE/'STUDENT_GLOBAL_BARRIER.json').exists(): raise FileExistsError('student barrier already exists; no resume through a stale seal')
    if any((BASE/'student'/f'{cid}.pt').exists() for cid in ids): raise FileExistsError('partial student outputs exist; preserve and review before retry')

    model=source=None; files=[]; started=time.perf_counter(); captures=fit_invocations=readout_calls=0; fit_steps={m:0 for m in ('M1','M2','M0matched')}
    for rec in roster:
        cid=rec['probe_alias']; case,source_path=load_case(rec); ref=refs[cid]
        assert case.get('GT_read') is False and case['source_case_sha256']==ref['source_case_sha256']
        ppath=PARENT/'student'/(cid+'.pt'); prel=str(ppath.relative_to(ROOT))
        assert prel in parent_files and sha(ppath)==parent_files[prel]['sha256']
        parent=torch.load(ppath,map_location='cpu',weights_only=False)
        assert parent.get('GT_read') is False and parent.get('source_case_sha256')==case['source_case_sha256']
        lr1path=LR1/'student'/(cid+'.pt'); lr1rel=str(lr1path.relative_to(ROOT))
        assert lr1rel in lr1_files and sha(lr1path)==lr1_files[lr1rel]['sha256']
        lr1_payload=torch.load(lr1path,map_location='cpu',weights_only=False)
        assert lr1_payload.get('GT_read') is False and lr1_payload.get('source_case_sha256')==case['source_case_sha256']
        assert lr1_payload.get('reference_records')==ref

        arms={name:copy.deepcopy(parent['arms'][old]) for name,old in [('F','F'),('T','T'),('S_M0','S_new_crop'),('TS_M0','TS_new_crop')]}
        for m in ('M1','M2'):
            arms['S_'+m]=None; arms['TS_'+m]=None
        arms['S_M0matched']=None; arms['TS_M0matched']=None
        results={m:None for m in ('M1','M2','M0matched')}; outputs={m:None for m in ('M1','M2','M0matched')}
        source_seconds=0.0; ids_grid=list(map(int,case['frame_ids'])); expected_state=case['source_model_state_sha256']; pixel=case['pixel_sha256']; anchors_json=ref['anchors']

        if anchors_json:
            transient=dict(case,source=source_helper.SOURCE_MODEL_NAME[rec['dataset']],_source_case_sha256=case['source_case_sha256'])
            _,row=source_helper._validate_roster_record(rec); transient['parent']=row['source']
            old_catalog=h.catalog
            def catalog(record):
                assert record['case_id']==cid and record['source_case_sha256']==case['source_case_sha256']
                return transient,{}
            h.catalog=catalog
            try:
                model,source,_,cat,raw,frames,ids_actual,records,batch,base,native,got_state,got_pixel,source_seconds=h._capture_source(
                    dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256']),
                    model,source,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
            finally: h.catalog=old_catalog
            captures+=1
            assert got_pixel==pixel and got_state==expected_state and list(map(int,ids_actual))==ids_grid
            assert np.array_equal(arms['F']['boxes'],h._xyxy(native['boxes'])) and h._same_logits(native['logits'],case['zero']['logits'])
            initial={k:v.clone() for k,v in base.initial.items()}
            anchors=h._anchors(dict(anchors=anchors_json),ids_grid,torch)
            cfg=locked_config(rec['dataset'],'current5'); cfg['lr']=float(cfg['lr'])*2.0
            assert cfg==design['configs'][rec['dataset']]
            settings=design['response_settings']; key=stable_fit_key(rec['dataset'],rec['query_key_sha256'])
            refsha=sha(BASE/'REFERENCE_LOCK.json')
            fit_specs=[('M1',lambda cb:fit_soft(base,initial,anchors,ids_grid,key,cfg,'M1',settings,qualification=False,checkpoint=cb)),
                ('M2',lambda cb:fit_soft(base,initial,anchors,ids_grid,key,cfg,'M2',settings,qualification=False,checkpoint=cb)),
                ('M0matched',lambda cb:fit_validated(base,initial,anchors,ids_grid,key,cfg,validate=True,reference_sha256=refsha,checkpoint_callback=cb))]
            for name,fn in fit_specs:
                f=_fit_cache(base,cid,name,fn); fit_invocations+=1; fit_steps[name]+=int(f.get('optimizer_steps',0))
                base.restore(initial)
                out=h._readout(model,batch,ids_grid,records,f,raw['parses']['subject'],core,torch,
                    transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,
                    frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,
                    query_subject=query_subject)
                readout_calls+=1
                assert state_hash(model.state_dict())==expected_state
                assert h._same_logits(out['logits'],native['logits']) and torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu())
                results[name]=f; outputs[name]=out
                arms['S_'+name]=dict(boxes=h._xyxy(out['boxes']),interval=list(arms['F']['interval']))
                arms['TS_'+name]=dict(boxes=h._xyxy(out['boxes']),interval=list(arms['T']['interval']))
        else:
            for name in results:
                arms['S_'+name]=copy.deepcopy(arms['F']); arms['TS_'+name]=copy.deepcopy(arms['T'])

        # Ensure M0 exact-reuse controls match both sealed parent trials; no new M0 fit is made.
        for dst,src in [('F','F'),('T','T'),('S_M0','S_M0'),('TS_M0','TS_M0')]:
            if src in lr1_payload['arms']:
                assert np.array_equal(arms[dst]['boxes'],lr1_payload['arms'][src]['boxes']) and list(arms[dst]['interval'])==list(lr1_payload['arms'][src]['interval'])
        payload=dict(schema='stvg-soft-ep-lr2-case-v1',case_id=cid,dataset=rec['dataset'],arms=arms,fits=results,readouts=outputs,
            frame_ids=ids_grid,reference_records=ref,source_capture_seconds=source_seconds,source_case_sha256=case['source_case_sha256'],
            source_file_sha256=sha(source_path),source_model_state_sha256=expected_state,pixel_sha256=pixel,
            parent_student_sha256=sha(ppath),lr1_trial_student_sha256=sha(lr1path),lr1_trial_runtime_sha256=sha(LR1/'RUNTIME.json'),
            source_restored=True,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False)
        outpath=BASE/'student'/(cid+'.pt'); item=save_pt(outpath,payload)
        receipt=dict(status='complete',case_id=cid,sha256=sha(outpath),file=item,fit_invocations=sum(x is not None for x in results.values()),
            full_readout_calls=sum(x is not None for x in outputs.values()),GT_read=False)
        write(outpath.with_suffix('.receipt.json'),receipt); files.append(item)
        write(BASE/'STATUS.json',dict(status='running',stage='development',done=len(files),total=32,fit_invocations=fit_invocations,
            fit_steps=fit_steps,source_captures=captures,full_readout_calls=readout_calls,GT_read=False,time=time.time()),mutable=True)
        print('SOFT_EP_LR2',cid,'anchors',len(anchors_json),'fits', {k:(v.get('optimizer_steps') if v else 0) for k,v in results.items()},flush=True)
        gc.collect(); torch.cuda.empty_cache()
    assert len(files)==32 and len({x['path'] for x in files})==32 and {x['path'] for x in files}==expected_files
    assert fit_invocations==69 and captures==23 and readout_calls==69
    barrier=dict(schema='stvg-soft-ep-lr2-student-barrier-v1',status='complete',count=32,files=files,stage='development',
        fit_invocations=fit_invocations,fit_steps=fit_steps,source_captures=captures,full_readout_calls=readout_calls,
        wall_seconds=time.perf_counter()-started,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False)
    write(BASE/'STUDENT_GLOBAL_BARRIER.json',barrier)

if __name__=='__main__': main()
