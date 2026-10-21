"""Input-only workers for native teacher features and fixed-budget adaptation."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from stvg.runtime import setup,lease,sha,ROOT
from stvg.tube_runner import digest,implementation_hash,write,save_prediction
from stvg.initialization import evidence_hash
from stvg.tube_decode import decode
from stvg.inputs import validate_input
from tools.layout_trial.reward import LayoutReward,fit_layout,fit_direct


def checked_plan(path):
    p=json.loads(Path(path).read_text())
    if implementation_hash()!=p['implementation_sha256']:raise ValueError('Frozen implementation changed')
    if sha(ROOT/p['teacher_checkpoint'])!=p['teacher_checkpoint_sha256']:
        raise ValueError('Single ReferDINO checkpoint changed')
    if sha(ROOT/p['configuration']['source_checkpoint'])!=p['configuration']['source_checkpoint_sha256']:
        raise ValueError('TubeDETR source checkpoint changed')
    for name,h in p['code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Layout experiment code changed after freeze')
    return p


def records(plan):
    path=ROOT/plan['manifest'];assert sha(path)==plan['manifest_sha256']
    return [json.loads(s)for s in path.read_text().splitlines()]


def reference(plan):
    root=ROOT/plan['reference_predictions'];seal=json.loads((root/'sealed.json').read_text())
    if seal['queries']!=10303 or seal['manifest_sha256']!=plan['reference_full_manifest_sha256']:
        raise ValueError('Complete full prediction seal with exact population required')
    if seal['identity']['configuration']!=plan['configuration']:
        raise ValueError('Reference paired configuration changed')
    return root,seal


def original(row,root,seal):
    e=seal['files'][row['id']];p=root/e['path'];r=root/e['receipt_path']
    if sha(p)!=e['sha256']or sha(r)!=e['receipt_sha256']:raise ValueError('Reference prediction changed')
    with np.load(p,allow_pickle=False)as z:value={k:z[k].copy()for k in z.files}
    return value,json.loads(r.read_text())


def feature_worker(plan_path):
    p=checked_plan(plan_path);setup();lock=lease();out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=True)
    from stvg.teacher_worker import deny_detector
    sys.addaudithook(deny_detector)
    from stvg.teacher import SingleRefer
    from stvg.tensors import state_hash
    from tools.layout_trial.features import capture
    root,seal=reference(p);teacher=SingleRefer();before=state_hash(teacher.model.state_dict());entries={}
    for index,row in enumerate(records(p)):
        path=out/'features'/f"{row['id']}.npz";receipt=path.with_suffix('.json')
        if path.exists()or receipt.exists():
            if not(path.exists()and receipt.exists()):raise ValueError('Partial teacher cache')
            m=json.loads(receipt.read_text())
            if m['sha256']!=sha(path)or m['request_sha256']!=digest(row):raise ValueError('Teacher cache changed')
        else:
            value,old=original(row,root,seal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
            positions=value['positions8'].astype(int);rgb=frames[positions];answer=capture(teacher,rgb,row['metadata']['caption'])
            if evidence_hash(answer['boxes'].numpy())!=old['initialization']['evidence_sha256']:
                raise ValueError('Feature instrumentation changed single-teacher evidence')
            arrays={name:answer[name].numpy()for name in ['features','boxes','valid_ratio','alternative_boxes']}
            arrays.update(positions8=positions,observed_frame_ids=np.asarray(fids)[positions])
            path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays)
            m=dict(sha256=sha(path),request_sha256=digest(row),evidence_sha256=old['initialization']['evidence_sha256'],
                RGB_sha256=answer['RGB_sha256'],feature_space=answer['feature_space'],feature_shape=list(answer['features'].shape),
                input_shape=answer['input_shape'],native_same_track_all8=True,expert_instances=1,
                feature_hook_preserves_native_boxes_bitexact=True,teacher_cost=answer['cost'],GT_read=False)
            write(receipt,m);del frames,answer,arrays
        entries[row['id']]=dict(path=str(path.relative_to(out)),sha256=sha(path),receipt_path=str(receipt.relative_to(out)),receipt_sha256=sha(receipt))
        write(out/'worker_progress.json',dict(stage='features',complete=index+1,total=128,labels_read=False))
        print(f'Layout features {index+1}/128',flush=True)
    if state_hash(teacher.model.state_dict())!=before:raise ValueError('Teacher weights changed')
    write(out/'features_seal.json',dict(status='complete',queries=128,files=entries,teacher_weights_unchanged=True,expert_instances=1,GT_read=False))
    lock.close()


def load_feature(row,out,seal):
    e=seal['files'][row['id']];path=out/e['path'];receipt=out/e['receipt_path']
    if sha(path)!=e['sha256']or sha(receipt)!=e['receipt_sha256']:raise ValueError('Feature seal mismatch')
    with np.load(path,allow_pickle=False)as z:v={k:z[k].copy()for k in z.files}
    return v


def adapt_worker(plan_path):
    p=checked_plan(plan_path);setup();lock=lease();out=ROOT/p['output'];root,oldseal=reference(p)
    fseal=json.loads((out/'features_seal.json').read_text());cal=json.loads((out/'temperature_calibration.json').read_text())
    if not cal['target_GT_used']is False or cal['feature_seal_sha256']!=sha(out/'features_seal.json'):
        raise ValueError('Temperature calibration must be unlabeled and feature-bound')
    from stvg.tubedetr import load_source,native_forward,TubeReplay
    from stvg.policy import fit
    from stvg.tensors import state_hash
    from tools import tubedetr_vid_development as shared
    model=load_source(p['configuration']);before=state_hash(model.state_dict());files={name:{}for name in p['adaptation_arms']};qualified=False
    for index,row in enumerate(records(p)):
        old,receipt=original(row,root,oldseal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        native=native_forward(model,frames,fids,row['metadata']['caption'],224)
        if not np.array_equal(native['source'].numpy(),old['Source'])or native['interval']!=old['Native_interval'].tolist():raise ValueError('Native source interface differs')
        f=load_feature(row,out,fseal);positions=old['positions8'].astype(int)
        if not np.array_equal(positions,f['positions8'])or not np.array_equal(np.asarray(fids)[positions],f['observed_frame_ids']):raise ValueError('Physical observation alignment changed')
        if hashlib.sha256(frames[positions].tobytes()).hexdigest()!=json.loads((out/fseal['files'][row['id']]['receipt_path']).read_text())['RGB_sha256']:raise ValueError('Teacher/student RGB differs')
        if not qualified:
            replay=TubeReplay(model,native,old['E_Ref'],positions);check=fit(replay,f['boxes'],row['key'],'Live',41,p['configuration'])
            if not np.array_equal(check['final'].numpy(),old['Live_raw']):raise ValueError('Frozen geometric OPD reference replay differs; stop before reward comparison')
            write(out/'qualification.json',dict(status='pass',geometric_OPD_replay_bitexact=True,physical_frames_bitexact=True,feature_hook_teacher_boxes_bitexact=True,GT_read=False));qualified=True;del replay,check
        features=torch.from_numpy(f['features']).cuda();target=torch.from_numpy(f['boxes']).cuda().double()
        for name in p['adaptation_arms']:
            dest=out/'predictions'/name;path=dest/'predictions'/f"{row['id']}.npz";cfg=dict(p['configuration'])
            cfg.update(reward=name,plan_sha256=sha(plan_path),temperature_calibration_sha256=sha(out/'temperature_calibration.json'))
            if name.endswith('OPD'):cfg['tau']=cal['layout1'if name=='Layout1OPD'else'layout3']['tau']
            identity=dict(configuration=cfg,implementation_sha256=p['implementation_sha256'],manifest_sha256=p['manifest_sha256'],scope='fixed128_layout_reward_pilot',plan_sha256=sha(plan_path))
            if shared.saved(dest,row,identity):
                files[name][row['id']]=True;continue
            replay=TubeReplay(model,native,old['E_Ref'],positions);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            if name=='GeometryDirect':result=fit_direct(replay,f['boxes'],cfg);reward=None
            else:
                bins=1 if name=='Layout1OPD'else 3
                reward=LayoutReward(features,target,bins,torch.from_numpy(f['valid_ratio']).cuda())
                if name.endswith('OPD'):
                    identity['configuration']=cfg
                    result=fit_layout(replay,f['boxes'],row['key'],41,cfg,reward)
                else:result=fit_direct(replay,f['boxes'],cfg,reward)
            if replay.state.count_nonzero()or any(v.requires_grad or v.grad is not None for v in model.parameters()):raise ValueError('Frozen model or per-query reset violated')
            if 'rounds'in result:
                norms=[float(v['gradient'].norm())if v['gradient']is not None else 0. for v in result['rounds']]
                gradient=dict(mean_state_gradient_norm=float(np.mean(norms)),nonzero_gradient_steps=sum(v>0 for v in norms))
            else:gradient=dict(mean_state_gradient_norm=result['gradient_norm_mean'],any_nonzero_gradient=result['gradient_nonzero'])
            value={**old,'Live_raw':result['final'].numpy(),'Live':result['final'].numpy(),'checkpoint':model.load_audit,
                'initialization':receipt['initialization'],'cost':dict(adaptation_seconds=time.perf_counter()-tick,actual_updates=result['actual_updates'],
                    pseudo_loss_before=result['path'][0]['anchor_loss'],pseudo_loss_after=result['path'][-1]['anchor_loss'],
                    reward=name,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**gradient,
                    Strong_exact_cache_reuse=True,not_fresh_end_to_end_timing=True,GT_read=False)}
            save_prediction(path,value,dict(identity_sha256=digest(identity),request_sha256=digest(row)))
            write(dest/'run.json',identity)if not(dest/'run.json').exists()else None
            files[name][row['id']]=True;del result,replay,reward
        del native,frames,features,target
        write(out/'worker_progress.json',dict(stage='adaptation',complete=index+1,total=128,labels_read=False))
        print(f'Layout adaptation {index+1}/128',flush=True)
    if state_hash(model.state_dict())!=before:raise ValueError('Frozen student weights changed')
    for name in p['adaptation_arms']:
        dest=out/'predictions'/name;identity=json.loads((dest/'run.json').read_text());shared.seal(dest,records(p),identity)
    write(out/'adaptation_receipt.json',dict(status='complete',queries=128,arms=p['adaptation_arms'],frozen_student_state_unchanged=True,per_query_reset=True,GT_read=False))
    lock.close()
