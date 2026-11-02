"""Seal three fixed-box full readouts; no task labels or model forwards."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from full_common50 import ROOT,EXP,read,sha,dump,npz,seal
from pool_weights50 import legacy_operator,exact
sys.path.insert(0,str(ROOT/'research/round49/independent_audit'))
from reference49 import extract_original_relation,image_only_scene
ARMS=['source','old_G','wideentry_oldW']
SEALS={}
def member(folder,relative):
    folder=Path(folder)
    if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
    p=folder/relative;assert sha(p)==SEALS[folder]['files'][relative]
    return p
def main():
    auth=read(EXP/'FULL_PREDICTION_AUTHORIZATION.json');pp=EXP/'FULL_PREDICTION_PLAN_LOCK.json'
    assert auth['state']=='ROOT_AUTHORIZED_R50_FULL5912_THREE_FIXED_BOX_READOUTS'
    assert auth['plan_sha256']==sha(pp);plan=read(pp)
    assert plan['arms']==ARMS and plan['n']==5912 and plan['no_GT_or_model_or_optimizer']
    for group in ('programs_sha256','inputs_sha256'):
        for rel,h in plan[group].items():
            assert 'FINAL384' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h
    for rel,h in plan['sealed_roots'].items():assert sha(ROOT/rel/'SEALED.json')==h
    runroot=ROOT/plan['runroot'];rows=read(ROOT/plan['manifest']);assert len(rows)==5912
    index=read(ROOT/plan['evidence_case_index']);out=ROOT/plan['output']
    assert not out.exists();out.mkdir(parents=True)
    for d in ('samples_PRIVATE','cases_PRIVATE'):(out/d).mkdir()
    relation=extract_original_relation((ROOT/'crossplatform/round23/rank23.py').read_text(),(ROOT/'crossplatform/round20/detect.py').read_text())
    devrows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
    identity=lambda r:tuple(r[k] for k in ('imagepath','pointpath','metapath','caption'))
    dev={identity(r):r for r in devrows};assert len(dev)==336
    bridges=0;counts={'n':0,'old_pool_masks':0,'wide_pool_masks':0};start=time.perf_counter()
    try:
        for i,row in enumerate(rows):
            sid=row['id'];ap=member(runroot/'prepared','arrays_PRIVATE/'+sid+'.npz');cp=member(runroot/'prepared','cases_PRIVATE/'+sid+'.json');tp=member(runroot/'text','cases_PRIVATE/'+sid+'.json')
            a=npz(ap);c=read(cp);g=read(tp);ref=index[sid];ep=ROOT/ref['file'];assert sha(ep)==ref['sha256']
            covered=False
            for rel,digest in plan['sealed_roots'].items():
                folder=ROOT/rel
                if ep.is_relative_to(folder):
                    assert sha(folder/'SEALED.json')==digest
                    assert member(folder,ep.relative_to(folder).as_posix())==ep
                    covered=True;break
            assert covered
            evidence=read(ep);old=a['old_pool'];wide=a['wide_pool'];s=a['source_scores'];boxes=a['source_boxes']
            assert c['mask_pool_PRIVATE']==old.tolist() and c['wide_pool_PRIVATE']==wide.tolist()
            assert hashlib.sha256(g['subject_query'].encode()).hexdigest()==c['actual_subject_query_sha256']==evidence['actual_subject_query_sha256']
            assert set(evidence['targets_PRIVATE'])==set(map(str,wide))
            mp=member(runroot/'frames','masks_PRIVATE/'+sid+'.npz');masks=npz(mp);W,H,objects=image_only_scene(masks);M=len(objects)
            e=np.full(M,np.nan);pr=np.zeros((M,2))
            for m in wide:
                vv=evidence['targets_PRIVATE'][str(m)];assert len(vv)==2
                for v in vv:
                    assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
                    assert all(v[mode]['generated_tokens']==0 and np.isfinite(v[mode]['log_odds_yes_no']) for mode in ('full','gray'))
                e[m]=np.mean([v['visual_evidence'] for v in vv]);pr[m]=relation(g,objects[m],evidence,W,H)
            assert np.isfinite(e[wide]).all()
            oldstate=legacy_operator(s,a['old_weights'],old,e,pr)
            newstate=legacy_operator(s,a['wide_old_weights'],wide,e,pr)
            scores=np.stack([s.astype(float),oldstate['scores'],newstate['scores']]);q=np.argmax(scores,axis=1).astype(np.int64)
            selected=boxes[q];devrow=dev.get(identity(row))
            if devrow:
                before=npz(ROOT/'crossplatform/round50/predicted336_attempt001/samples_PRIVATE'/(devrow['id']+'.npz'))
                exact(scores,before['scores'][:3],'fullDEV_three_scores');exact(q,before['queries'][:3],'fullDEV_three_queries');exact(selected,before['selected_original_boxes'][:3],'fullDEV_three_boxes');bridges+=1
            np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS),scores=scores,queries=q,selected_original_boxes=selected,
                source_boxes=boxes,source_scores=s,old_pool=old,wide_pool=wide,old_weights=a['old_weights'],wide_old_weights=a['wide_old_weights'],image_e=e,raw_mask_PR=pr,
                vision=np.stack([np.zeros(256),oldstate['vision'],newstate['vision']]),position=np.stack([np.zeros(256),oldstate['position'],newstate['position']]),relation=np.stack([np.zeros(256),oldstate['relation'],newstate['relation']]))
            dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'platform':row['platform'],'old_pool_count':len(old),'wide_pool_count':len(wide),'is_DEV336':bool(devrow),'evidence_file_sha256':sha(ep),'geometry_file_sha256':sha(ap),'text_file_sha256':sha(tp),'mask_file_sha256':sha(mp),'all_selected_boxes_original':True})
            counts['n']+=1;counts['old_pool_masks']+=len(old);counts['wide_pool_masks']+=len(wide)
            if (i+1)%128==0:print('FULL_PREDICT50',i+1,flush=True)
    except BaseException as exc:
        dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':counts['n'],'no_GT_read_or_automatic_retry':True});raise
    assert counts['n']==5912 and bridges==336
    dump(out/'COUNTS.json',dict(counts,DEV336_three_score_and_original_box_bridges_bitexact=bridges))
    seal(out,'R50_FULL5912_THREE_FIXED_BOX_READOUTS_BEFORE_TASK_SCORE',n=5912,arms=ARMS,DEV336_bitexact_bridges=bridges,seconds=time.perf_counter()-start,plan_sha256=sha(pp),no_GT_models_optimizer_or_new_boxes=True)
if __name__=='__main__':main()
