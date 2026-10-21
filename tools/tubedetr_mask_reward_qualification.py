"""Bounded native-mask signal qualification; no student adaptation or tuning.

Capture -> reward bank -> independent offline score in separate processes.
"""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha,setup,lease
from stvg.tube_runner import write,digest
from tools.layout_trial.worker import checked_plan,records,reference,original
from tools.mask_reward import MaskAreaReward,box_overlap

PUBLIC=ROOT/'results/tubedetr_mask_reward_qualification';OUT=ROOT/'outputs/tubedetr_mask_reward_qualification';PLAN=PUBLIC/'plan.json'
CACHE=ROOT/'outputs/tubedetr_layout_trial';PILOT=ROOT/'results/tubedetr_layout_trial'


def inputs():
    p=json.loads(PLAN.read_text())
    for name,h in p['code_sha256'].items():assert sha(ROOT/name)==h,name
    for folder,h in p['parent_final_seals'].items():
        f=ROOT/folder/'final_seal.json';assert sha(f)==h;seal=json.loads(f.read_text());assert seal['status']=='complete'
        for name,v in seal['files'].items():assert sha(ROOT/name)==v,name
    assert sha(CACHE/'diagnostic_seal.json')==p['reference_diagnostic_seal_sha256']
    assert sha(CACHE/'features_seal.json')==p['reference_feature_seal_sha256']
    old=checked_plan(PILOT/'plan.json');rows=records(old);assert len(rows)==128
    return p,old,rows


def save_npz(path,**values):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**values)
    return dict(path=str(path.relative_to(OUT)),sha256=sha(path))


def read_npz(seal,q):
    e=seal['files'][q];assert sha(OUT/e['path'])==e['sha256']
    with np.load(OUT/e['path'],allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}


def masks():
    p,old,rows=inputs();setup();lock=lease()
    from stvg.teacher import SingleRefer
    from stvg.teacher_worker import deny_detector
    from stvg.tensors import state_hash
    from stvg.initialization import evidence_hash
    from stvg.tube_decode import decode
    from stvg.inputs import validate_input
    sys.addaudithook(deny_detector)
    teacher=SingleRefer();before=state_hash(teacher.model.state_dict());root,oldseal=reference(old)
    featureseal=json.loads((CACHE/'features_seal.json').read_text());entries={};cost=[]
    for i,row in enumerate(rows):
        value,receipt=original(row,root,oldseal);frames,fids=decode(row['metadata'],'vidstg');validate_input(frames,fids,row['metadata'])
        positions=value['positions8'].astype(int);rgb=frames[positions];fe=featureseal['files'][row['id']]
        assert sha(CACHE/fe['receipt_path'])==fe['receipt_sha256'];prior=json.loads((CACHE/fe['receipt_path']).read_text())
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==prior['RGB_sha256']
        captured=[];native=teacher.model.infer
        def capture(*args,**kwargs):
            out=native(*args,**kwargs);captured.append(out);return out
        teacher.model.infer=capture
        try:
            torch.manual_seed(42);answer=teacher(rgb,row['metadata']['caption'])
        finally:teacher.model.infer=native
        assert len(captured)==1 and evidence_hash(answer['boxes'].numpy())==receipt['initialization']['evidence_sha256']
        selected=answer['selected_query'];others=[int(q)for q in answer['native_query_scores'].argsort(descending=True)if int(q)!=selected][:3]
        raw=captured[0];indices=[selected,*others]
        prob=raw['pred_masks'][0,:,indices].sigmoid().float().cpu()
        boxes=raw['pred_boxes'][0,:,indices].float().cpu();h,w=answer['input_shape'][-2:]
        assert prob.shape[:2]==(8,len(indices)) and list(prob.shape[-2:])==prior['feature_shape'][-2:]
        assert torch.isfinite(prob).all() and torch.equal(boxes[:,0],answer['boxes'])
        ratio=np.tile([w/(4*prob.shape[-1]),h/(4*prob.shape[-2])],(8,1)).astype(np.float32)
        entries[row['id']]=save_npz(OUT/'masks'/f"{row['id']}.npz",masks=prob.numpy(),boxes=boxes.numpy(),valid_ratio=ratio,
            selected_frame_scores=answer['selected_frame_scores'].reshape(8,-1).max(-1).values.numpy(),
            observed_frame_ids=np.array(fids)[positions],positions8=positions)
        entries[row['id']].update(request_sha256=digest(row),evidence_sha256=receipt['initialization']['evidence_sha256'],RGB_sha256=prior['RGB_sha256'])
        cost.append(answer['cost']);del frames,raw,captured,answer,prob,boxes
        write(OUT/'worker_progress.json',dict(stage='masks',complete=i+1,total=128,labels_read=False));print('Native mask evidence',i+1,flush=True)
    assert state_hash(teacher.model.state_dict())==before
    write(OUT/'mask_seal.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),labels_read=False,
        teacher_weights_unchanged=True,old_boxes_and_RGB_bitexact=True,expert_instances=1,expert_forwards=128,
        mean_expert_seconds=float(np.mean([c['expert_seconds']for c in cost]))));lock.close()


def bank():
    p,old,rows=inputs();setup();ms=json.loads((OUT/'mask_seal.json').read_text());assert ms['status']=='complete' and ms['queries']==128 and ms['plan_sha256']==sha(PLAN)
    prior=json.loads((CACHE/'diagnostic_seal.json').read_text());entries={}
    for i,row in enumerate(rows):
        m=read_npz(ms,row['id']);e=prior['files'][row['id']];assert sha(CACHE/e['path'])==e['sha256']
        with np.load(CACHE/e['path'],allow_pickle=False)as z:
            candidates=torch.from_numpy(z['boxes'].copy());assert np.array_equal(z['observed_frame_ids'],m['observed_frame_ids'])
        b=torch.from_numpy(m['boxes']);eligible=box_overlap(b[:,0],b[:,1:])<p['duplicate_box_IoU_threshold']
        reward=MaskAreaReward(torch.from_numpy(m['masks']),torch.from_numpy(m['valid_ratio']),eligible)
        scores={k:v.detach().numpy()for k,v in reward.scores(candidates).items()}
        entries[row['id']]=save_npz(OUT/'rewards'/f"{row['id']}.npz",**scores,selected_frame_scores=m['selected_frame_scores'],
            observed_frame_ids=m['observed_frame_ids'],eligible_alternative_count=eligible.sum(-1).numpy())
        write(OUT/'worker_progress.json',dict(stage='bank',complete=i+1,total=128,labels_read=False))
    write(OUT/'reward_seal.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),mask_seal_sha256=sha(OUT/'mask_seal.json'),labels_read=False))


def score():
    p,old,rows=inputs();seal=json.loads((OUT/'reward_seal.json').read_text());assert seal['status']=='complete' and seal['queries']==128 and seal['plan_sha256']==sha(PLAN)
    assert seal['mask_seal_sha256']==sha(OUT/'mask_seal.json')
    for e in seal['files'].values():assert sha(OUT/e['path'])==e['sha256']
    prior=json.loads((CACHE/'diagnostic_seal.json').read_text())
    for e in prior['files'].values():assert sha(CACHE/e['path'])==e['sha256']
    labels_path=ROOT/old['labels'];assert sha(labels_path)==old['labels_sha256'];labels={v['id']:v for v in map(json.loads,labels_path.read_text().splitlines())}
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    names=['Geometry','Layout3','MaskDice','MaskContrast'];details=[];negative_counts=[]
    for row in rows:
        new=read_npz(seal,row['id']);truth=labels[row['id']];vals={name:[]for name in names};teacher_delta=[]
        with np.load(CACHE/prior['files'][row['id']]['path'],allow_pickle=False)as z:
            rewards=dict(Geometry=z['geometry_reward'],Layout3=z['layout3_reward'],MaskDice=new['MaskDice'],MaskContrast=new['MaskContrast'])
            for i,fid in enumerate(z['observed_frame_ids']):
                if str(int(fid))not in truth['boxes']:continue
                iou=box_iou(xyxy(z['boxes'][i],row['metadata']['width'],row['metadata']['height']),truth['boxes'][str(int(fid))])
                teacher_delta.append(float(iou[32]-iou[33]))
                for name in names:
                    s=rewards[name][i,:32];tied=np.isclose(s,s.max(),rtol=1e-10,atol=1e-12)
                    vals[name].append({**paired_ranking(s,iou[:32]),'top1_IoU_regret':float(iou[:32].max()-iou[:32][tied].mean())})
        active=(new['observed_frame_ids']>=truth['interval'][0])&(new['observed_frame_ids']<truth['interval'][1]);scores=new['selected_frame_scores']
        diff=scores[active,None]-scores[None,~active]
        auc=float(np.where(diff>0,1.,np.where(diff==0,.5,0.)).mean())if diff.size else None
        details.append(dict(id=row['id'],parent=truth['parent'],eligible_GT_observations=len(vals['Geometry']),teacher_minus_source=mean(teacher_delta),
            event_support_AUC=auc,rewards={name:{k:mean([v[k]for v in values])for k in ['pair_accuracy','spearman','top1_IoU_regret']}for name,values in vals.items()}))
        negative_counts.extend(new['eligible_alternative_count'].tolist())
    result=dict(status='complete_qualification_only',queries=128,eligible_queries=sum(r['eligible_GT_observations']>0 for r in details),
        rewards={name:{k:mean([v['rewards'][name][k]for v in details])for k in ['pair_accuracy','spearman','top1_IoU_regret']}for name in names},qualification={},
        eligible_alternatives_mean=float(np.mean(negative_counts)),frames_without_eligible_alternatives=int(np.sum(np.array(negative_counts)==0)),
        no_student_adaptation=True,no_full_population_evaluation=True,not_independent_confirmation=True,
        alternative_query_not_claimed_distinct_semantic_identity=True,frame_scores_not_claimed_event_probabilities_or_occlusion_labels=True,
        aggregation='mean eligible frames per query then equal eligible queries; policy32 only; ties averaged for top1 regret')
    baseline=result['rewards']['Geometry'];parents=[v['parent']for v in details]
    for name in ['MaskDice','MaskContrast']:
        values=result['rewards'][name];a=values['pair_accuracy'];r=values['top1_IoU_regret']
        passed=a is not None and r is not None and a>=baseline['pair_accuracy']-1e-12 and r<=baseline['top1_IoU_regret']+1e-12 and (a>baseline['pair_accuracy']+1e-12 or r<baseline['top1_IoU_regret']-1e-12)
        result['qualification'][name]=dict(passed_development_signal_gate=passed,not_an_adaptation_success_claim=True,
            pair_accuracy_difference_pp=cluster_difference([None if v['rewards'][name]['pair_accuracy']is None or v['rewards']['Geometry']['pair_accuracy']is None else v['rewards'][name]['pair_accuracy']-v['rewards']['Geometry']['pair_accuracy']for v in details],parents),
            regret_difference_pp=cluster_difference([None if v['rewards'][name]['top1_IoU_regret']is None else v['rewards'][name]['top1_IoU_regret']-v['rewards']['Geometry']['top1_IoU_regret']for v in details],parents))
    for group,predicate in dict(no_GT_observation=lambda v:v['eligible_GT_observations']==0,teacher_not_better=lambda v:v['teacher_minus_source']is not None and v['teacher_minus_source']<=0).items():
        subset=[v for v in details if predicate(v)];result[group]=dict(queries=len(subset),rewards={name:{k:mean([v['rewards'][name][k]for v in subset])for k in ['pair_accuracy','top1_IoU_regret']}for name in names})
    result['native_text_score_event_support']=dict(mean_query_AUC=mean([r['event_support_AUC']for r in details]),eligible_queries=sum(r['event_support_AUC']is not None for r in details),
        AUC_minus_chance_pp=cluster_difference([None if r['event_support_AUC']is None else r['event_support_AUC']-.5 for r in details],parents),not_used_as_a_gate=True)
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result)
    lines=['# TubeDETR 原生指代mask奖励资格检查','',
        '仅同一Vid128/同八帧/单ReferDINO；原框与RGB逐位一致后补存原生mask。无学生适应、无参数搜索、无新全量。完整奖励封存后独立GT诊断。','',
        '| 奖励 | 成对排序准确率 | Spearman | Top1 IoU regret（越低越好） |','|---|---:|---:|---:|']
    for name,v in result['rewards'].items():lines.append('| '+name+' | '+' | '.join('null'if v[k]is None else f'{v[k]:.6f}'for k in ['pair_accuracy','spearman','top1_IoU_regret'])+' |')
    lines+=['','资格：'+json.dumps({k:v['passed_development_signal_gate']for k,v in result['qualification'].items()},ensure_ascii=False),
        '逐帧文本分数事件支持：'+json.dumps(result['native_text_score_event_support'],ensure_ascii=False),
        '开发资格不证明OPD优势。软mask复制错误教师的风险仍在，替代query不是已知错误身份，事件外缺少GT不能当遮挡。两奖励均有同样的Direct连续梯度。','']
    (PUBLIC/'report.md').write_text('\n'.join(lines));write(PUBLIC/'final_seal.json',dict(status='complete',queries=128,
        files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},reward_seal_sha256=sha(OUT/'reward_seal.json'),publication_pending=True,qualification_only=True))


def controller():
    inputs()
    for stage in ['masks','bank','score']:
        command=[sys.executable,'-B','-u','tools/tubedetr_mask_reward_qualification.py','--stage',stage];log=ROOT/'logs'/f'tubedetr_mask_reward_qualification_{stage}.log';env=dict(os.environ)
        if stage!='masks':env['CUDA_VISIBLE_DEVICES']=''
        write(OUT/'status.json',dict(status=stage,no_student_adaptation=True))
        with log.open('a')as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            path=ROOT/'runtime/tubedetr_active_jobs.json';r=json.loads(path.read_text());r['jobs']['mask_reward_qualification_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage);write(path,r)
            if process.wait():raise RuntimeError('Mask qualification stage failed: '+stage)
    write(OUT/'status.json',dict(status='complete',queries=128,no_student_adaptation=True,old_queues_still_deferred=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['masks','bank','score']);a=p.parse_args()
    try:globals()[a.stage]()if a.stage else controller()
    except Exception as e:
        v=dict(status='failed',error=type(e).__name__,message=str(e));write(OUT/'failure.json',v);write(OUT/'status.json',v);raise
