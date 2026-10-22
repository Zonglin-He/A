"""Frozen fixed128 scoring/admission study, with unchanged candidate banks."""
import argparse,copy,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha,setup
from stvg.tube_runner import write,digest
from tools.boxfixed_tube.profiles import profile_id
from tools.boxfixed_tube.spatial import evidence_dp
from tools.boxfixed_followup.selection import rescore,signals
from tools.boxfixed_followup.selection_capture import capture_bank
import tools.tubedetr_boxfixed_development as original
OLD_PLAN=original.PLAN;OLD_OUT=original.OUT;OLD_PUBLIC=original.PUBLIC
PREVIOUS=ROOT/'results/tubedetr_spatial_refinement';PREV_OUT=ROOT/'outputs/tubedetr_spatial_refinement'
PUBLIC=ROOT/'results/tubedetr_selection';OUT=ROOT/'outputs/tubedetr_selection'
QUAL=PUBLIC/'qualification_plan.json';PLAN=PUBLIC/'plan.json'
original.PUBLIC=PUBLIC;original.OUT=OUT;original.PLAN=PLAN


def prepare():
    assert not QUAL.exists()
    p=copy.deepcopy(json.loads((PREVIOUS/'plan.json').read_text()))
    for f in ['tools/boxfixed_followup/selection.py','tools/boxfixed_followup/selection_capture.py','tools/boxfixed_followup/selection_run.py','tests/test_boxfixed_selection.py']:
        p['code_sha256'][f]=sha(ROOT/f)
    p.update(status='frozen_bank_capture_and_diagnosis_before_scoring_trial',previous_completion_sha256=sha(PREVIOUS/'completion_seal.json'),
             candidate_profiles={ds:json.loads((PREVIOUS/ds/'selected_profile.json').read_text())for ds in p['dataset_order']},
             scoring_scope='Only scoring/admission; exact old candidate banks and Native/Global6 intervals fixed',new_full_allowed=False,new_neural_calls=0)
    write(QUAL,p);print('Qualification frozen',sha(QUAL),flush=True)


def checked(stage=False):
    path=PLAN if stage else QUAL;p=json.loads(path.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert sha(PREVIOUS/'completion_seal.json')==p['previous_completion_sha256']
    for ds,d in p['datasets'].items():assert sha(ROOT/d['manifest'])==d['manifest_sha256']
    return p


def cache(ds):
    setup();p=checked();torch.set_num_threads(1)
    s=json.loads((PREV_OUT/ds/'geometry/sealed.json').read_text())
    caps=OLD_OUT/ds/'captures';cs=json.loads((caps/'sealed.json').read_text());selection=p['candidate_profiles'][ds]
    oldroot=PREV_OUT/ds/'profiles'/selection['profile_id'];oldseal=json.loads((oldroot/'sealed.json').read_text())
    assert sha(oldroot/'sealed.json')==selection['metrics']['prediction_seal_sha256']
    rows=[json.loads(x)for x in (ROOT/p['datasets'][ds]['manifest']).read_text().splitlines()]
    assert len(rows)==128;entries={};target=OUT/ds/'banks';target.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        q=row['id'];ce=cs['files'][q];ge=s['files'][digest(row)];oe=oldseal['files'][q]
        for f,h in [(caps/ce['path'],ce['sha256']),(Path(ge['path']),ge['sha256']),(oldroot/oe['path'],oe['sha256'])]:assert sha(f)==h
        c=torch.load(caps/ce['path'],map_location='cpu',weights_only=False);g=torch.load(ge['path'],map_location='cpu',weights_only=False)
        assert c['request_sha256']==digest(row)
        b,r=capture_bank(g,c['evidence'],c['Source'],c['raw'],c['Global6_interval'],selection['parameters'])
        bn,rn=capture_bank(g,c['evidence'],c['Source'],c['raw'],c['Native_interval'],selection['parameters'])
        with np.load(oldroot/oe['path'])as z:
            assert np.array_equal(r['final'].numpy(),z['BoxFixed']) and np.array_equal(rn['final'].numpy(),z['BoxFixedNative'])
            assert np.array_equal(r['actions'].numpy(),z['actions'])
        f=target/(q+'.pt');torch.save(dict(bank=b,native_bank=bn,Source=c['Source'],frame_ids=c['frame_ids'],positions8=c['positions8'],Native_interval=c['Native_interval'],Global6_interval=c['Global6_interval'],request_sha256=digest(row)),f)
        entries[q]=dict(path=str(f),sha256=sha(f),request_sha256=digest(row))
        write(OUT/'worker_progress.json',dict(stage='banks',dataset=ds,complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,qualification_plan_sha256=sha(QUAL),old_prediction_seal_sha256=sha(oldroot/'sealed.json'),candidates_and_original_outputs_bitexact=True,labels_read=False))


def verified_banks(ds):
    f=OUT/ds/'banks/sealed.json';s=json.loads(f.read_text());assert s['queries']==128 and s['qualification_plan_sha256']==sha(QUAL)
    for e in s['files'].values():assert sha(Path(e['path']))==e['sha256']
    return s


def diagnose():
    p=checked();torch.set_num_threads(1);out={}
    from stvg.metrics_base import xyxy
    from stvg.metrics import box_iou
    for ds in p['dataset_order']:
        s=verified_banks(ds);d=p['datasets'][ds]
        assert sha(ROOT/d['labels'])==d['labels_sha256']
        labels={r['id']:r for r in map(json.loads,(ROOT/d['labels']).read_text().splitlines())}
        requests={r['id']:r for r in map(json.loads,(ROOT/d['manifest']).read_text().splitlines())}
        frames=[];comparisons=[]
        for q,e in s['files'].items():
            c=torch.load(e['path'],map_location='cpu',weights_only=False);b=c['bank'];r=evidence_dp(b);sig=signals(b)
            truth={int(k):v for k,v in labels[q]['boxes'].items()};nodes=[t for t,f in enumerate(c['frame_ids'])if int(f)in truth and b.event[t]]
            meta=requests[q]['metadata'];cand=xyxy(b.boxes.numpy().reshape(-1,4),meta['width'],meta['height']).reshape(b.frames,3,4)
            if ds=='hc2':cand=np.maximum(cand,0)
            for t in nodes:
                iou=box_iou(cand[t],np.asarray(truth[int(c['frame_ids'][t])])[None]);k=int(r['actions'][t])
                if k:
                    frames.append(dict(observed=bool(b.observed[t]),gain=float(iou[k]-iou[0]),own=float(sig['own_advantage'][t,k]),cross=float(sig['advantage'][t,k]),independent=bool(sig['independent'][t]),cycle=float(b.confidence[t]),cycle_min=float(sig['joint_reliability'][t]),agreement=float(sig['agreement'][t])))
                if b.active[t] and not b.observed[t]:
                    for k in (1,2):
                        if b.valid[t,k]:comparisons.append(dict(gain=float(iou[k]-iou[0]),own=float(sig['own_advantage'][t,k]),cross=float(sig['advantage'][t,k]),independent=bool(sig['independent'][t])))
        def group(a):
            n=len(a)
            return dict(nodes=n,mean_node_IoU_change_pp=float(100*np.mean([x['gain']for x in a]))if n else None,
                        improved=sum(x['gain']>1e-12 for x in a),harmed=sum(x['gain']< -1e-12 for x in a),
                        mean_own_advantage=float(np.mean([x['own']for x in a]))if n else None,
                        mean_cross_advantage=float(np.mean([x['cross']for x in a]))if n else None,
                        own_positive_cross_nonpositive=sum(x['own']>0 and x['cross']<=0 for x in a),
                        independently_supported=sum(x['independent']for x in a))
        out[ds]=dict(selected_observed=group([x for x in frames if x['observed']]),selected_unobserved=group([x for x in frames if not x['observed']]),
                    harmed_unobserved=group([x for x in frames if not x['observed']and x['gain']<0]),
                    improved_unobserved=group([x for x in frames if not x['observed']and x['gain']>0]),
                    all_allowed_unobserved_candidates=group(comparisons),
                    cross_positive_candidates=group([x for x in comparisons if x['independent']and x['cross']>0]),
                    mean_cycle_harmed_unobserved=float(np.mean([x['cycle']for x in frames if not x['observed']and x['gain']<0]))if any(not x['observed']and x['gain']<0 for x in frames)else None,
                    GT_groups_used_only_for_diagnosis=True,node_scope='GT-annotated physical nodes inside frozen Global6 predicted interval; not dense full-query estimand')
        write(OUT/ds/'diagnostic_private.json',dict(selected=frames,candidates=comparisons))
    write(PUBLIC/'diagnosis.json',dict(status='complete',datasets=out,no_new_predictions_or_neural_calls=True))
    write(PUBLIC/'diagnosis_seal.json',dict(status='complete',diagnosis_sha256=sha(PUBLIC/'diagnosis.json'),banks={ds:sha(OUT/ds/'banks/sealed.json')for ds in p['dataset_order']}))
    print(json.dumps(out,indent=2),flush=True)


def freeze_trial():
    assert not PLAN.exists();p=checked();assert json.loads((PUBLIC/'diagnosis.json').read_text())['status']=='complete'
    p.update(status='frozen_before_new_profile_scores',qualification_plan_sha256=sha(QUAL),diagnosis_seal_sha256=sha(PUBLIC/'diagnosis_seal.json'),
             max_profiles_per_dataset=12,margin_grid=[0.,.02,.05],agreement_grid=[.5,.8],cycle_floor=.1,guard_grid=[.4,1.,2.],
             selection='Max Full dense sIoU among configs with dense>Temporal, unobserved>Temporal, Full v>=max(Temporal, frozen candidate incumbent). If none qualifies retain best diagnostic only. Practical positive target additionally requires dense/unobserved >= +.1pp over Temporal; CIs are descriptive after selection.',
             order='original and observation-protection controls; cross-score/cross-gate margin grids; best cross-score with two agreement thresholds; global best guard grid; no expansion',
             new_full_allowed=False,candidate_banks={ds:sha(OUT/ds/'banks/sealed.json')for ds in p['dataset_order']})
    write(PLAN,p);print('Trial frozen',sha(PLAN),flush=True)


def profile(ds,request):
    setup();p=checked(True);torch.set_num_threads(1);params=json.loads(Path(request).read_text());s=verified_banks(ds)
    assert sha(OUT/ds/'banks/sealed.json')==p['candidate_banks'][ds]
    target=OUT/ds/'profiles'/profile_id(params);target.mkdir(parents=True,exist_ok=True);entries={}
    for i,(q,e)in enumerate(s['files'].items()):
        c=torch.load(e['path'],map_location='cpu',weights_only=False);b=c['bank'];bn=c['native_bank']
        r=rescore(b,params);rn=rescore(bn,params)
        assert torch.equal(r['final'],b.boxes[torch.arange(b.frames),r['actions']])
        path=target/(q+'.npz');rp=path.with_suffix('.json');original.save_npz(path,c,r,rn)
        write(rp,dict(plan_sha256=sha(PLAN),request_sha256=e['request_sha256'],parameters=params,profile_id=profile_id(params),labels_read=False,new_GPU_forwards=0,candidate_bank_sha256=e['sha256'],admitted_frames=int((b.active&b.valid[:,1:].any(-1)).sum()),changed_from_ERef_frames=int((r['final']!=b.E).any(-1).sum()),independent_admitted_nodes=r.get('independent_admitted_nodes')))
        entries[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp))
        write(OUT/'worker_progress.json',dict(dataset=ds,stage='profile',profile_id=profile_id(params),complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,parameters=params,plan_sha256=sha(PLAN),candidate_banks_seal_sha256=sha(OUT/ds/'banks/sealed.json'),labels_read=False))


def child(ds,stage,request=None):
    cmd=[sys.executable,'-B','-u','tools/boxfixed_followup/selection_run.py','--stage',stage,'--dataset',ds]
    if request:cmd+=['--request',str(request)]
    log=ROOT/'logs'/f'tubedetr_selection_{ds}_{stage}.log';write(OUT/'status.json',dict(stage=stage,dataset=ds,request=str(request),new_GPU=False))
    with log.open('a')as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        path=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(path.read_text());j['jobs']['selection_controller']=dict(pid=os.getpid(),stage='fixed128_selection',log='logs/complete_tubedetr_selection.log');j['jobs']['selection_worker']=dict(pid=proc.pid,command=shlex.join(cmd),dataset=ds,stage=stage,log=str(log.relative_to(ROOT)));write(path,j)
        if proc.wait():raise RuntimeError(f'{ds} {stage} failed; {log}')


def run():
    p=checked(True);summary={}
    def feasible(m,inc):return m['means']['Full']['s']>m['means']['Temporal']['s'] and m['unobserved_sIoU']['Full']>m['unobserved_sIoU']['Temporal'] and m['means']['Full']['v']>=max(inc['means']['Full']['v'],m['means']['Temporal']['v'])
    for ds in p['dataset_order']:
        tested={};decisions=[];pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True)
        base=dict(selection_mode='original',margin=0.,cycle_floor=p['cycle_floor'],agreement_floor=0.,guard=p['candidate_profiles'][ds]['parameters']['guard'])
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                assert len(tested)<p['max_profiles_per_dataset'];req=OUT/ds/'requests'/(key+'.json');write(req,params);child(ds,'profile',req);child(ds,'score',req)
                tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text());write(pub/'all_metrics.json',list(tested.values()))
            return tested[key]
        incumbent=evaluate(base)
        def choose(values):
            pool=[m for m in values if feasible(m,incumbent)]or list(values)
            return max(pool,key=lambda m:(m['means']['Full']['s'],m['parameters']==base))
        evaluate(dict(base,selection_mode='observe_protection'))
        for mode in ['cross_score','cross_gate']:
            batch=[evaluate(dict(base,selection_mode=mode,margin=x))for x in p['margin_grid']]
            decisions.append(dict(stage=mode,reason_before='Remove self-reference advantage or use independent advantage only for admission; compare every candidate against E on the same other anchor',candidates=[m['parameters']for m in batch],selected=choose(batch)['parameters']));write(pub/'decisions.json',decisions)
        start=choose([m for m in tested.values()if m['parameters']['selection_mode']=='cross_score'])['parameters']
        batch=[evaluate(dict(start,selection_mode='cross_agreement',agreement_floor=x))for x in p['agreement_grid']]
        decisions.append(dict(stage='agreement',reason_before='Check whether two different anchors must agree in extent, without semantic identity claims',candidates=[m['parameters']for m in batch]));write(pub/'decisions.json',decisions)
        start=choose(list(tested.values()))['parameters']
        # Observation/original controls are already fixed original scoring; do
        # not count fake guard changes that the control mode does not execute.
        if start['selection_mode'] in ['cross_score','cross_agreement']:
            batch=[evaluate(dict(start,guard=x))for x in p['guard_grid']]
            decisions.append(dict(stage='protection',reason_before='One final predeclared guard-weight check on the selected evidence score',candidates=[m['parameters']for m in batch]));write(pub/'decisions.json',decisions)
        win=choose(list(tested.values()));m=win['means'];u=win['unobserved_sIoU']
        qualified=feasible(win,incumbent);practical=qualified and m['Full']['s']-m['Temporal']['s']>=.1 and u['Full']-u['Temporal']>=.1
        result=dict(status='complete_fixed128_only',parameters=win['parameters'],profile_id=profile_id(win['parameters']),metrics=win,profiles=len(tested),incumbent=incumbent,qualifies=qualified,practical_positive_target=practical,candidate_parameters=p['candidate_profiles'][ds]['parameters'],plan_sha256=sha(PLAN),prediction_seal_sha256=win['prediction_seal_sha256'],new_full_predictions=0,independent_confirmation=False)
        write(pub/'selected_profile.json',result);summary[ds]=result
    write(PUBLIC/'summary.json',dict(status='complete',datasets=summary,no_new_full=True,no_new_neural_calls=True))
    files=[QUAL,PLAN,PUBLIC/'diagnosis.json',PUBLIC/'diagnosis_seal.json',PUBLIC/'summary.json',*[PUBLIC/ds/f for ds in p['dataset_order']for f in ['decisions.json','all_metrics.json','selected_profile.json']]]
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete_fixed128_only',new_full=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','cache','diagnose','freeze','profile','score','run'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='cache':cache(args.dataset)
        elif args.stage=='diagnose':diagnose()
        elif args.stage=='freeze':freeze_trial()
        elif args.stage=='profile':profile(args.dataset,args.request)
        elif args.stage=='score':checked(True);original.score(args.dataset,json.loads(Path(args.request).read_text()))
        else:run()
    except Exception as e:
        write(OUT/'failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e)));raise
