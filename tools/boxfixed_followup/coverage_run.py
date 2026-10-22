"""Original fixed128 panels; diagnose admission coverage, then bounded rescore."""
import argparse,copy,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha,setup
from stvg.tube_runner import write,digest
from tools.boxfixed_tube.profiles import profile_id
from tools.boxfixed_followup.coverage import appearance,geometric_valid,rescore_coverage
from tools.boxfixed_followup.selection import signals,rescore
import tools.tubedetr_boxfixed_development as original
PREVIOUS=ROOT/'results/tubedetr_selection';P_OUT=ROOT/'outputs/tubedetr_selection'
PUBLIC=ROOT/'results/tubedetr_selection_coverage';OUT=ROOT/'outputs/tubedetr_selection_coverage'
QUAL=PUBLIC/'qualification_plan.json';PLAN=PUBLIC/'plan.json'
original.PUBLIC=PUBLIC;original.OUT=OUT;original.PLAN=PLAN


def prepare():
    assert not QUAL.exists();p=copy.deepcopy(json.loads((PREVIOUS/'plan.json').read_text()))
    p.update(status='frozen_coverage_evidence_before_diagnosis',previous_completion_sha256=sha(PREVIOUS/'completion_seal.json'),
             previous_selections={ds:json.loads((PREVIOUS/ds/'selected_profile.json').read_text())for ds in p['dataset_order']},
             scope='Score/admission only; same raw candidate coordinates, teacher, Source, eight anchors, time; allow preexisting geometric proposals to recover from old cycle mask',
             new_full_allowed=False,new_neural_calls=0,appearance='fixed Source prototype contrast with 2x native-box outer ring; valid-image outside-box fallback; no renormalization of prototype difference')
    for f in ['tools/boxfixed_followup/coverage.py','tools/boxfixed_followup/coverage_run.py','tests/test_boxfixed_coverage.py']:p['code_sha256'][f]=sha(ROOT/f)
    write(QUAL,p);print('Qualification frozen',sha(QUAL),flush=True)


def checked(trial=False):
    p=json.loads((PLAN if trial else QUAL).read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert sha(PREVIOUS/'completion_seal.json')==p['previous_completion_sha256']
    for ds,d in p['datasets'].items():assert sha(ROOT/d['manifest'])==d['manifest_sha256']
    return p


def cache(ds):
    setup();p=checked();torch.set_num_threads(1);parent=P_OUT/ds/'banks/sealed.json'
    assert sha(parent)==p['candidate_banks'][ds];s=json.loads(parent.read_text());assert s['queries']==128
    rawroot=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'captures';rawseal=json.loads((rawroot/'sealed.json').read_text());entries={};target=OUT/ds/'evidence';target.mkdir(parents=True,exist_ok=True)
    for i,(q,e)in enumerate(s['files'].items()):
        re=rawseal['files'][q];assert sha(Path(e['path']))==e['sha256'] and sha(rawroot/re['path'])==re['sha256']
        c=torch.load(e['path'],weights_only=False,map_location='cpu');raw=torch.load(rawroot/re['path'],weights_only=False,map_location='cpu');assert c['request_sha256']==raw['request_sha256']
        a=appearance(c['bank'],raw['evidence']);assert torch.equal(c['bank'].boxes,c['native_bank'].boxes)
        # These features do not depend on a temporal readout. Verify both views.
        assert torch.equal(c['bank'].features,c['native_bank'].features)
        path=target/(q+'.pt');torch.save(a,path)
        entries[q]=dict(**e,appearance_path=str(path),appearance_sha256=sha(path),raw_capture_sha256=re['sha256'])
        write(OUT/'worker_progress.json',dict(stage='appearance_cache',dataset=ds,complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,qualification_plan_sha256=sha(QUAL),bank_seal_sha256=sha(parent),labels_read=False))


def verified(ds):
    f=OUT/ds/'evidence/sealed.json';s=json.loads(f.read_text());assert s['queries']==128 and s['qualification_plan_sha256']==sha(QUAL)
    for e in s['files'].values():assert sha(Path(e['path']))==e['sha256'] and sha(Path(e['appearance_path']))==e['appearance_sha256']
    return s


def diagnose():
    from stvg.metrics import box_iou,DenseTube
    from stvg.metrics_base import xyxy
    p=checked();torch.set_num_threads(1);results={}
    for ds in p['dataset_order']:
        seal=verified(ds);d=p['datasets'][ds];assert sha(ROOT/d['labels'])==d['labels_sha256']
        gt={r['id']:r for r in map(json.loads,(ROOT/d['labels']).read_text().splitlines())};requests={r['id']:r for r in map(json.loads,(ROOT/d['manifest']).read_text().splitlines())}
        rows=[];query=[];funnel={'geometric_slots':0,'old_valid_slots':0,'two_anchor_slots':0,'cross_positive_slots':0,'contrast_positive_slots':0,'GT_improving_slots':0,'GT_improving_but_old_masked':0,'positive_cross_GT_good_unary_guard_defeated':0,'positive_cross_GT_good':0}
        for q,e in seal['files'].items():
            c=torch.load(e['path'],weights_only=False,map_location='cpu');b=c['bank'];a=torch.load(e['appearance_path'],weights_only=False,map_location='cpu');s=signals(b);valid=geometric_valid(b).numpy();old=b.valid.numpy()&b.active.numpy()[:,None];old[:,0]=True
            ids=np.asarray(c['frame_ids']);truth={int(k):v for k,v in gt[q]['boxes'].items()};nodes=[t for t,f in enumerate(ids)if int(f)in truth and b.event[t]and not b.observed[t]]
            meta=requests[q]['metadata'];cand=xyxy(b.boxes.numpy().reshape(-1,4),meta['width'],meta['height']).reshape(b.frames,3,4)
            if ds=='hc2':cand=np.maximum(cand,0)
            path=b.E.clone();pot_all=pot_old=0;n=max(1,int((b.event&~b.observed).sum()))
            for t in nodes:
                iou=box_iou(cand[t],np.asarray(truth[int(ids[t])])[None]);choices=np.where(valid[t],iou,-1);best=int(choices.argmax());path[t]=b.boxes[t,best]
                pot_all+=int(iou[best]>iou[0]+.01);pot_old+=int(np.max(np.where(old[t],iou,-1))>iou[0]+.01)
                for k in (1,2):
                    if not valid[t,k]:continue
                    gain=float(iou[k]-iou[0]);cross=float(s['advantage'][t,k]);contrast=float(a['cross'][t,k]);own=float(a['own'][t,k]);two=bool(s['independent'][t]);cycle=float(s['joint_reliability'][t]);isold=bool(old[t,k])
                    guard=float(p['candidate_profiles'][ds]['parameters']['guard']*b.guard[t,k]/b.frames);reward=.5*cycle*cross/n
                    row=dict(gain=gain,old=isold,two=two,cross=cross,contrast=contrast,own=own,cycle=cycle,appearance_valid=bool(a['valid'][t]),guard_defeats_reward=reward<=guard)
                    rows.append(row);funnel['geometric_slots']+=1;funnel['old_valid_slots']+=isold;funnel['two_anchor_slots']+=two;funnel['cross_positive_slots']+=two and cross>0;funnel['contrast_positive_slots']+=two and contrast>0 and own>0
                    funnel['GT_improving_slots']+=gain>.01;funnel['GT_improving_but_old_masked']+=gain>.01 and not isold
                    funnel['positive_cross_GT_good']+=gain>.01 and two and cross>0
                    funnel['positive_cross_GT_good_unary_guard_defeated']+=gain>.01 and two and cross>0 and reward<=guard
            truthboxes={int(k):v for k,v in gt[q]['boxes'].items()};mrow=dict(frame_ids=c['frame_ids'],input=meta)
            oracle=DenseTube(path.numpy(),mrow,truthboxes,gt[q]['interval'],clip=ds=='hc2').score(c['Global6_interval']);eref=DenseTube(b.E.numpy(),mrow,truthboxes,gt[q]['interval'],clip=ds=='hc2').score(c['Global6_interval'])
            query.append(dict(id=q,potential_all_nodes=pot_all,potential_old_nodes=pot_old,GT_local_path_vs_ERef_s_pp=100*(oracle['s']-eref['s']),GT_local_path_vs_ERef_v_pp=100*(oracle['v']-eref['v'])))
        def describe(a):
            return dict(candidate_slots=len(a),mean_GT_IoU_gain_pp=100*float(np.mean([r['gain']for r in a]))if a else None,improved=sum(r['gain']>1e-12 for r in a),harmed=sum(r['gain']< -1e-12 for r in a))
        results[ds]=dict(funnel=funnel,queries_with_existing_better_nodes=sum(x['potential_all_nodes']>0 for x in query),queries_with_old_admitted_better_nodes=sum(x['potential_old_nodes']>0 for x in query),
                        all_geometric=describe(rows),old_admitted=describe([r for r in rows if r['old']]),
                        cross_positive=describe([r for r in rows if r['two']and r['cross']>0]),contrast_positive=describe([r for r in rows if r['two']and r['appearance_valid']and r['contrast']>0 and r['own']>0]),
                        GT_local_node_oracle_dense_gain_vs_ERef_pp=float(np.mean([r['GT_local_path_vs_ERef_s_pp']for r in query])),
                        GT_local_node_oracle_v_gain_vs_ERef_pp=float(np.mean([r['GT_local_path_vs_ERef_v_pp']for r in query])),
                        GT_local_path_positive_queries=sum(x['GT_local_path_vs_ERef_s_pp']>1e-10 for x in query),
                        oracle_scope='GT-local physical-node choices then dense interpolation; diagnostic only, not global dense optimum or deployable method',
                        node_scope='GT-annotated physical nodes inside frozen Global6 interval; candidate-slot-weighted, not query-weighted full metric')
        write(OUT/ds/'diagnosis_private.json',dict(candidates=rows,queries=query))
    write(PUBLIC/'diagnosis.json',dict(status='complete',datasets=results,GT_for_offline_diagnosis_only=True));write(PUBLIC/'diagnosis_seal.json',dict(status='complete',diagnosis_sha256=sha(PUBLIC/'diagnosis.json'),evidence={ds:sha(OUT/ds/'evidence/sealed.json')for ds in p['dataset_order']}));print(json.dumps(results,indent=2))


def freeze():
    assert not PLAN.exists();p=checked();assert json.loads((PUBLIC/'diagnosis.json').read_text())['status']=='complete'
    p.update(status='frozen_before_new_prediction_scores',qualification_plan_sha256=sha(QUAL),diagnosis_seal_sha256=sha(PUBLIC/'diagnosis_seal.json'),
             max_profiles_per_dataset=16,cycle_grid=[.1,.25,.4],contrast_grid=[.5,1.],guard_grid=[.1,.4,1.],
             trial_order='previous control; relaxed geometric cross score with/without unary cycle weight at each floor; contrast .5/1 at each floor without unary cycle weight; one guard check on best expanded policy; no further search',
             selection='Require dense/unobserved >= previous and Temporal, v >= previous and Temporal. Maximize minimum of dense/unobserved gain over Temporal. Exact ties retain previous. Practical target: both spatial gains >=.25pp, at least10 parents improve over same ERef, descriptive CI and leave-one-parent-out reported after selection.',
             new_full_allowed=False,evidence_seals={ds:sha(OUT/ds/'evidence/sealed.json')for ds in p['dataset_order']})
    write(PLAN,p);print('Trial frozen',sha(PLAN),flush=True)


def profile(ds,request):
    setup();p=checked(True);torch.set_num_threads(1);params=json.loads(Path(request).read_text());s=verified(ds);assert sha(OUT/ds/'evidence/sealed.json')==p['evidence_seals'][ds]
    target=OUT/ds/'profiles'/profile_id(params);target.mkdir(parents=True,exist_ok=True);entries={}
    for i,(q,e)in enumerate(s['files'].items()):
        c=torch.load(e['path'],weights_only=False,map_location='cpu');a=torch.load(e['appearance_path'],weights_only=False,map_location='cpu');b=c['bank']
        r=rescore_coverage(b,a,params);rn=rescore_coverage(c['native_bank'],a,params);assert torch.equal(r['final'],b.boxes[torch.arange(b.frames),r['actions']])
        path=target/(q+'.npz');rp=path.with_suffix('.json');original.save_npz(path,c,r,rn)
        write(rp,dict(plan_sha256=sha(PLAN),parameters=params,request_sha256=e['request_sha256'],labels_read=False,candidate_bank_sha256=e['sha256'],appearance_sha256=e['appearance_sha256'],admitted_frames=r.get('admitted_frames'),changed_from_ERef_frames=int((r['final']!=b.E).any(-1).sum()),new_GPU_forwards=0))
        entries[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp));write(OUT/'worker_progress.json',dict(stage='profile',dataset=ds,profile_id=profile_id(params),complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,parameters=params,plan_sha256=sha(PLAN),evidence_seal_sha256=sha(OUT/ds/'evidence/sealed.json'),labels_read=False))


def child(ds,stage,request):
    cmd=[sys.executable,'-B','-u','tools/boxfixed_followup/coverage_run.py','--stage',stage,'--dataset',ds,'--request',str(request)];log=ROOT/'logs'/f'tubedetr_coverage_{ds}_{stage}.log'
    write(OUT/'status.json',dict(stage=stage,dataset=ds,request=str(request),new_GPU=False))
    with log.open('a')as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});path=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(path.read_text());j['jobs']['coverage_controller']=dict(pid=os.getpid(),command='tools/boxfixed_followup/coverage_run.py --stage run',log='logs/complete_tubedetr_coverage.log');j['jobs']['coverage_worker']=dict(pid=proc.pid,command=shlex.join(cmd),dataset=ds,stage=stage,log=str(log.relative_to(ROOT)));write(path,j)
        if proc.wait():raise RuntimeError(f'{ds} {stage} failed; {log}')


def run():
    p=checked(True);summary={}
    for ds in p['dataset_order']:
        tested={};decisions=[];pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True)
        previous=p['previous_selections'][ds];base=dict(policy='previous',selection=previous['parameters'])
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                assert len(tested)<p['max_profiles_per_dataset'];req=OUT/ds/'requests'/(key+'.json');write(req,params);child(ds,'profile',req);child(ds,'score',req);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text());write(pub/'all_metrics.json',list(tested.values()))
            return tested[key]
        incumbent=evaluate(base)
        def feasible(m):return m['means']['Full']['s']>=max(incumbent['means']['Full']['s'],m['means']['Temporal']['s'])and m['unobserved_sIoU']['Full']>=max(incumbent['unobserved_sIoU']['Full'],m['unobserved_sIoU']['Temporal'])and m['means']['Full']['v']>=max(incumbent['means']['Full']['v'],m['means']['Temporal']['v'])
        def choose(values):
            pool=[m for m in values if feasible(m)]or list(values)
            return max(pool,key=lambda m:(min(m['means']['Full']['s']-m['means']['Temporal']['s'],m['unobserved_sIoU']['Full']-m['unobserved_sIoU']['Temporal']),m['parameters']==base))
        guard=p['candidate_profiles'][ds]['parameters']['guard']
        for weight in [True,False]:
            batch=[evaluate(dict(policy='expanded',contrast=0.,cycle_floor=c,unary_cycle_weight=weight,guard=guard))for c in p['cycle_grid']]
            decisions.append(dict(stage='geometry_cycle_weight_'+str(weight),reason_before='Recover same preexisting geometric candidates masked by old cycle threshold; distinguish gate from repeated unary attenuation',candidates=[m['parameters']for m in batch],selected=choose(list(tested.values()))['parameters']));write(pub/'decisions.json',decisions)
        batch=[evaluate(dict(policy='expanded',contrast=a,cycle_floor=c,unary_cycle_weight=False,guard=guard))for a in p['contrast_grid']for c in p['cycle_grid']]
        decisions.append(dict(stage='foreground_background_contrast',reason_before='Use local background to test whether raw foreground cosine rewards common image features; fixed candidate coordinates and evidence',candidates=[m['parameters']for m in batch],selected=choose(list(tested.values()))['parameters']));write(pub/'decisions.json',decisions)
        start=choose(list(tested.values()))['parameters']
        if start['policy']=='expanded':
            batch=[evaluate(dict(start,guard=g))for g in p['guard_grid']];decisions.append(dict(stage='bounded_guard',reason_before='One predefined guard check after ranking/gating changed; do not retune other factors',candidates=[m['parameters']for m in batch]));write(pub/'decisions.json',decisions)
        win=choose(list(tested.values()));d=win['means'];u=win['unobserved_sIoU'];result=dict(status='complete_fixed128_only',parameters=win['parameters'],profile_id=profile_id(win['parameters']),metrics=win,profiles=len(tested),previous=incumbent,point_guardrails_pass=feasible(win),spatial_0_25pp_target=d['Full']['s']-d['Temporal']['s']>=.25 and u['Full']-u['Temporal']>=.25,plan_sha256=sha(PLAN),prediction_seal_sha256=win['prediction_seal_sha256'],new_full_predictions=0,independent_confirmation=False)
        write(pub/'selected_profile.json',result);summary[ds]=result
    write(PUBLIC/'summary.json',dict(status='complete',datasets=summary,new_full_predictions=0,new_neural_calls=0));files=[QUAL,PLAN,PUBLIC/'diagnosis.json',PUBLIC/'diagnosis_seal.json',PUBLIC/'summary.json',*[PUBLIC/ds/f for ds in p['dataset_order']for f in ['all_metrics.json','decisions.json','selected_profile.json']]]
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete_fixed128_only',new_full=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','cache','diagnose','freeze','profile','score','run'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='cache':cache(args.dataset)
        elif args.stage=='diagnose':diagnose()
        elif args.stage=='freeze':freeze()
        elif args.stage=='profile':profile(args.dataset,args.request)
        elif args.stage=='score':checked(True);original.score(args.dataset,json.loads(Path(args.request).read_text()))
        else:run()
    except Exception as e:
        write(OUT/'failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e)));raise
