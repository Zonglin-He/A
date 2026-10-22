"""One additional bounded teacher-only score-calibration validation, no full."""
import argparse,copy,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha,setup
from stvg.tube_runner import write
from tools.boxfixed_tube.profiles import profile_id
from tools.boxfixed_followup.coverage import rescore_coverage
from tools.boxfixed_followup.teacher_calibration import calibration,rescore_calibrated
import tools.tubedetr_boxfixed_development as original
PARENT=ROOT/'results/tubedetr_selection_coverage';P_OUT=ROOT/'outputs/tubedetr_selection_coverage'
PUBLIC=ROOT/'results/tubedetr_selection_calibration';OUT=ROOT/'outputs/tubedetr_selection_calibration';PLAN=PUBLIC/'plan.json'
original.PUBLIC=PUBLIC;original.OUT=OUT;original.PLAN=PLAN


def prepare():
    assert not PLAN.exists();p=copy.deepcopy(json.loads((PARENT/'plan.json').read_text()))
    assert json.loads((PARENT/'final_seal.json').read_text())['status']=='complete'
    p.update(status='frozen_before_teacher_calibration_or_scores',protocol_version=2,parent_coverage_final_sha256=sha(PARENT/'final_seal.json'),
             engineering_recovery_sha256=sha(PARENT/'calibration_recovery.json'),
             coverage_selections={ds:json.loads((PARENT/ds/'selected_profile.json').read_text())for ds in p['dataset_order']},
             scoring_scope='Same fixed candidate coordinates; per-query four-coefficient closed-form ridge score using held-out teacher-target pseudo responses; no dataset GT online',
             max_profiles_per_dataset=7,noise_margin_grid=[0.,.25,.5],link_multiplier_grid=[0.,1.],ridge=1.,feature_RMS_floor=.01,minimum_training_pairs=8,minimum_teacher_targets=4,
             admission='Keep observed ERef; original existing side must be geometrically available with own cycle>=.1; teacher-only CV excludes held observation from both source and target training roles, beats zero; predicted IoU advantage minus noise buffer >0',
             selection='Previous coverage incumbent always eligible. Require both spatial means and v at least previous and Temporal; maximize minimum dense/unobserved gain. Practical goal both >=.25pp, >=10 improving parents vs ERef and stability audit. No score search after fixed6 combinations.',
             new_full_allowed=False,new_neural_calls=0)
    for f in ['tools/boxfixed_followup/teacher_calibration.py','tools/boxfixed_followup/calibration_run.py','tests/test_teacher_score_calibration.py']:p['code_sha256'][f]=sha(ROOT/f)
    write(PLAN,p);print('Calibration plan frozen',sha(PLAN),flush=True)


def checked():
    p=original.checked();assert sha(PARENT/'final_seal.json')==p['parent_coverage_final_sha256'];return p


def cache(ds):
    setup();p=checked();torch.set_num_threads(1);f=P_OUT/ds/'evidence/sealed.json';assert sha(f)==p['evidence_seals'][ds];s=json.loads(f.read_text())
    rawroot=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'captures';rs=json.loads((rawroot/'sealed.json').read_text());entries={};stats=[];target=OUT/ds/'calibration';target.mkdir(parents=True,exist_ok=True)
    for i,(q,e)in enumerate(s['files'].items()):
        rawfile=rawroot/rs['files'][q]['path']
        assert sha(Path(e['path']))==e['sha256'] and sha(Path(e['appearance_path']))==e['appearance_sha256'] and sha(rawfile)==e['raw_capture_sha256']
        c=torch.load(e['path'],map_location='cpu',weights_only=False);raw=torch.load(rawfile,map_location='cpu',weights_only=False);a=torch.load(e['appearance_path'],map_location='cpu',weights_only=False)
        cal=calibration(c['bank'],raw,a,p['candidate_profiles'][ds]['parameters']);path=target/(q+'.pt');torch.save(cal,path)
        entries[q]=dict(**e,calibration_path=str(path),calibration_sha256=sha(path));m=cal['model'];stats.append({k:v for k,v in m.items()if not torch.is_tensor(v)})
        write(OUT/'worker_progress.json',dict(stage='teacher_calibration',dataset=ds,complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),parent_evidence_sha256=sha(f),labels_read=False))
    write(PUBLIC/ds/'teacher_calibration_summary.json',dict(status='complete',queries=128,teacher_CV_pass_queries=sum(x['valid']for x in stats),mean_pairs=float(np.mean([x['samples']for x in stats])),mean_teacher_CV_skill=float(np.mean([x.get('teacher_cv_skill',0.)for x in stats])),no_dataset_GT_used=True,closed_form_coefficients_per_query=4,teacher_target_holdout_not_statistical_independence=True))


def verified(ds):
    f=OUT/ds/'calibration/sealed.json';s=json.loads(f.read_text());assert s['queries']==128 and s['plan_sha256']==sha(PLAN)
    for e in s['files'].values():
        for key,h in [('path','sha256'),('appearance_path','appearance_sha256'),('calibration_path','calibration_sha256')]:assert sha(Path(e[key]))==e[h]
    return s


def profile(ds,request):
    setup();checked();torch.set_num_threads(1);params=json.loads(Path(request).read_text());s=verified(ds);target=OUT/ds/'profiles'/profile_id(params);target.mkdir(parents=True,exist_ok=True);files={}
    for i,(q,e)in enumerate(s['files'].items()):
        c=torch.load(e['path'],weights_only=False,map_location='cpu');a=torch.load(e['appearance_path'],weights_only=False,map_location='cpu');cal=torch.load(e['calibration_path'],weights_only=False,map_location='cpu')
        def decode(b):return rescore_coverage(b,a,params['coverage'])if params['policy']=='previous'else rescore_calibrated(b,cal,params)
        r=decode(c['bank']);rn=decode(c['native_bank']);path=target/(q+'.npz');rp=path.with_suffix('.json');original.save_npz(path,c,r,rn)
        write(rp,dict(parameters=params,plan_sha256=sha(PLAN),request_sha256=e['request_sha256'],labels_read=False,admitted_frames=r.get('admitted_frames'),changed_from_ERef_frames=int((r['final']!=c['bank'].E).any(-1).sum()),calibration_sha256=e['calibration_sha256'],candidate_bank_sha256=e['sha256'],temporary_score_coefficients=4,no_neural_weight_updates=True,new_GPU_forwards=0))
        files[q]=dict(path=str(path.relative_to(target)),sha256=sha(path),receipt_path=str(rp.relative_to(target)),receipt_sha256=sha(rp));write(OUT/'worker_progress.json',dict(stage='profile',dataset=ds,profile_id=profile_id(params),complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,parameters=params,files=files,plan_sha256=sha(PLAN),calibration_seal_sha256=sha(OUT/ds/'calibration/sealed.json'),labels_read=False))


def child(ds,stage,request=None):
    cmd=[sys.executable,'-B','-u','tools/boxfixed_followup/calibration_run.py','--stage',stage,'--dataset',ds]
    if request:cmd+=['--request',str(request)]
    log=ROOT/'logs'/f'tubedetr_score_calibration_{ds}_{stage}.log';write(OUT/'status.json',dict(stage=stage,dataset=ds,request=str(request),new_GPU=False))
    with log.open('a')as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});path=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(path.read_text());j['jobs']['score_calibration_controller']=dict(pid=os.getpid(),command='tools/boxfixed_followup/calibration_run.py --stage run',log='logs/complete_tubedetr_score_calibration.log');j['jobs']['score_calibration_worker']=dict(pid=proc.pid,command=shlex.join(cmd),stage=stage,dataset=ds,log=str(log.relative_to(ROOT)));write(path,j)
        if proc.wait():raise RuntimeError(f'{ds} {stage} failed; {log}')


def run():
    p=checked();summary={}
    for ds in p['dataset_order']:
        child(ds,'cache');tested={};pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True);base=dict(policy='previous',coverage=p['coverage_selections'][ds]['parameters'])
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                assert len(tested)<p['max_profiles_per_dataset'];request=OUT/ds/'requests'/(key+'.json');write(request,params);child(ds,'profile',request);child(ds,'score',request);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text());write(pub/'all_metrics.json',list(tested.values()))
            return tested[key]
        incumbent=evaluate(base)
        for margin in p['noise_margin_grid']:
            for link in p['link_multiplier_grid']:evaluate(dict(policy='teacher_calibrated',noise_margin=margin,link_multiplier=link))
        def feasible(m):return m['means']['Full']['s']>=max(incumbent['means']['Full']['s'],m['means']['Temporal']['s'])and m['unobserved_sIoU']['Full']>=max(incumbent['unobserved_sIoU']['Full'],m['unobserved_sIoU']['Temporal'])and m['means']['Full']['v']>=max(incumbent['means']['Full']['v'],m['means']['Temporal']['v'])
        pool=[m for m in tested.values()if feasible(m)];win=max(pool,key=lambda m:(min(m['means']['Full']['s']-m['means']['Temporal']['s'],m['unobserved_sIoU']['Full']-m['unobserved_sIoU']['Temporal']),m['parameters']==base))
        result=dict(status='complete_fixed128_only',parameters=win['parameters'],profile_id=profile_id(win['parameters']),metrics=win,profiles=len(tested),previous=incumbent,plan_sha256=sha(PLAN),prediction_seal_sha256=win['prediction_seal_sha256'],new_full_predictions=0,independent_confirmation=False)
        write(pub/'selected_profile.json',result);summary[ds]=result
    write(PUBLIC/'summary.json',dict(status='complete',datasets=summary,new_full_predictions=0,new_neural_calls=0));files=[PLAN,PUBLIC/'summary.json',*[PUBLIC/ds/f for ds in p['dataset_order']for f in ['teacher_calibration_summary.json','all_metrics.json','selected_profile.json']]];write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True));write(OUT/'status.json',dict(status='complete_fixed128_only',new_full=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','cache','profile','score','run'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='cache':cache(args.dataset)
        elif args.stage=='profile':profile(args.dataset,args.request)
        elif args.stage=='score':checked();original.score(args.dataset,json.loads(Path(args.request).read_text()))
        else:run()
    except Exception as e:write(OUT/'failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e)));raise
