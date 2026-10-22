"""Frozen fixed128-only candidate study; no full/GPU/teacher entry point."""
import argparse, copy, json, os, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha, setup
from stvg.tube_runner import write, digest
from tools.boxfixed_tube.profiles import profile_id
from tools.boxfixed_followup.refinement import predict
import tools.tubedetr_boxfixed_development as original
OLD_PLAN=original.PLAN;OLD_OUT=original.OUT;OLD_PUBLIC=original.PUBLIC
PUBLIC=ROOT/'results/tubedetr_spatial_refinement';OUT=ROOT/'outputs/tubedetr_spatial_refinement';PLAN=PUBLIC/'plan.json'
PREVIOUS=ROOT/'results/tubedetr_boxfixed_followup'
original.PLAN=PLAN;original.PUBLIC=PUBLIC;original.OUT=OUT
build_original_geometry=original.geometry


def prepare():
    assert not PLAN.exists(), 'Frozen plans cannot be overwritten'
    p=copy.deepcopy(json.loads(OLD_PLAN.read_text()))
    p.update(status='frozen_before_new_predictions_or_scores',run_scope='two fixed128 panels only; small candidate modifications and bounded parameters',
             parent_plan_sha256=sha(OLD_PLAN),parent_final_seal_sha256=sha(OLD_PUBLIC/'final_seal.json'),
             previous_parameter_completion_sha256=sha(PREVIOUS/'completion_seal.json'),
             diagnosis_sha256={str(f.relative_to(ROOT)):sha(f) for f in [PREVIOUS/'diagnosis.json',PREVIOUS/'extent_diagnosis.json']},
             new_full_allowed=False,full_evaluation_after_both128_profile_locks=False,no_new_GPU_or_teacher_calls=True,
             max_profiles_per_dataset=16,mechanisms=['center_preserve','relative_residual'],alpha_grid=[.25,.5,1.],beta_grid=[0.,.25,.5],
             guard_grid=[.4,1.,2.],admission_grid=[.4,.7],link_grid=[1.],
             selection='Maximize equal-query Full dense sIoU among configurations with Full m_vIoU >= max(Temporal, incumbent), unobserved sIoU >= Temporal, and dense sIoU >= Temporal. Exact ties retain incumbent then earliest candidate. If none qualifies, no promotion; record best observed diagnostic configuration.',
             refinement_order='incumbent and gamma0 controls; each family alpha grid with beta0; each family best alpha with beta .25/.5; best diagnostic configuration guard grid then admission grid; no further search',
             speculative_modifications='Only unobserved candidate coordinates; same evidence, three states, upstream scoring definitions and exact DP. Observed teacher proposals unchanged. New candidates use gamma0 to isolate inherited initialization calibration.',
             exposure='Both panels previously exposed repeatedly. Post-selection bootstrap is descriptive, not independent confirmation.',
             incumbents={ds:json.loads((PREVIOUS/ds/'parameter_selection.json').read_text()) for ds in ['vidstg','hc2']})
    for f in ['tools/boxfixed_followup/refinement.py','tools/boxfixed_followup/refinement_run.py','tests/test_boxfixed_spatial_refinement.py']:
        p['code_sha256'][f]=sha(ROOT/f)
    write(PLAN,p)
    for ds in p['dataset_order']:
        root=OLD_OUT/ds/'captures';s=json.loads((root/'sealed.json').read_text())
        assert s['queries']==128 and s['plan_sha256']==sha(OLD_PLAN)
        files={q:{**e,'path':str(root/e['path']),'receipt_path':str(root/e['receipt_path'])}for q,e in s['files'].items()}
        for e in files.values():
            assert sha(Path(e['path']))==e['sha256'] and sha(Path(e['receipt_path']))==e['receipt_sha256']
        write(OUT/ds/'captures/sealed.json',dict(s,files=files,plan_sha256=sha(PLAN),raw_capture_seal_sha256=sha(root/'sealed.json')))
    print('Frozen',sha(PLAN),flush=True)


def checked():
    p=original.checked()
    assert sha(OLD_PLAN)==p['parent_plan_sha256'] and sha(OLD_PUBLIC/'final_seal.json')==p['parent_final_seal_sha256']
    assert sha(PREVIOUS/'completion_seal.json')==p['previous_parameter_completion_sha256']
    return p


def cache(ds):
    setup();p,rows=original.records(ds);root,s=original.capture_seal(ds)
    import torch
    torch.set_num_threads(1);entries={};target=OUT/ds/'geometry';target.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        e=s['files'][row['id']];c=torch.load(root/e['path'],map_location='cpu',weights_only=False)
        assert c['request_sha256']==digest(row)
        g=build_original_geometry(c,p,ds);path=target/(row['id']+'.pt');torch.save(g,path)
        entries[digest(row)]=dict(path=str(path),sha256=sha(path),capture_sha256=e['sha256'])
        write(OUT/'worker_progress.json',dict(dataset=ds,stage='geometry_cache',complete=i+1,total=128,labels_read=False))
    write(target/'sealed.json',dict(status='complete',queries=128,files=entries,plan_sha256=sha(PLAN),capture_seal_sha256=sha(root/'sealed.json'),labels_read=False))


def profile(ds,request):
    setup();checked()
    import torch
    params=json.loads(Path(request).read_text());gs=OUT/ds/'geometry/sealed.json';s=json.loads(gs.read_text())
    assert s['queries']==128 and s['plan_sha256']==sha(PLAN)
    count=[0]
    def cached_geometry(c,p,dataset):
        assert dataset==ds
        e=s['files'][c['request_sha256']];assert sha(Path(e['path']))==e['sha256']
        count[0]+=1
        write(OUT/'worker_progress.json',dict(stage='profile',dataset=ds,profile_id=profile_id(params),complete=count[0],total=128,labels_read=False))
        return torch.load(e['path'],map_location='cpu',weights_only=False)
    original.geometry=cached_geometry;original.predict=predict
    original.profile_worker(ds,request)


def child(ds,stage,request=None):
    cmd=[sys.executable,'-B','-u','tools/boxfixed_followup/refinement_run.py','--stage',stage,'--dataset',ds]
    if request:cmd+=['--request',str(request)]
    log=ROOT/'logs'/f'tubedetr_spatial_refinement_{ds}_{stage}.log'
    write(OUT/'status.json',dict(status=stage,dataset=ds,request=str(request),new_GPU=False))
    with log.open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        runtime=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(runtime.read_text())
        j['jobs']['spatial_refinement_controller']=dict(pid=os.getpid(),command=shlex.join([sys.executable,'-B','-u','tools/boxfixed_followup/refinement_run.py','--stage','run']),log='logs/complete_tubedetr_spatial_refinement.log')
        j['jobs']['spatial_refinement_worker']=dict(pid=proc.pid,command=shlex.join(cmd),dataset=ds,stage=stage,log=str(log.relative_to(ROOT)))
        write(runtime,j)
        if proc.wait():raise RuntimeError(f'{ds} {stage} failed; inspect {log}')


def feasible(m,inc):
    return (m['means']['Full']['s']>=m['means']['Temporal']['s'] and
            m['unobserved_sIoU']['Full']>=m['unobserved_sIoU']['Temporal'] and
            m['means']['Full']['v']>=max(m['means']['Temporal']['v'],inc['means']['Full']['v']))


def run():
    p=checked();summary={}
    for ds in p['dataset_order']:
        child(ds,'cache');tested={};decisions=[];pub=PUBLIC/ds;pub.mkdir(parents=True,exist_ok=True)
        base=dict(p['incumbents'][ds]['parameters'],mode='original')
        def evaluate(params):
            key=profile_id(params)
            if key not in tested:
                assert len(tested)<p['max_profiles_per_dataset']
                req=OUT/ds/'requests'/(key+'.json');write(req,params);child(ds,'profile',req)
                if params==base:
                    import numpy as np
                    newroot=OUT/ds/'profiles'/key;oldroot=ROOT/'outputs/tubedetr_boxfixed_followup/parameters'/ds/'profiles'/p['incumbents'][ds]['profile_id']
                    ns=json.loads((newroot/'sealed.json').read_text());old=json.loads((oldroot/'sealed.json').read_text())
                    assert sha(oldroot/'sealed.json')==p['incumbents'][ds]['prediction_seal_sha256']
                    for q,e in ns['files'].items():
                        oe=old['files'][q];assert sha(oldroot/oe['path'])==oe['sha256']
                        with np.load(newroot/e['path'])as a,np.load(oldroot/oe['path'])as b:
                            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k])for k in a.files),q
                child(ds,'score',req);tested[key]=json.loads((OUT/ds/'profiles'/key/'metrics.json').read_text())
                write(pub/'all_metrics.json',list(tested.values()))
            return tested[key]
        incumbent=evaluate(base)
        def choose(values):
            qualified=[m for m in values if feasible(m,incumbent)]
            pool=qualified or list(values)
            return max(pool,key=lambda m:(m['means']['Full']['s'],m['parameters']==base))
        control={**base,'gamma':0.};evaluate(control)
        families={}
        for mode in p['mechanisms']:
            start={**control,'mode':mode,'alpha':.5,'beta':0.}
            batch=[evaluate({**start,'alpha':a})for a in p['alpha_grid']]
            selected=choose(batch)['parameters'];families[mode]=selected
            decisions.append(dict(stage=mode+'_centers',reason_before='Test conservative center corrections at preserved Source size; same evidence/score/DP',candidates=[m['parameters']for m in batch],selected=selected))
            write(pub/'decisions.json',decisions)
        for mode,start in families.items():
            batch=[evaluate({**start,'beta':b})for b in p['beta_grid']]
            decisions.append(dict(stage=mode+'_sizes',reason_before='Test limited log-size correction after center stage, without changing transforms or searching full population',candidates=[m['parameters']for m in batch],selected=choose(batch)['parameters']))
            write(pub/'decisions.json',decisions)
        for key,grid in [('guard',p['guard_grid']),('admission',p['admission_grid'])]:
            start=choose(list(tested.values()))['parameters'].copy()
            batch=[evaluate({**start,key:x})for x in grid]
            decisions.append(dict(stage=key,reason_before='One bounded scalar refinement on selected candidate family; no expanded grid',candidates=[m['parameters']for m in batch],selected=choose(list(tested.values()))['parameters']))
            write(pub/'decisions.json',decisions)
        winner=choose(list(tested.values()));qualified=feasible(winner,incumbent)
        assert len(tested)<=p['max_profiles_per_dataset']
        result=dict(status='complete_fixed128_only',parameters=winner['parameters'],profile_id=profile_id(winner['parameters']),metrics=winner,
                    profiles=len(tested),qualifies_for_spatial_promotion=qualified,incumbent=incumbent,
                    plan_sha256=sha(PLAN),incumbent_all_arrays_bitexact=True,
                    no_full_run=True,no_new_neural_calls=True,independent_confirmation=False,
                    note='Selection is exposed-development evidence only. If qualification fails, best diagnostic configuration is not promoted.')
        write(pub/'selected_profile.json',result);summary[ds]=result
    write(PUBLIC/'summary.json',dict(status='complete_fixed128_only',datasets=summary,new_full_predictions=0,new_neural_calls=0))
    files=[PLAN,PUBLIC/'summary.json',*[PUBLIC/ds/f for ds in p['dataset_order']for f in ['all_metrics.json','decisions.json','selected_profile.json']]]
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in files},publication_pending=True))
    write(OUT/'status.json',dict(status='complete_fixed128_only',new_full=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','cache','profile','score','run'],required=True);a.add_argument('--dataset',choices=['vidstg','hc2']);a.add_argument('--request');args=a.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='cache':cache(args.dataset)
        elif args.stage=='profile':profile(args.dataset,args.request)
        elif args.stage=='score':checked();original.score(args.dataset,json.loads(Path(args.request).read_text()))
        else:run()
    except Exception as e:
        write(OUT/'failure.json',dict(stage=args.stage,dataset=args.dataset,error=type(e).__name__,message=str(e)));raise
