"""Finite spatial follow-up on the same Vid128, after both ongoing full runs.

Worker imports the frozen Vid replay code in a fresh process and rebinds only its
output/plan paths. The running original worker and all implementation files are
untouched. New profiles really fit content; no interpolation masquerades as fit.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import digest,write
from tools import tubedetr_vid_development as dev
from tools import complete_tubedetr_vid_development as shared

PUBLIC=ROOT/'results/tubedetr_vid_spatial_followup'
OUT=ROOT/'outputs/tubedetr_vid_spatial_followup'
PLAN=PUBLIC/'plan.json'
ORIGINAL=ROOT/'outputs/tubedetr_vid_development'
MANIFEST=ROOT/'data/evaluation/tubedetr_vid_dev128.jsonl'
LABELS=ROOT/'data/evaluation/tubedetr_vid_dev128_labels.jsonl'
GAMMAS=[.0125,.025];RATES=[.01,.02];STEPS=[15,30,60]


def worker_module():
    dev.OUT=OUT;dev.PUBLIC=PUBLIC;dev.PLAN=PLAN
    shared.dev=dev
    return dev


def prepare():
    p=copy.deepcopy(json.loads((ROOT/'results/tubedetr_vid_development/plan.json').read_text()))
    reference=ROOT/'results/tubedetr_vid_development/selected_profile.json'
    selected=json.loads(reference.read_text())
    p.update(status='frozen_bounded_spatial_followup_before_new_candidates',
        prior_selected_sha256=sha(reference),reference_profile=selected['profile'],reference_parameters=selected['parameters'],
        diagnosis_sha256=sha(ROOT/'results/tubedetr_spatial_analysis/vidstg.json'),
        development_code_sha256={str(f.relative_to(ROOT)):sha(f)for f in [
            ROOT/'tools/tubedetr_vid_spatial_followup.py',ROOT/'tools/tubedetr_vid_development.py',
            ROOT/'tools/complete_tubedetr_vid_development.py']},
        stages=[dict(name='weaker_initialization_optimizer_coupling',gamma=GAMMAS,lr=RATES,tau=1.,steps=STEPS)],
        max_new_development_fit_paths=4*128+1,selected_profile_full_evaluations=1,
        dependency_seals=['results/tubedetr_vid_development/final_seal.json','results/tubedetr_hc2_development/final_seal.json'],
        selection_objective='maximize dev128 dense m_sIoU, subject to m_vIoU and unobserved_actual_sIoU each no worse than the previous locked Vid profile',
        tie_policy='retain prior profile on exact tie; then fewer steps',
        no_query_specific_GT_gating=True,spatial_net_benefit_requires_comparison_to_Source=True,
        reason='60-step OPD improves observed frames but harms dense/unobserved spatial IoU; compare weaker gamma and smaller LR, with real shared-step fits.',
        original_results_overwritten=False,output_alpha=1.,new_full_outputs=str(OUT.relative_to(ROOT)/'full_selected'))
    if PLAN.exists()and json.loads(PLAN.read_text())!=p:raise ValueError('Frozen follow-up plan changed')
    if not PLAN.exists():write(PLAN,p)
    return p


def ready():
    for name in ['results/tubedetr_vid_development/final_seal.json','results/tubedetr_hc2_development/final_seal.json']:
        path=ROOT/name
        if not path.exists():return False
        v=json.loads(path.read_text())
        if v['status']!='complete':return False
        if any(sha(ROOT/p)!=h for p,h in v['files'].items()):raise ValueError('Upstream final seal changed')
    registry=json.loads((ROOT/'runtime/tubedetr_active_jobs.json').read_text())
    for name in ['vid_development_controller','vid_development_gpu','hc2_development_controller','hc2_development_gpu']:
        job=registry['jobs'].get(name)
        if not job:continue
        try:
            argv=(Path('/proc')/str(job['pid'])/'cmdline').read_bytes().split(b'\0')
            if [s.decode()for s in argv if s]==shlex.split(job['command']):return False
        except FileNotFoundError:pass
    return True


def choose(values,reference):
    floor=values[reference]['means']['Full'];eligible=[]
    for name,value in values.items():
        x=value['means']['Full']
        if x['m_vIoU']>=floor['m_vIoU'] and x['unobserved_actual_sIoU']>=floor['unobserved_actual_sIoU']:
            eligible.append(name)
    return max(eligible,key=lambda n:(values[n]['means']['Full']['m_sIoU'],n==reference,-values[n]['params']['steps'],n))


def copy_teachers(manifest):
    for row in map(json.loads,Path(manifest).read_text().splitlines()):
        for suffix in ['.npz','.json']:
            src=ORIGINAL/'teacher_cache'/(row['id']+suffix);dst=OUT/'teacher_cache'/src.name
            if not src.exists():raise ValueError('Complete prior single-expert evidence is required; no new teacher variant')
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists()and sha(dst)!=sha(src):raise ValueError('Teacher evidence changed')
            if not dst.exists():shutil.copy2(src,dst)


def launch(params,manifest,stage,base,scope):
    pending=[p for p in params if not (base/dev.pid(p)/'sealed.json').exists()]
    if not pending:return
    request=OUT/'worker_requests'/f'{stage}.json';value=dict(manifest=str(manifest),profiles=pending,
        output_base=str(base),scope=scope,receipt=str(OUT/'worker_receipts'/f'{stage}.json'))
    if request.exists()and json.loads(request.read_text())!=value:raise ValueError('Worker request already frozen')
    if not request.exists():write(request,value)
    cmd=[sys.executable,'-B','-u',str(ROOT/'tools/tubedetr_vid_spatial_followup.py'),'--worker-request',str(request)]
    log=ROOT/'logs'/f'tubedetr_vid_spatial_followup_{stage}.log'
    write(OUT/'status.json',dict(status=stage,new_GPU_work_started=True,as_of=dev.now()))
    with log.open('a')as stream:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        registry=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(registry.read_text())
        j['jobs']['vid_spatial_followup_gpu']=dict(pid=process.pid,command=shlex.join(cmd),log=str(log.relative_to(ROOT)))
        write(registry,j)
        if process.wait():raise RuntimeError('Spatial follow-up worker failed: '+str(log))


def main(wait):
    p=prepare();worker_module();OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runtime/tubedetr_vid_spatial_followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    final=PUBLIC/'final_seal.json'
    if final.exists():
        s=json.loads(final.read_text())
        if s['status']!='complete' or any(sha(ROOT/f)!=h for f,h in s['files'].items()):raise ValueError('Final results changed')
        return
    while not ready():
        for ds in ['vid','hc2']:
            if (ROOT/f'outputs/tubedetr_{ds}_development/failure.json').exists():raise RuntimeError('Upstream failed; keep follow-up deferred')
        write(OUT/'status.json',dict(status='waiting_for_both_selected_full_terminal_seals',new_GPU_work_started=False,
                                   tuning_candidates_evaluated=0,as_of=dev.now()))
        if not wait:return
        time.sleep(30)
    dev.guard_plan(p)
    if sha(ROOT/'results/tubedetr_vid_development/selected_profile.json')!=p['prior_selected_sha256']:
        raise ValueError('Prior profile changed')
    copy_teachers(MANIFEST)
    reference=p['reference_profile'];ref_params=p['reference_parameters'];base=OUT/'candidates'
    shared.bind_seeded_view(MANIFEST,base/reference,ref_params,p,source=ORIGINAL/'candidates'/reference)
    params=[dev.parameters(gamma=g,lr=lr,tau=1.,steps=n)for g in GAMMAS for lr in RATES for n in STEPS]
    launch(params,MANIFEST,'coupling',base,'dev128_spatial_followup')
    values={}
    for cfg in [ref_params,*params]:
        name=dev.pid(cfg);root=base/name
        result=shared.score(MANIFEST,LABELS,root,['Source','Spatial','Temporal','WithoutOPD','Full'])
        values[name]=dict(params=cfg,means=result['datasets']['vidstg']['query_means'],metrics_sha256=sha(root/'metrics.json'))
    selected=choose(values,reference);best=values[selected]['params'];m=values[selected]['means']
    selection=dict(status='locked_before_new_full',selected_profile=selected,parameters=best,
        reference_profile=reference,all_outcomes=values,selection_objective=p['selection_objective'],
        dense_spatial_gain_over_Source_pp=100*(m['Full']['m_sIoU']-m['Source']['m_sIoU']),
        dev_exposed=True,no_retuning_from_new_full=True,plan_sha256=sha(PLAN))
    chosen=PUBLIC/'selected_profile.json'
    if chosen.exists()and json.loads(chosen.read_text())!=selection:raise ValueError('Selected profile is immutable')
    if not chosen.exists():write(chosen,selection)
    if selected==reference:
        root=ORIGINAL/'full_selected'/reference
        full=json.loads((root/'metrics.json').read_text());reused=True
    else:
        manifest=ROOT/'data/evaluation/vidstg.jsonl';full_base=OUT/'full_selected';root=full_base/selected
        copy_teachers(manifest)
        shared.bind_seeded_view(MANIFEST,root,best,p,source=base/selected,scope='full_spatial_followup',
                               identity_manifest=manifest,seal_when_complete=False)
        launch([best],manifest,'full_selected',full_base,'full_spatial_followup')
        full=shared.score(manifest,ROOT/'data/evaluation/vidstg_labels.jsonl',root,
                          ['Source','Spatial','Temporal','WithoutOPD','Full']);reused=False
    public=copy.deepcopy(full)
    for group in public['datasets'].values():
        for cluster in group['clusters']:cluster['parent']=digest(cluster['parent'])
    for row in public['rows']:row['parent']=digest(row['parent'])
    write(PUBLIC/'full_metrics.json',public)
    lines=['# TubeDETR Vid128 空间耦合复查与全量','',
        '沿用固定128、seed41；原全量已曝光。四条新fit路径每query，每条保存15/30/60步。',
        '按空间均值选择，同时要求完整定位与未观测帧指标不劣于此前锁定配置。所有正负结果见selected_profile。',
        '选中：'+json.dumps(best),'原选中配置全量复用：'+str(reused),'',
        '| 设置 | m_vIoU | m_sIoU | 观测sIoU | 未观测sIoU |','|---|---:|---:|---:|---:|']
    for arm,v in full['datasets']['vidstg']['query_means'].items():
        lines.append('| '+arm+' | '+' | '.join(f'{100*v[k]:.5f}'if v[k]is not None else'null'for k in
            ['m_vIoU','m_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'])+' |')
    lines+=['','若空间指标仍低于Source，保留该负结果，不以m_vIoU提高声称空间问题已解决。缓存执行成本不是全新端到端效率。','']
    (PUBLIC/'report.md').write_text('\n'.join(lines))
    files=[PLAN,chosen,PUBLIC/'full_metrics.json',PUBLIC/'report.md']
    write(final,dict(status='complete',queries=10303,reference_reused=reused,selected_profile=selected,
        files={str(f.relative_to(ROOT)):sha(f)for f in files},prediction_seal_sha256=sha(root/'sealed.json'),publication_pending=True))
    write(OUT/'status.json',dict(status='complete',queries=10303,selected_profile=selected,publication_pending=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--wait',action='store_true');a.add_argument('--prepare-only',action='store_true');a.add_argument('--worker-request')
    args=a.parse_args()
    try:
        if args.worker_request:worker_module().worker(json.loads(Path(args.worker_request).read_text()))
        elif args.prepare_only:print(json.dumps(prepare(),indent=2))
        else:main(args.wait)
    except BaseException as e:
        write(OUT/'failure.json',dict(status='failed',error_type=type(e).__name__,message=str(e),as_of=dev.now()));raise
