"""After the current Vid full run, execute one bounded ROI reward pilot.

No old parameter queues are resumed. No new full-population evaluation or GT
hyperparameter search is scheduled here. Feature/reward/inference workers never
open labels; independent CPU scoring follows complete prediction seals.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import implementation_hash,write,digest,save_prediction
from tools import tubedetr_vid_development as storage
from tools import complete_tubedetr_vid_development as scoring

PUBLIC=ROOT/'results/tubedetr_layout_trial';OUT=ROOT/'outputs/tubedetr_layout_trial';PLAN=PUBLIC/'plan.json'
ARMS=['Layout3OPD','Layout1OPD','Layout3Direct','GeometryDirect']


def now():return storage.now()


def prepare():
    prior=ROOT/'results/tubedetr_vid_development/plan.json';selected=ROOT/'results/tubedetr_vid_development/selected_profile.json'
    source=json.loads(prior.read_text());selection=json.loads(selected.read_text());cfg={**source['baseline_configuration'],**selection['parameters']}
    files=[ROOT/'tools/complete_tubedetr_layout_trial.py',ROOT/'tools/tubedetr_vid_development.py',ROOT/'tools/complete_tubedetr_vid_development.py',*sorted((ROOT/'tools/layout_trial').glob('*.py'))]
    plan=dict(status='frozen_fixed128_layout_reward_pilot',dataset='vidstg',backbone='TubeDETR',queries=128,parents=128,
        manifest='data/evaluation/tubedetr_vid_dev128.jsonl',manifest_sha256=source['manifest_sha256'],
        labels='data/evaluation/tubedetr_vid_dev128_labels.jsonl',labels_sha256=source['labels_sha256'],
        population_sha256=source['population_sha256'],seed=41,
        reference_full_manifest_sha256=sha(ROOT/'data/evaluation/vidstg.jsonl'),
        teacher_checkpoint='checkpoints/referdino/ryt_mevis_swinb.pth',
        teacher_checkpoint_sha256=json.loads((ROOT/'docs/assets.json').read_text())['required_files']['checkpoints/referdino/ryt_mevis_swinb.pth'],
        implementation_sha256=implementation_hash(),configuration=cfg,
        prior_selected_sha256=sha(selected),reference_profile=selection['profile'],
        reference_predictions='outputs/tubedetr_vid_development/full_selected/'+selection['profile'],
        output='outputs/tubedetr_layout_trial',code_sha256={str(f.relative_to(ROOT)):sha(f)for f in files},
        adaptation_arms=ARMS,geometry_OPD='exact existing selected-profile cache on identical128',
        feature='ReferDINO.pixel_decoder output from the same original eight-frame inference, stride4, float32',
        reward='mean cosine at ordered3x3 ROI cells; both ROIs sample one same-frame feature map',
        ROI_sampling='shared6x6 bilinear lattice, each2x2 average gives3x3; pooled1x1 averages same36 samples; normalize after pooling',
        coordinate_policy='original normalized cxcywh; resize preserves aspect; stride4 feature canvas valid_ratio; no box clipping, outside samples zero',
        temperature='one global value per layout reward, match initial32-candidate entropy to geometric OPD tau on fixed128; no GT',
        OPD='original Live loop/Adam/noise/sampling/mixture-gradient/reset; replace feedback weights only',
        initialization='same single ReferDINO eight-box consensus/E_Ref as current selected profile, unchanged',
        direct='same frozen student/content/initialization and same layout3 or geometry objective; equal fixed lr and60-step budget, no separate Direct optimization search',
        direct_comparison_scope='pilot only; not a fair-tuned final proof that OPD is necessary',
        new_expert_instances=1,expert_evidence_must_match_frozen_native_boxes_bitexact=True,
        geometry_reward_not_mixed_into_layout_weights=True,network_weights_frozen=True,per_query_reset=True,
        extra_full_population_evaluation=False,GT_parameter_search=False,old_queues_deferred=True,
        maximum_new_adaptation_paths=4*128+1,qualification='first fixed query reproduces current full geometric OPD exactly before new reward comparisons',
        reward_diagnostic='common32 policy candidates plus fixed teacher/source shifts, scales, aspect changes and native alternative-query boxes; offline GT ranking only',
        minimum_controls=['geometry OPD','pooled1 OPD','layout3 OPD','layout3 Direct','geometry Direct'],
        cross_attention_not_in_this_pilot=True,not_claimed_new_relative_to_historical_Fine_ROI=True,
        no_claim_of_adversarial_robustness_or_superiority=True,development_panel_and_full_target_previously_exposed=True,
        literature=[dict(title='Distilling Object Detectors With Fine-Grained Feature Imitation',year=2019,url='https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html'),
                    dict(title='Focal and Global Knowledge Distillation for Detectors',year=2022,url='https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html')])
    if PLAN.exists()and json.loads(PLAN.read_text())!=plan:raise ValueError('Frozen ROI pilot plan changed')
    if not PLAN.exists():write(PLAN,plan)
    return plan


def ready():
    path=ROOT/'results/tubedetr_vid_development/final_seal.json'
    if not path.exists():return False
    seal=json.loads(path.read_text())
    if seal['status']!='complete':return False
    if any(sha(ROOT/name)!=h for name,h in seal['files'].items()):raise ValueError('Current full final artifacts changed')
    jobs=json.loads((ROOT/'runtime/tubedetr_active_jobs.json').read_text())['jobs']
    for key in ['vid_development_controller','vid_development_gpu','hc2_development_controller','vid_spatial_followup_controller']:
        job=jobs.get(key)
        if not job:continue
        try:
            argv=[v.decode()for v in(Path('/proc')/str(job['pid'])/'cmdline').read_bytes().split(b'\0')if v]
            if argv==shlex.split(job['command']):return False
        except FileNotFoundError:pass
    return True


def launch(stage):
    command=[sys.executable,'-B','-u','tools/complete_tubedetr_layout_trial.py','--worker-stage',stage]
    log=ROOT/'logs'/f'tubedetr_layout_trial_{stage}.log'
    write(OUT/'status.json',dict(status=stage,as_of=now(),new_GPU_work_started=stage in ['features','adaptation'],labels_read=False))
    with log.open('a')as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        reg=ROOT/'runtime/tubedetr_active_jobs.json';j=json.loads(reg.read_text())
        j['jobs']['layout_trial_worker']=dict(pid=process.pid,command=shlex.join(command),log=str(log.relative_to(ROOT)),stage=stage)
        write(reg,j)
        if process.wait():raise RuntimeError('ROI pilot stage failed: '+str(log))


def bind_reference(plan):
    from tools.layout_trial.worker import records,reference,original
    root,seal=reference(plan);out=OUT/'predictions/GeometryOPD'
    identity=dict(configuration={**plan['configuration'],'reward':'GeometryOPD'},implementation_sha256=plan['implementation_sha256'],
        manifest_sha256=plan['manifest_sha256'],scope='fixed128_layout_reward_pilot',plan_sha256=sha(PLAN))
    for row in records(plan):
        if storage.saved(out,row,identity):continue
        value,r=original(row,root,seal);value.update(cost={**r['cost'],'exact_prediction_cache_reuse':True,'not_fresh_end_to_end_timing':True},
            checkpoint=r['checkpoint'],initialization=r['initialization'])
        save_prediction(out/'predictions'/f"{row['id']}.npz",value,dict(identity_sha256=digest(identity),request_sha256=digest(row)))
    storage.seal(out,records(plan),identity)


def main(wait):
    plan=prepare();OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runtime/tubedetr_layout_trial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    final=PUBLIC/'final_seal.json'
    if final.exists():
        s=json.loads(final.read_text())
        if s['status']!='complete' or any(sha(ROOT/f)!=h for f,h in s['files'].items()):raise ValueError('Final pilot seal changed')
        return
    while not ready():
        if(ROOT/'outputs/tubedetr_vid_development/failure.json').exists():raise RuntimeError('Current full failed; do not start ROI pilot')
        write(OUT/'status.json',dict(status='waiting_for_current_Vid_full_terminal_seal',as_of=now(),new_GPU_work_started=False,
            adaptation_queries=0,old_tuning_queues_deferred=True))
        if not wait:return
        time.sleep(30)
    for stage,completion in [('features','features_seal.json'),('calibration','diagnostic_seal.json'),('adaptation','adaptation_receipt.json')]:
        if not(OUT/completion).exists():launch(stage)
    bind_reference(plan)
    results={}
    for arm in ['GeometryOPD',*ARMS]:
        result=scoring.score(ROOT/plan['manifest'],ROOT/plan['labels'],OUT/'predictions'/arm,['Source','Spatial','Temporal','WithoutOPD','Full'])
        results[arm]=result
    # Independent CPU diagnostics after every adaptation is sealed.
    from tools.layout_trial.diagnostics import diagnose
    diagnostic=diagnose(PLAN);write(PUBLIC/'reward_diagnostics.json',diagnostic)
    write(PUBLIC/'temperature_calibration.json',json.loads((OUT/'temperature_calibration.json').read_text()))
    summary=dict(status='complete_fixed128_pilot',queries=128,parents=128,main_estimand='equal queries',arms={},
        Direct_is_fixed_budget_not_fair_tuned=True,cross_attention_not_evaluated=True,
        no_full_population_evaluation=True,no_adversarial_robustness_claim=True,development_exposure_disclosed=True)
    for arm,result in results.items():
        public=copy.deepcopy(result)
        for group in public['datasets'].values():
            for cluster in group['clusters']:cluster['parent']=digest(cluster['parent'])
        for row in public['rows']:row['parent']=digest(row['parent'])
        write(PUBLIC/'metrics'/f'{arm}.json',public)
        group=result['datasets']['vidstg'];summary['arms'][arm]=dict(means=group['query_means'],equal_parent_means=group['equal_parent_means'],
            denominators=group['query_denominators'],paired=group['paired'],prediction_seal_sha256=sha(OUT/'predictions'/arm/'sealed.json'))
    write(PUBLIC/'summary.json',summary)
    lines=['# TubeDETR ReferDINO ROI布局奖励固定128初试','',
        '单ReferDINO同帧有序布局；原OPD仅替换候选反馈，无几何奖励混合。全部128封存后离线评分。',
        'Direct仅相同固定预算，不是已公平调参的最终必要性证明；未评价Cross-attention，也不宣称对抗鲁棒。','',
        '| 方法 | m_vIoU | vIoU@0.3 | vIoU@0.5 | m_sIoU | 观测sIoU | 未观测sIoU |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,v in summary['arms'].items():
        m=v['means']['Full'];lines.append('| '+arm+' | '+' | '.join(f'{100*m[k]:.5f}'if m[k]is not None else'null'for k in
            ['m_vIoU','vIoU@0.3','vIoU@0.5','m_sIoU','observed_actual_sIoU','unobserved_actual_sIoU'])+' |')
    lines+=['','奖励排序诊断：'+json.dumps(diagnostic['rewards'],ensure_ascii=False),
        '本面板和原完整目标结果已曝光。伪损失下降、教师自匹配或单个开发增量不能证明真实定位改善。旧Fine ROI负结果没有被此实现准备所推翻。',
        '特征缓存、温度和推理均不读GT；GT只在完整封存后的评分/奖励排序使用。Strong及原生输出缓存复用，运行成本不冒充新端到端部署效率。','']
    (PUBLIC/'report.md').write_text('\n'.join(lines))
    files=[PLAN,PUBLIC/'summary.json',PUBLIC/'report.md',PUBLIC/'reward_diagnostics.json',PUBLIC/'temperature_calibration.json',
           *sorted((PUBLIC/'metrics').glob('*.json'))]
    write(final,dict(status='complete',queries=128,files={str(f.relative_to(ROOT)):sha(f)for f in files},
        current_full_unchanged=True,pilot_only=True,publication_pending=True))
    write(OUT/'status.json',dict(status='complete',queries=128,publication_pending=True,old_tuning_queues_deferred=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker-stage',choices=['features','calibration','adaptation']);a=parser.parse_args()
    try:
        if a.worker_stage:
            if a.worker_stage=='calibration':
                from tools.layout_trial.calibration import calibrate
                calibrate(PLAN)
            else:
                from tools.layout_trial.worker import feature_worker,adapt_worker
                (feature_worker if a.worker_stage=='features'else adapt_worker)(PLAN)
        elif a.prepare_only:print(json.dumps(prepare(),indent=2))
        else:main(a.wait)
    except BaseException as e:
        value=dict(status='failed',error_type=type(e).__name__,message=str(e),as_of=now());write(OUT/'failure.json',value);write(OUT/'status.json',value);raise
