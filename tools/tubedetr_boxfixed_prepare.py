"""Bind already fixed128 inputs and GitHub method before any new profile score."""
from pathlib import Path
import json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    r=ROOT;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();origin=json.loads((r/'results/tubedetr_boxfixed_development/upstream_port.json').read_text());main=json.loads(subprocess.check_output(['git','show',origin['upstream_commit']+':configs/default.json'],cwd=r));tube=json.loads((r/'configs/tubedetr.json').read_text());datasets={}
    for ds,name,reference in [('vidstg','vid','hc2_to_vid'),('hc2','hc2','vid_to_hc2')]:
        old=json.loads((r/f'results/tubedetr_{name}_development/plan.json').read_text());manifest=f'data/evaluation/tubedetr_{name}_dev128.jsonl';labels=f'data/evaluation/tubedetr_{name}_dev128_labels.jsonl';full=f'data/evaluation/{ds}.jsonl';fl=f'data/evaluation/{ds}_labels.jsonl';assert sha(r/manifest)==old['manifest_sha256']and sha(r/labels)==old['labels_sha256'];assert len((r/manifest).read_text().splitlines())==128
        config={**main['datasets'][ds],**tube['datasets'][ds],**{k:tube['common'][k]for k in ['stride','native_resolution','upstream_revision']},'backbone':'TubeDETR','temporal_mode':'native_Global6'}
        datasets[ds]=dict(manifest=manifest,manifest_sha256=sha(r/manifest),labels=labels,labels_sha256=sha(r/labels),full_manifest=full,full_manifest_sha256=sha(r/full),full_labels=fl,full_labels_sha256=sha(r/fl),full_queries=len((r/full).read_text().splitlines()),full_parents=732 if ds=='vidstg' else 227,development_queries=128,development_parents=128,old_reference='outputs/tubedetr/'+reference,old_prediction_seal_sha256=sha(r/'outputs/tubedetr'/reference/'sealed.json'),configuration=config,old_development_plan_sha256=sha(r/f'results/tubedetr_{name}_development/plan.json'))
    configuration=dict(method='TubeDETR BoxFixed + native224 Global6 + physical MBR',upstream_commit=origin['upstream_commit'],datasets={ds:d['configuration']for ds,d in datasets.items()},fixed_affinity=dict(temperature=.1,topk=20),fixed_temporal=dict(native_resolution=224,layers=6,temperature=1,no448=True,no_fake_offsets=True))
    cfg=r/'configs/tubedetr_boxfixed.json';text=json.dumps(configuration,ensure_ascii=False,indent=2)+'\n'
    if cfg.exists():assert cfg.read_text()==text
    else:cfg.write_text(text)
    code=[str(f.relative_to(r))for f in(r/'tools/boxfixed_tube').glob('*.py')]+['tools/tubedetr_boxfixed_development.py','tools/tubedetr_boxfixed_qualify.py','tools/tubedetr_boxfixed_prepare.py']+[str(f.relative_to(r))for f in(r/'stvg').glob('*.py')]+['tests/test_boxfixed_tube_upstream.py','tests/test_boxfixed_tube_profiles.py','tests/test_boxfixed_tube_isolation.py']
    teacher_fp=hashlib.sha256(json.dumps({f:sha(r/f)for f in ['stvg/teacher.py','tools/boxfixed_tube/teacher_worker.py','checkpoints/referdino/ryt_mevis_swinb.pth']},sort_keys=True).encode()).hexdigest()
    p=dict(protocol_version=2,engineering_recovery_sha256=sha(r/'results/tubedetr_boxfixed_development/engineering_recovery.json'),status='frozen_before_new128_outcomes',date='2026-10-22',upstream_commit=origin['upstream_commit'],upstream_source_provenance_sha256=sha(r/'results/tubedetr_boxfixed_development/upstream_port.json'),interface_qualification_sha256=sha(r/'results/tubedetr_boxfixed_development/interface_qualification.json'),configuration_sha256=sha(cfg),method=configuration['method'],datasets=datasets,dataset_order=['vidstg','hc2'],code_sha256={f:sha(r/f)for f in sorted(code)},teacher_fingerprint=teacher_fp,gamma_grid=[0.,.0125,.025,.05,.1,.25,.5,.75,1.],admission_grid=[.05,.1,.2,.4],guard_grid=[.025,.05,.1,.2,.4],link_grid=[0.,.125,.25,.5],max_profiles_per_dataset=24,max_coupling_profiles=2,feature_affinity=configuration['fixed_affinity'],temporal=configuration['fixed_temporal'],one_teacher=True,no_optimizer_or_OPD=True,all_old_results_immutable=True,GT_only_after_each_complete_prediction_seal=True,full_evaluation_after_both128_profile_locks=True,no_full_feedback_into_tuning=True,all_development_and_prior_full_exposure_disclosed=True,selection_objective='maximum complete128 equal-query Full m_vIoU with original profile always eligible; exact tie retains original')
    path=r/'results/tubedetr_boxfixed_development/plan.json';text=json.dumps(p,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text()==text,'Already frozen plan differs'
    else:path.write_text(text)
    print(sha(path))

if __name__=='__main__':main()
