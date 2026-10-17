"""Original noCoverage algorithm, only complementary224 development parents."""
from pathlib import Path
import copy,hashlib,json,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as old
BASE=ROOT/'artifacts/stvg_caption_coverage256_v1';EXP=BASE/'expansion224';CORE=old.CORE
read,sha,put=old.read,old.sha,old.write
def verify():
    old.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME001.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==read(BASE/'STUDY_LOCK.json')['CURRENT_sha256']
def setup():
    BASE.mkdir(exist_ok=True);allr=read(CORE/'ROSTER_LOCK.json')['records'];first=read(old.BASE/'ablation_noCoverage/ROSTER_LOCK.json')['records'];ids={r['probe_alias'] for r in first};rr=[r for r in allr if r['probe_alias'] not in ids];assert len(ids)==32 and len(rr)==224 and all(sum(r['dataset']==ds for r in rr)==112 for ds in ('hc2','vidstg'));EXP.mkdir(exist_ok=False)
    put(EXP/'ROSTER_LOCK.json',dict(status='locked',count=224,per_domain=112,records=rr,selection='exact core256 complement of existing fixed noCoverage32; all development, no new confirmation',GT_read=False))
    cfg=read(old.BASE/'ablation_noCoverage/SCOPE_CONFIG.json');put(EXP/'SCOPE_CONFIG.json',cfg)
    put(BASE/'STUDY_LOCK.json',dict(status='locked',count=256,initial_count=32,additional_count=224,primary='B-noCoverage minus same-source A',secondary='B-noCoverage minus same-source B',reports=['initial32','additional224','complete256'],single_intervention='delete local candidate-priority coverage loop only',same_original_Prompt_clock_global64_local96_and_Student=True,missing_caption_fallback_A='saved-prediction whole-query offline diagnostic only, not combined intervention',old_confirmation_negative_preserved=True,no_new_Global_C_BT_Prompt_LR_budget_search=True,no_auto_new_confirmation_or_fullset=True,semantic_panel_sha256=sha(CORE/'SEMANTIC_PANEL_LOCK.json'),CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),existing32_result_sha256=sha(old.BASE/'ablation_noCoverage/POSTSEAL_RESULTS.json'),existing32_root_sha256=sha(old.BASE/'ablation_noCoverage/ROOT_COMPONENT_READBACK002.json'),old32_predictions_not_recomputed=True,only_224_new_Spatial_replays=True,GT_read=False))
    pins=copy.deepcopy(read(old.BASE/'EXECUTION_RUNTIME001.json')['pins'])
    for p in [Path(__file__),ROOT/'scripts/continue_stvg_coverage224_v1.py',ROOT/'scripts/score_stvg_coverage256_v1.py',ROOT/'scripts/stvg_overnight_global_v1.py',ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/stvg_overnight_score_v1.py',EXP/'ROSTER_LOCK.json',EXP/'SCOPE_CONFIG.json',BASE/'STUDY_LOCK.json',CORE/'SEMANTIC_PANEL_LOCK.json',ROOT/'scripts/stvg_pipeline_speed_v1.py']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=224,pins=pins,GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',stage='await_saved_caption_decomposition',count=224,GT_read=False));print('COMPLEMENT224_AND_FROZEN_SINGLE_INTERVENTION_LOCKED')
def configure(module):
    verify();cfg=read(EXP/'SCOPE_CONFIG.json');module.BASE=EXP;module.CORE=CORE;module.CFG=cfg;module.N=224
    proxy=types.SimpleNamespace(BASE=BASE,CORE=CORE,OLDGLOBAL=old.OLDGLOBAL,verify=verify,read=read,write=put,sha=sha)
    if hasattr(module,'common'):module.common=proxy
    return module
def run(stage):
    if stage in ('jobs','qualification','teacher','teacher_root'):
        from scripts import stvg_overnight_global_v1 as g
        configure(g);return dict(jobs=g.jobs,qualification=g.qualify,teacher=g.teacher,teacher_root=g.teacher_root)[stage]()
    from scripts import stvg_overnight_student_v1 as s
    configure(s);return dict(student_prepare=s.prepare,student_worker=s.worker,student_root=s.root)[stage]()
if __name__=='__main__':setup() if sys.argv[1]=='setup' else run(sys.argv[1])
