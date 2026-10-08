"""Bind already authorized later rosters to the new HC2 lock, without inference/GT."""
import copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import *

def run():
    activate()
    import scripts.prepare_stvg_opd_later_stages_v1 as prep
    prep.prepare()
    if not (BASE/'APPENDIX_STAGE_LOCK.json').exists():
        d=read(OLD/'APPENDIX_STAGE_LOCK.json');d=copy.deepcopy(d)
        d.update(original_appendix_lock_sha256=sha(OLD/'APPENDIX_STAGE_LOCK.json'),
            current_HC2_main_config_may_differ=True,not_selected_on_new_confirmation=True,time=time.time())
        write(BASE/'APPENDIX_STAGE_LOCK.json',d)
    if not (BASE/'COMPONENT_RUNTIME_LOCK.json').exists():
        original=read(OLD/'COMPONENT_RUNTIME_LOCK.json')
        for f,h in original['pins'].items():assert sha(ROOT/f)==h,f
        write(BASE/'COMPONENT_RUNTIME_LOCK.json',dict(**{k:v for k,v in original.items() if k not in ['design_sha256','time']},
            design_sha256=sha(BASE/'LATER_DESIGN_LOCK.json'),
            original_component_runtime_sha256=sha(OLD/'COMPONENT_RUNTIME_LOCK.json'),
            only_human_authorized_HC2_configuration_revision=True,time=time.time()))
    if not (BASE/'COMPONENT_RUNTIME_LOCK_revision001.json').exists():
        files=['scripts/stvg_opd_paper_later_common_v1.py','scripts/run_stvg_opd_paper_components_v2.py',
            'scripts/run_stvg_opd_revision_later_v2.py','scripts/continue_stvg_opd_revision_later_v2.py',
            'scripts/test_stvg_opd_later_contracts_v1.py','scripts/prepare_stvg_opd_revision_later_v2.py']
        write(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json',dict(status='prepared_not_GPU_qualified',
            original_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK.json'),
            appendix_design_sha256=sha(BASE/'APPENDIX_STAGE_LOCK.json'),pins={f:sha(ROOT/f) for f in files},time=time.time()))
    if not (BASE/'LATER_CPU_RUNTIME_LOCK.json').exists():
        files=['scripts/score_stvg_opd_later_phase_v1.py','scripts/finalize_stvg_opd_later_phase_v1.py',
            'scripts/run_stvg_opd_revision_later_v2.py','scripts/test_stvg_opd_later_contracts_v1.py']
        write(BASE/'LATER_CPU_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},GT_after_all_phase_deployment_arms=True,time=time.time()))
    write(BASE/'LATER_IMPLEMENTATION_PROGRESS.json',dict(status='prepared_not_executed',P2_P5_runners=True,
        P6_deployment_and_offline_oracle_runner_still_requires_root_implementation=True,
        no_new_GPU_or_GT_work=True,original_preparations_preserved=True,time=time.time()))
    print('REVISION_LATER_PLANS_PINNED_NOT_EXECUTED')

if __name__=='__main__':run()
