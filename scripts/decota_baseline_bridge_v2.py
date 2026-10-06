"""Process-local registry verification override; original baseline code intact."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.decota_opd_tuning_common_v1 import bridge,verify

if __name__=='__main__':
    verify();bridge();action=sys.argv[1]
    if action=='resume_TENT_vidstg':
        from scripts.resume_decota_paper_baseline_saved_v1 import run
        run()
    else:
        from scripts.run_decota_paper_baselines_v1 import lock,smoke,fisher,execute
        if action=='lock':lock()
        elif action.startswith('smoke_'):
            _,method,ds=action.split('_');smoke(method,ds)
        elif action.startswith('fisher_'):fisher(action.split('_',1)[1])
        else:execute(*action.split('_',1))
