"""Original complete-model M2 fit/redecode on newly sealed Native references."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_component_native007_common import BASE, verify

if __name__ == '__main__':
    verify()
    from scripts import run_stvg_soft_ep_coordinate_trial003 as worker
    from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
    from scripts import stvg_component_spatial_fit_v1 as ablation
    original = dispatch.fit
    def fitter(*args, **kwargs):
        return ablation.fit(*args, **kwargs, flags={})
    try:
        dispatch.fit = fitter
        worker.TRIALS = BASE / 'trials'
        worker.MODES = ('M2',)
        worker.ARM_NAMES = ('F', 'T', 'S_M2', 'TS_M2')
        sys.argv = ['native_student', 'N000']
        worker.main()
    finally:
        dispatch.fit = original
