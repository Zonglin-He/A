"""Only the frozen main-method stages; historical experiment launchers are absent."""
import argparse
from . import pipeline

STAGES=('source','inputs','temporal_teacher','temporal_fit','temporal_readout',
    'spatial_inputs','spatial_teacher','spatial_student','score')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare',)+STAGES)
    p.add_argument('--scope',default='engineering32')
    a=p.parse_args()
    pipeline.path(a.scope)
    if a.stage!='prepare':
        import sys
        from pathlib import Path
        from .runtime import interpreter
        expected=Path(interpreter(a.stage)).parent.parent.resolve()
        assert Path(sys.prefix).resolve()==expected, 'use the stage-specific compatible environment'
    lease=None
    if a.stage in ('temporal_teacher','spatial_teacher'):
        # The Student loader imports its pinned old Transformers stack. It must
        # never be installed into the Qwen process, including during lease setup.
        from scripts.run_final_simplification_v1 import lease as acquire
        from scripts.stvg_native_instance_common_v1 import guard
        import sys
        sys.addaudithook(guard)
        lease=acquire()
    elif a.stage in ('source','temporal_readout','spatial_student'):
        from scripts.run_decota_paper_main_v1 import gpu
        lease=gpu()
    if a.stage=='prepare':pipeline.prepare()
    else:getattr(pipeline,a.stage)(a.scope)

if __name__=='__main__':main()
