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
    lease=None
    if a.stage in ('source','temporal_teacher','temporal_readout','spatial_teacher','spatial_student'):
        from scripts.run_decota_paper_main_v1 import gpu
        lease=gpu()
    if a.stage=='prepare':pipeline.prepare()
    else:getattr(pipeline,a.stage)(a.scope)

if __name__=='__main__':main()
