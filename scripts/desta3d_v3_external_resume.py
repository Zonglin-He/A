"""Resume the unchanged pixel-baseline stages with the latent-oracle receipt included.

This same-process entry point changes accounting/registration only, never data,
loader, recipe, geometry, predictions or scores. No old scientific pins are edited.
"""
import argparse,importlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta import external_qualification_io as io
from vg_tta.desta3d_v3_oracle_io import total_prior
STAGES={'smoke':'scripts.desta3d_v3_external_loader_smoke','teacher':'scripts.desta3d_v3_external_teacher_qualification','pixel':'scripts.desta3d_v3_privileged_ptd_qualification'}

def install_accounting():
    io.prior_seconds=total_prior
    original=io.register_base
    def register(name,config,rows,paths):
        return original(name,{**config,'accounting_entry':'scripts/desta3d_v3_external_resume.py','includes_latent_oracle_receipts':True},rows,
            [*paths,Path(__file__),ROOT/'protocols/desta3d_v3_external_resume_accounting_v1.md'])
    io.register_base=register

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=STAGES);p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);p.add_argument('--smoke');p.add_argument('--teacher')
    a=p.parse_args();install_accounting();module=importlib.import_module(STAGES[a.stage])
    if a.action=='run':
        cfg=io.read(io.OUT/a.name/'CONFIG.json');assert cfg['includes_latent_oracle_receipts'] and cfg['accounting_entry']=='scripts/desta3d_v3_external_resume.py'
        io.check_pins(io.read(io.OUT/a.name/'LOCK.json')['pins']);module.run(a.name)
    elif a.stage=='smoke':module.register(a.name)
    elif a.stage=='teacher':module.register(a.name,a.smoke)
    else:module.register(a.name,a.teacher)
if __name__=='__main__':main()
