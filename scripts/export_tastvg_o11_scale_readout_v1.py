"""Allowlisted fixed-readout code and anonymous scalar evidence."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_o11_scale_readout_v1';DEST=ROOT/'artifacts/github_public_A_o11_scale_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A');RESULT='results/tastvg_o11_scale_readout/2026-09-29'

def run():
    files={}
    def add(rel,data):
        f=DEST/'files'/rel;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();f.write_bytes(data);files[rel]=dict(sha256=sha(f),bytes=len(data))
    for rel in ['scripts/run_tastvg_o11_scale_readout_v1.py','scripts/audit_tastvg_o11_public_v1.py','scripts/report_tastvg_o11_scale_readout_v1.py','scripts/export_tastvg_o11_scale_readout_v1.py','protocols/tastvg_o11_scale_readout_v1.md']:add(rel,(ROOT/rel).read_bytes())
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for name in ['ROWS.json','SUMMARY.json','REFERENCE.json','PUBLIC_AUDIT.json','DECISION.json','RESOURCES.json','PROVENANCE.json','SELECTION_BARRIER.json','AGREEMENT_BARRIER.json','SCORE_RECEIPT.json']:
        add(f'{RESULT}/{name}',(json.dumps(anon(read(OUT/name)),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    add(f'{RESULT}/REPORT.md',(OUT/'REPORT.md').read_bytes());add('docs/TA_O11_SCALE_READOUT_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1)
    entry='## Latest TA-STVG: O1.1 fixed-state scale readout completed\n\n[Report](results/tastvg_o11_scale_readout/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_O11_SCALE_READOUT_UPDATE.md), [protocol](protocols/tastvg_o11_scale_readout_v1.md). CPU-only alpha1/8/16/32 on the saved O1 arrival states: many choices move, but no amplified scale improves mean vIoU. No retraining, new expert calls, online trajectory or normalization rerun. Earlier latest/running entries below are historical snapshots.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),result_directory=RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','raw hidden features and state tensors','private conversations']))
    for rel in files:
        f=PUBLIC/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,f)
    print('EXPORTED',len(files),'files')

if __name__=='__main__':run()
