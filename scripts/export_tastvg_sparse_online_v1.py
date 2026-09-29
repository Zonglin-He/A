"""Allowlisted O1 code and anonymous scalar results; private tensors stay local."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_sparse_online_o1_v1';DEST=ROOT/'artifacts/github_public_A_sparse_online_o1_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A');RESULT='results/tastvg_sparse_online_o1/2026-09-29'

def run():
    files={}
    def add(rel,data):
        f=DEST/'files'/rel;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();f.write_bytes(data);files[rel]=dict(sha256=sha(f),bytes=len(data))
    code=['vg_tta/tastvg_sparse_online_v1.py','scripts/run_tastvg_sparse_online_capture_v1.py','scripts/run_tastvg_sparse_online_teacher_v1.py','scripts/run_tastvg_sparse_online_v1.py',
          'scripts/audit_tastvg_sparse_online_v1.py','scripts/analyze_tastvg_sparse_online_v1.py','scripts/audit_tastvg_sparse_online_public_v1.py','scripts/report_tastvg_sparse_online_v1.py',
          'scripts/diagnose_tastvg_sparse_online_readout_v1.py','scripts/export_tastvg_sparse_online_v1.py','tests/test_tastvg_sparse_online_v1.py','protocols/tastvg_sparse_online_o1_v1.md']
    for f in code:add(f,(ROOT/f).read_bytes())
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for f in sorted((OUT/'analysis').glob('*.json')):add(f'{RESULT}/{f.name}',(json.dumps(anon(read(f)),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    for name in ['REPORT.md','STATE_AUDIT.json','UTILITY_AUDIT.json','PUBLIC_AUDIT.json','RESOURCES.json','DECISION.json','PROVENANCE.json','CPU_TESTS.txt']:add(f'{RESULT}/{name}',(OUT/name).read_bytes())
    add('docs/TA_SPARSE_ONLINE_O1_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1)
    entry='## Latest TA-STVG: O1 sparse-critic online transfer completed\n\n[Report](results/tastvg_sparse_online_o1/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPARSE_ONLINE_O1_UPDATE.md), [protocol](protocols/tastvg_sparse_online_o1_v1.md), [persistent ranker](vg_tta/tastvg_sparse_online_v1.py). One32-source stream, eight specialist writes; primary future non-expert t/v gain is exactly zero in this fixed first implementation. State and chronology are verified; no added mechanisms or production promotion. Earlier latest/running statements below are historical snapshots.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),result_directory=RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','raw hidden features and online state tensors','private conversations']))
    for rel in files:
        f=PUBLIC/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,f)
    print('EXPORTED',len(files),'files')

if __name__=='__main__':run()
