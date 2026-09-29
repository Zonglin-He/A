"""Publish current O3 plus separately labeled superseded measurements, scalar only."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_conditional_opd_o3_v1';LEGACY=ROOT/'artifacts/tastvg_preference_memory_o3_v1';DEST=ROOT/'artifacts/github_public_A_conditional_opd_o3_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')
RESULT='results/tastvg_conditional_opd_o3/2026-09-29';OLD_RESULT='results/tastvg_preference_memory_superseded/2026-09-29'


def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    paths=['vg_tta/tastvg_conditional_opd_v1.py','scripts/run_tastvg_conditional_opd_o3_v1.py','scripts/audit_tastvg_conditional_opd_o3_v1.py','scripts/score_tastvg_conditional_opd_o3_v1.py','scripts/report_tastvg_conditional_opd_o3_v1.py','scripts/audit_tastvg_conditional_opd_public_v1.py','scripts/export_tastvg_conditional_opd_o3_v1.py','protocols/tastvg_conditional_opd_o3_v1.md','tests/test_tastvg_conditional_opd_v1.py',
      'vg_tta/tastvg_preference_memory_v1.py','scripts/run_tastvg_preference_memory_o3_v1.py','scripts/audit_tastvg_preference_memory_o3_v1.py','scripts/audit_tastvg_preference_memory_public_v1.py','protocols/tastvg_preference_memory_o3_v1.md','tests/test_tastvg_preference_memory_v1.py']
    for f in paths:add(f,(ROOT/f).read_bytes())
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    def jsonadd(rel,x):add(rel,(json.dumps(anon(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
    for n in ['ROWS.json','MACRO_ROWS.json','SUMMARY.json','DIAGNOSTICS.json','LEARNING.json','PUBLIC_AUDIT.json','STATE_AUDIT.json','SCORE_RECEIPT.json','RESOURCES.json','DECISION.json','PROVENANCE.json']:jsonadd(f'{RESULT}/{n}',read(OUT/n))
    for n in ['REPORT.md','CPU_TESTS.txt']:add(f'{RESULT}/{n}',(OUT/n).read_bytes())
    oldrows=[{k:v for k,v in r.items() if k!='pairs'} for r in read(LEGACY/'ROWS.json')];jsonadd(f'{OLD_RESULT}/ROWS.json',oldrows)
    for n in ['SUMMARY.json','STATE_AUDIT.json','PUBLIC_AUDIT.json','SCORE_RECEIPT.json','SCOPE_REVISION.json']:jsonadd(f'{OLD_RESULT}/{n}',read(LEGACY/n))
    add(f'{OLD_RESULT}/REPORT.md',(LEGACY/'REPORT.md').read_bytes())
    add('docs/TA_CONDITIONAL_OPD_O3_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    jsonadd(f'{RESULT}/S0_SCOPE_PAUSE.json',read(ROOT/'artifacts/tastvg_spatial_expansion_s0_v1/SCOPE_PAUSE.json'))
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1)
    entry='## Latest TA-STVG: O3 Conditional Online OPD completed\n\n[Report](results/tastvg_conditional_opd_o3/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_CONDITIONAL_OPD_O3_UPDATE.md), [protocol](protocols/tastvg_conditional_opd_o3_v1.md). Matched768->128->1 Pairwise/Reverse-KL students on original O2 streams:40 verified CPU SGD updates, but both preserve all60 nonexpert choices and yield zero t/v gain. Reverse-KL remains an implementation candidate per user tie preference, not a validated benefit or production promotion. [Earlier KNN measurements](results/tastvg_preference_memory_superseded/2026-09-29/REPORT.md) completed before steering and are separately archived as superseded. S0 deferred; no spatial inference. Earlier latest/running entries below are historical.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files.values()),result_directory=RESULT,superseded_directory=OLD_RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','raw cached features and online state tensors','full KNN per-pair neighbor traces','private conversations']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(x['bytes'] for x in files.values()))

if __name__=='__main__':run()
