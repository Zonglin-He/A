"""Explicit public allowlist: implementation and anonymous results only."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_temporal_fourarm_v1'
DEST=ROOT/'artifacts/github_public_A_temporal_fourarm_20260929'
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
RESULT='results/tastvg_temporal_fourarm/2026-09-29'

def run():
    files={}
    def add(rel,data):
        f=DEST/'files'/rel;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();f.write_bytes(data);files[rel]=dict(sha256=sha(f),bytes=len(data))
    code=['scripts/run_tastvg_transient_capture_v1.py','scripts/run_tastvg_fourarm_critic_v1.py','scripts/run_tastvg_temporal_fourarm_v1.py',
          'vg_tta/tastvg_temporal_fourarm_v1.py','scripts/analyze_tastvg_temporal_fourarm_v1.py','scripts/audit_tastvg_fourarm_public_v1.py',
          'scripts/report_tastvg_temporal_fourarm_v1.py','scripts/audit_tastvg_fourarm_losses_v1.py','scripts/export_tastvg_fourarm_v1.py','tests/test_tastvg_temporal_fourarm_v1.py',
          'protocols/tastvg_temporal_fourarm_v1.md','protocols/tastvg_transient_c25_c3_c06_v1.md']
    for f in code:add(f,(ROOT/f).read_bytes())
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for f in sorted((OUT/'analysis').glob('*.json')):add(f'{RESULT}/{f.name}',(json.dumps(anon(read(f)),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    for name in ['REPORT.md','RESOURCES.json','PROVENANCE.json','DECISION.json','AUDIT.json','PUBLIC_SCALAR_AUDIT.json','CPU_TESTS.txt','LOSS_READBACK.json']:
        add(f'{RESULT}/{name}',(OUT/name).read_bytes())
    revision=read(ROOT/'artifacts/tastvg_transient_c25_c3_c06_v1/SCOPE_REVISION.json');add(f'{RESULT}/SCOPE_REVISION.json',(json.dumps(revision,indent=2)+'\n').encode())
    add('docs/TA_TEMPORAL_FOURARM_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    navigation=(PUBLIC/'REVIEW_START_HERE.md').read_text()
    entry='## Latest TA-STVG: minimal temporal four-arm experiment completed\n\n[Results](results/tastvg_temporal_fourarm/2026-09-29/REPORT.md), [current protocol](protocols/tastvg_temporal_fourarm_v1.md), [one-step implementation](vg_tta/tastvg_temporal_fourarm_v1.py), [anonymous per-cell results](results/tastvg_temporal_fourarm/2026-09-29/ROWS.json). Frozen/Rerank/Hard/OPD on the unchanged transient setting plus clean, only motion-H updates. Earlier C2.5 gates and C0.6 multiseed plans are superseded; no production change. Older running/next statements below are historical snapshots.\n\n'
    first,rest=navigation.split('\n',1);add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(z['bytes'] for z in files.values()),result_directory=RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','raw predictions and features','private conversation attachments']))
    for rel in files:
        f=PUBLIC/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,f)
    print('EXPORTED',len(files),'files')

if __name__=='__main__':run()
