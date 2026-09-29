"""Allowlisted public export: code and anonymous scalars, never media/labels."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2'
DEST=ROOT/'artifacts/github_public_A_deployment_c05_c2t_20260929'
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
RESULT='results/tastvg_deployment_c05_c2t/2026-09-29'

def run():
    files={}
    def add(rel,data):
        f=DEST/'files'/rel;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();f.write_bytes(data);files[rel]=dict(sha256=sha(f),bytes=len(data))
    code=['vg_tta/tastvg_temporal_qualification_v1.py','vg_tta/tastvg_deployment_corruption_v2.py',
          'scripts/run_tastvg_c05_v1.py','scripts/run_tastvg_c05_v2.py','scripts/run_tastvg_c2t_v2.py',
          'scripts/analyze_tastvg_c05_v2.py','scripts/analyze_tastvg_c2t_v2.py','scripts/audit_tastvg_deployment_v2.py',
          'scripts/audit_tastvg_deployment_public_v2.py','scripts/report_tastvg_deployment_v2.py','scripts/export_tastvg_deployment_v2.py',
          'tests/test_tastvg_temporal_qualification_v1.py','protocols/tastvg_c05_c2t_v1.md','protocols/tastvg_deployment_c05_c2t_v2.md']
    for f in code:add(f,(ROOT/f).read_bytes())
    for f in sorted((OUT/'analysis').glob('*.json')):
        d=read(f)
        def anon(x):
            if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
            if isinstance(x,list):return [anon(v) for v in x]
            return x
        add(f'{RESULT}/{f.name}',(json.dumps(anon(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    for name in ['REPORT.md','C05_EFFECTS.svg','RESOURCES.json','PROVENANCE.json','C05_DECISION.json','C2_DECISION.json','C05_AUDIT.json','C2_AUDIT.json','INDEPENDENT_C05_AUDIT.json','CPU_TESTS.txt']:
        add(f'{RESULT}/{name}',(OUT/name).read_bytes())
    add(f'{RESULT}/SUPERSEDED_GT_CENTER_SMOKE.json',(ROOT/'artifacts/tastvg_c05_c2t_v1/SUPERSEDED.json').read_bytes())
    context=(OUT/'RESEARCH_UPDATE.md').read_bytes();add('docs/TA_DEPLOYMENT_C05_C2T_UPDATE.md',context)
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(z['bytes'] for z in files.values()),result_directory=RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','raw predictions and features','private conversation attachments']))
    for rel in files:
        f=PUBLIC/rel;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,f)
    print('EXPORTED',len(files),'files')

if __name__=='__main__':run()
