"""Explicit S0 code/scalar allowlist, excluding all private visual/label caches."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_spatial_expansion_s0_v1';DEST=ROOT/'artifacts/github_public_A_spatial_s0_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A');RESULT='results/tastvg_spatial_expansion_s0/2026-09-29'

def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for rel in ['protocols/tastvg_spatial_expansion_s0_v1.md','vg_tta/tastvg_spatial_expansion_s0_v1.py','scripts/run_tastvg_spatial_expansion_s0_v1.py','scripts/score_tastvg_spatial_expansion_s0_v1.py','scripts/audit_tastvg_spatial_reinsertion_s0_v1.py','scripts/summarize_tastvg_spatial_s0_components_v1.py','scripts/report_tastvg_spatial_expansion_s0_v1.py','scripts/audit_tastvg_spatial_expansion_public_v1.py','scripts/export_tastvg_spatial_expansion_s0_v1.py','scripts/download_sa2va_s0_weights.py','tests/test_tastvg_spatial_expansion_s0_v1.py']:add(rel,(ROOT/rel).read_bytes())
    for n in ['ROWS.json','SUMMARY.json','AUDIT.json','PUBLIC_AUDIT.json','RESOURCES.json','PROVENANCE.json','DECISION.json','REINSERTION_AUDIT.json','H_AUDIT.json','EXPERT_COVERAGE.json','COMPONENTS.json']:
        x=read(OUT/n)
        if n=='RESOURCES.json':x['failures']=[dict(status=a['status'],stage=a['stage'],done=a['done'],seconds=a['seconds'],error=a['failure']['error']) for a in x['failures']]
        add(f'{RESULT}/{n}',(json.dumps(anon(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    for n in ['REPORT.md','CPU_TESTS.txt','REPRODUCE.md']:add(f'{RESULT}/{n}',(OUT/n).read_bytes())
    add('docs/TA_SPATIAL_EXPANSION_S0_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1)
    entry='## Latest TA-STVG: S0 RVOS-guided spatial expansion\n\n[Report](results/tastvg_spatial_expansion_s0/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPATIAL_EXPANSION_S0_UPDATE.md), [protocol](protocols/tastvg_spatial_expansion_s0_v1.md). Real Sa2VA-4B sparse-video masks guide three fixed appearance-only native trajectories on16 previously exposed sources × clean/five existing5% corruptions. Native B0 always retained; whole-tube spatial oracle compared against matched six-layer support. This measures candidate headroom, not a deployed selector or online-TTA benefit. Temporal research frozen at native candidates + UniversalVTG reranking; prior OPD/KNN reports remain historical. No S1 or production change.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files.values()),result_directory=RESULT,exclusions=['media','captions','source IDs','GT coordinates','weights','masks','H and native raw predictions','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(x['bytes'] for x in files.values()))

if __name__=='__main__':run()
