"""Publish explicit v2 and superseded-v1 code/scalar allowlist, never raw data."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
DEST=ROOT/'artifacts/github_public_A_spatial_s05_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')

def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for v in ['v1','v2']:
        out=ROOT/f'artifacts/tastvg_spatial_propagation_s05_{v}';result=f'results/tastvg_spatial_propagation_s05_superseded_{v}/2026-09-29'
        for rel in [f'protocols/tastvg_spatial_propagation_s05_{v}.md',f'vg_tta/tastvg_spatial_propagation_s05_{v}.py',f'scripts/run_tastvg_spatial_propagation_s05_{v}.py',f'scripts/score_tastvg_spatial_propagation_s05_{v}.py',f'scripts/audit_tastvg_spatial_propagation_s05_{v}.py',f'scripts/report_tastvg_spatial_propagation_s05_{v}.py',f'tests/test_tastvg_spatial_propagation_s05_{v}.py']:add(rel,(ROOT/rel).read_bytes())
        for n in ['ROWS.json','SUMMARY.json','COMPONENTS.json','COMPARISON.json','AUDIT.json','PUBLIC_AUDIT.json','PAIRED_PUBLIC_AUDIT.json','PROPAGATION_AUDIT.json','REINSERTION_AUDIT.json','RESOURCES.json','PROVENANCE.json','DIAGNOSTICS.json','DECISION.json']+['SCOPE_REVISION.json']:add(f'{result}/{n}',(json.dumps(anon(read(out/n)),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        for n in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:
            data=(out/n).read_bytes()
            if n.endswith('.md'):data=b'> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.\n\n'+data
            add(f'{result}/{n}',data)
    out=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1';result='results/tastvg_native_spatial_rollout_s05/2026-09-29'
    for rel in ['protocols/tastvg_native_spatial_rollout_s05_v1.md','vg_tta/tastvg_native_spatial_rollout_s05_v1.py','scripts/run_tastvg_native_spatial_rollout_s05_v1.py','scripts/score_tastvg_native_spatial_rollout_s05_v1.py','scripts/report_tastvg_native_spatial_rollout_s05_v1.py','scripts/audit_tastvg_native_spatial_rollout_public_v1.py','tests/test_tastvg_native_spatial_rollout_s05_v1.py']:add(rel,(ROOT/rel).read_bytes())
    for n in ['ROWS.json','SUMMARY.json','AUDIT.json','PUBLIC_AUDIT.json','PARAMETER_SUPPORT.json','PARAMETER_AUDIT.json','REINSERTION_AUDIT.json','RESOURCES.json','PROVENANCE.json','DECISION.json','SCORING_RECOVERY.json']:add(f'{result}/{n}',(json.dumps(anon(read(out/n)),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    for n in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:add(f'{result}/{n}',(out/n).read_bytes())
    for rel in ['scripts/audit_tastvg_spatial_propagation_public_v1.py','scripts/export_tastvg_spatial_propagation_s05_v2.py']:add(rel,(ROOT/rel).read_bytes())
    add('docs/TA_SPATIAL_PROPAGATION_S05_UPDATE.md',(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/RESEARCH_UPDATE.md').read_bytes())
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1)
    entry='## Latest TA-STVG: S0.5 Native Spatial Rollout Support Test\n\n[Latest report](results/tastvg_native_spatial_rollout_s05/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPATIAL_PROPAGATION_S05_UPDATE.md), [protocol](protocols/tastvg_native_spatial_rollout_s05_v1.md). Student-only1792D parameter neighborhood,4 orthogonal antithetic pairs plus native, one5% radius,96cells/864candidates. No experts or GT during generation; post-seal oracle support remains limited. No S1/OPD/online learning or production promotion. Completed [threshold v1](results/tastvg_spatial_propagation_s05_superseded_v1/2026-09-29/REPORT.md) and [soft-moments v2](results/tastvg_spatial_propagation_s05_superseded_v2/2026-09-29/REPORT.md) are preserved and explicitly superseded by the latest user route.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files.values()),exclusions=['media','captions','source IDs','GT coordinates','weights','masks','H and raw predictions','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(x['bytes'] for x in files.values()))

if __name__=='__main__':run()
