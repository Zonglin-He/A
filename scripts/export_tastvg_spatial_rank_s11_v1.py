"""Explicit allowlist export of S1.1 code and anonymous scalar evidence."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
BASE=ROOT/'artifacts/tastvg_spatial_rank_s11_v1';RAW=ROOT/'artifacts/tastvg_spatial_online_opd_s1_v1';DEST=ROOT/'artifacts/github_public_A_spatial_s11_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')

def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    def js(rel,path):add(rel,(json.dumps(anon(read(path)),ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())
    for rel in ['protocols/tastvg_spatial_rank_s11_v1.md','vg_tta/tastvg_spatial_rank_s11_v1.py','tests/test_tastvg_spatial_rank_s11_v1.py']+[f'scripts/{prefix}_tastvg_spatial_rank_s11_v1.py' for prefix in ['run','score','analyze','report','export']]+['scripts/audit_tastvg_spatial_rank_s11_public_v1.py']:add(rel,(ROOT/rel).read_bytes())
    result='results/tastvg_spatial_rank_s11/2026-09-29'
    for name in ['ROWS.json','SUMMARY.json','SOURCE_EFFECTS.json','DIAGNOSTICS.json','RESOURCES.json','PROVENANCE.json','PUBLIC_AUDIT.json','DECISION.json']:js(f'{result}/{name}',BASE/name)
    for name in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:add(f'{result}/{name}',(BASE/name).read_bytes())
    for arm in ['raw','rank','norm']:
        out=RAW if arm=='raw' else BASE/arm
        for name in ['ROWS.json','SUMMARY.json','AUDIT.json','PUBLIC_AUDIT.json','CURRENT_POLICY_AUDIT.json','STATE_CHAIN.json','REINSERTION_AUDIT.json']:js(f'{result}/{arm}/{name}',out/name)
    add('docs/TA_SPATIAL_RANK_S11_UPDATE.md',(BASE/'RESEARCH_UPDATE.md').read_bytes())
    first,rest=(PUBLIC/'REVIEW_START_HERE.md').read_text().split('\n',1)
    entry='## Latest TA-STVG: S1.1 rank preference and normalized spatial updates\n\n[Chinese interpretation](docs/TA_SPATIAL_RANK_S11_UPDATE.md), [three-arm report](results/tastvg_spatial_rank_s11/2026-09-29/REPORT.md), [fixed protocol](protocols/tastvg_spatial_rank_s11_v1.md). Raw-RKL reused; two new on-policy persistent1792D arms,192arrivals/36updates. Rank preference gives small positive future-nonexpert transfer; normalized1%-probe steps do not improve over Rank-SGD. No sweep, parameter-space OPD, joint run or production promotion.\n\n'
    add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),exclusions=['media','query text/source IDs','labels/GT coordinates','masks','H','raw prediction tubes','parameters/gradients','weights','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print(len(files),sum(v['bytes'] for v in files.values()))

if __name__=='__main__':run()
