"""Explicit code/anonymized scalar publication; never raw caches or parameters."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
DEST=ROOT/'artifacts/github_public_A_spatial_s06_s1_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')

def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for base in ['tastvg_spatial_critic_s06_v1','tastvg_spatial_online_opd_s1_v1']:
        for folder,suffix in [('protocols','.md'),('vg_tta','.py')]:rel=f'{folder}/{base}{suffix}';add(rel,(ROOT/rel).read_bytes())
        rel=f'scripts/run_{base}.py';add(rel,(ROOT/rel).read_bytes());rel=f'tests/test_{base}.py';add(rel,(ROOT/rel).read_bytes())
    for rel in ['scripts/analyze_tastvg_spatial_critic_s06_v1.py','scripts/score_tastvg_spatial_online_opd_s1_v1.py','scripts/audit_tastvg_spatial_s06_s1_public_v1.py','scripts/report_tastvg_spatial_s06_s1_v1.py','scripts/export_tastvg_spatial_s06_s1_v1.py']:add(rel,(ROOT/rel).read_bytes())
    for base in ['tastvg_spatial_critic_s06','tastvg_spatial_online_opd_s1']:
        out=ROOT/'artifacts'/f'{base}_v1';result=f'results/{base}/2026-09-29'
        names=['ROWS.json','SUMMARY.json','PUBLIC_AUDIT.json','RESOURCES.json','PROVENANCE.json','DECISION.json']
        names+=['REWARDS.json','MARGIN_CUTS.json','PAIRS.json','REWARD_BARRIER.json'] if base.endswith('s06') else ['AUDIT.json','DIAGNOSTICS.json','REINSERTION_AUDIT.json','STATE_CHAIN.json','CURRENT_POLICY_AUDIT.json','AUDIT_NUMERICAL_RECOVERY.json']
        for n in names:
            x=anon(read(out/n));data=json.dumps(x,ensure_ascii=False,allow_nan=False,**({'separators':(',',':')} if n=='PAIRS.json' else {'indent':2}))+'\n';add(f'{result}/{n}',data.encode())
        for n in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:add(f'{result}/{n}',(out/n).read_bytes())
    add('docs/TA_SPATIAL_CRITIC_S06_ONLINE_S1_UPDATE.md',(ROOT/'artifacts/tastvg_spatial_online_opd_s1_v1/RESEARCH_UPDATE.md').read_bytes())
    nav=(PUBLIC/'REVIEW_START_HERE.md').read_text();first,rest=nav.split('\n',1);entry='## Latest TA-STVG: spatial critic qualification and persistent Reverse-KL\n\n[Chinese interpretation](docs/TA_SPATIAL_CRITIC_S06_ONLINE_S1_UPDATE.md), [S0.6 critic report](results/tastvg_spatial_critic_s06/2026-09-29/REPORT.md), [S1 online report](results/tastvg_spatial_online_opd_s1/2026-09-29/REPORT.md). Cached critic antithetic ordering is informative, so the user-authorized persistent1792D spatial RKL was run.18 actual SGD updates across six16-arrival streams, current-policy probes regenerated, no expert regression target. Fixed tau1/SGD.005 yields negligible practical future-nonexpert gains. Both positive critic evidence and weak online outcome retained; no joint run or production promotion.\n\n';add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),exclusions=['media','captions','source IDs','GT coordinates','masks','H','raw candidate tubes','raw gradients/parameter states','model weights','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(v['bytes'] for v in files.values()))

if __name__=='__main__':run()
