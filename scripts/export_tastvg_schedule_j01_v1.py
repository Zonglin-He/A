"""Publish five prelocked schedules and all scalar outcomes without selecting an order."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_schedule_j01_v1';DEST=ROOT/'artifacts/github_public_A_schedule_j01_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')


def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else ([f'Q{z+1:02}' if isinstance(z,int) else z for z in v] if k=='parents' else anon(v))) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for name in ['FREEZE_J01.json']:
        rel='methods/tastvg_dual_evidence_j0_v1/'+name;add(rel,(ROOT/rel).read_bytes())
    for rel in ['protocols/tastvg_schedule_j01_v1.md','tests/test_tastvg_schedule_j01_v1.py','scripts/audit_tastvg_schedule_j01_public_v1.py']+[f'scripts/{x}_tastvg_schedule_j01_v1.py' for x in ['run','score','tier0','report','export']]:add(rel,(ROOT/rel).read_bytes())
    result='results/tastvg_schedule_j01/2026-09-29'
    for name in ['ROWS.json','SUMMARY.json','SOURCE_EFFECTS.json','ACROSS_ORDERS.json','INCLUDING_J0.json','J0_REFERENCE.json','ORDERS.json','TIER0.json','TIER0_SOURCE_ROWS.json','TIER0_SUBSETS.json','TIER0_MARGIN_ROWS.json','RESOURCES.json','AUDIT.json','PUBLIC_AUDIT.json','STATE_CHAIN.json','REINSERTION_AUDIT.json','SLOW_REPLAY_AUDIT.json','PROVENANCE.json','DECISION.json','SCORING_RECOVERY.json']:
        add(f'{result}/{name}',(json.dumps(anon(read(OUT/name)),ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())
    for name in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:add(f'{result}/{name}',(OUT/name).read_bytes())
    add('docs/TA_SCHEDULE_J01_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    first,rest=(PUBLIC/'REVIEW_START_HERE.md').read_text().split('\n',1);entry='## Latest TA-STVG: J0.1 online schedule robustness\n\n[Chinese results](docs/TA_SCHEDULE_J01_UPDATE.md), [five-order report](results/tastvg_schedule_j01/2026-09-29/REPORT.md), [recipe freeze](methods/tastvg_dual_evidence_j0_v1/FREEZE_J01.json). The exact J0 method was tested on five prelocked source-hash orders with 25% specialist availability. Whole-stream Fast and Final improve in 5/5 orders: Final gain +1.5177 pp, sample SD 0.8599 pp. Future spatial transfer averages +0.0255 pp but is positive in 4/5 orders, negative in one. Original J0 negative result retained separately; all six also summarized. Freeze the existing research recipe for subsequent evaluation; no favorable-order selection, new gate or production promotion. Same 16 exposed sources, not five independent cohorts.\n\n';add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),exclusions=['media','captions/source IDs','GT coordinates','model weights','H/masks','raw tubes/parameter states/gradients','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(v['bytes'] for v in files.values()))

if __name__=='__main__':run()
