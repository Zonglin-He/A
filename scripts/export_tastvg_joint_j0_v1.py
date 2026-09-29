"""Publish J0 implementation, fixed recipe and anonymized scalar positive/negative evidence."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
OUT=ROOT/'artifacts/tastvg_joint_j0_v1';DEST=ROOT/'artifacts/github_public_A_joint_j0_20260929';PUBLIC=Path('/home/wwww/visual-grounding-public-A')


def run():
    files={}
    def add(rel,data):
        p=DEST/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(data);files[rel]=dict(bytes=len(data),sha256=sha(p))
    def anon(x):
        if isinstance(x,dict):return {k:(f'Q{v+1:02}' if k=='parent' and isinstance(v,int) else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    for name in ['__init__.py','config.json','method.py','RECIPE_LOCK.json']:
        rel='methods/tastvg_dual_evidence_j0_v1/'+name;add(rel,(ROOT/rel).read_bytes())
    for rel in ['protocols/tastvg_joint_j0_v1.md','tests/test_tastvg_joint_j0_v1.py','scripts/audit_tastvg_joint_j0_public_v1.py']+[f'scripts/{x}_tastvg_joint_j0_v1.py' for x in ['run','score','diagnose','report','export']]:add(rel,(ROOT/rel).read_bytes())
    result='results/tastvg_joint_j0/2026-09-29'
    for name in ['ROWS.json','SUMMARY.json','SOURCE_EFFECTS.json','TEMPORAL_REFERENCE_ROWS.json','TEMPORAL_DIAGNOSIS.json','RESOURCES.json','AUDIT.json','PUBLIC_AUDIT.json','STATE_CHAIN.json','REINSERTION_AUDIT.json','PROVENANCE.json','DECISION.json','IMPLEMENTATION_REVISION.json']:
        add(f'{result}/{name}',(json.dumps(anon(read(OUT/name)),ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode())
    for name in ['REPORT.md','RESEARCH_UPDATE.md','REPRODUCE.md','CPU_TESTS.txt']:add(f'{result}/{name}',(OUT/name).read_bytes())
    add('docs/TA_JOINT_J0_UPDATE.md',(OUT/'RESEARCH_UPDATE.md').read_bytes())
    first,rest=(PUBLIC/'REVIEW_START_HERE.md').read_text().split('\n',1);entry='## Latest TA-STVG: J0 integrated sparse-expert online method\n\n[Chinese interpretation](docs/TA_JOINT_J0_UPDATE.md), [four-arm report](results/tastvg_joint_j0/2026-09-29/REPORT.md), [fixed recipe](methods/tastvg_dual_evidence_j0_v1/config.json). Current-policy temporal reranking + persistent spatial Rank-RKL executes correctly. Future Final−Fast vIoU+0.0472pp reproduces S1.1, but whole-stream Final−Frozen is−0.0363pp (CI crosses0): the positive final-method criterion is not met. Fixed25%expert availability selects four sources with negative temporal reranking gain, despite a positive prior full-availability reference. No favorable-schedule selection, new module, parameter-space OPD or production promotion. Components and implementation archived; mechanism optimization stopped.\n\n';add('REVIEW_START_HERE.md',(first+'\n\n'+entry+rest.lstrip('\n')).encode())
    write(DEST/'MANIFEST.json',dict(files=files,file_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),exclusions=['media','captions/source IDs','GT coordinates','model weights','H/masks','raw tubes/parameter states/gradients','private conversation']))
    for rel in files:
        p=PUBLIC/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/'files'/rel,p)
    print('EXPORTED',len(files),sum(v['bytes'] for v in files.values()))

if __name__=='__main__':run()
