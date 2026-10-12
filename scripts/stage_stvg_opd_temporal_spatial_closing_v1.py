"""Add real bounded-closing/archive receipts to the unchanged science whitelist."""
import hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.stage_stvg_opd_o_q0_phase_public_v1 import receipt_metadata
from scripts.publish_stvg_opd_supplement_v1 import safe
BASE=ROOT/'artifacts/stvg_opd_temporal_spatial_complementarity_v1'
OUT=ROOT/'results/stvg_opd_temporal_spatial_complementarity/2026-10-12'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()

def run():
    previous=read(BASE/'PUBLIC_STAGE.json');remote=read(BASE/'FINAL_GITHUB_RECEIPT.json');assert remote['status']=='pass'
    assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
    assert read(BASE/'ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json')['status']=='pass'
    base=git('rev-parse','HEAD');assert base==remote['commit'];assert not git('diff','HEAD','--name-only')
    owned={r['path'] for r in previous['files']}
    def copy(source,relative):
        target=CHECKOUT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(source)==sha(target);owned.add(relative)
    for relative in ['scripts/close_stvg_opd_temporal_spatial_v1.py','scripts/stage_stvg_opd_temporal_spatial_closing_v1.py','docs/STVG_OPD_TEMPORAL_SPATIAL_CACHED_REVIEW.md']:
        copy(ROOT/relative,relative)
    for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
        original=BASE/name;target=OUT/'closing'/name;projected=dict(original_private_receipt_sha256=sha(original),projection=receipt_metadata(read(original)));safe(projected);write(target,projected);copy(target,str(target.relative_to(ROOT)))
    entries=[]
    for relative in sorted(owned):
        p=CHECKOUT/relative;data=p.read_bytes();entries.append(dict(path=relative,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    for e in previous['files']:assert next(v for v in entries if v['path']==e['path'])==e
    with tempfile.TemporaryDirectory(prefix='stvg-cached-closing-index-') as tmp:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(tmp)/'index')};git('read-tree',base,env=env)
        for e in entries:
            blob=git('hash-object','-w',str(CHECKOUT/e['path']));assert blob==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',blob,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'CLOSING_PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=previous['branch'],base_commit=base,expected_tree=tree,
        files=entries,file_count=len(entries),bytes=sum(v['bytes'] for v in entries),
        scope='all bounded cached T science/source/full positive-negative results/two viewed PDFs/paper working revision and genuine ROOT_CLOSING/FINAL/archive receipts; unavailable fullO and S-C-N not complete',time=time.time()))
    print('CLOSING_WHITELIST',len(entries),sum(v['bytes'] for v in entries),tree)

if __name__=='__main__':run()
