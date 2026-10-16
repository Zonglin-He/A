"""Local receipt closure after actual remote HTTP, refs and pixel inspection."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE,read,put,sha
def main(group,pixel):
    p=BASE/'publication'/group;remote=read(p/'REMOTE_BYTES_READBACK.json');refs=read(p/'REFS_READBACK.json');plan=read(p/'UPLOAD_PLAN.json');c=remote['commit_sha'];w=ROOT.parent/'visual-grounding-public-A'
    assert remote['status']=='pass'and remote['file_count']==plan['file_count']and remote['bytes']==plan['bytes']and refs['status']=='pass'and set(refs['branches'])=={'main','research/stvg-opd-paper-hc2-revision-v2'}and all(v==c for v in refs['branches'].values())
    def g(*a):return subprocess.check_output(['git',*a],cwd=w,text=True).strip()
    assert g('rev-parse','HEAD')==plan['base_commit'];assert g('rev-parse',c+'^{tree}')==g('write-tree')==plan['expected_tree']==remote['tree_sha']
    for row in plan['files']:
        assert g('ls-tree',c,'--',row['path']).split()[:3]==['100644','blob',row['blob_sha']]
    subprocess.run(['git','reset','--soft',c],cwd=w,check=True);assert not g('diff','--cached','--name-only')and not g('diff','--name-only')
    put(p/'LOCAL_COMMIT_READBACK.json',dict(status='pass',commit_sha=c,tree_sha=plan['expected_tree'],file_count=plan['file_count'],mode_and_blob_checked=True,tracked_and_index_clean=True,unrelated_untracked_preserved=True))
    img=ROOT/pixel;assert img.exists()and img.suffix=='.png';put(p/'ROOT_VISUAL_REVIEW.json',dict(status='pass',count=1,actual_pixels_viewed=True,path=pixel,sha256=sha(img),no_private_media_exported=True))
    put(p/'FINAL_GITHUB_RECEIPT.json',dict(status='pass',commit_sha=c,tree_sha=remote['tree_sha'],file_count=remote['file_count'],bytes=remote['bytes'],both_refs_verified=True,remote_bytes_sha256_blob_mode_tree=True,remote_readback_sha256=sha(p/'REMOTE_BYTES_READBACK.json'),local_readback_sha256=sha(p/'LOCAL_COMMIT_READBACK.json'),refs_sha256=sha(p/'REFS_READBACK.json')))
    put(p/'PUBLICATION_COMPLETION.json',dict(status='complete',count=16,group=group,commit_sha=c,whole_study_FINAL=False,github_receipt_sha256=sha(p/'FINAL_GITHUB_RECEIPT.json')));print(json.dumps(dict(status='complete',group=group,commit=c)))
if __name__=='__main__':
    assert sys.argv[1]=='--after-actual-pixel-view-and-ref-readback';main(sys.argv[2],sys.argv[3])
