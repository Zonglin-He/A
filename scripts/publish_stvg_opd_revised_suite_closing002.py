"""Publish actual final suite closing/archive/verification receipts without scientific changes."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import BASE,read,write,sha
from scripts.publish_stvg_opd_p6_existing_root001 import CHECKOUT,BRANCH,git,safe_json
from scripts.assemble_stvg_opd_revised_suite001 import DEST


def run():
    final=read(BASE/'FINAL_COMPLETION.json');ar=read(BASE/'PAPER_SUITE_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json')
    assert final['status']=='complete' and final['paper_suite_complete'] and ar['status']=='pass'
    assert ar['FINAL_COMPLETION_sha256']==sha(BASE/'FINAL_COMPLETION.json') and ar['history_sha256']==sha(ROOT/'docs/RESEARCH_HISTORY.md')
    assert final['actual_root_closing_sha256']==sha(BASE/'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json')
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)==final['actual_public_commit']
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    names=['FINAL_COMPLETION.json','PAPER_SUITE_ROOT_CLOSING_RECEIPT.json','PAPER_SUITE_ARCHIVE_VERIFICATION_RECEIPT.json',
        'PAPER_SUITE_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json','PAPER_SUITE_FINAL_GITHUB_RECEIPT.json','PAPER_SUITE_PUBLIC_AUDIT.json']
    files=['scripts/finalize_stvg_opd_revised_suite001.py','scripts/publish_stvg_opd_revised_suite_closing002.py']
    for name in names:
        target=DEST/'receipts'/name;shutil.copy2(BASE/name,target);files.append(str(target.relative_to(ROOT)))
    write(DEST/'FINAL_CLOSING_BINDING.json',dict(status='actual_complete_suite_receipts_pending_remote_byte_verification',
        paper_suite_complete=True,original_actual_FINAL_COMPLETION_sha256=sha(BASE/'FINAL_COMPLETION.json'),
        files={rel:dict(bytes=(ROOT/rel).stat().st_size,sha256=sha(ROOT/rel)) for rel in files},
        all_negative_results_preserved=True,current_method_changed=False,EATA_and_historical_queues_paused=True))
    files.append(str((DEST/'FINAL_CLOSING_BINDING.json').relative_to(ROOT)))
    for rel in files:
        p=ROOT/rel;assert p.suffix in {'.py','.json'}
        if p.suffix=='.json':safe_json(read(p))
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'PAPER_SUITE_FINAL_CLOSING_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='actual original fixed P1-P6 FINAL_COMPLETION root closing public verification and final archive receipts',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),changed_files=changed,files=records,file_count=len(records),bytes=sum(r['bytes'] for r in records),paper_suite_complete=True,time=time.time()))
    print(json.dumps(dict(status='ready',actual_complete_suite=True,files=len(records),bytes=sum(r['bytes'] for r in records))))


if __name__=='__main__':run()
