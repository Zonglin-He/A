"""Export audited scalar evidence and register the verified GitHub publication."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from scope172 import HERE, NEW_INPUT, OUT, PRIOR_INPUT, ROOT, atomic_json, read, sha, status
from audit172 import audit

sys.path.insert(0, str(ROOT / 'research/round154'))
from package154 import scan_release


PREFIX = 'studies/3eed_joint_score_attribution_cross_query_verification_20261118'
CODE = ('scope172.py','evidence172.py','geometry172.py','analyze172.py','audit172.py','release172.py')
PROOFS = ('EVIDENCE_LOCK.json','EVIDENCE_SEAL.json','ATTRIBUTION_LOCK.json','ANALYSIS_SEAL.json',
          'AUDIT_LOCK.json','SCALAR_AUDIT.json','IDENTITY_GATE.json')


def private_literals():
    reserved = set()
    for entry in read(NEW_INPUT)+read(PRIOR_INPUT):
        row = entry['row']
        values = [row[k] for k in ('id','caption','scene','imagepath','pointpath','metapath')]
        values.append(Path(row['metapath']).parent.name)
        values.extend(v['text'] for k,v in entry['rewrite'].items() if k in ('B','C','D'))
        reserved.update(v.encode() for v in values if isinstance(v,str) and len(v)>=12)
    return reserved


def export():
    result = audit()
    decision = read(HERE/'DECISION.json')
    gate = read(HERE/'IDENTITY_GATE.json')
    assert gate['independent_readout_gate_pass'] == decision['C_D_identity_gate_passed'] is False
    assert not decision['FINAL_access'] and not decision['C_D_joint_readout_executed']
    parent = HERE/'publication'
    parent.mkdir(exist_ok=True)
    dest = parent/'public_export_attempt001'
    assert not dest.exists()
    temp = Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(src, rel):
            path = temp/rel
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,path)
        for name in CODE:
            copy(HERE/name,'own_code/research/round172/'+name)
        for name in (*PROOFS,'REPORT.md','REPORT_ZH.md','DECISION.json','experiment/PROTOCOL.json'):
            copy(HERE/name,name)
        for name,d in result['sealed_artifacts'].items():
            assert sha(OUT/name)==d
            copy(OUT/name,'results/'+name)
        atomic_json(temp/'PUBLIC_EXPORT_AUDIT.json',dict(scalar_audit=result,
            raw_sensors_text_geometry_model_state_dataset_ids_or_credentials_exported=False))
        files = [dict(path=p.relative_to(temp).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
                 for p in sorted(temp.rglob('*')) if p.is_file()]
        atomic_json(temp/'MANIFEST.json',dict(prefix=PREFIX,files=files))
        scan_release(temp,private_literals())
        temp.rename(dest)
        status('local_audited_pending_publication', direction_records=60, report='round172/REPORT_ZH.md')
        print('R172 public files',len(files)+1,flush=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def register():
    receipt = read(HERE/'publication/PUBLICATION_RECEIPT.json')
    verdict = read(HERE/'DECISION.json')
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==PREFIX
    assert receipt['commit']==receipt['mainRefObserved'] and receipt['immutableVerifiedBeforeRefUpdate'] and receipt['remoteByteVerified']
    commit = receipt['commit']
    remote = 'https://github.com/Zonglin-He/A/blob/'+commit+'/'+PREFIX+'/REPORT_ZH.md'
    relative = 'round172/publication/PUBLICATION_RECEIPT.json'
    status('completed_published_verified', direction_records=60, physical_frames=30, public_report=remote,
           public_commit=commit, report='round172/REPORT_ZH.md', decision='round172/DECISION.json', publication_receipt=relative)
    path = ROOT/'research/STATE.json'
    s = read(path)
    s.update(latest_completed_round='R172', measurement_state='R172_SEALED_PAIR_AND_IDENTITY_AUDITED',
             publication_state='R172_REMOTE_BYTES_VERIFIED', claim_state=verdict['finding'], next_action=verdict['decision'],
             public_study=PREFIX, publication_receipt=relative)
    s['directions']['crossplatform'].update(latest_completed_round='R172',latest_completed_publication=commit,
                                           latest_public_commit=commit,latest_public_report=remote)
    atomic_json(path,s)
    archive = ROOT/'research/ARCHIVE.md'
    tag = 'R172 completed and published '+commit
    if tag not in archive.read_text():
        with archive.open('a',encoding='utf-8') as out:
            out.write('\n## R172 — Joint score attribution and C/D verification\n\n'+tag+'. '+
                      verdict['finding']+' [Report](round172/REPORT_ZH.md) · [Verified receipt]('+relative+')\n')


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('export','register'))
    {'export':export,'register':register}[p.parse_args().phase]()
