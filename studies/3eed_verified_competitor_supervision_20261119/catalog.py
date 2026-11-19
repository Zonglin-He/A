"""Bridge to the sealed source-only study without exposing identity to feature capture."""
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
PRIVATE = HOME / 'private'
RESULTS = HOME / 'results'
PROTOCOL = HOME / 'protocol.json'
FULL = HOME.parent / '3eed_full_pool_reference_and_winner_verification_20261119'
OVERNIGHT = HOME.parent / 'codex_overnight_20261119'
sys.path.insert(0, str(FULL))
import inputs as earlier
import rank as earlier_rank
from inputs import PREVIOUS, SUPPLEMENT, atomic_json, csv_rows, read, sha, write_table


def verify_prior():
    evidence = read(FULL/'results/AUDIT.json')
    assert evidence['status']=='PASS' and not evidence['protected_data_access'] and evidence['higher_budget_replay_verified']
    assert all(sha(FULL/'results'/f)==s for f,s in evidence['files'].items())
    assert read(FULL/'publication/PUBLICATION_RECEIPT.json')['commit']==read(PROTOCOL)['parent']
    assert read(PROTOCOL)['protected_data_access'] is False
    return sha(FULL/'results/AUDIT.json')


def source_roster():
    """Read preparation entries and validate the original identity-free prediction seals."""
    audit = verify_prior()
    base = PREVIOUS/'private'
    initial_pred = base/'predictions'
    launch = read(initial_pred/'LOCK.json')
    first = []
    for source, digest in launch['inputs'].items():
        path = base/'prep'/f'{source}.json'
        assert sha(path)==digest
        rows = read(path)
        assert all(row['row']['platform']==source for row in rows)
        first.extend(rows)
    second_path = SUPPLEMENT/'RECRUITED.json'
    second_pred = SUPPLEMENT/'predictions'
    assert sha(second_path)==read(second_pred/'LOCK.json')['input']
    second = read(second_path)
    assert len(first)==108 and len(second)==36
    out = {}
    checksums = {}
    groups = (
        ('A',3,first,initial_pred,'source_parameters_unchanged','prediction_reads_truth',False),
        ('B',2,second,second_pred,'weights_unchanged','truth_not_loaded',True),
    )
    for prefix,digits,rows,folder,freeze,label_field,label_value in groups:
        sfile = folder/'SEAL.json'
        p = read(sfile)
        assert not p['protected_data_access'] and p[freeze] is True and p[label_field] is label_value
        assert p['pin']==sha(folder/'LOCK.json') and set(p['files'])=={x['case'] for x in rows}
        checksums[prefix] = sha(sfile)
        for i,item in enumerate(sorted(rows,key=lambda x:x['case']),1):
            alias = prefix + str(i).zfill(digits)
            key = item['case']
            packet = folder/(key+'.npz')
            assert sha(packet)==p['files'][key]
            assert 'reference_identity' not in item and 'verified_foil_ids' not in item
            out[alias] = (item, packet)
    return out, dict(prior_audit=audit,prediction_seals=checksums)


def label_truths():
    data, published, seals = earlier.load()
    truth = {}
    for i, (file,prefix,width) in enumerate(((PREVIOUS/'private/VERIFIED_ROSTER.json','A',3),
                                     (SUPPLEMENT/'VERIFIED_ROSTER.json','B',2))):
        for k,x in enumerate(sorted(read(file),key=lambda row:row['case']),1):
            truth[prefix+str(k).zfill(width)] = x
    assert set(truth)=={x.alias for x in data}
    return data, truth, seals
