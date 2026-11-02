"""One guarded CPU batch of the already registered literal source adapter."""
from full_frontend50 import ROOT, EXP, context, read, sha, dump, source_chunk

def main():
    plan, rows, source_seal = context()
    auth = read(EXP/'FULL_SOURCE_BATCH_AUTHORIZATION.json')
    assert auth['state'] == 'ROOT_AUTHORIZED_R50_REGISTERED_FULL_SOURCE_BATCH_NO_FORWARD'
    assert auth['program_sha256'] == sha(__file__)
    registry = ROOT/plan['output_root']/'registry'
    assert auth['registry_seal_sha256'] == sha(registry/'SEALED.json')
    seal = read(registry/'SEALED.json')
    for rel, digest in seal['files'].items():
        assert sha(registry/rel) == digest
    jobs = [j for j in read(registry/'STAGE_JOBS_PRIVATE.json') if j['stage'] == 'source']
    assert sum(j['n'] for j in jobs) == 5912
    for job in jobs:
        source_chunk(plan, source_seal, job['chunk'])
        print('SOURCE50_CHUNK', job['chunk'], job['n'], flush=True)
    out = ROOT/plan['output_root']/'source_chunks'
    dump(out/'ALL_CHUNKS_SEALED.json', {'state':'R50_FULL5912_LITERAL_R8_SOURCE_ADAPTER_COMPLETE_NO_FORWARD',
        'n':len(rows), 'chunks':{j['chunk']:sha(out/j['chunk']/'SEALED.json') for j in jobs},
        'new_model_calls':0, 'new_features_available':False})

if __name__ == '__main__':
    main()
