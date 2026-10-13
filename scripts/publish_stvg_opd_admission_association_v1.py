"""Scoped anonymous real-forward outcomes/code and closure export only."""
import ast,hashlib,json,os,shutil,subprocess,tempfile,time
from scripts.stvg_opd_admission_association_common_v1 import *
from scripts.publish_stvg_opd_actionability_audit_v1 import CHECKOUT,BRANCH,git,put,safe,private_attachment_path
from scripts.publish_stvg_opd_LN_state_intervention_v1 import metadata
from scripts.audit_stvg_opd_admission_association_public_v1 import public

def run(closing=False):
    verify();assert read(BASE/'ROOT_READBACK.json')['status']==read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'MECHANISM_DECISION.json')['status']=='bounded_admission_association_task_evaluated'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(p,rel=None):
        rel=str(p.relative_to(ROOT)) if rel is None else rel;data=p.read_bytes()
        if p.suffix=='.json':safe(read(p))
        elif p.suffix in ['.py','.md']:
            assert not private_attachment_path(data.decode())
            if p.suffix=='.py':ast.parse(data.decode())
        elif p.suffix=='.gz':
            import gzip
            with gzip.open(p,'rt') as f:
                for line in f:safe(json.loads(line))
        elif p.suffix=='.csv':
            import csv
            with p.open(newline='') as f:
                for row in csv.DictReader(f):safe(row)
        else:assert p.suffix=='.pdf'
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==sha(p);owned.add(rel)
    sources=SCIENCE+['scripts/close_stvg_opd_admission_association_v1.py','scripts/report_stvg_opd_admission_association_v1.py','scripts/publish_stvg_opd_admission_association_v1.py','scripts/audit_stvg_opd_admission_association_public_v1.py','scripts/audit_stvg_opd_admission_association_oracle_v1.py']
    for n in sources:copy(ROOT/n)
    from scripts.publish_stvg_opd_projection_qualification_v1 import source_proof
    source_proof()  # Read-check all inherited science pins, including the documented prepare-only public alias.
    for n in sources:assert sha(CHECKOUT/n)==sha(ROOT/n)
    if not (BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json').exists():
        write(BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',scope='all new scientific sources exact bytes; inherited full qualified capture/model/OPD helpers retain exact science/known prepare-only alias; no original source changed',
          count=len(sources),inherited_source_receipt_sha256=sha(prior.BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json'),
          hashes={n:sha(ROOT/n) for n in sources},CPU_only=True,GT_read=False,time=time.time()))
    for n in ['ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','ROUND_ROWS.jsonl.gz','EVIDENCE_ROWS.jsonl.gz','SUMMARY.json','REPORT.md','ROWS.csv']:copy(OUT/n)
    if not closing:
        for n in ['RUNTIME.json','CPU_CONTRACTS.json','CPU_SELECTION_READBACK.json','SOURCE_STATE_CUSTODY_READBACK.json','ARCHIVE_PRELOCK.json','QUALIFICATION_BARRIER.json','ROOT_QUALIFICATION.json','GLOBAL_PREDICTION_BARRIER.json','ORACLE_PREDICTION_BARRIER.json','GT_EXPOSURE.json',
                  'ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_ORACLE_SELECTION_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','MECHANISM_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json','CPU_COMPLETION.json','qualification_worker_exit.json','formal_worker_exit.json','oracle_worker_exit.json']:
            p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        for p in sorted(BASE.glob('*/RESOLUTION.json')):
            put(OUT/'receipts'/(p.parent.name+'_RESOLUTION.json'),dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        protocol=[dict(parent_id=r['index'],dataset=r['dataset'],source_model_sha256=r['source_model_sha256'],pixel_sha256=r['pixel_sha256'],original_fit_sha256=r['A_sha256'],config={('actions_per_round' if k=='samples' else k):v for k,v in r['config'].items()},entering_state_label='Inherited_order1') for r in read(BASE/'DESIGN_LOCK.json')['records']]
        barriers={k:read(BASE/n) for k,n in [('qualification','QUALIFICATION_BARRIER.json'),('formal','GLOBAL_PREDICTION_BARRIER.json'),('Oracle','ORACLE_PREDICTION_BARRIER.json')]}
        put(OUT/'ANONYMOUS_PROTOCOL.json',dict(records=protocol,arms=ARMS,parents=64,beta=1,
          phase_counts={k:{f:v[f] for f in ['logical_complete_fits','logical_complete_rounds','actual_complete_GPU_fits','actual_complete_rounds','actual_frozen_captures','GT_read']} for k,v in barriers.items()},
          original_H_control_replays=64,actual_new_DINO_calls=0,historical_exposure=True,primary='I1 mvIoU',secondary='I0 mvIoU',
          Oracle_GT_assisted_diagnostic_only=True,GT_online=False,monitor_recreated=False,production_promoted=False,
          excluded_categories=['private media','annotations','weights','raw inputs','candidate or GT geometry','complete fits','state vectors','gradient and sample vectors','personal conversation']))
        copy(OUT/'figures/ADMISSION_ASSOCIATION_VALIDATION.pdf')
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
            p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
    for p in OUT.rglob('*.json'):
        if p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(p)
    audit=public(CHECKOUT/OUT.relative_to(ROOT))
    if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit)
    else:
        old=read(OUT/'PUBLIC_SCALAR_AUDIT.json')
        assert audit['files']==old['files'] and audit['root_readback_receipts']==old['root_readback_receipts']
        for k,v in old['checks'].items():
            if k!='exact_additional_JSON_attachments':assert audit['checks'][k]==v
        write(BASE/'CLOSING_PUBLIC_SCALAR_AUDIT.json',audit)
    copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
    relative=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='complete fixed64 parents/six fixed supervision trajectories/separate GT-assisted original-pool Oracle/all signs and unknowns/source/config/actual viewed PDF; exclude self hash',
      files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=relative}));copy(manifest,relative)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='forward-candidate-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
      base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,
      scope='complete fixed64 historical admission-association task/six label-free arms/separate GT-assisted Oracle/all outcomes; no promotion or monitor recreation',time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',files=len(entries),bytes=sum(e['bytes'] for e in entries),expected_tree=tree,base_commit=base)))

if __name__=='__main__':run(len(__import__('sys').argv)>1 and __import__('sys').argv[1]=='closing')
