"""Scoped anonymous real-forward outcomes/code and closure export only."""
import ast,hashlib,json,os,shutil,subprocess,tempfile,time
from scripts.stvg_opd_forward_candidate_common_v1 import *
from scripts.publish_stvg_opd_actionability_audit_v1 import CHECKOUT,BRANCH,git,put,safe,private_attachment_path
from scripts.publish_stvg_opd_LN_state_intervention_v1 import metadata
from scripts.audit_stvg_opd_forward_candidate_v1 import public

def run(closing=False):
    verify();assert read(BASE/'ROOT_READBACK.json')['status']==read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'TECHNICAL_DECISION.json')['status']=='stop_fixed_forward_candidate_recipe_before_GT'
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
        else:assert p.suffix=='.pdf'
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==sha(p);owned.add(rel)
    sources=SCIENCE+['scripts/report_stvg_opd_forward_candidate_v1.py','scripts/publish_stvg_opd_forward_candidate_v1.py']
    for n in sources:copy(ROOT/n)
    from scripts.publish_stvg_opd_projection_qualification_v1 import source_proof
    source_proof()  # Read-check all inherited science pins, including the documented prepare-only public alias.
    for n in sources:assert sha(CHECKOUT/n)==sha(ROOT/n)
    if not (BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json').exists():
        write(BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',scope='all new scientific sources exact bytes; inherited full qualified capture/model/OPD helpers retain exact science/known prepare-only alias; no original source changed',
          count=len(sources),inherited_source_receipt_sha256=sha(prior.BASE/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json'),
          hashes={n:sha(ROOT/n) for n in sources},CPU_only=True,GT_read=False,time=time.time()))
    for n in ['BOUNDARY_ROWS.json','STATISTICS.json','REPORT.md']:copy(OUT/n)
    if not closing:
        for n in ['RUNTIME.json','CPU_CONTRACTS.json','SOURCE_STATE_CUSTODY_READBACK.json','ARCHIVE_PRELOCK.json','GLOBAL_READOUT_BARRIER.json',
                  'ROOT_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json','TECHNICAL_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json','CPU_COMPLETION.json','worker_exit.json']:
            p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        protocol=[]
        for r in read(BASE/'DESIGN_LOCK.json')['records']:
            protocol.append(dict(parent_id=r['index'],dataset=r['dataset'],source_model_sha256=r['source_model_sha256'],pixel_sha256=r['pixel_sha256'],
              original_fit_sha256=r['A_sha256'],config=r['config'],state='Inherited_order1',boundaries=[0,r['config']['steps']//2,r['config']['steps']-1]))
        put(OUT/'ANONYMOUS_PROTOCOL.json',dict(records=protocol,candidates=['A','LN','Query','Half'],rho=.9,
          criterion_fraction_per_domain=.5,historically_exposed=True,queries=8,complete_candidate_readouts=192,
          fixed_boundary_comparisons=48,actual_fresh_captures=16,new_fits=0,new_backward_calls=0,new_optimizer_steps=0,new_DINO_calls=0,
          GT_read=False,exclusions='media/annotations/weights/raw inputs/complete fits/state vectors/gradient and sample vectors/private conversation'))
        copy(OUT/'figures/FORWARD_CANDIDATE_SCREEN.pdf')
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
            p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        for e in read(BASE/'PUBLIC_STAGE.json')['files']:assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
    for p in OUT.rglob('*.json'):
        if p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json','PUBLIC_SCALAR_AUDIT.json']:copy(p)
    audit=public(CHECKOUT/OUT.relative_to(ROOT))
    if not closing:write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit)
    else:assert audit==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
    copy(OUT/'PUBLIC_SCALAR_AUDIT.json')
    relative=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='all fixed eight-query real-forward outcomes/source/config/negative and unknowns/one PDF two viewed pages; no fitting or GT efficacy; exclude self hash',
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
      scope='fixed real-forward candidate screen complete; current coverage criterion failed; no new fits or GT',time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',files=len(entries),bytes=sum(e['bytes'] for e in entries),expected_tree=tree,base_commit=base)))

if __name__=='__main__':run(len(__import__('sys').argv)>1 and __import__('sys').argv[1]=='closing')
