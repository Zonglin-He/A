"""Publish reviewed anonymous scalar results and verify every remote file byte."""
import sys,json,time,shutil,gzip,hashlib,subprocess,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_baseline_scoring_common_v1 import BASE,PUB,read,write,sha,verify,PAPER
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/decota-paper-baseline-scoring-v1'
def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()
def fetch(url):
    error=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-byte-audit'}),timeout=45) as r:return r.read()
        except Exception as e:error=e;time.sleep(attempt+1)
    raise error

def caption_guard(value,path=()):
    """Only the aggregate query-type category may have a caption key."""
    if isinstance(value,dict):
        for key,v in value.items():
            if key=='caption':
                assert len(path)==5 and path[-3:]==('strata','query_type','groups'),path
                assert isinstance(v,dict) and set(v)=={'queries','parent_sources','metrics'}
                assert type(v['queries']) is int and type(v['parent_sources']) is int
            caption_guard(v,path+(key,))
    elif isinstance(value,list):
        for v in value:caption_guard(v,path+('array',))

def stage():
    runtime,_=verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_STATISTICS_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert git('branch','--show-current')==BRANCH and not git('status','--porcelain')
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    original=read(PAPER/'baselines/RUNTIME_LOCK.json')
    owned=sorted(set(runtime['pins'])|set(original['pins'])|{
        'scripts/audit_decota_paper_baseline_scoring_public_v1.py','scripts/publish_decota_paper_baseline_scoring_v1.py',
        'scripts/decota_public_result_io_v1.py','protocols/decota_paper_baseline_scoring_v1.md'})
    dependencies={p:sha(ROOT/p) for p in owned if p.startswith('external/')}
    owned=[p for p in owned if not p.startswith('external/')]
    for p in owned:assert not p.startswith(('data/','artifacts/','checkpoints/')) and Path(p).suffix in ('.py','.md')
    write(PUB/'CODE_BINDING.json',dict(code={p:sha(ROOT/p) for p in owned},official_evaluator_dependency_pins=dependencies,
        external_clone_excluded=True,official_repository=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT/'external/TA-STVG').decode().strip(),
        official_repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'external/TA-STVG').decode().strip(),
        evaluation_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),original_baseline_port_runtime_sha256=sha(PAPER/'baselines/RUNTIME_LOCK.json'),
        private_media_captions_GT_weight_gradient_state_payloads_exported=False))
    for name in ['EVALUATION_BARRIER.json','SYNTHETIC_TEST_RECEIPT.json','ROOT_VISUAL_REVIEW.json']:
        shutil.copy2(BASE/name,PUB/name)
    write(PUB/'CONFIGURATION.json',dict(baseline_version=original['version'],baseline_configs=original['configs'],
      adapted_scope='ground decoder and time decoder LayerNorm affine; 19968 coordinates; encoders/heads frozen',
      entropy='mean across independent native offsets of H(start)+H(end)+mean binary actionness entropy, one sigmoid, valid frames only',
      source_checkpoint={'vidstg_target':'official HC2-trained TA-STVG','hc2_validation_target':'official VidSTG-trained TA-STVG'},
      reference='same-dataset trained frozen checkpoint; supervised comparator, not TTA or mathematical upper bound',
      DINO_Refine='sealed admitted Uniform4 frame Top1 boxes, physical cxcywh interpolation and nearest admitted extension; no additional expert calls',
      target_queries={'vidstg_test':10303,'hc2_validation':3482},parent_sources={'vidstg':732,'hc2':237},
      original_query_orders=3,no_new_hyperparameter_selection=True,EATA_inference_user_paused=True,
      EATA_hc2_source_Fisher='sealed VidSTG 2000 official training queries / 1488 videos; original locked inputs',
      EATA_vidstg='unavailable and paused; no HC-source Fisher performed',current_OPD_full_roster_evaluation=False))
    (PUB/'README.md').write_text('''# Baseline scoring of sealed predictions\n\n[Report](REPORT.md), [source and query table](TABLE.csv), [paired comparisons](PAIRED_COMPARISONS.json), [recorded cost](RECORDED_COST.json), and the three PNG/PDF figures are actual post-seal results. EATA is only the completed HC2 direction; the other direction remains paused. OPD has no full-roster result here.\n\nReproduce statistics without private assets:\n\n```bash\npython -B scripts/audit_decota_paper_baseline_scoring_public_v1.py results/decota_paper_baseline_scoring/2026-10-08\n```\n\nDependencies: Python 3 and NumPy. Anonymous compressed JSONL retains all 217221 logical rows, including negative results and three complete orders. Large JSON files are losslessly compressed as described in PUBLIC_DATA_FORMAT.json; use scripts/decota_public_result_io_v1.py. No video, caption, GT annotation, trained weight, optimizer, parameter vector or raw cache is distributed. The raw evaluator requires separately obtained official/private inputs and preserved stage receipts; the public scalar audit does not claim to reconstruct unavailable production gradients or model Jacobians.\n''')
    formats={}
    for f in PUB.rglob('*.json'):
        if f.stat().st_size<300000:continue
        raw=f.read_bytes();gz=Path(str(f)+'.gz');gz.write_bytes(gzip.compress(raw,9,mtime=0));assert gzip.decompress(gz.read_bytes())==raw
        formats[str(f.relative_to(PUB))]=dict(public_file=str(gz.relative_to(PUB)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_files=formats,reader='scripts/decota_public_result_io_v1.py',rows_filtered=False))
    files=owned+[str(p.relative_to(ROOT)) for p in PUB.rglob('*') if p.is_file() and str(p.relative_to(PUB)) not in formats]
    for p in files:
        dst=CHECKOUT/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
        if p.endswith(('.json','.md','.gz')):
            raw=gzip.decompress(dst.read_bytes()) if p.endswith('.gz') else dst.read_bytes()
            assert b'/home/wwww/.codex/attachments/' not in raw
            if b'"caption":' in raw:caption_guard(json.loads(raw))
            for forbidden in [b'"video_path":',b'"state_before":',b'"state_after":',b'"parameters":']:
                assert forbidden not in raw,(p,forbidden)
    # Validate the copied, publishable artifacts before committing.
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_decota_paper_baseline_scoring_public_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert set(changed)<=set(files) and changed
    rows=[dict(path=p,bytes=(CHECKOUT/p).stat().st_size,sha256=sha(CHECKOUT/p),blob_sha=git('hash-object',p)) for p in files]
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),expected_tree=git('write-tree'),files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),time=time.time()))
    print('PUBLIC_STAGE_READY',len(rows),sum(r['bytes'] for r in rows),flush=True)

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json')
    for branch in ['main',BRANCH]:
        ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/'+branch));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert cm['parents'][0]['sha']==meta['base_commit'] and cm['tree']['sha']==meta['expected_tree']
    def one(r):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(r['path'])}")
        assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'] and raw==(CHECKOUT/r['path']).read_bytes()
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==r['blob_sha']
        return dict(**r,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(one,meta['files']))
    verify();assert git('rev-parse','HEAD')==commit and not git('status','--porcelain')
    write(BASE/'FINAL_GITHUB_RECEIPT.json',dict(status='pass',commit=commit,repository='Zonglin-He/A',branches=['main',BRANCH],
        files=receipts,file_count=len(receipts),bytes=sum(r['bytes'] for r in receipts),remote_tree_verified=True,time=time.time()))
    print('REMOTE_BYTES_VERIFIED',commit,len(receipts),flush=True)

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
