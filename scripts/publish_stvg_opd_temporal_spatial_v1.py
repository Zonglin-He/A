"""Finite anonymous scalar/source/PDF whitelist; no private trajectories/media."""
import ast,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
from scripts.stage_stvg_opd_o_q0_phase_public_v1 import receipt_metadata
BASE=ROOT/'artifacts/stvg_opd_temporal_spatial_complementarity_v1';OUT=ROOT/'results/stvg_opd_temporal_spatial_complementarity/2026-10-12'
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()

def run():
    assert read(BASE/'ROOT_READBACK.json')['status']==read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set();bindings={}
    def copy(source,relative):
        data=source.read_bytes()
        if source.suffix=='.json':safe(read(source))
        if source.suffix=='.gz':
            import gzip
            with gzip.open(source,'rt') as stream:
                for line in stream:safe(json.loads(line))
        if source.suffix in ['.py','.md','.tex']:assert not private_attachment_path(data.decode())
        if source.suffix=='.py':ast.parse(data.decode())
        dest=CHECKOUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        assert sha(dest)==sha(source);owned.add(relative);bindings[relative]=dict(sha256=sha(source),bytes=len(data))
    for name in ['REPORT.md','ROWS.jsonl.gz','SUMMARY.json','Q0_GROUP_ROWS.jsonl.gz','Q0_PAIR_ROWS.jsonl.gz','Q0_DIAGNOSTIC_SUMMARY.json']:
        copy(OUT/name,str((OUT/name).relative_to(ROOT)))
    sources=['scripts/analyze_stvg_opd_temporal_spatial_v1.py','scripts/review_stvg_opd_temporal_spatial_v1.py',
        'scripts/report_stvg_opd_temporal_spatial_v1.py','scripts/render_stvg_opd_q0_geometry_layout002.py',
        'scripts/audit_stvg_opd_temporal_spatial_public_v1.py','scripts/publish_stvg_opd_temporal_spatial_v1.py',
        'protocols/stvg_opd_temporal_spatial_complementarity_v1.md','paper/stvg_opd_evidence_revision_20261012.tex',
        'paper/stvg_opd_revised_claims_20261012.tex','docs/STVG_OPD_CLAIM_EVIDENCE_REVISION_20261012.md']
    for relative in sources:copy(ROOT/relative,relative)
    receipt_dir=OUT/'receipts';receipt_dir.mkdir(exist_ok=False)
    for rel in ['CPU_CONTRACTS.json','RUNTIME.json','CACHE_SCOPE_READBACK.json','GLOBAL_READOUT_BARRIER.json',
        'CPU_COMPLETION.json','Q0_CANCELLATION_READBACK.json','ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json',
        'PAPER_REVISION_RECEIPT.json','PAPER_PREVIEW_COMPILER_RECEIPT.json','FIGURE_LAYOUT_REVISION002.json',
        'recovery/root_readiness001/CAPTURE_RECEIPT.json','ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
        f=BASE/rel
        if f.exists():
            target=receipt_dir/rel;write(target,dict(original_private_receipt_sha256=sha(f),projection=receipt_metadata(read(f))))
            copy(target,str(target.relative_to(ROOT)))
    inventory=read(BASE/'CACHE_SCOPE_READBACK.json')
    write(OUT/'DESIGN_PUBLIC.json',dict(status='complete_bounded_cached_study',scope='P6 all64 and exact two-order O128; unavailable full O extension not executed',
        primary_cells=64,secondary_cells=128,distinct_parents=64,matched_O_cells=128,total_O_cells=1792,unavailable_O_cells=1664,
        missing_time_fields=['two-offset frozen logits','two-offset raw actionness','physical offset frame IDs'],
        temporal_recipe_unchanged=True,original_Full_unchanged=True,new_GPU_fits=0,new_model_calls=0,source_state_pixel_ID_exact_join=True,
        source_inventory_sha256=sha(BASE/'CACHE_SCOPE_READBACK.json'),bootstrap_replicates=10000,seed=20261012,historical_exposure=True,
        source_checkpoint_labels=['official VidSTG TA-STVG','official HC-STVG-v2 TA-STVG'],
        Native_is_frozen_source=True,adapted_arms_conditional_on_Full_history=True,independent_online_streams=False,
        private_payloads_exported=False,GT_geometry_exported=False,personal_conversation_exported=False))
    copy(OUT/'DESIGN_PUBLIC.json',str((OUT/'DESIGN_PUBLIC.json').relative_to(ROOT)))
    for name in ['factorial_readouts.pdf','q0_mean_readout_geometry_layout002.pdf']:
        p=ROOT/'output/pdf/stvg_opd_temporal_spatial_complementarity'/name;copy(p,str((OUT/'figures'/name).relative_to(ROOT)))
    result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_temporal_spatial_public_v1.py'),str(CHECKOUT)],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    assert result.returncode==0,result.stdout+result.stderr;audit=json.loads(result.stdout);assert audit['status']=='pass'
    write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()))
    write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit);copy(OUT/'PUBLIC_SCALAR_AUDIT.json',str((OUT/'PUBLIC_SCALAR_AUDIT.json').relative_to(ROOT)))
    write(OUT/'SOURCE_MANIFEST.json',dict(status='pass',scope='exact source/scalar/PDF whitelist; no raw vector witnesses',files=bindings))
    copy(OUT/'SOURCE_MANIFEST.json',str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT)))
    entries=[]
    for relative in sorted(owned):
        p=CHECKOUT/relative;data=p.read_bytes();entries.append(dict(path=relative,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='stvg-temporal-spatial-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            blob=git('hash-object','-w',str(CHECKOUT/e['path']));assert blob==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',blob,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
        files=entries,file_count=len(entries),bytes=sum(v['bytes'] for v in entries),unrelated_untracked_preserved=unrelated,
        scope='complete cached64-parent factorial/Q0 algebra/negative results/two actual viewed PDFs and bounded working paper revision; fullO extension and S-C-N incomplete',time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',base=base,tree=tree,files=len(entries),bytes=sum(v['bytes'] for v in entries))))

if __name__=='__main__':run()
