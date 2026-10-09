"""Publish presentation code and integrity evidence without dataset pixels."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_motivation_backbone_names_v11'
PUB=ROOT/'results/stvg_motivation_backbone_names/2026-10-09'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()


def stage():
    review=read(BASE/'ROOT_VISUAL_REVIEW.json');assert review['status']=='pass'
    assert review['number_labels_only_saved_native_scores']==6 and review['actual_PNG_and_PDF_view']
    assert review['explicit_spatial_GT_badges']==15 and not review['training_subtitle_on_figure']
    assert read(ROOT/'artifacts/stvg_motivation_cross_domain_v6/FINAL_GITHUB_RECEIPT.json')['status']=='pass'
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only')
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not PUB.exists();PUB.mkdir(parents=True)
    for n in ['DESIGN_LOCK.json','HUMAN_LAYOUT_RUNTIME001.json','HUMAN_GT_OVERLAY_RUNTIME002.json']:
        shutil.copy2(BASE/n,PUB/n)
    public_review={k:v for k,v in review.items() if k!='files'}
    public_review['output_integrity']={k:v for k,v in review['files'].items() if not k.startswith('PUBLIC_LAYOUT')}
    write(PUB/'VISUAL_REVIEW.json',public_review)
    code=['scripts/render_stvg_motivation_backbone_names_v11.py',
          'scripts/publish_stvg_motivation_backbone_names_v11.py',
          'docs/STVG_MOTIVATION_BACKBONE_NAMES_V11.md']
    write(PUB/'SOURCE_BINDING.json',dict(status='bound',code={n:sha(ROOT/n) for n in code},
        original_native_outputs_and_statistics_changed=False,
        prior_results_commit=read(ROOT/'artifacts/stvg_motivation_cross_domain_v6/FINAL_GITHUB_RECEIPT.json')['commit'],
        actual_PNG_PDF_SVG_reviewed=True,private_media_query_GT_boxes_predictions_exported=False,
        GPU_model_optimizer_calls=0,new_GT_scoring=0,active_P1_payload_access=False,
        scientific_scope='qualitative same-case illustration, not aggregate evidence or efficacy proof'))
    (PUB/'README.md').write_text("""# Final named backbones, larger text and explicit spatial GT overlays

The latest human edits name the third row PTD / Qwen3-VL, remove the training
subtitle from the graphic itself, enlarge the query and content labels,
and mark every correct spatial GT box clearly. Each of fifteen unchanged
official GT rectangles has a heavier green dashed line with white contrast
stroke and a green GT badge. The coral native prediction rectangles retain
geometry. Prediction/GT event ranges and all six native score labels are
unchanged, on the original common linear scale.

PTD uses the Qwen3-VL-4B backbone and joint VidSTG/HC-STVG supervision. This
training scope remains disclosed in the paper caption and this record.
It is a separately authorized reference, excluded from the strict source-only
cross-domain cohort. The other two rows are TA-STVG and TubeDETR, using
HC2-source-only to VidSTG native outputs. No model is rerun.

The actual private PNG, independent PDF raster and editable SVG have been
reviewed. Dataset RGB, exact query, GT coordinates, raw temporal ranges and
prediction arrays remain local. This public folder contains only code,
factual caption, design/runtime bindings and integrity evidence.

The original post-hoc illustrative case and all negative aggregate findings
are unchanged. Case score readback remains public in
results/stvg_motivation_sketch/2026-10-09. All 512 native cross-domain scalar
rows, registered negative contrasts and bootstrap results remain in
results/stvg_motivation_cross_domain/2026-10-09. The conditional companion in
results/stvg_motivation_filmstrip/2026-10-09 remains labeled post hoc.
This figure illustrates that event coverage does not ensure instance
grounding; it does not establish prevalence, OPD efficacy, expert reliability
or an on-policy causal mechanism.

The independent public primary and conditional scalar check remains:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

Only this finite visual revision closes here. Original P1-P6 actual root,
math/state/dense/view/publication and archive obligations remain active.
""")
    owned=code+[str(p.relative_to(ROOT)) for p in sorted(PUB.iterdir())]
    for rel in owned:
        raw=(ROOT/rel).read_bytes()
        if rel.endswith('.json'):
            for key in [b'"caption":',b'"query":',b'"GT_box":',b'"boxes":',b'"gradient":']:
                assert key not in raw,(rel,key)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    subprocess.run(['git','add','--',*owned],cwd=CHECKOUT,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(owned)
    audit=json.loads(subprocess.check_output([str(ROOT/'.conda/tubedetr/bin/python'),'-B',
        'scripts/audit_stvg_motivation_filmstrip_public_v7.py'],cwd=CHECKOUT))
    assert audit['status']=='pass' and audit['total_scalar_comparisons']==144
    files=[dict(path=p,bytes=(CHECKOUT/p).stat().st_size,sha256=sha(CHECKOUT/p),blob_sha=git('hash-object',p))
           for p in sorted(owned)]
    receipt=dict(status='reviewed_code_caption_integrity_only_staged',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=files,file_count=len(files),bytes=sum(f['bytes'] for f in files),
        public_scalar_audit=audit,old_untracked_stage_preserved=True,private_media_exported=False,
        whole_P1_or_paper_complete=False,time=time.time())
    write(BASE/'PUBLIC_STAGE.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','base_commit','expected_tree','file_count','bytes']}))


def verify(commit):
    receipt=read(BASE/'PUBLIC_STAGE.json')
    subprocess.run(['git','fetch','origin','main',BRANCH],cwd=CHECKOUT,check=True,stdout=subprocess.DEVNULL)
    assert git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)==commit
    assert git('rev-parse',commit+'^{tree}')==receipt['expected_tree']
    def check(f):
        url='https://raw.githubusercontent.com/Zonglin-He/A/'+commit+'/'+urllib.parse.quote(f['path'])
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-root-presentation-review'}),timeout=30) as r:
            raw=r.read()
        assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],f['path']
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==f['blob_sha']==git('rev-parse',commit+':'+f['path']),f['path']
        return f
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(check,receipt['files']))
    result=dict(status='pass',scope='each remote code/caption/integrity file bytes SHA256 Gitblob and both refs',
        repository='Zonglin-He/A',commit=commit,main_sha=commit,research_sha=commit,branch=BRANCH,
        file_count=len(files),bytes=sum(f['bytes'] for f in files),files=files,
        private_media_exported=False,all_prior_negative_results_retained=True,
        whole_P1_or_paper_complete=False,time=time.time())
    write(BASE/'FINAL_GITHUB_RECEIPT.json',result)
    subprocess.run(['git','update-ref','refs/heads/'+BRANCH,commit,receipt['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','read-tree',commit],cwd=CHECKOUT,check=True)
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only')
    print(json.dumps({k:result[k] for k in ['status','commit','file_count','bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','verify']);p.add_argument('--commit')
    args=p.parse_args()
    if args.mode=='stage':stage()
    else:assert args.commit;verify(args.commit)
