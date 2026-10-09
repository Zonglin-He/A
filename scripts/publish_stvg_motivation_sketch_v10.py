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
BASE=ROOT/'artifacts/stvg_motivation_sketch_v10'
PUB=ROOT/'results/stvg_motivation_sketch/2026-10-09'
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
    assert read(BASE/'ROOT_CASE_READBACK.json')['status']=='pass'
    assert review['MLLM_marked_joint_trained_reference']
    assert read(ROOT/'artifacts/stvg_motivation_cross_domain_v6/FINAL_GITHUB_RECEIPT.json')['status']=='pass'
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only')
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not PUB.exists();PUB.mkdir(parents=True)
    for n in ['DESIGN_LOCK.json','HUMAN_SCORE_LABEL_RUNTIME001.json','ROOT_CASE_READBACK.json']:
        shutil.copy2(BASE/n,PUB/n)
    public_review={k:v for k,v in review.items() if k!='files'}
    public_review['output_integrity']={k:v for k,v in review['files'].items() if not k.startswith('PUBLIC_LAYOUT')}
    write(PUB/'VISUAL_REVIEW.json',public_review)
    code=['scripts/render_stvg_motivation_sketch_v10.py',
          'scripts/publish_stvg_motivation_sketch_v10.py',
          'docs/STVG_MOTIVATION_SKETCH_V10.md']
    write(PUB/'SOURCE_BINDING.json',dict(status='bound',code={n:sha(ROOT/n) for n in code},
        original_native_outputs_and_statistics_changed=False,
        prior_results_commit=read(ROOT/'artifacts/stvg_motivation_cross_domain_v6/FINAL_GITHUB_RECEIPT.json')['commit'],
        actual_PNG_PDF_SVG_reviewed=True,private_media_query_GT_boxes_predictions_exported=False,
        GPU_model_optimizer_calls=0,new_GT_scoring=0,active_P1_payload_access=False,
        scientific_scope='qualitative same-case illustration, not aggregate evidence or efficacy proof'))
    scalars=read(BASE/'CASE_SCALAR_READBACK.json')
    write(PUB/'CASE_SCALAR_READBACK.json',{k:v for k,v in scalars.items() if k!='GT_and_prediction_intervals'})
    (PUB/'README.md').write_text("""# Three native model filmstrip rows and case score columns

The final human sketch requests one row each for TA-STVG, TubeDETR and an
MLLM, real GT/native box overlays, Prediction/GT event bars, and two case
score columns. The subsequent human instruction allows the six native score
labels. They are rounded to three decimals and the columns retain a common
linear zero-to-one scale. Timelines have no timestamps or numbered ticks.

TA-STVG and TubeDETR use HC2-source-only to VidSTG native cross-domain outputs.
The explicitly authorized PTD/Qwen3-VL row is marked joint-trained reference;
it is not source-only cross-domain and is excluded from the strict aggregate.
All three real video/query inputs, frames and GT instances match. Candidate
zero is reused without new model calls. The actual private PNG, editable SVG
and independent PDF raster have been reviewed. Their integrity metadata is
published; dataset RGB, query, GT geometry, raw temporal ranges, prediction
arrays and weights remain local.

This illustration is selected post hoc by a documented deterministic ranking
among five existing cases where all three temporal scores exceed .5 and
spatial scores do not. It illustrates that event coverage does not ensure
instance grounding. It does not estimate prevalence or establish aggregate
spatial-only-error dominance, OPD benefit, expert reliability or an on-policy
mechanism. All 512 original anonymous cross-domain outputs, negative primary
contrasts and bootstrap results remain unchanged and published in
results/stvg_motivation_cross_domain/2026-10-09. The conditional endpoint in
results/stvg_motivation_filmstrip/2026-10-09 remains labeled post hoc.

The scalar records below are case scores, not aggregate quadrant percentages.
See docs/STVG_MOTIVATION_SKETCH_V10.md for the factual paper caption. Run the
preserved independent public primary/conditional scalar audit with:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

Only this finite figure presentation closes here. The P1-P6 paper stages keep
their separate actual root/math/state/dense/view/publication completion gates.
""")
    owned=code+[str(p.relative_to(ROOT)) for p in sorted(PUB.iterdir())]
    for rel in owned:
        raw=(ROOT/rel).read_bytes()
        if rel.endswith('.json'):
            for key in [b'"caption":',b'"query":',b'"GT_box":',b'"boxes":',b'"gradient":',b'"GT_and_prediction_intervals":',b'"frames":']:
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
