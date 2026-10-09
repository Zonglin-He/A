"""Explicit anonymous scalar/statistics export; private filmstrip stays local."""
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
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
OLD=ROOT/'artifacts/stvg_motivation_cross_domain_v6'
NEW=ROOT/'artifacts/stvg_motivation_filmstrip_v7'
PUB=ROOT/'results/stvg_motivation_cross_domain/2026-10-09'
NEWPUB=ROOT/'results/stvg_motivation_filmstrip/2026-10-09'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'
PYTHON=ROOT/'.conda/tubedetr/bin/python'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()


def stage():
    assert read(OLD/'ROOT_AUDIT.json')['status']=='pass'
    assert read(NEW/'ROOT_SCALAR_READBACK.json')['total_scalar_comparisons']==144
    assert all(read(b/'ROOT_VISUAL_REVIEW.json')['status']=='pass' for b in [OLD,NEW])
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert git('branch','--show-current')==BRANCH and not git('diff','--cached','--name-only')
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not PUB.exists() and not NEWPUB.exists()
    PUB.mkdir(parents=True);NEWPUB.mkdir(parents=True)
    for n in ['SCALAR_ROWS.json','QUADRANT_STATISTICS.json','PORTABLE_SCALAR_AUDIT.json',
              'ROOT_AUDIT.json','DESIGN_LOCK.json','RUNTIME_LOCK.json','ROOT_RUNTIME.json',
              'CPU_CONTRACTS.json','QUALIFICATION_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json',
              'GT_EXPOSURE.json','CPU_COMPLETION.json','SCORING_AUDIT.json']:
        shutil.copy2(OLD/n,PUB/n)
    qualification={}
    for direction in ['vidstg_to_hc2','hc2_to_vidstg']:
        for model in ['tastvg','tubedetr']:
            q=read(OLD/direction/model/'QUALIFICATION.json')
            qualification[direction+'/'+model]={k:q[k] for k in ['status','actual_GPU_qualification_cells',
                'parameter_and_buffer_sha256_before','parameter_and_buffer_sha256_after',
                'parameter_versions_unchanged','parameter_updates','optimizer_created',
                'accepted_formal_predictions','GT_read','native_parity','repeated_native_outputs']}
    write(PUB/'QUALIFICATION_SUMMARY.json',dict(status='pass',cells=qualification,
        original_qualification_barrier_sha256=sha(OLD/'QUALIFICATION_BARRIER.json')))
    for b,out,stem in [(OLD,PUB,'panel_B_cross_domain'),(NEW,NEWPUB,'panel_B_conditional_cross_domain')]:
        for ext in ['png','pdf','svg']:
            f=b/(stem+'.'+ext)
            if ext=='svg':assert b'<image' not in f.read_bytes()
            shutil.copy2(f,out/f.name)
        v=read(b/'ROOT_VISUAL_REVIEW.json')
        write(out/'VISUAL_REVIEW.json',{k:v[k] for k in ['status','scope','main_png_size',
            'actual_image_panels','unique_real_frames','editable_svg_text_nodes','actual_PNG_and_PDF_view',
            'original_error_dominance_hypothesis_not_supported','private_pixels_query_GT_overlays_not_public',
            'active_OPD_P1_payload_access']})
    for n in ['CONDITIONAL_STATISTICS.json','DESIGN_LOCK.json','ROOT_RUNTIME_ADDITIONS.json','ROOT_SCALAR_READBACK.json']:
        shutil.copy2(NEW/n,NEWPUB/n)
    (PUB/'README.md').write_text('''# Sealed source-only cross-domain native error composition

TA-STVG and TubeDETR use two directions: VidSTG-source → HC2 validation and
HC2-source → VidSTG test. Each direction has128 historically exposed parent
sources, one query per parent and identical uniform-at-most64 RGB frames.
Eight real GPU qualification cells passed before512 formal native outputs;
all512 sealed before GT scoring. No PTD joint-SFT output is relabeled source-only.
Native preprocessing and final outputs are retained, with no test-time updates.

The predeclared T+/S− dominance hypothesis is NOT supported. Counts are
[31,19,44,34], [22,17,58,31], [17,21,29,61], [30,16,31,51] in order
T+S+, T+S−, T−S+, T−S−. Strict tIoU>.5 and fixed-GT-frame meanIoU>.5 remain.
All categories, registered contrasts,10000 paired-parent intervals and every
anonymous scalar row are retained. Missing support scores zero over all GT
frames. This is a descriptive historical panel, not fresh/full-benchmark evidence.

Root independently checked source/checkpoint/RGB/receipt integrity and3072
metric components over126272 frame/model observations, maximum error
1.2212453270876722e-15. Independent scalar audit reproduced104 values.
Code, locked protocol, anonymous scalars and the statistics-only panel are
public. Dataset pixels/query/GT overlays/raw predictions/weights remain local.

The human's subsequent filmstrip request is implemented in v7 using the same
data. Its expressly post-hoc conditional panel observes spatial failures even
after temporal success; it does not replace this negative primary result.
See docs/STVG_MOTIVATION_FILMSTRIP_V7.md and
results/stvg_motivation_filmstrip/2026-10-09. Reproduce all144 primary and
conditional scalar checks without models, labels, media or prediction payload:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

This closes only the finite figure evidence task. P1 GPU prediction production
is globally sealed, but its CPU scoring/root review and P2–P6 remain separate.
No overall paper completion is claimed; EATA and historical queues stay paused.
''')
    (NEWPUB/'README.md').write_text('''# Filmstrip presentation and conditional cross-domain spatial failures

The local actual-frame Figure1 uses five real frames as GT/TA-STVG/TubeDETR
filmstrips. This public folder contains its statistics-only panel, source pins
and the post-hoc conditional readback. Media, exact query and GT overlays are
not exported. The E3M Figure1 layout is a style reference, not source evidence.

All128 parents per direction and original thresholds are retained. Conditional
spatial failures among temporal successes are19/50 (38.0%),17/39 (43.5897%),
21/38 (55.2632%),16/46 (34.7826%), in source-direction/model order specified
by CONDITIONAL_STATISTICS.json. Denominators are explicit. Conditional95%CIs
use10000 paired-parent draws; this supplementary endpoint is post hoc and uses
seed20261009+direction index. Original v6 primary bootstrap/results are unchanged.

The original all-query T+/S− dominance hypothesis failed and remains published
in results/stvg_motivation_cross_domain/2026-10-09. This panel motivates the
need for spatial correction after event localization. It does not establish
OPD efficacy, DINO reliability, native timing reliability overall, or on-policy
optimizer necessity. Locked P2 mechanisms provide the latter tests.

See docs/STVG_MOTIVATION_FILMSTRIP_V7.md for the factual caption/design mapping.
The local PNG and independent PDF render were actually viewed. Independent
multiplicity-bootstrap readback verified144 scalar values including all104
original values. No new model, optimizer, expert or active P1 payload access
occurred. Different frozen training/preprocessing prevents pure architecture
attribution; the128-parent panels have historical exposure.
''')
    owned={p for p in read(OLD/'RUNTIME_LOCK.json')['code'] if 'stvg_motivation_cross_domain' in p or p=='vg_tta/stvg_motivation_quadrants_v6.py'}
    owned|={'scripts/audit_stvg_motivation_cross_domain_v6.py',
        'scripts/prepare_stvg_motivation_cross_domain_case_v6.py',
        'scripts/publish_stvg_motivation_cross_domain_v6.py','docs/STVG_MOTIVATION_CROSS_DOMAIN_V6.md',
        'scripts/render_stvg_motivation_filmstrip_v7.py','scripts/audit_stvg_motivation_filmstrip_public_v7.py',
        'docs/STVG_MOTIVATION_FILMSTRIP_V7.md','protocols/stvg_motivation_filmstrip_v7.md'}
    write(PUB/'SOURCE_BINDING.json',dict(status='bound',code={p:sha(ROOT/p) for p in sorted(owned)},
        original_native_runtime_sha256=sha(OLD/'RUNTIME_LOCK.json'),original_design_sha256=sha(OLD/'DESIGN_LOCK.json'),
        root_audit_sha256=sha(OLD/'ROOT_AUDIT.json'),new_presentation_design_sha256=sha(NEW/'DESIGN_LOCK.json'),
        current_registry_changed=False,native_parameters_outputs_thresholds_unchanged=True,
        all_negative_findings_retained=True,media_query_GT_overlay_raw_predictions_weights_exported=False,
        whole_P1_or_paper_complete=False))
    owned|={str(p.relative_to(ROOT)) for b in [PUB,NEWPUB] for p in b.iterdir() if p.is_file()}
    for rel in sorted(owned):
        raw=(ROOT/rel).read_bytes()
        if rel.endswith('.json'):
            for key in [b'"caption":',b'"query":',b'"video_path":',b'"GT_box":',b'"boxes":',
                        b'"gradient":',b'"optimizer_state":',b'"raw_logits":']:
                assert key not in raw,(rel,key)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    subprocess.run(['git','add','--',*sorted(owned)],cwd=CHECKOUT,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==owned
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B','scripts/audit_stvg_motivation_filmstrip_public_v7.py'],cwd=CHECKOUT))
    assert audited['status']=='pass' and audited['total_scalar_comparisons']==144
    files=[dict(path=p,bytes=(CHECKOUT/p).stat().st_size,sha256=sha(CHECKOUT/p),blob_sha=git('hash-object',p)) for p in sorted(owned)]
    record=dict(status='reviewed_anonymous_scalar_and_statistics_only_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        files=files,file_count=len(files),bytes=sum(f['bytes'] for f in files),public_portable_audit=audited,
        other_untracked_v5_preserved=True,media_bearing_figure_exported=False,whole_P1_or_paper_complete=False,time=time.time())
    write(OLD/'PUBLIC_STAGE.json',record)
    print(json.dumps({k:record[k] for k in ['status','file_count','bytes','base_commit','expected_tree']}))


def verify_remote(commit):
    record=read(OLD/'PUBLIC_STAGE.json')
    subprocess.run(['git','fetch','origin','main',BRANCH],cwd=CHECKOUT,check=True,stdout=subprocess.DEVNULL)
    assert git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)==commit
    assert git('rev-parse',commit+'^{tree}')==record['expected_tree']
    def check(f):
        url='https://raw.githubusercontent.com/Zonglin-He/A/'+commit+'/'+urllib.parse.quote(f['path'])
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-root-scalar-verification'}),timeout=30) as r:raw=r.read()
        assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],f['path']
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==f['blob_sha'] and git('rev-parse',commit+':'+f['path'])==blob,f['path']
        return f
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:verified=list(pool.map(check,record['files']))
    receipt=dict(status='pass',scope='each remote public file byte/SHA256/Gitblob and both branch refs',
        repository='Zonglin-He/A',commit=commit,branch=BRANCH,main_sha=commit,research_sha=commit,
        file_count=len(verified),bytes=sum(f['bytes'] for f in verified),files=verified,
        all_negative_findings_public=True,media_bearing_figure_exported=False,public_scalar_comparisons=144,
        whole_P1_or_paper_complete=False,time=time.time())
    write(OLD/'FINAL_GITHUB_RECEIPT.json',receipt)
    write(NEW/'FINAL_GITHUB_RECEIPT.json',{k:v for k,v in receipt.items() if k!='files'})
    subprocess.run(['git','update-ref','refs/heads/'+BRANCH,commit,record['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','read-tree',commit],cwd=CHECKOUT,check=True)
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only')
    print(json.dumps({k:receipt[k] for k in ['status','commit','file_count','bytes']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','verify']);parser.add_argument('--commit')
    a=parser.parse_args()
    if a.mode=='stage':stage()
    else:assert a.commit;verify_remote(a.commit)
