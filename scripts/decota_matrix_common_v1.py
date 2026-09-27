import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/decota_full_matrix_v1'
COHORTS={'vid707':('official_dense_evaluation_v2','hc_to_vid',707,384),'hc413':('dense_expansion_evaluation_v1','vid_to_hc',413,135),'hc16':('hc_missing16_evaluation_v2','vid_to_hc',16,16)}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def status(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def save(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    t=p.with_suffix('.tmp');torch.save(x,t);t.replace(p)
def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def plan():
    p=read(OUT/'lock.json')
    for f,h in p['pins'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Versioned implementation changed: '+f)
    return p
def queries(p,cohort):return p['cohorts'][cohort]['queries']
def folder(backbone,cohort,condition):return OUT/'runs'/backbone/cohort/condition
def prepare():
    from vg_tta.foreground_runtime import QuerySubjectParser
    from vg_tta.unanchored_shift_predictor_v1 import DEFAULT_BASELINE_CONFIGS
    parser=QuerySubjectParser(ROOT/'.cache/stanza')
    p={'version':'decota_full_matrix_v1','protocol':read(ROOT/'protocols/decota_full_matrix_v1.json'),'cohorts':{},'dev':{},'ta_checkpoints':{},'baseline_configs':DEFAULT_BASELINE_CONFIGS,'tastvg_selection':str(OUT/'tuning/selected.json')}
    old=read(ROOT/'artifacts/backbone_joint_falsification_v1/lock.json')
    p['dev']=old['groups'];p['dev_label_manifest']=old['score_manifest'];p['dev_label_manifest_sha256']=old['score_manifest_sha256'];p['ta_checkpoints']=old['checkpoints']
    for key,(parent,g,n,ns) in COHORTS.items():
        root=ROOT/'artifacts'/parent;l=read(root/'lock.json');spec=l['groups'][g];qs=spec['queries']
        assert len(qs)==n and len({q['source'] for q in qs})==ns
        assert not {r['input']['source'] for r in old['groups'][g]} & {q['source'] for q in qs}
        rows=[]
        for q in qs:
            rows.append({'input':q,'subject':parser(q['caption'])})
        p['cohorts'][key]={'parent':str(root),'parent_lock_sha256':sha(root/'lock.json'),'group':g,'queries':rows,'config':spec['selected_config'],'checkpoint':spec['checkpoint'],'checkpoint_sha256':spec['checkpoint_sha256']}
        if 'conditions' not in p:p['conditions']=l['conditions']
        assert p['conditions']==l['conditions']
        print('LOCK',key,n,ns,flush=True)
    # Prioritize representative severity3, then finish every remaining cell.
    p['conditions']=sorted(p['conditions'],key=lambda c:(0 if c['name']=='clean' else 1 if c['severity']==3 else 2,c['name']))
    files=['protocols/decota_full_matrix_v1.json','methods/decota_v1/api.py','methods/decota_v1/_fullspan.py','methods/decota_v1/__init__.py',
        'scripts/decota_matrix_common_v1.py','scripts/run_decota_matrix_v1.py','scripts/score_decota_matrix_v1.py','scripts/tune_decota_tastvg_v1.py',
        'vg_tta/decota_tastvg_episode_v1.py','vg_tta/tastvg_fullspan_multistep.py','vg_tta/tastvg_baseline_expansion.py','vg_tta/external_tta_baselines.py',
        'vg_tta/unanchored_shift_predictor_v1.py','vg_tta/shift_corruptions_v2.py','vg_tta/dense_expansion_data_v1.py','vg_tta/metrics.py','vg_tta/foreground_runtime.py']
    p['pins']={f:sha(ROOT/f) for f in files};p['query_backbone_conditions']=1136*len(p['conditions'])*2;p['created_unix']=time.time()
    write(OUT/'lock.json',p);print('PREPARED',p['query_backbone_conditions'],'query-backbone-condition episodes; online inputs',flush=True)
