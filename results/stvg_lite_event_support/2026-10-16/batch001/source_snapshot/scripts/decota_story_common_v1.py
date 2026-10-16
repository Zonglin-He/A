"""Write-once small-study metadata; target labels are not in inference records."""
import hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import sha,read,write,status,save,load
OUT=ROOT/'artifacts/decota_story_v1'
GROUPS=['hc_to_vid','vid_to_hc'];BACKBONES=['tubedetr','tastvg']
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def cell(b,g,split,c='clean'):return OUT/'runs'/b/g/split/c
def config_path(b,g):return OUT/'tuning'/b/g/'selected.json'
def verify():
    p=read(OUT/'lock.json')
    pins=dict(p['pins'])
    for amendment in sorted((OUT/'amendments').glob('*.json')):
        for f,change in read(amendment)['changes'].items():
            assert pins[f]==change['before'];pins[f]=change['after']
    for f,h in pins.items():assert sha(f)==h,'Changed locked input: '+f
    return p
def cohort(p,g,split,condition='clean'):
    rs=p['groups'][g][split]
    return rs[:16] if condition!='clean' else rs
def prepare():
    from vg_tta.foreground_runtime import QuerySubjectParser
    from vg_tta.unanchored_shift_predictor_v1 import DEFAULT_BASELINE_CONFIGS
    old=read(ROOT/'artifacts/backbone_joint_falsification_v1/lock.json')
    native=read(ROOT/'artifacts/native_coverage_calibration_v1/lock.json')
    parser=QuerySubjectParser(ROOT/'.cache/stanza');groups={};parents={}
    for parent,g in [('official_dense_evaluation_v2','hc_to_vid'),('dense_expansion_evaluation_v1','vid_to_hc'),('hc_missing16_evaluation_v2','vid_to_hc')]:
        f=ROOT/'artifacts'/parent/'lock.json';parents[parent]={'path':str(f),'sha256':sha(f),'group':g,'spec':read(f)['groups'][g]}
    for g in GROUPS:
        dev=[]
        for r in old['groups'][g]:
            f=next(x for x in native['groups'][g]['rows'] if x['index']==r['input']['index'])
            dev.append({'input':r['input'],'subject':r['subject'],'tube_cache':f['feature_path'],'tube_cache_sha256':f['feature_sha256']})
        seen={r['input']['source'] for r in dev};pool=[]
        for parent,s in parents.items():
            if s['group']==g:
                pool.extend({'input':q,'parent':parent} for q in s['spec']['queries'] if q['source'] not in seen)
        pool.sort(key=lambda r:digest(f'20260909:{g}:{r["input"]["source"]}:{r["input"]["index"]}'))
        confirm=[]
        for r in pool:
            q=r['input']
            if q['source'] in seen:continue
            seen.add(q['source']);r['subject']=parser(q['caption']);confirm.append(r)
            if len(confirm)==64:break
        assert len(dev)==len({r['input']['source'] for r in dev})==32
        assert len(confirm)==len({r['input']['source'] for r in confirm})==64
        assert not {r['input']['source'] for r in dev}&{r['input']['source'] for r in confirm}
        groups[g]={'development':dev,'confirmation':confirm,'previous_config':native['groups'][g]['config']}
    proto=ROOT/'protocols/decota_story_v1.json'
    pins=[proto,ROOT/'methods/decota_v1/api.py',ROOT/'methods/decota_v1/_fullspan.py',
      ROOT/'scripts/decota_story_common_v1.py',ROOT/'scripts/run_decota_story_v1.py',
      ROOT/'vg_tta/decota_tastvg_episode_v1.py',ROOT/'vg_tta/native_coverage_calibration_v1.py',
      ROOT/'vg_tta/shift_corruptions_v2.py',ROOT/'vg_tta/unanchored_shift_predictor_v1.py',
      ROOT/'artifacts/backbone_joint_falsification_v1/lock.json',ROOT/'artifacts/native_coverage_calibration_v1/lock.json']
    pins += [Path(r['path']) for r in parents.values()]
    p={'version':'decota_story_v1','created_unix':time.time(),'protocol':read(proto),'groups':groups,'parents':parents,
       'ta_checkpoints':old['checkpoints'],'dev_label_manifest':old['score_manifest'],'dev_label_manifest_sha256':old['score_manifest_sha256'],
       'baseline_configs':DEFAULT_BASELINE_CONFIGS,'historically_untouched':False,'GT_used_for_roster':False,
       'pins':{str(f):sha(f) for f in pins},'conditions':[{'name':'clean','corruption':'clean','severity':0,'seed':20260908}]+[
          {'name':f'{k}_s{s}_seed20260908','corruption':k,'severity':s,'seed':20260908} for k in ['low_light','gaussian_blur','frame_dropping'] for s in [1,3,5]]}
    write(OUT/'lock.json',p);print('LOCKED development32 and confirmation64 unique sources per direction; corruption16.',flush=True)
