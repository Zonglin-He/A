"""One authorized cache-only bridge; no media, task labels, model or predictions."""
from pathlib import Path
import hashlib,json,collections,datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/round49/inventory_attempt001'
SEALS={};BINDINGS={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def cached(p):
    p=Path(p);folder=p.parent.parent if p.parent.name in ('samples_PRIVATE','cases_PRIVATE','arrays_PRIVATE') else p.parent
    sp=folder/'SEALED.json'
    if folder not in SEALS:
        SEALS[folder]=read(sp);BINDINGS[str(sp.relative_to(ROOT))]=sha(sp)
    got=sha(p);expected=SEALS[folder]['files'][str(p.relative_to(folder))]
    assert got==expected,('sealed_file_changed',str(p))
    BINDINGS[str(p.relative_to(ROOT))]=got
    if p.suffix=='.json':return read(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def comparison(a,b):
    out={}
    for k in ('boxes','scores','query','tokens','token_mask'):
        if k not in a or k not in b:out[k]={'present_both':False};continue
        x,y=np.asarray(a[k]),np.asarray(b[k]);same=x.shape==y.shape
        out[k]={'present_both':True,'shape_equal':same,'dtype_equal':str(x.dtype)==str(y.dtype),'exact':bool(same and np.array_equal(x,y)),
                'max_abs_difference':float(np.max(np.abs(x.astype(float)-y.astype(float)))) if same and x.size else None,
                'different_elements':int(np.count_nonzero(x!=y)) if same else None}
    return out

def original_source(r):
    folder={'A':'round17/features96_attempt001','H':'round19/sourceH32_attempt001','V':'round19/sourceV16_attempt001',
            'W':'round20/sourceW32_attempt001','Z':'round20/sourceZ64_attempt001','K':'round23/sourceK64_attempt001',
            'M':'round24/capture336_attempt001'}[r['stage']]
    return ROOT/'crossplatform'/folder/'samples_PRIVATE'/(r['id']+'.npz')

def main():
    assert not OUT.exists(),'Never overwrite an inventory attempt'
    OUT.mkdir(parents=True)
    manifests={s:ROOT/'research/round24/input'/(s+'_PRIVATE.json') for s in ('ALL336','D208','V64','M64')}
    jointmanifest=ROOT/'research/round27/input/ALL528_PRIVATE.json'
    lock={'state':'AUTHORIZED_ONE_CACHE_INVENTORY_BEFORE_ARRAY_ACCESS','script_sha256':sha(__file__),
          'human_scope_sha256':sha(ROOT/'research/round49/experiment/HUMAN_SCOPE_AUTHORIZATION.json'),
          'manifests_sha256':{s:sha(p) for s,p in manifests.items()},'R27_join_manifest_sha256':sha(jointmanifest),
          'no_GT_media_models_prediction_scoring':True,'source_exact_no_tolerance':True,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(OUT/'INVENTORY_LOCK.json',lock)
    rr=read(manifests['ALL336']);assert len(rr)==336 and len({r['id'] for r in rr})==336
    split={r['id']:s for s in ('D208','V64','M64') for r in read(manifests[s])};assert len(split)==336
    lookup={r['id']:r for r in read(jointmanifest)};records=[];errors=[]
    for r in rr:
        rid=r['id'];rec={'id_PRIVATE':rid,'split':split[rid],'platform':r['platform'],'stage':r['stage']}
        try:
            assert rid in lookup,'old336_id_absent_from_R27'
            jr=lookup[rid];rec['R27_caption_exact']=jr['caption']==r['caption'];assert rec['R27_caption_exact']
            legacy=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23')
            suffix='M64' if r['stage']=='M' else '272'
            z=cached(ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(rid+'.npz'))
            orig=cached(original_source(r));r25=cached(ROOT/'crossplatform/round25/source_features_attempt002/samples_PRIVATE'/(rid+'.npz'))
            prep=cached(legacy/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            arr=cached(legacy/('prepared'+suffix+'_attempt001')/'arrays_PRIVATE'/(rid+'.npz'))
            ev=cached(legacy/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            text=cached(legacy/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            dino=cached(legacy/('dino'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'))
            bank=cached(legacy/('bank'+suffix+'_attempt001')/'samples_PRIVATE'/(rid+'.npz'))
            joint=cached(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(rid+'.json'))
            jointread=cached(ROOT/'crossplatform/round27/scores528_attempt001/samples_PRIVATE'/(rid+'.npz'))
            rec['R24_vs_original_legacy_source']=comparison(z,orig)
            rec['R24_vs_R25_joint_source']=comparison(z,r25)
            rec['R24_vs_prepared_source']=comparison(z,{'boxes':arr['source_boxes'],'scores':arr['source_scores']})
            rec['original_legacy_vs_prepared_source']=comparison(orig,{'boxes':arr['source_boxes'],'scores':arr['source_scores']})
            rec['R24_vs_joint_readout_source']=comparison(z,{'boxes':jointread['boxes'],'scores':jointread['source']})
            pool=prep['mask_pool_PRIVATE'];w=arr['weights'];rec['pool_size']=len(pool);rec['W_shape']=list(w.shape)
            assert w.shape==(256,len(pool)) and len(set(pool))==len(pool) and np.isfinite(w).all() and (w>=0).all()
            rec['mapped_queries']=int((w.sum(1)>0).sum());rec['zero_W_queries']=int((w.sum(1)==0).sum())
            rec['W_row_sum_max_abs_from_one_mapped']=float(abs(w.sum(1)[w.sum(1)>0]-1).max()) if rec['mapped_queries'] else None
            assert set(ev['targets_PRIVATE'])=={str(m) for m in pool}
            assert set(joint['targets_PRIVATE'])=={str(m) for m in pool}
            visual=[];maxres=0.;evexact=True
            for m in pool:
                views=ev['targets_PRIVATE'][str(m)];assert len(views)==2
                for v in views:
                    delta=v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
                    maxres=max(maxres,abs(delta-v['visual_evidence']));evexact &= delta==v['visual_evidence']
                    assert np.isfinite([delta,v['visual_evidence']]).all()
                visual.append(float(np.mean([v['visual_evidence'] for v in views])))
                assert np.isfinite(joint['targets_PRIVATE'][str(m)]['crop_scene_joint']['log_odds_yes_no'])
            rec['raw_visual_evidence_subtraction_exact']=bool(evexact);rec['raw_visual_evidence_subtraction_maxerror']=maxres
            rec['raw_visual_mask_n']=len(visual);rec['reference_entities']=len(ev['references']);rec['dino_entities']=len(dino['objects'])
            rec['subject_query_present']=isinstance(text.get('subject_query'),str);rec['base_text_present']='base_text' in text
            rec['bank_shape_finite']=all(bank[k].shape==(256,) and np.isfinite(bank[k]).all() for k in ('vision','position','relation'))
            assert rec['bank_shape_finite']
            for key in ('R24_vs_original_legacy_source','R24_vs_R25_joint_source','R24_vs_prepared_source','original_legacy_vs_prepared_source','R24_vs_joint_readout_source'):
                rec[key+'_boxes_scores_exact']=all(rec[key][v].get('exact',False) for v in ('boxes','scores'))
            rec['structural_inventory_pass']=True
        except Exception as exc:
            rec['structural_inventory_pass']=False;rec['error_type']=type(exc).__name__;rec['error']=str(exc);errors.append(rec.copy())
        records.append(rec)
    groups={}
    for s in ('ALL336','D208','V64','M64'):
        rows=records if s=='ALL336' else [r for r in records if r['split']==s]
        groups[s]={'n':len(rows),'structural_pass':sum(r['structural_inventory_pass'] for r in rows)}
        for key in ('R24_vs_original_legacy_source','R24_vs_R25_joint_source','R24_vs_prepared_source','original_legacy_vs_prepared_source','R24_vs_joint_readout_source'):
            groups[s][key+'_exact_n']=sum(r.get(key+'_boxes_scores_exact',False) for r in rows)
    stop=bool(errors) or groups['ALL336']['R24_vs_prepared_source_exact_n']!=336
    summary={'state':'CACHE_INVENTORY_COMPLETE_NO_REPLAY_NO_SCORE','n':336,'by_split':groups,'structural_errors':len(errors),
             'R24_exact_W_bridge_pass':not stop,'R24_W_branch_STOP_pending_canonical_review':stop,
             'no_tolerance_or_remapping_applied':True,'no_media_GT_model_rank_prediction_access':True,
             'notes':['query/tokens differences logged independently; boxes+scores are mandatory for exact W/source bridge',
                      'No branch automatically switched canonical pool; root and independent reviewer must decide before stage replay']}
    write(OUT/'ROWS_PRIVATE.json',records);write(OUT/'ERRORS_PRIVATE.json',errors);write(OUT/'SUMMARY.json',summary)
    write(OUT/'SOURCE_BINDINGS_PRIVATE.json',BINDINGS)
    write(OUT/'SEALED.json',{'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
                           'state':summary['state'],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
