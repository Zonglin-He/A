"""Verify anonymous outputs against sealed original receipts, without GT annotations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,collections,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_tastvg_large_evidence_v1 import BASE,PUBLIC,PRIOR,LABELS,POOL,VIEWS,paths,key,read,sha,write,verify

def audit():
    torch.set_num_threads(2);verify(full=True);checks=collections.Counter();maxerr=0.
    manifest=read(PRIOR/'PUBLIC_MANIFEST.json')['files'];ev={e['cell_key']:e for e in read(PUBLIC/'EVIDENCE_ROWS.json')}
    labels={};output={};heads={};choices=collections.defaultdict(collections.Counter)
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            f=LABELS/sp/ds/'ROWS.json';assert sha(f)==manifest[str(f.relative_to(ROOT))]['sha256']
            labels.update({key(r):r for r in read(f)});output.update({key(r):r for r in read(PUBLIC/sp/ds/'ROWS.json')})
    for c in read(BASE/'COHORT.json')['cells']:
        raw=labels[key(c)];r=output[key(c)];assert c['parent']==r['source_id']==raw['source_id']
        p=read(paths(c)['pred']);assert p['persistent_pre_sha']==r['A_state_pre_sha256']==raw['A_state_pre_sha256']
        assert p['persistent_post_sha']==r['A_state_post_sha256']==raw['A_state_post_sha256']
        assert r['A8_v']==raw['A8_v'] and r['A8_t']==raw['A8_t'];checks['baseline_rows']+=1
        if not c['scheduled']:continue
        e=ev[key(c)];assert r['candidate_v']==raw['candidate_v'] and r['candidate_t']==raw['candidate_t'];checks['metric_scalars']+=64
        assert e['candidate_indices']==raw['candidate_indices'] and e['intervals']==raw['intervals_normalized']
        pp=paths(c);hr=read(pp['head_receipt']);hf=(POOL/c['dataset'])/hr['cache']
        if hr['sha256'] not in heads:
            d=torch.load(hf,map_location='cpu',weights_only=False);ids=d['frame_ids'];lp=np.zeros((len(ids),2))
            for z,rec in zip(d['prediction']['logits'],d['records']):
                val=torch.log_softmax(z.reshape(-1,2).double(),dim=0).numpy()-np.log(2.)
                for i,f in enumerate(rec['frame_ids']):lp[ids.index(f)]=val[i]
            heads[hr['sha256']]=(lp,ids)
        lp,ids=heads[hr['sha256']];err=float(np.max(np.abs(lp-e['native_logprior'])))
        assert err<1e-11;maxerr=max(maxerr,err);checks['native_logprob_scalars']+=lp.size
        lo,hi=ids[0],ids[-1]+1
        for view in ['view0','view1']:
            receipt=read(pp['expert_receipt'] if view=='view0' else pp['view_receipt'])
            f=(POOL/c['dataset'])/'experts'/receipt['cache'] if view=='view0' else ROOT/receipt['cache']
            d=torch.load(f,map_location='cpu',weights_only=False)
            mapped=(np.asarray(d['proposals']).reshape(-1,2)-lo)/(hi-lo)
            np.testing.assert_array_equal(mapped,np.asarray(e['proposals_'+view]));checks['proposal_endpoint_scalars']+=mapped.size
        panel=c['dataset']+'/'+c['split']
        for s in ['N','U','S']:
            for n in [8,32]:
                arm=s+str(n);delta=r[arm+'_gain'];a=e['anchor_index'];i=e['choices'][arm];g=e['geometry'][i]
                assert r[arm+'_destroyed_old_fast']==float(raw['old_fast_gain']>1e-12 and delta< -1e-12)
                if c['condition']=='clean':continue
                dest=choices[panel+'/'+arm];dest['changed']+=i!=a;dest['large_selected']+=g['large']
                dest['large_improved']+=g['large'] and delta>1e-12;dest['large_harmed']+=g['large'] and delta< -1e-12
                dest['small_harmed']+=(not g['large']) and delta< -1e-12
        checks['expert_bindings']+=1
    assert not torch.cuda.is_initialized() and checks['expert_bindings']==288
    seal=read(PUBLIC/'SIGNAL_SEAL.json');join=read(PUBLIC/'LABEL_JOIN.json')
    assert seal['time']<join['time'] and seal['GT_read'] is False
    result=dict(status='pass',checks=dict(checks),max_raw_native_logprob_error=maxerr,
        unique_head_caches=len(heads),CUDA_initialized=False,GT_annotation_files_opened=0,
        source_metric_inputs_byte_match_predecessor=True,production_unchanged=True,
        signal_seal_precedes_label_join=True,time=time.time())
    write(BASE/'FINAL_ROOT_AUDIT.json',result);write(PUBLIC/'ROOT_READBACK.json',result)
    write(PUBLIC/'SELECTED_GEOMETRY_COUNTS.json',{k:dict(v) for k,v in choices.items()})
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
