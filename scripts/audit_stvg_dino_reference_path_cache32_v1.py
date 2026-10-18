"""Independent cached-coordinate census and within-mask control audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.diagnose_stvg_dino_reference_path_cache32_v1 import D,P,PRE,verify,read_guard
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def main():
    import torch
    p=verify();read_guard();data=load(D/'SCALAR_ROWS_PRIVATE.pt');diag=read(P/'DIAGNOSIS.json');byalias={r['alias']:r for r in p['records']};count=collections.Counter();maxerr=0.
    def overlap(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        al,ah=a[:2]-a[2:]*.5,a[:2]+a[2:]*.5;bl,bh=b[:2]-b[2:]*.5,b[:2]+b[2:]*.5
        d=np.maximum(np.minimum(ah,bh)-np.maximum(al,bl),0);it=d[0]*d[1];u=a[2]*a[3]+b[2]*b[3]-it
        return it/u if u>0 else 0.
    payload={};native={}
    for rec in p['records']:
        a=rec['alias'];z=load(PRE/'predictions'/(a+'.pt'));s=load(ROOT/rec['prior_Source']);e=load(ROOT/rec['prior_prediction']);payload[a]=(z,s,e)
        assert not any(k in z for k in ['intermediate_reference_points','intermediate_hidden_states','init_reference_points','decoder_hidden_states','decoder_attentions'])
        for c in z['controls']:
            ob=s['expert']['observations'][('original',c['position'])];pr=ob['probe']
            if ob['receipt']['context_active']and c['native_top_slot']in pr['candidate_ids']:
                j=list(pr['candidate_ids']).index(c['native_top_slot']);assert c['native_top_score']==float(pr['target_scores'][j]);native[(a,c['position'])]=np.asarray(pr['boxes'][j],float)
        count['cache_inventories']+=1
    for x in data['rows']:
        a=x['alias'];rec=byalias[a];z,s,e=payload[a];k=x['sample'];pos=x['position'];base=np.asarray(e['outputs']['Source_no_correction'],float)[pos];shift=z['actions'][k]*np.asarray(p['scale'][rec['dataset']])
        inp=np.r_[base[:2]+shift[:2],np.exp(np.log(base[2:])+shift[2:])].astype(np.float32).astype(float)
        out=np.asarray(z['matched']['target_high']['details'][k]['anchors'][pos]['box'],float);n=native[(a,pos)]
        actual=dict(input_native_iou=overlap(inp,n),output_native_iou=overlap(out,n),
            input_center_relative=float(np.sqrt(sum(((inp[:2]-n[:2])/n[2:])**2))),output_center_relative=float(np.sqrt(sum(((out[:2]-n[:2])/n[2:])**2))),
            input_log_size=float(np.sqrt(sum(np.log(inp[2:]/n[2:])**2))),output_log_size=float(np.sqrt(sum(np.log(out[2:]/n[2:])**2))))
        for key,v in actual.items():maxerr=max(maxerr,abs(v-x[key]));count['independent_geometry_scalars']+=1
        assert hashlib.sha256(inp.astype(np.float32).tobytes()).hexdigest()==x['input_digest'];count['exact_FP32_reference_digest']+=1
        assert x['accepted']==(x['score']>=.35)
    assert maxerr<1e-12,maxerr
    for ds,g in diag['groups'].items():
        rr=[x for x in data['rows']if x['dataset']==ds];assert g['all_matched']['references']==len(rr);assert g['all_matched']['admitted']==sum(x['accepted']for x in rr)
        for field in ['input_native_iou','output_native_iou','input_center_relative','input_log_size','output_center_relative','output_log_size']:
            outkey={'input_native_iou':'input_native_final_IoU','output_native_iou':'output_native_final_IoU'}.get(field,field)
            assert abs(g['all_matched'][outkey]['mean']-np.mean([x[field]for x in rr]))<1e-12;count['domain_summary_means']+=1
        qq=[x for x in data['queries']if x['dataset']==ds]
        assert paired([x['near_minus_far_score']for x in qq])==g['fixed_content_and_mask']['query_near_minus_far_score'];count['paired_query_summary']+=1
    for w in data['within']:
        rr=[x for x in data['rows']if all(x[k]==w[k]for k in ['alias','position','content_slot','mask_slot'])];rr=list({x['input_digest']:x for x in rr}.values())
        assert len(rr)==w['unique_refs'];near=max(rr,key=lambda x:x['input_native_iou']);far=min(rr,key=lambda x:x['input_native_iou'])
        assert near['score']-far['score']==w['near_minus_far_score'];count['fixed_content_mask_groups']+=1
    assert not torch.cuda.is_initialized()
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(count),max_geometry_error=maxerr,CUDA_initialized=False,
        source_weights_media_GT_not_read=True,no_model_forward=True,layer_failure_location_unknown=True,private_scalar_SHA256=sha(D/'SCALAR_ROWS_PRIVATE.pt')))
    print(dict(status='pass',checks=dict(count),maxerr=maxerr))


if __name__=='__main__':main()
