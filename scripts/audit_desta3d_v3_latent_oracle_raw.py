"""Root readback of all oracle supports, sparse merger changes and native logits."""
import argparse,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import *
import numpy as np
import torch
ARMS=('original','temporal','spatial','dual','wrong_temporal','wrong_spatial')

def cell_mask(boxes,valid,T,H,W):
    out=np.ones((1,T,H,W),dtype=np.float64)
    for t in range(T):
        if not valid[t]:continue
        a,b,c,d=map(float,boxes[t])
        for y in range(H):
            for x in range(W):
                out[0,t,y,x]=max(0,min(c,(x+1)/W)-max(a,x/W))*max(0,min(d,(y+1)/H)-max(b,y/H))*H*W
    return out

def KL(a,b):
    a=a.detach().double().numpy();b=b.detach().double().numpy();assert a.shape==b.shape
    la=a-a.max(-1,keepdims=True);lb=b-b.max(-1,keepdims=True)
    la-=np.log(np.exp(la).sum(-1,keepdims=True));lb-=np.log(np.exp(lb).sum(-1,keepdims=True))
    value=float((np.exp(la)*(la-lb)).sum(-1).mean());assert value>=-1e-12;return value

def run(name):
    dest=OUT/name;verify_seal(dest);rows=read(dest/'INPUTS.json');labels=read(PANEL/'SOURCE_RECORDS.json')
    cases=[];max_mask=0.;max_norm=0.;fixed_s=0
    for i,(row,lab) in enumerate(zip(rows,labels)):
        ep=dest/'episodes'/f'{i:02}';m=torch.load(ep/'ORACLE_MASKS.pt',map_location='cpu',weights_only=False)
        w=torch.load(ep/'WRONG_MASKS.pt',map_location='cpu',weights_only=False)
        _,T,H,W=m['event'].shape;ids=lab['frame_ids'];assert ids==row['input']['frame_ids'] and row['key']==lab['key']
        active=np.array([lab['event_interval']['begin_fid']<=f<lab['event_interval']['end_fid'] for f in ids])
        assert np.array_equal(active,np.asarray(lab['event_active']))
        expected=cell_mask(lab['boxes_xyxy'],lab['box_valid'],T,H,W)
        err=float(np.max(np.abs(expected-m['spatial'].numpy())));max_mask=max(max_mask,err);assert err<1e-7
        assert np.array_equal(m['event'].numpy(),np.broadcast_to(active[None,:,None,None],(1,T,H,W)))
        dt=w['diagnostic']['temporal'];start,end=dt['wrong_interval'];wrong_active=np.array([start<=f<end for f in ids])
        assert end-start==lab['event_interval']['end_fid']-lab['event_interval']['begin_fid']
        assert wrong_active.sum()==active.sum() and dt['eligible_changed_support']==bool(np.any(active!=wrong_active))
        assert np.array_equal(w['temporal']['event'].numpy(),np.broadcast_to(wrong_active[None,:,None,None],(1,T,H,W)))
        boxes=[list(b) for b in lab['boxes_xyxy']]
        for sh in w['diagnostic']['spatial']['shifts']:
            k=sh['position'];old=np.array(boxes[k]);new=np.array(sh['to']);assert np.allclose(old[2:]-old[:2],new[2:]-new[:2],rtol=0,atol=1e-14)
            assert lab['box_valid'][k] and np.all(new>=0) and np.all(new<=1+1e-14);boxes[k]=sh['to']
        wrong_expected=cell_mask(boxes,lab['box_valid'],T,H,W)
        err=float(np.max(np.abs(wrong_expected-w['spatial']['spatial'].numpy())));max_mask=max(max_mask,err);assert err<1e-7
        assert np.allclose(expected.sum((-2,-1)),wrong_expected.sum((-2,-1)),rtol=0,atol=1e-10)
        base=torch.load(ep/'original.pt',map_location='cpu',weights_only=False);data={}
        for arm in ARMS:
            p=torch.load(ep/(arm+'.pt'),map_location='cpu',weights_only=False)
            effect=torch.load(ep/(arm+'_INJECTION_EFFECT.pt'),map_location='cpu',weights_only=False);inj={}
            for branch,e in effect.items():
                idx=e['indices'].numpy();a=e['before'].float().double().numpy();b=e['after'].float().double().numpy()
                assert len(idx)==e['changed_elements'] and len(np.unique(idx))==len(idx)
                assert len(idx)==0 or (idx.min()>=0 and idx.max()<e['numel'])
                assert np.isfinite(a).all() and np.isfinite(b).all() and np.all(a!=b)
                norm=float(np.sqrt(np.sum((b-a)**2)));err=abs(norm-e['delta_l2']);max_norm=max(max_norm,err);assert err<1e-10
                if arm=='original' or (arm in ('spatial','wrong_spatial') and branch=='event') or (arm in ('temporal','wrong_temporal') and branch=='spatial'):
                    assert len(idx)==0 and e['current_sha']==e['baseline_sha']
                inj[branch]={'changed':len(idx),'total':e['numel'],'fraction':len(idx)/e['numel'],'delta_l2':norm}
            same_ref=p['readout']['spatial_reference_token_ids']==base['readout']['spatial_reference_token_ids']
            same_support=same_ref and p['interval']==base['interval'] and p['positions']==base['positions']
            td=p['time_distribution'];bd=base['time_distribution'];timekl=coordkl=None
            timeids=set(bd['time_token_ids'])
            reference=lambda x:[t for t in x['readout']['spatial_reference_token_ids'] if t not in timeids]
            same_semantic=reference(p)==reference(base)
            if same_semantic and td['time_token_ids']==bd['time_token_ids'] and td['endpoint_logits'] is not None and bd['endpoint_logits'] is not None:
                timekl=KL(bd['endpoint_logits'],td['endpoint_logits'])
            cl=p['readout']['coordinate_logits'];bl=base['readout']['coordinate_logits']
            if same_support and cl is not None and bl is not None:coordkl=KL(bl,cl)
            if arm in ('spatial','wrong_spatial'):
                assert same_support and p['event_completion']==base['event_completion'] and torch.equal(td['endpoint_logits'],bd['endpoint_logits']);fixed_s+=1
            data[arm]={'interval':p['interval'],'positions':p['positions'],'same_semantic_reference':same_semantic,'same_spatial_condition_support':same_support,
                'time_KL_original_to_arm':timekl,'coordinate_KL_original_to_arm_on_identical_support':coordkl,
                'coordinate_KL_missing_reason':None if coordkl is not None else 'native reference/interval/positions or valid logits support differs',
                'injection':inj,'latent_log':p['latent_log'],'format_ok':p['format_ok'],
                'boxes_changed_on_same_support':int((p['boxes_cxcywh']!=base['boxes_cxcywh']).sum()) if same_support else None}
        cases.append({'key':row['key'],'source':row['source'],'THW':[T,H,W],'negative_control':w['diagnostic'],'arms':data})
    result={'status':'passed','cases':cases,'mask_scalar_loop_max_error':max_mask,'sparse_merger_norm_max_error':max_norm,
        'fixed_spatial_support_checks':fixed_s,'predictions_checked':len(cases)*6,'source_GT_used_for_oracle_audit':True,'target_read':False,
        'limitation':'Unchanged merger endpoints are worker-verified hashes; all changed positions/values retained. KL is conditional, not a full tube-policy KL.'}
    write(dest/'ROOT_RAW_READBACK.json',result);print({k:v for k,v in result.items() if k!='cases'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
