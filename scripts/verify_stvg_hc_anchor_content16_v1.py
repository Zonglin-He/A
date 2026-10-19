"""Postseal independent reference, dense/actual metrics and parent statistics."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_hc_anchor_content16_v1 import D,P,PRE,OBS,verify


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    torch.set_num_threads(2);p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==16
    result=read(P/'RESULTS_PUBLIC.json');gt,spans,_gt_provenance=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
    plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
    saved={r['alias']:r for r in result['rows']};count=dict(teacher_reconstructions=0,full_metrics=0,actual_metrics=0,paired_statistics=0)
    model_hash=read(ROOT/'results/stvg_error_state/2026-10-19/observation_factorial32/MODEL_PROVENANCE.json')['TA']['hc2']['model_state_sha256']
    for r in p['records']:
        a=r['alias'];q=r['query_ordinal'];row=plans[q];z=load(D/'predictions'/(a+'.pt'));obs=load(OBS/'predictions'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'))
        assert z['model_hash']==model_hash
        original={f['position']:src['expert']['observations'][('original',f['position'])]['probe']for f in src['expert']['anchors']['single4']}
        expected=[]
        for ob in obs['spatial']['observations']:
            pos=ob['position'];cp=ob.get('probe',{});pr=cp if cp.get('accepted',False)else original.get(pos)
            if pr is None:continue
            j=int(np.argmax(pr['target_scores']));expected.append((pos,np.asarray(pr['boxes'][j],float)))
        assert len(expected)==len(z['targets'])
        for (pos,b),f in zip(expected,z['targets']):assert pos==f['position'] and np.array_equal(b,f['box']);count['teacher_reconstructions']+=1
        W,H=row['input']['width'],row['input']['height']
        def iou(b,frame):
            b=np.asarray(b,float);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*[W,H,W,H];xy=np.maximum(xy,0)
            tr=np.asarray(gt[q][frame]);inter=np.maximum(0,np.minimum(xy[2:],tr[2:])-np.maximum(xy[:2],tr[:2])).prod()
            area=lambda bb:np.maximum(bb[2:]-bb[:2],0).prod()
            return float(inter/max(area(xy)+area(tr)-inter,1e-12))
        for arm,boxes in z['outputs'].items():
            calc=DenseTube(boxes,row,gt[q],spans[q],clip=True).score(z['interval'])
            for k in ['v','t','s']:assert abs(calc[k]-saved[a]['values'][arm][k])<2e-10;count['full_metrics']+=1
            for name,observed in [('observed_actual_s',True),('unobserved_actual_s',False)]:
                values=[iou(boxes[j],fid)for j,fid in enumerate(z['frame_ids'])if fid in gt[q] and ((j in z['planned_positions'])==observed)]
                val=float(np.mean(values))if values else None;old=saved[a]['values'][arm][name]
                assert val is None and old is None or val is not None and old is not None and abs(val-old)<1e-12;count['actual_metrics']+=1
    for name,eps in result['groups']['hc2']['paired'].items():
        a,b=name.split('_minus_')
        for ep,expected in eps.items():
            v=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in result['rows']if r['values'][a][ep]is not None and r['values'][b][ep]is not None])
            assert len(v)==expected['parents'];assert abs(v.mean()-expected['mean'])<1e-12
            ids=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
            assert np.max(abs(np.quantile(v[ids].mean(1),[.025,.975])-expected['CI95']))<1e-12;count['paired_statistics']+=1
    verify(True);assert not torch.cuda.is_initialized()
    out=dict(status='pass',count=16,checks=count,teacher_not_GT_filtered=True,all_old_E_metrics_exact=True,
        checkpoint_model_hash_exact=True,GT_postseal_only=True,new_models_or_GPU=False,
        scope='independent teacher construction, alternative dense-tube and actual-node metrics, parent bootstrap; not GPU replication')
    put(P/'SCIENTIFIC_READBACK_AUDIT.json',out);print(out,flush=True)


if __name__=='__main__':main()
