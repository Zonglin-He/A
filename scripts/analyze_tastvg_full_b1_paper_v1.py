"""Append-only post-completion sampled-grid readouts; never rename them official."""
import sys,json,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha,load
from vg_tta.tastvg_paper_readouts_v1 import standard_columns,arrival_kind,quartile
B=ROOT/'artifacts/tastvg_full_b1_v1';OUT=ROOT/'artifacts/tastvg_paper_matrix_v1/b1_readouts'

def run():
    assert read(B/'COMPLETION.json')['status']=='completed'
    assert read(B/'AUDIT.json')['status']=='pass'
    p=read(B/'ROSTER_LOCK.json');bar=read(B/'PREDICTION_BARRIER.json');assert bar['cells']==451728
    # Each bucket holds source-level accumulators; query and source means both retained.
    sums={};counts={};harm={};drift=[];cells=0
    for cond in p['conditions']:
      for order,seq in p['orders'].items():
        f=B/'metrics'/cond/(order+'.jsonl');assert sha(f)==read(f.with_suffix('.audit.json'))['metrics_sha256']
        drift_sums=np.zeros((4,3));drift_counts=np.zeros(4,dtype=int)
        with f.open() as stream:
          for line in stream:
            x=json.loads(line);arrival=x['arrival'];parent=x['parent'];assert seq[arrival]==parent
            row=p['rows'][parent];value=standard_columns(x['metrics']);delta=np.asarray(x['metrics'])[:,2]-x['metrics'][0][2]
            kind=arrival_kind(x['first_query_of_source'],x['expert_scheduled']);quarter=quartile(arrival,len(seq))
            from vg_tta.tastvg_deployment_corruption_v2 import burst_spec
            dose=0. if cond=='clean' else burst_spec(row['source'],row['input']['frame_count'],row['frame_ids'],int(cond.rsplit('_',1)[1]))['actual_observed_fraction']
            binname='zero' if dose==0 else ('0-5%' if dose<=.05 else ('5-10%' if dose<=.1 else '>10%'))
            groups=['all',kind,'quartile'+str(quarter+1),'query_type:'+str(row['query_type']),'dose:'+binname]
            for group in groups:
                key=(cond,order,group,row['source'])
                if key not in sums:sums[key]=np.zeros_like(value);counts[key]=0;harm[key]=np.zeros((4,2),dtype=int)
                sums[key]+=value;counts[key]+=1;harm[key]+=np.stack([delta<-.05,delta<-.20],axis=-1)
            rf=B/'online'/cond/order/f'{arrival:05}.json';rel=str(rf.relative_to(B));assert sha(rf)==bar['files'][rel]
            receipt=read(rf);raw=rf.with_suffix('.pt');assert sha(raw)==receipt['sha256'];z=load(raw)
            # Parameter source center is saved once and is not GT.
            if cells==0:initial=load(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/PARAMETER_SUPPORT.pt')['center']
            displacement=float(np.sqrt(sum(float((v.double()-initial[k].double()).square().sum()) for k,v in z['post_state'].items())))
            drift_sums[quarter]+=np.array([displacement,delta[3],float(x['expert_scheduled'])]);drift_counts[quarter]+=1;cells+=1
        assert sum(drift_counts)==len(seq)
        drift.append(dict(condition=cond,order=order,quarters=[dict(count=int(n),post_state_l2=float(v[0]/n),Final_minus_Frozen_v=float(v[1]/n),expert_fraction=float(v[2]/n)) for v,n in zip(drift_sums,drift_counts)]))
    assert cells==451728
    grouped=collections.defaultdict(list)
    for key in sums:grouped[key[:3]].append(key)
    output=[]
    for (cond,order,group),keys in grouped.items():
        count=sum(counts[k] for k in keys);query=sum(sums[k] for k in keys)/count;source=np.stack([sums[k]/counts[k] for k in keys]).mean(0)
        output.append(dict(condition=cond,order=order,group=group,queries=count,sources=len(keys),query_macro=query.tolist(),source_macro=source.tolist(),harm_counts=sum(harm[k] for k in keys).tolist()))
    write(OUT/'READOUTS.json',dict(status='completed',cells=cells,arms=['Frozen','Fast-only','Slow-only','Final'],metrics=['m_tIoU','sampled_m_vIoU','sampled_vIoU@0.3','sampled_vIoU@0.5'],rows=output,drift=drift,official_dense_equivalence='pending_separate_audit',clean_retention='requires_corruption_final_state_clean_replay; not the reset clean stream',time=time.time()))
if __name__=='__main__':run()
