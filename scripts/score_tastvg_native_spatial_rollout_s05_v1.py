"""Post-seal support scoring: no GT enters rollout generation."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_native_spatial_rollout_s05_v1 import OUT,S0,OLD,CONDS,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats

def write_new_or_equal(path,value):
    if path.exists():
        previous=read(path)
        if path.name=='GT_EXPOSURE.json':value['time']=previous['time']
        assert previous==value
    else:write(path,value)

def macro(rows,key):
    by=collections.defaultdict(list)
    for r in rows:
        if r[key] is not None:by[r['parent']].append(r[key])
    return stats([np.mean(by[p]) for p in sorted(by)])

def run():
    import ijson
    torch.set_num_threads(4);lock=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==96 and bar['candidates']==864
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    oldbar=read(OLD/'CAPTURE_BARRIER.json');previous={(r['parent'],r['condition']):r for r in read(S0/'ROWS.json')}
    keys={r['key'] for r in lock['rows']};labelpath=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
    with labelpath.open('rb') as f:labels={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
    assert set(labels)==keys
    write_new_or_equal(OUT/'GT_EXPOSURE.json',dict(retained_queries=16,historically_exposed=True,GT_for_scoring_only=True,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),container_sha256=sha(labelpath),time=time.time()))
    rows=[];calls=0
    for row in lock['rows']:
        label=labels[row['key']];valid=np.asarray(label['valid'],bool);seen=np.zeros(len(valid),bool);seen[np.rint(np.linspace(0,len(valid)-1,5)).astype(int)]=True;obs=valid&seen;unobs=valid&~seen
        for cond in CONDS:
            name=f"{row['ordinal']:03}.pt";x=load(OUT/'ta'/cond/name);rel=f'capture/{cond}/{name}';assert sha(OLD/rel)==oldbar['files'][rel];old=load(OLD/rel);assert x['pixel_sha256']==old['pixel_sha']
            base=x['trajectory'][0]['prediction'];assert torch.equal(base['boxes'],old['native']['boxes']);assert all(torch.equal(a,b) for a,b in zip(base['logits'],old['native']['logits']))
            metrics=[];quality=[]
            for candidate in x['trajectory']:
                pred=candidate['prediction'];m,q=checked_score(pred['boxes'],label,row['frame_ids'],pred['indices']);calls+=1;metrics.append({k:m[k] for k in ['sIoU','tIoU','vIoU_corrected']});quality.append(q)
            six=[]
            for candidate in old['candidates']['spatial']:
                m,_=checked_score(candidate['boxes'],label,row['frame_ids'],old['native']['indices']);six.append(m['sIoU']);calls+=1
            v=[m['sIoU'] for m in metrics];best=int(np.argmax(v));native=v[0];oracle=v[best];union=max(oracle,max(six));prior=previous[row['ordinal'],cond]
            assert abs(native-prior['native_s'])<1e-12 and abs(max(six)-prior['six_layer_oracle_s'])<1e-12
            un=float((quality[best]-quality[0])[unobs].mean()) if unobs.any() else None;ob=float((quality[best]-quality[0])[obs].mean()) if obs.any() else None
            rows.append(dict(parent=row['ordinal'],condition=cond,native_s=native,rollout_oracle_s=oracle,six_layer_oracle_s=max(six),union_oracle_s=union,rollout_gain=oracle-native,six_layer_gain=max(six)-native,union_gain=union-native,rollout_minus_six_layer=oracle-max(six),union_minus_six_layer=union-max(six),union_minus_s0=union-prior['union_oracle_s'],rollout_minus_s0=oracle-prior['expanded_oracle_s'],best_candidate=best,candidate_metrics=metrics,six_layer_s=six,observed_gain=ob,unobserved_gain=un,matched12_unobserved_gain=un if ob is not None else None,matched12_unobserved_minus_s0=un-prior['unobserved_gain_at_s_oracle'] if ob is not None and un is not None else None,unique_tubes=x['unique_tubes'],self_sIoU=x['self_sIoU'],minimum_self_sIoU=min(x['self_sIoU']),fixed_candidate_gains=[z-native for z in v],fixed_unobserved_gains=[float((q-quality[0])[unobs].mean()) if unobs.any() else None for q in quality],v_gain_at_s_oracle=metrics[best]['vIoU_corrected']-metrics[0]['vIoU_corrected'] if metrics[best]['vIoU_corrected'] is not None and metrics[0]['vIoU_corrected'] is not None else None))
    summary={};metric_keys=['native_s','rollout_oracle_s','six_layer_oracle_s','union_oracle_s','rollout_gain','six_layer_gain','union_gain','rollout_minus_six_layer','union_minus_six_layer','union_minus_s0','rollout_minus_s0','observed_gain','unobserved_gain','matched12_unobserved_gain','matched12_unobserved_minus_s0','minimum_self_sIoU','v_gain_at_s_oracle']
    for group in ['corruption']+CONDS:
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        arms={}
        for k in range(9):
            source_values=[]
            for p in sorted({r['parent'] for r in rr}):source_values.append(np.mean([r['fixed_candidate_gains'][k] for r in rr if r['parent']==p]))
            arms[str(k)]=dict(gain=stats(source_values),sources_better=int(sum(v>.001 for v in source_values)),sources_worse=int(sum(v<-.001 for v in source_values)),sources_worse_gt5pp=int(sum(v<-.05 for v in source_values)))
        summary[group]=dict(cells=len(rr),sources=16,metrics={k:macro(rr,k) for k in metric_keys},best_candidate_counts=dict(collections.Counter(r['best_candidate'] for r in rr)),unique_tube_counts=dict(collections.Counter(r['unique_tubes'] for r in rr)),fixed_arms=arms)
    write_new_or_equal(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summary);write(OUT/'AUDIT.json',dict(status='pass',cells=96,candidates=864,dual_metric_calls=calls,native_exact_cells=96,full_reinsertion_checks=len(read(OUT/'REINSERTION_AUDIT.json')['cells']),post_seal_scoring=True,expert_read_in_generation=False))
    print({g:{k:summary[g]['metrics'][k] for k in ['rollout_gain','union_gain','union_minus_s0','matched12_unobserved_gain']} for g in ['corruption','clean']})

if __name__=='__main__':run()
