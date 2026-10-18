"""Independent saved-vector, admission and arithmetic-score-path readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.analyze_stvg_dino_endpoint_sampling_diagnosis32_v1 import D,P,PRE,OLD,guard,verify

def main():
    guard();plan=verify();x=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in x['rows']};old={r['alias']:r for r in read(OLD/'RESULTS_PUBLIC.json')['rows']};checks=collections.Counter();largest=0.
    def enc(b):
        b=np.array(b,dtype=float);return np.r_[b[:2],np.log(b[2:])]
    for rec in plan['records']:
        a=rec['alias'];o=load(PRE/'predictions'/(a+'.pt'));r=rows[a];L=sum(v['legal']for v in o['sample_feedback']['details']);assert L==r['legal_states']
        for j,pos in enumerate(o['planned_positions']):
            node=r['nodes'][j];accepted=[(k,v['anchors'][pos])for k,v in enumerate(o['sample_feedback']['details'])if pos in v['anchors']and v['anchors'][pos]['accepted']]
            n=len(accepted);p=n/24;e=enc(o['outputs']['E_MBR'][pos]);c=o['center_feedback']['anchors'].get(pos);M=bool(c and c['accepted']);dc=enc(c['box'])-e if M else np.zeros(4)
            ds=sum((enc(v['box'])-e for _,v in accepted),np.zeros(4))/24;dp=p*dc;scale=np.asarray(o['scale'])
            assert node['sample_accepted']==n and node['p']==p and node['center_accepted']==M;checks['independent_support_nodes']+=1
            assert n==0 if L==0 else abs(p-(L/24)*(n/L))<1e-15;checks['independent_p_factorization']+=1
            for name,d in [('Center_full_local',dc),('Samples24_endpoint',ds),('Center_p_local',dp)]:
                actual=enc(o['outputs'][name][pos])-e;err=float(abs(actual-d).max());largest=max(largest,err);assert err<1e-13
                assert abs(np.linalg.norm(actual/scale)-node['actual_displacement'][name]['source_scaled_norm'])<1e-12
                checks['independent_applied_vectors_and_norms']+=1
            assert abs(np.linalg.norm((dp-dc)/scale)-node['amplitude_term_norm'])<1e-12
            assert abs(np.linalg.norm((ds-dp)/scale)-node['endpoint_and_support_term_norm'])<1e-12;checks['independent_component_norms']+=2
            if n and M:
                t=np.array([enc(v['box'])for _,v in accepted]).mean(0);cc=enc(c['box']);assert abs(np.linalg.norm((t-cc)/scale)-node['endpoint_mean_center_source_scaled_distance'])<1e-12;checks['independent_endpoint_geometry']+=1
            if n>=2:
                tr=o['traces']['samples'][pos];raw=[];te=[]
                for k,v in accepted:
                    ii=list(tr['candidate_indices']).index(k);idx=tr['inverse'][ii];raw.append(enc(tr['B_aligned']['initial_reference'][idx])/scale);te.append(enc(v['box'])/scale);checks['independent_raw_trace_mappings']+=1
                raw=np.array(raw);te=np.array(te);a1=np.sqrt(np.sum(np.var(raw,axis=0)));b1=np.sqrt(np.sum(np.var(te,axis=0)));ratio=b1/a1 if a1>1e-13 else None
                got=node['dispersion']['teacher_over_input_RMS'];assert got is None if ratio is None else abs(got-ratio)<1e-12;checks['independent_dispersion_nodes']+=1
        v=old[a]['values'];gap=100*(v['Samples24_endpoint']['v']-v['Center_full_local']['v']);amp=100*(v['Center_p_local']['v']-v['Center_full_local']['v']);end=100*(v['Samples24_endpoint']['v']-v['Center_p_local']['v'])
        for k,val in [('total_gap',gap),('p_scaling_path',amp),('endpoint_and_support_path',end)]:assert abs(r['outcome_pp'][k]-val)<1e-12;checks['saved_query_metric_paths']+=1
        assert abs(gap-amp-end)<1e-12;checks['independent_scalar_identity']+=1
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    for ds,g in x['groups'].items():
        rr=[r for r in x['rows']if r['dataset']==ds];assert len(rr)==16
        for k,v in g['existing_complete_path'].items():assert paired([r['outcome_pp'][k]for r in rr])==v;checks['independent_paired_summaries']+=1
        for k,v in g['common_saved_sIoU_path'].items():assert paired([r['common_saved_sIoU_pp'][k]for r in rr])==v;checks['independent_paired_summaries']+=1
        ns=[n for r in rr for n in r['nodes']];assert dict(collections.Counter(n['support']for n in ns))==g['support_counts']
        for k in ['total_gap','p_scaling_path','endpoint_and_support_path']:
            assert abs(sum(v['path_contribution_to_full16_mean_pp'][k]for v in g['center_outcome_descriptive_strata'].values())-g['existing_complete_path'][k]['mean'])<1e-12;checks['strata_exact_full16_accounting']+=1
    verify();import torch;assert not torch.cuda.is_initialized()
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),maximum_vector_error=largest,
        saved_scores_only=True,new_GT_reads=0,new_scoring=0,new_predictions=0,new_models=0,new_media_reads=0,new_weights_reads=0,inputs_unchanged=True,GPU_initialized=False))
    print(dict(checks),flush=True)
if __name__=='__main__':main()
