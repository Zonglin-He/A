"""One prelocked CPU-only native probability pushforward probe, not new B."""
from pathlib import Path
import gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_temporal_b256_v1 import BASE,B,read,write,sha,numstats
OUT=BASE/'native_marginal_probe001'
def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f,review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    assert not OUT.exists();OUT.mkdir();write(OUT/'LOCK.json',dict(status='locked_before_probe_scores',count=256,one_rule_only='exact sum of independent Native offset-state probabilities for each merged physical interval; stable row-major ties',interval_readout='MAP of pushed-forward Native posterior; no Actionness',one_alternative='highest interval mass with different actual uniform8 frames from original Native; fixed same8-frame eligibility',no_Teacher_or_Student_called=True,not_new_B_method_result=True,no_parameter_or_rule_search=True,development_GT_already_exposed=True,source_sha256=sha(Path(__file__)),GT_read=False))
    rr=read(B/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();d=read(BASE/'RESULTS.json');prior={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};rows=[];details=[];torch.set_num_threads(2)
    for rec in rr:
        cid=rec['probe_alias'];c=cases[cid];off=f.native.offsets(c,_hidden_from_source_case(c));pp=[f.original._prob(o['base'])[0] for o in off];ids=list(map(int,c['frame_ids']));imap={x:i for i,x in enumerate(ids)};ss=[np.asarray([imap[o['ids'][i]] for i in o['ij'][0]]) for o in off];ee=[np.asarray([imap[o['ids'][i]] for i in o['ij'][1]]) for o in off];joint=pp[0][:,None]*pp[1][None,:];key=np.minimum(ss[0][:,None],ss[1][None,:])*len(ids)+np.maximum(ee[0][:,None],ee[1][None,:]);mass=np.bincount(key.reshape(-1),weights=joint.reshape(-1),minlength=len(ids)**2).reshape(len(ids),len(ids));assert abs(float(mass.sum())-1)<1e-12;flat=int(mass.argmax());s,e=divmod(flat,len(ids));mode=[ids[s],ids[e]+1];native=list(c['Native_interval']);req=read(ROOT/'artifacts/stvg_spatial_controls256_v1/temporal/requests'/(cid+'.json'));available=sorted(r['physical_frame_id']for r in req['frame_map']);native_set=f.sampled(native,available);alt=None;alt_key=None
        for flat in np.argsort(-mass.reshape(-1),kind='stable'):
            s,e=divmod(int(flat),len(ids))
            if s>=e or mass[s,e]<=0:continue
            iv=[ids[s],ids[e]+1];sample=f.sampled(iv,available)
            if sample is not None and sample!=native_set:alt=iv;alt_key=int(flat);break
        def score(iv):
            if iv==native:return prior[cid,'Frozen']
            m,_=metric._score_student_arm(dict(boxes=np.asarray(c['native_boxes_xyxy_normalized']),interval=iv),c,targets[cid],set(),official,independent);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>=.3),R_vIoU_at_0p5=float(m['m_vIoU']>=.5),m_sIoU_common_Native_clock_GT=prior[cid,'Frozen']['m_sIoU_common_Native_clock_GT']);return m
        n=score(native);m=score(mode);a=score(alt) if alt else n
        for arm in ['Frozen','Projection','T2','B']:rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,metrics=prior[cid,arm]))
        rows.append(dict(case_id=cid,dataset=rec['dataset'],arm='Native_marginal_MAP_CPU',metrics=m));native_mass=mass[imap[native[0]],imap[native[1]-1]];altmass=float(mass.reshape(-1)[alt_key])if alt_key is not None else None
        if alt is not None:
            latent=int(np.argmax(np.where(key.reshape(-1)==alt_key,joint.reshape(-1),-1.)));g0=math.log(float(joint.max()));g1=math.log(float(joint.reshape(-1)[latent]));gap=g0-g1
        else:gap=None
        details.append(dict(case_id=cid,dataset=rec['dataset'],mode_changed=mode!=native,alternative_available=alt is not None,alternative_duration_over_Native=(alt[1]-alt[0])/(native[1]-native[0]) if alt else None,native_interval_mass=float(native_mass),mode_interval_mass=float(mass.max()),alternative_interval_mass=altmass,representative_joint_logit_gap=gap,current_bounded_reward_could_not_flip_joint_target=gap>=2 if gap is not None else None,posthoc_pair_oracle_vIoU_gain_pp=100*(max(a['m_vIoU'],n['m_vIoU'])-n['m_vIoU']),posthoc_pair_oracle_tIoU_gain_pp=100*(max(a['m_tIoU'],n['m_tIoU'])-n['m_tIoU'])));del c,off,joint,key,mass;gc.collect()
    summary,paired=subset_aggregate(rows);stats=[]
    for ds in ['hc2','vidstg']:
        dd=[r for r in details if r['dataset']==ds];stats.append(dict(dataset=ds,count=128,mode_changed=sum(r['mode_changed']for r in dd),candidate_available=sum(r['alternative_available']for r in dd),bounded_joint_target_cannot_flip=sum(r['current_bounded_reward_could_not_flip_joint_target']is True for r in dd),numerical={k:numstats(r[k]for r in dd)for k in ['alternative_duration_over_Native','native_interval_mass','mode_interval_mass','representative_joint_logit_gap','posthoc_pair_oracle_vIoU_gain_pp','posthoc_pair_oracle_tIoU_gain_pp']}))
    write(OUT/'RESULTS.json',dict(status='complete',count=256,rows=rows,summary=summary,paired=paired,details=details,statistics=stats,GT_bindings=bindings,new_models=0,new_Teacher_calls=0,new_Student_fits=0,not_new_B_method_result=True,not_an_independent_confirmation=True,GT_read=True));write(OUT/'CPU_COMPLETION.json',dict(status='complete',count=256,result_sha256=sha(OUT/'RESULTS.json'),GT_read=True))
if __name__=='__main__':main()
