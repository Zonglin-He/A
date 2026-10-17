"""CPU audit of fixed hypothesis identity/mass and boundary geometry; no search."""
from pathlib import Path
import gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_temporal_b256_v1 import BASE,B,T2,read,write,sha,numstats
def merged_mass(off,ps,iv):
    import numpy as np
    a,b=iv;terms=[]
    for strict_s,strict_e in [(False,False),(True,False),(False,True),(True,True)]:
        z=1.
        for o,p in zip(off,ps):
            ids=np.asarray(o['ids']);s=ids[o['ij'][0]];e=ids[o['ij'][1]]+1;mask=((s>a)if strict_s else(s>=a))&((e<b)if strict_e else(e<=b));z*=float(p[mask].sum())
        terms.append(z)
    return terms[0]-terms[1]-terms[2]+terms[3]
def main():
    import numpy as np,torch
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_hypothesis_fit32_v1 as f,review_stvg_sparse_r1_v2 as truth
    assert read(BASE/'CPU_COMPLETION.json')['count']==256;rr=read(B/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);jm={x['case_id']:x for x in read(B/'INPUT_BARRIER.json')['jobs']};fm={x['case_id']:x for x in read(B/'FIT_BARRIER.json')['files']};tm={x['case_id']:x for x in read(T2/'GLOBAL_PREDICTION_BARRIER.json')['files']};rows=[];direct_checks=0
    for r in rr:
        cid=r['probe_alias'];c=cases[cid];assert sha(ROOT/jm[cid]['path'])==jm[cid]['sha256'] and sha(ROOT/fm[cid]['path'])==fm[cid]['sha256'];j=read(ROOT/jm[cid]['path']);fit=torch.load(ROOT/fm[cid]['path'],map_location='cpu',weights_only=False)['fits']['HYP'];old=torch.load(ROOT/tm[cid]['path'],map_location='cpu',weights_only=False);off=f.native.offsets(c,_hidden_from_source_case(c));ps=[f.original._prob(o['base'])[0] for o in off];hs=j['selection']['hypotheses'];mass=[]
        for h in hs:
            joint=1.
            for o,p,pair in zip(off,ps,h['pairs']):
                ix=np.flatnonzero((o['ij'][0]==pair[0])&(o['ij'][1]==pair[1]));assert len(ix)==1;joint*=float(p[ix[0]])
            merged=merged_mass(off,ps,h['interval_frames']);assert merged>0 and merged>=joint-1e-14
            if direct_checks<4:
                a,b=h['interval_frames'];s=[np.asarray(o['ids'])[o['ij'][0]] for o in off];e=[np.asarray(o['ids'])[o['ij'][1]]+1 for o in off];brute=float((ps[0][:,None]*ps[1][None,:]*((np.minimum(s[0][:,None],s[1][None,:])==a)&(np.maximum(e[0][:,None],e[1][None,:])==b))).sum());assert abs(brute-merged)<1e-12;direct_checks+=1
            mass.append(dict(retained_joint_mass=joint,observable_merged_interval_mass=merged,retained_fraction_of_merged=joint/merged))
        qview=list(c['Native_interval']) if fit['no_information'] else hs[int(np.argmax(fit['Teacher_conditional']))]['interval_frames'];n=c['Native_interval'];p=c['Projection_interval'];a,b=targets[cid]['span'];duration=b-a;assert duration>0
        def geom(iv):return dict(duration_over_GT=(iv[1]-iv[0])/duration,start_error_over_GT=(iv[0]-a)/duration,end_error_over_GT=(iv[1]-b)/duration)
        rows.append(dict(case_id=cid,dataset=r['dataset'],B_actual_equals_target_hard=fit['final_interval']==qview,B_target_mass=mass,B_merged_probability_prefers_alternative=len(mass)==2 and mass[1]['observable_merged_interval_mass']>mass[0]['observable_merged_interval_mass'],T2_final_outside_pair=old['selected_interval'] not in [h['interval_frames'] for h in old['candidates']],Native_geometry=geom(n),Projection_geometry=geom(p),B_alternative_geometry=geom(hs[1]['interval_frames']) if len(hs)==2 else None,Projection_duration_over_Native=(p[1]-p[0])/(n[1]-n[0]),Projection_start_shift_over_Native=(p[0]-n[0])/(n[1]-n[0]),Projection_end_shift_over_Native=(p[1]-n[1])/(n[1]-n[0])));del c,off,fit,old;gc.collect()
    summary=[]
    for ds in ['hc2','vidstg']:
        dd=[r for r in rows if r['dataset']==ds];summary.append(dict(dataset=ds,count=len(dd),B_actual_equals_target_hard=sum(r['B_actual_equals_target_hard'] for r in dd),T2_final_outside_pair=sum(r['T2_final_outside_pair'] for r in dd),B_merged_probability_prefers_alternative=sum(r['B_merged_probability_prefers_alternative'] for r in dd),retained_fraction_of_merged=numstats(v['retained_fraction_of_merged'] for r in dd for v in r['B_target_mass']),retained_joint_mass=numstats(v['retained_joint_mass'] for r in dd for v in r['B_target_mass']),merged_interval_mass=numstats(v['observable_merged_interval_mass'] for r in dd for v in r['B_target_mass']),geometry={a:{k:numstats(r[a][k] for r in dd if r[a] is not None) for k in ['duration_over_GT','start_error_over_GT','end_error_over_GT']} for a in ['Native_geometry','Projection_geometry','B_alternative_geometry']},Projection_duration_over_Native=numstats(r['Projection_duration_over_Native'] for r in dd)))
    write(BASE/'OBSERVABILITY_AUDIT.json',dict(status='pass',count=256,rows=rows,summary=summary,merged_mass_direct_outer_product_checks=direct_checks,new_models=0,new_Teacher_calls=0,new_fits=0,new_candidates_evaluated=0,GT_only_posthoc_boundary_diagnosis=True,GT_bindings=bindings,GT_read=True))
if __name__=='__main__':main()
