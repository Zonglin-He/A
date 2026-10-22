"""Post-seal audit only: no fitting, selection, prediction or label reads."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.boxfixed_tube.profiles import profile_id
from tools.tubedetr_opd_direct_diagnosis import cluster_difference


def read(p):return json.loads(Path(p).read_text())
def details(root):return {x['id']:x for x in read(root/'details_private.json')}
def value(r,k,method='Full'):return r['unobserved'][method]if k=='unobserved'else r['values'][method][k]
def paired(rows,baseline,method='Full'):
    result={}
    for k in ['s','unobserved','v']:
        ids=sorted(rows);v=[value(rows[q],k)-value(baseline[q],k,method)for q in ids];parents=[rows[q]['parent']for q in ids]
        d=cluster_difference(v,parents);a=np.asarray(v);d['leave_one_parent_out_min_pp']=min(100*float(a[np.array(parents)!=p].mean())for p in set(parents));result[k]=d
    return result


def main():
    torch.set_num_threads(1);pub=ROOT/'results/tubedetr_selection_coverage';out=ROOT/'outputs';calpub=ROOT/'results/tubedetr_selection_calibration';counts={};results={};audit={};bound={}
    for name in ['tubedetr_selection_coverage','tubedetr_selection_calibration']:
        p=read(ROOT/'results'/name/'plan.json');f=ROOT/'results'/name/'final_seal.json';assert read(f)['status']=='complete'
        for path,h in read(f)['files'].items():assert sha(ROOT/path)==h,path
        for path,h in p['code_sha256'].items():assert sha(ROOT/path)==h,path
        assert sha(ROOT/'results/tubedetr_boxfixed_development/final_seal.json')==p['parent_final_seal_sha256']
        bound[str(f.relative_to(ROOT))]=sha(f)
    for ds in ['vidstg','hc2']:
        banks=read(out/'tubedetr_selection_coverage'/ds/'evidence/sealed.json')['files'];cache={}
        for q,e in banks.items():
            for key,h in [('path','sha256'),('appearance_path','appearance_sha256')]:assert sha(Path(e[key]))==e[h]
            cache[q]=torch.load(e['path'],map_location='cpu',weights_only=False)
        cs=out/'tubedetr_selection_calibration'/ds/'calibration/sealed.json';flags={}
        for q,e in read(cs)['files'].items():
            assert sha(Path(e['calibration_path']))==e['calibration_sha256'];flags[q]=torch.load(e['calibration_path'],map_location='cpu',weights_only=False)['model']['valid']
        for name in ['tubedetr_selection_coverage','tubedetr_selection_calibration']:
            p=read(ROOT/'results'/name/'plan.json');metrics=read(ROOT/'results'/name/ds/'all_metrics.json');counts[name+'_'+ds]=len(metrics)*128
            for m in metrics:
                root=out/name/ds/'profiles'/profile_id(m['parameters']);f=root/'sealed.json';s=read(f);assert sha(f)==m['prediction_seal_sha256'];assert s['queries']==128 and set(s['files'])==set(banks) and s['plan_sha256']==sha(ROOT/'results'/name/'plan.json')
                for q,e in s['files'].items():
                    assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256'];r=read(root/e['receipt_path']);assert not r['labels_read']and r['new_GPU_forwards']==0
                    c=cache[q];b=c['bank'];nb=c['native_bank']
                    with np.load(root/e['path'])as z:
                        for k in ['Source','frame_ids','positions8','Native_interval','Global6_interval']:assert np.array_equal(z[k],np.asarray(c[k])),(name,ds,q,k)
                        idx=np.arange(b.frames);assert np.array_equal(z['BoxFixed'],b.boxes.numpy()[idx,z['actions']]);assert np.array_equal(z['BoxFixed'][b.observed],b.E.numpy()[b.observed])
                        assert np.all(np.any(np.all(z['BoxFixedNative'][:,None,:]==nb.boxes.numpy(),axis=-1),axis=1))
                dr=details(root)
                for method in ['Source','Spatial','Temporal','Full']:
                    for k in ['s','v','t']:assert abs(100*np.mean([x['values'][method][k]for x in dr.values()])-m['means'][method][k])<1e-10
                    for x in dr.values():assert x['values'][method]['t']==x['values']['Source'if method in ['Source','Spatial']else'Temporal']['t']
        previous=read(ROOT/'results/tubedetr_selection'/ds/'selected_profile.json');old=out/'tubedetr_selection'/ds/'profiles'/previous['profile_id'];oldrows=details(old)
        selected=read(calpub/ds/'selected_profile.json');current=out/'tubedetr_selection_calibration'/ds/'profiles'/selected['profile_id'];rows=details(current)
        ef=out/'tubedetr_selection/controls'/ds/'profiles'/profile_id(dict(selection_mode='ERef_only_no_selection'));es=read(ef/'sealed.json');assert len(es['files'])==128
        for e in es['files'].values():assert sha(ef/e['path'])==e['sha256']and sha(ef/e['receipt_path'])==e['receipt_sha256']
        comparisons={'Temporal':paired(rows,rows,'Temporal'),'previous_selection':paired(rows,oldrows),'same_ERef':paired(rows,details(ef))}
        # Both incumbent controls must be exact array replays, not only metric ties.
        cover=read(pub/ds/'selected_profile.json');coverage_root=out/'tubedetr_selection_coverage'/ds/'profiles'/cover['profile_id']
        pc=out/'tubedetr_selection_coverage'/ds/'profiles'/profile_id(dict(policy='previous',selection=previous['parameters']))
        for q in rows:
            for a,b in [(old,pc),(coverage_root,current)]:
                with np.load(a/(q+'.npz'))as za,np.load(b/(q+'.npz'))as zb:
                    for key in za.files:assert np.array_equal(za[key],zb[key]),(ds,q,key)
        allcal=read(calpub/ds/'all_metrics.json');newcal=[m for m in allcal if m['parameters']['policy']=='teacher_calibrated'];ranges={k:[min(100*0+m['means']['Full'][k]-m['means']['Temporal'][k]for m in newcal),max(m['means']['Full'][k]-m['means']['Temporal'][k]for m in newcal)]for k in ['s','v']}
        # Fixed first calibration config; this diagnosis never feeds back into fit.
        first=out/'tubedetr_selection_calibration'/ds/'profiles'/profile_id(dict(policy='teacher_calibrated',noise_margin=0.,link_multiplier=0.));fr=details(first);proxy={}
        for valid in [True,False]:
            qs=[q for q in rows if flags[q]==valid];proxy['teacher_CV_pass'if valid else'teacher_CV_fail']={
                'queries':len(qs),'dense_gain_vs_same_ERef_pp':100*float(np.mean([value(fr[q],'s')-value(details(ef)[q],'s')for q in qs])),
                'dense_wins_vs_same_ERef':sum(value(fr[q],'s')>value(details(ef)[q],'s')+1e-12 for q in qs),
                'dense_losses_vs_same_ERef':sum(value(fr[q],'s')<value(details(ef)[q],'s')-1e-12 for q in qs)}
        practical=comparisons['Temporal']['s']['mean_pp']>=.25 and comparisons['Temporal']['unobserved']['mean_pp']>=.25 and comparisons['same_ERef']['s']['wins']>=10 and all(comparisons['Temporal'][k]['CI95_pp'][0]>0 for k in ['s','unobserved'])
        results[ds]=dict(comparisons=comparisons,practical_goal_met=practical,new_calibration_all6_range_vs_Temporal_pp=ranges,first_fixed_calibration_proxy_diagnosis=proxy,parameters=selected['parameters'])
        audit[ds]=dict(queries=128,independent_parents=len({x['parent']for x in rows.values()}),Source_frames_observations_intervals_exact=True,all_outputs_from_existing_candidates=True,previous_controls_array_exact=True,all_profile_hashes_verified=True,metrics_recomputed_from_private_rows=True)
    write(pub/'paired_analysis.json',dict(status='complete',datasets=results,bootstrap_repeats=5000,seed=41,selection_and_repeated_exposure_not_adjusted=True,independent_confirmation=False))
    write(pub/'integrity_audit.json',dict(status='pass',trial_predictions_and_receipts=counts,total=sum(counts.values()),datasets=audit,production_code_hashes_and_old_full_seal_unchanged=True,new_full_predictions=0,new_neural_calls=0,postseal_analysis_code_sha256=sha(Path(__file__)),parent_seals=bound))
    print(json.dumps(results,indent=2));print('Prediction/receipt pairs audited:',sum(counts.values()))

if __name__=='__main__':main()
