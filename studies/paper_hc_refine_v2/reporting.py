"""Read-only postseal validation and HC fold/cost reporting; no inference/scoring."""
import sys,json,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,R,read,sha,atomic,validate

def run():
    import numpy as np
    p=validate();assert(D/'score_receipt.json').exists(),'Complete sealed prediction+independent score required';score=read(D/'score_receipt.json');assert score['checks']==23154 and score['all227sealed_before_labels'];seals=[read(D/f'sealed_{i}.json')for i in range(2)];assert score['seals_sha256']==[sha(D/f'sealed_{i}.json')for i in range(2)];files={k:v for s in seals for k,v in s['files'].items()};rows=read(D/'metrics.json')['private_rows'];index=read(D/'index.json');selection=read(P/'selection.json');assert len(rows)==len(files)==len(index)==227;assert len({r['parent']for r in rows})==227;ids={r['id']:f'Q{i:03}'for i,r in enumerate(rows,1)};cost={};checks=0;parity=0
    for x in index:
        e=files[x['row']['id']];assert sha(D/e['path'])==e['sha256']and sha(x['capture'])==x['capture_sha256']and sha(x['teacher'])==x['teacher_sha256'];checks+=3
        if x['origin']in ['old_screen','followup_confirm']:
            path=R/'outputs/paper_followup_v1/cg_dev/B/hc2/beta_0.25'/(x['row']['id']+'.npz')if x['origin']=='old_screen'else Path(x['baseline'])
            with np.load(D/e['path'])as z,np.load(path)as old:
                assert np.array_equal(z['u0.5_l0.25_b0.25'],old['Full'])and np.array_equal(z['u0.5_l0.25_b0.25_interval'],old['Full_interval']);parity+=1
        w=f"{e['worker']['pid']}:{e['worker']['startticks']}";c=cost.setdefault(w,dict(queries=0,configuration_predictions=0,conditional_CPU_seconds=0.));c['queries']+=1;c['configuration_predictions']+=100;c['conditional_CPU_seconds']+=e['CPU_seconds']
    oof=[];fold_means={}
    for fold in range(5):
        train=[r for r in rows if r['fold']!=fold];test=[r for r in rows if r['fold']==fold];s=selection['folds'][str(fold)];assert s['train_parents']==len(train)and s['heldout_parents']==len(test);assert not{r['parent']for r in train}.intersection(r['parent']for r in test)
        ordered=[]
        for order,a in enumerate([t['id']for t in p['trials']]+['Source']):
            ordered.append(dict(profile=a,mean_v=float(np.mean([r['values'][a]['v']for r in train])),severe=sum(any(r['values'][a][k]-r['values']['Source'][k]<-.05 for k in ['v','t'])for r in train),order=order))
        ordered.sort(key=lambda a:(-a['mean_v'],a['severe'],a['order']));assert ordered==s['training_ranking']and ordered[0]['profile']==s['selected'];a=s['selected'];fold_means[str(fold)]=dict(selected=a,queries=len(test),Source_v=float(np.mean([r['values']['Source']['v']for r in test])),Current_v=float(np.mean([r['values']['Current']['v']for r in test])),Selected_v=float(np.mean([r['values'][a]['v']for r in test])))
        oof.extend([{**r,'chosen':a,'values':{k:r['values'][k]for k in ['Source','Original','Current']}|{'Selected':r['values'][a]}}for r in test])
    report=read(P/'out_of_fold/report.json')
    for a in ['Source','Original','Current','Selected']:
        for field,value in report['means'][a].items():
            vals=[r['values'][a][field]for r in oof if r['values'][a][field]is not None];assert(value is None and not vals)or abs(value-float(np.mean(vals)))<1e-12
    with(P/'out_of_fold/all_signed.tsv').open()as f:table=list(csv.reader(f,delimiter='\t'))
    assert len(table)==228
    for line,r in zip(table[1:],oof):line[0]=ids[r['id']]
    with(P/'out_of_fold/all_signed.tsv').open('w')as f:csv.writer(f,delimiter='\t',lineterminator='\n').writerows(table)
    with(P/'out_of_fold/choices_and_severe.tsv').open('w')as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['query','fold','chosen','timegroup','baseline','delta_t_pp','delta_v_pp','severe_union'])
        for r in oof:
            for base in ['Source','Original','Current']:
                t=100*(r['values']['Selected']['t']-r['values'][base]['t']);v=100*(r['values']['Selected']['v']-r['values'][base]['v']);writer.writerow([ids[r['id']],r['fold'],r['chosen'],r['duration_group'],base,t,v,t< -5 or v< -5])
    result=dict(status='verified_all227_postseal',prediction_Source_teacher_SHA_checks=checks,prior_saved_Current_array_matches=parity,train_holdout_parent_disjoint=True,fold_training_rank_recomputed_exact=True,folds=fold_means,worker_cost=cost,worker_wall_seconds=[s['seconds']for s in seals],direct_math_bridge_checks=sum(sum(s['bridges'].values())for s in seals),new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0,scope='All227historically exposed HCparent cases. OOFselection separation within this run, not new independent confirmation. Global fitted winner is descriptive. ParentCIs conditional on fold choices.',cost_scope='True cached CPU study, per-process walls overlap; conditional timings exclude source/teacher acquisition and some loading/geometry, not an end-to-end deployment benchmark.',code_sha256=sha(Path(__file__)))
    atomic(D/'reporting_receipt.json',result);atomic(P/'reporting_receipt.json',result);print(result)

if __name__=='__main__':run()
