"""HC-only conservative spatial-weight calibration on already exposed CG caches."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,hashlib,time,argparse,subprocess,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/paper_followup_v1'),str(R/'studies/paper_review_suite_v1')]
from common import read,sha,atomic,identity,check,digest,labels,report
D=R/'outputs/paper_hc_refine_v2';P=R/'results/paper_hc_refine_v2';OLD=R/'outputs/paper_review_suite_v1';F=R/'outputs/paper_followup_v1'
SEED=20261027

def configs():
    base=dict(id='u0.5_l0.25_b0.25',unobserved=.5,link=.25,beta=.25)
    return [base]+[dict(id=f'u{u}_l{l}_b{b}',unobserved=u,link=l,beta=b)for u in [0.,.05,.1,.25,.5]for l in [0.,.25,.5,1.,2.]for b in [0.,.125,.25,.5]if(u,l,b)!=(.5,.25,.25)]

def prepare():
    import torch
    assert not(D/'plan.json').exists();check();old=read(OLD/'selection.json')['hc2'];new=read(F/'cg_selection.json')['hc2'];index=[]
    sources=[('old_screen',old['screen'],OLD,read(OLD/'cgstvg_v9_screen_sealed.json')['files']['hc2']),('old_confirm',old['confirm'],OLD,read(OLD/'cgstvg_v9_confirm_sealed.json')['files']['hc2']),('followup_confirm',new['confirm'],F,read(F/'cg_confirm_sealed.json')['files']['hc2'])]
    for origin,rows,root,files in sources:
        for row in rows:
            e=files[row['id']];cp=root/e['capture_path'];assert sha(cp)==e['capture_sha256'];cap=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
            if root==OLD:tp=OLD/'cgstvg_teacher_cache'/(cap['teacher_evidence_key']+'.pt')
            else:
                aq=torch.load((root/e['path']).with_suffix('.acquired.pt'),map_location='cpu',weights_only=False,mmap=True);tp=F/'cg_teacher_cache'/(aq['teacher']['cache_key']+'.pt')
            meta=read(tp.with_suffix('.json'));assert sha(tp)==meta['sha256'];assert meta['context']['frame_ids']==cap['frame_ids']and meta['context']['positions']==cap['positions8'];bp=root/e['path'];assert sha(bp)==e['sha256']
            index.append(dict(row=row,parent=str(row['metadata']['source']),origin=origin,capture=str(cp),capture_sha256=e['capture_sha256'],teacher=str(tp),teacher_sha256=meta['sha256'],baseline=str(bp),baseline_sha256=e['sha256']))
    assert len(index)==len({x['parent']for x in index})==227
    index.sort(key=lambda x:hashlib.sha256(f'{SEED}:{x["parent"]}'.encode()).hexdigest())
    for i,x in enumerate(index):x.update(fold=i%5,shard=i%2)
    atomic(D/'index.json',index)
    code=['studies/paper_hc_refine_v2/run.py','stvg/spatial.py','stvg/_spatial_math.py','stvg/evidence.py','stvg/prototype.py','stvg/representation.py','stvg/temporal.py','studies/paper_followup_v1/common.py','studies/paper_followup_v1/cg_calibrate.py','studies/paper_review_suite_v1/cgstvg_runtime_v9.py','studies/paper_review_suite_v1/cgstvg_interfaces_v9.py']
    p=dict(seed=SEED,scope='Latest direct user explicitly reopens ONLY further HC CG parameter calibration. No Vid, Source/expert/model/optimizer/production changes.',mechanism='With gamma0, residual damage is at spatial paths. Reduce unsupported trust in unobserved appearance; vary smoothness, then temporal evidence strength. No new scoring module, teacher, observation or GT inference gate.',frozen=dict(gamma=0,kappa=64,context=1.5,precision='originalCG420FP32',expert_K=8),trials=configs(),index_sha256=sha(D/'index.json'),code_sha256={n:sha(R/n)for n in code},prior_complete_sha256=sha(R/'outputs/publication/paper_followup_complete_20261024/completion.json'),data='ALL227HC parents previously CG-evaluated and TA-exposed; one existing query/parent. Development/recalibration only, NOT unseen confirmation. No data/parents silently replaced.',validation='Fixed metadata-hash5fold. For each fold select on remaining4folds only, evaluate heldout fold. All100configuration predictions for all227 seal BEFORE CPUlabels. No changing grid/folds based on these scores. Historical/manual exposure prevents claiming a fresh blind test.',selection='Source no-adaptation is eligible control; maximize train meanv, tie fewer severeSourceunion then declared order (Current profile first). Report all folds, allprofiles/all signed/severe/timegroups/parentCI. Global best on227 is explicitly fitted/descriptive, not independent evidence.',zero_parameters='beta0 disables temporal feedback, u0/lambda0 reduces to independent observed-frame selection and Source outside observations; report any selected disabled components explicitly, never relabel as validated full feedback.',budget=dict(new_Source=0,new_expert=0,new_model=0,new_optimizer=0,configuration_query_predictions=22700,CPU_workers=2,max_seconds=10800,max_private_bytes=10*2**30),Fig1='excluded',third_benchmark='deferred_by_user')
    p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',p);(D/'source_snapshot/run.py').write_bytes(Path(__file__).read_bytes());atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'],flush=True)

def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'});assert sha(D/'index.json')==p['index_sha256']
    for n,h in p['code_sha256'].items():assert sha(R/n)==h,n
    check();return p

def predict(shard):
    import numpy as np,torch,gc
    from stvg.runtime import guard
    from stvg.evidence import SourceEvidence
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.prototype import one_step
    from stvg.temporal import final_mbr
    from cgstvg_interfaces_v9 import pool_single_grid
    from cgstvg_runtime_v9 import clip_native_boxes
    from cg_calibrate import feedback
    p=validate();torch.set_num_threads(1);sys.addaudithook(guard);index=[x for x in read(D/'index.json')if x['shard']==shard];files={};started=time.perf_counter();atomic(D/f'worker_{shard}.json',identity());bridges={};profile=dict(calibration='Transferred',gamma=0.,link=.25,context=1.5,kappa=64,beta=.25)
    for i,x in enumerate(index,1):
        assert time.perf_counter()-started<p['budget']['max_seconds'];assert sha(x['capture'])==x['capture_sha256']and sha(x['teacher'])==x['teacher_sha256']and sha(x['baseline'])==x['baseline_sha256'];cap=torch.load(x['capture'],map_location='cpu',weights_only=False,mmap=True);teacher=torch.load(x['teacher'],map_location='cpu',weights_only=False,mmap=True);ids=cap['frame_ids'];pos=cap['positions8'];source=cap['Source'];g=cap['grids'];ref=np.asarray(teacher['boxes'],np.float64);ev=SourceEvidence.from_native_masks(g['spatial'],g['spatial_valid'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],g['shape'],selected_query=teacher['selected_query']);geo=SpatialGeometry(ev,source,ref);arrays={};tick=time.perf_counter()
        def solve(iv,u,l):
            bank=geo.for_interval(iv)
            if u!=.5:
                observed=bank.event&bank.observed;unobserved=bank.event&~bank.observed;ow=(1-u)*observed.double()/max(1,int(observed.sum()));uw=u*bank.supported.double()*unobserved/max(1,int(unobserved.sum()));bank.score_unary=(ow+uw)[:,None]*bank.u
            if l!=.25:bank.score_pair*=l/.25
            return evidence_dp(bank)['final'].numpy()
        spatial_cache={};feature_cache={}
        for t in p['trials']:
            u,l,b=t['unobserved'],t['link'],t['beta'];key=(u,l)
            if key not in spatial_cache:spatial_cache[key]=clip_native_boxes(solve(cap['native_mbr'],u,l))
            if b==0:iv=cap['native_mbr']
            else:
                if key not in feature_cache:feature_cache[key]=pool_single_grid(g['projected'],g['valid'],solve([ids[0],ids[-1]+1],u,l))
                q,_=one_step(feature_cache[key],cap['q0'],kappa=64,beta=b);iv=final_mbr(ids,q)['interval']
            full=spatial_cache[key]if list(iv)==cap['native_mbr']else clip_native_boxes(solve(iv,u,l));arrays[t['id']]=full;arrays[t['id']+'_interval']=np.asarray(iv)
        if bridges.get(x['origin'],0)<2:
            teacher['positions']=pos;reference=feedback(cap,teacher,'hc2',profile);assert np.array_equal(arrays['u0.5_l0.25_b0.25'],reference['Full'])and np.array_equal(arrays['u0.5_l0.25_b0.25_interval'],reference['Full_interval']);bridges[x['origin']]=bridges.get(x['origin'],0)+1
        if x['origin']in ['old_screen','followup_confirm']:
            with np.load((F/'cg_dev/B/hc2/beta_0.25'/(x['row']['id']+'.npz'))if x['origin']=='old_screen'else x['baseline'])as z:assert np.array_equal(arrays['u0.5_l0.25_b0.25'],z['Full'])and np.array_equal(arrays['u0.5_l0.25_b0.25_interval'],z['Full_interval'])
        with np.load(x['baseline'])as z:arrays.update(Original=z['TransferredFull'].copy()if x['origin']=='followup_confirm'else z['Full'].copy(),Original_interval=z['TransferredFull_interval'].copy()if x['origin']=='followup_confirm'else z['Full_interval'].copy())
        arrays.update(Source=source,Source_interval=np.asarray(cap['native_mbr']),frame_ids=np.asarray(ids),positions8=np.asarray(pos));out=D/'predictions'/(x['row']['id']+'.npz');out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();np.savez_compressed(out,**arrays);files[x['row']['id']]=dict(path=str(out.relative_to(D)),sha256=sha(out),CPU_seconds=time.perf_counter()-tick,capture_sha256=x['capture_sha256'],teacher_sha256=x['teacher_sha256'],worker=identity(),GT_read=False);atomic(D/f'progress_{shard}.json',dict(completed=i,total=len(index),profiles=100,seconds=time.perf_counter()-started,GT_read=False,**identity()));print(shard,i,flush=True);del cap,teacher,g,geo,ev,arrays,feature_cache;gc.collect()
    atomic(D/f'sealed_{shard}.json',dict(files=files,queries=len(index),bridges=bridges,seconds=time.perf_counter()-started,fingerprint=p['fingerprint'],GT_read=False))

def score():
    import numpy as np,torch
    from full_score import metric,cluster
    from cgstvg_score import native_official_metric
    torch.set_num_threads(1);p=validate();seals=[read(D/f'sealed_{i}.json')for i in range(2)];assert sum(s['queries']for s in seals)==227 and all(not s['GT_read']for s in seals);files={k:v for s in seals for k,v in s['files'].items()};gt=labels('hc2');rows=[];maxerr=0.;checks=0;names=['Source','Original']+[t['id']for t in p['trials']]
    for x in read(D/'index.json'):
        row=x['row'];e=files[row['id']];assert sha(D/e['path'])==e['sha256'];g=gt[row['id']];truth={int(k):v for k,v in g['boxes'].items()};vals={}
        with np.load(D/e['path'])as z:
            r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for a in names:
                iv=z[a+'_interval'].tolist();vals[a],err=metric(z[a],r,truth,g['interval'],iv,z['positions8']);ref=native_official_metric(z[a],z['frame_ids'].tolist(),r['metadata'],truth,g['interval'],iv,'hc2');err=max(err,max(abs(ref[k]-vals[a][k])for k in ['t','v','s']));assert err<2e-10;maxerr=max(maxerr,err);checks+=1
        vals['Current']=vals['u0.5_l0.25_b0.25'];rows.append(dict(id=row['id'],parent=x['parent'],fold=x['fold'],origin=x['origin'],duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
    def ranking(rr):
        out=[]
        for order,a in enumerate([t['id']for t in p['trials']]+['Source']):
            v=float(np.mean([r['values'][a]['v']for r in rr]));severe=sum(any(r['values'][a][k]-r['values']['Source'][k]<-.05 for k in ['v','t'])for r in rr);out.append(dict(profile=a,mean_v=v,severe=severe,order=order))
        return sorted(out,key=lambda x:(-x['mean_v'],x['severe'],x['order']))
    folds={};oof=[]
    for fold in range(5):
        train=[r for r in rows if r['fold']!=fold];test=[r for r in rows if r['fold']==fold];rr=ranking(train);chosen=rr[0]['profile'];folds[str(fold)]=dict(train_parents=len(train),heldout_parents=len(test),selected=chosen,training_ranking=rr)
        for r in test:oof.append({**r,'values':{a:r['values'][a]for a in ['Source','Original','Current']}|{'Selected':r['values'][chosen]}})
    global_rank=ranking(rows);winner=global_rank[0]['profile'];fitted=[{**r,'values':{a:r['values'][a]for a in ['Source','Original','Current']}|{'Selected':r['values'][winner]}}for r in rows];report(oof,['Source','Original','Current','Selected'],[('Selected','Source'),('Selected','Current'),('Selected','Original')],P/'out_of_fold');report(fitted,['Source','Original','Current','Selected'],[('Selected','Source'),('Selected','Current')],P/'fitted_all_exposed');atomic(P/'selection.json',dict(folds=folds,global_fitted_winner=winner,global_ranking=global_rank,scope='All data historically exposed; out-of-fold selection separation within this run, not unseen confirmation. Global best is fitted/descriptive.'))
    with(P/'all_profiles_signed.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','fold','timegroup','profile','t','v','s','unobserved_s','delta_t_Source_pp','delta_v_Source_pp','severe_union']);w.writerows([[f'Q{i:03}',r['fold'],r['duration_group'],a,*[r['values'][a][k]for k in ['t','v','s','unobserved_actual_sIoU']],100*(r['values'][a]['t']-r['values']['Source']['t']),100*(r['values'][a]['v']-r['values']['Source']['v']),any(r['values'][a][k]-r['values']['Source'][k]<-.05 for k in ['t','v'])]for i,r in enumerate(rows,1)for a in names])
    atomic(D/'metrics.json',dict(private_rows=rows));atomic(D/'score_receipt.json',dict(checks=checks,max_error=maxerr,all227sealed_before_labels=True,seals_sha256=[sha(D/f'sealed_{i}.json')for i in range(2)],new_model=0,new_expert=0));atomic(P/'score_receipt.json',read(D/'score_receipt.json'))

def run():
    p=validate();atomic(D/'launch.json',identity());atomic(D/'state.json',dict(status='predict',fingerprint=p['fingerprint'],**identity()));children=[]
    for i in range(2):
        log=(D/f'worker_{i}.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'predict','--shard',str(i)],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'child_{i}.json',identity(c.pid));children.append(c)
    assert[c.wait()for c in children]==[0,0],'Preserve failure, no automatic retry';atomic(D/'state.json',dict(status='CPU_score',fingerprint=p['fingerprint'],**identity()));c=subprocess.Popen([sys.executable,'-B',__file__,'score']);atomic(D/'score_child.json',identity(c.pid));assert c.wait()==0;atomic(D/'state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint'],**identity()))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','run']);a.add_argument('--shard',type=int,choices=[0,1]);z=a.parse_args();{'prepare':prepare,'predict':lambda:predict(z.shard),'score':score,'run':run}[z.command]()
