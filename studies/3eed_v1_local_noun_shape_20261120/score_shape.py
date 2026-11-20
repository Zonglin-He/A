"""Compare all sealed single-sentence predictions with object identities and official oriented boxes."""
import argparse
import csv
import gzip
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np
from paths_shape import *
from predict_shape import CODE, observations
from evaluate_witness import metadata, PANELS, COHORTS, historical_records
from score_roles import score_boxes, state, sensor_frame_box, csv_file, record_intervals
from endpoint_score import arrays_to_owners,write_gz
from score_swap import best
from tables_scene import grouped,paired_counts
from projection_witness import projection


def subsets(rows):
    result={k[0]:v for k,v in grouped(rows,['panel']).items()}
    a=result.get('ordinary_uniform',[])+result.get('ordinary_relation',[])
    if a:result['ordinary_combined']=a
    return result


def summarize(rows,pairs,prefix=''):
    metrics=[];changes=[];bound=[];members=[];summary=[]
    for group,sub in subsets(rows).items():
        mats={};recs={}
        idx=ARMS.index('local_shape')
        vals=[]
        for (src,target),rr in grouped(sub,['source','target']).items():
            by=[sorted([r for r in rr if r['arm']==a],key=lambda d:d['case']) for a in ARMS]
            mat=np.array([[r['raw_iou'] for r in x] for x in by]).T
            geo=np.array([[r['final_iou'] for r in x] for x in by]).T
            per=np.asarray([np.mean(m,axis=0)*100 for m in (mat>.25,mat>.5,mat,geo>.25,geo>.5,geo)])
            vals.append(per)
            members.append([group,src+'→'+target,len(mat)])
            for i,a in enumerate(ARMS):
                metrics.append(dict(panel=group,direction=src+'_to_'+target,arm=a,n=len(mat),
                    **dict(zip(('acc25','acc50','miou','geo_acc25','geo_acc50','geo_miou'),per[:,i]))))
            mats[src,target]=mat
            recs[src,target]=dict(recording=np.array([r['recording'] for r in by[0]]))
        mat=np.vstack(list(mats.values()))
        for i,a in enumerate(ARMS):
            rate=np.mean(vals,axis=0)[:,i]
            metrics.append(dict(panel=group,direction='macro',arm=a,n=len(mat),
                **dict(zip(('acc25','acc50','miou','geo_acc25','geo_acc50','geo_miou'),rate))))
            ci=record_intervals(recs,mats,idx,i)
            for tag,m in [('all_records',mat)]+[(s+'_to_'+t,m) for (s,t),m in mats.items()]:
                changes.append(dict(panel=group,direction=tag,method='local_shape',control=a,n=len(m),
                    better=int((m[:,idx]>m[:,i]+1.e-10).sum()),worse=int((m[:,idx]<m[:,i]-1.e-10).sum()),
                    **paired_counts(m,idx,i),macro_delta50_ci_low=ci[0] if tag=='all_records' else '',
                    macro_delta50_ci_high=ci[1] if tag=='all_records' else ''))
        for (a,),rr in grouped(sub,['arm']).items():
            known=[r for r in rr if r['known']]
            edited=[r for r in rr if r['changed_geometry']]
            bound.append(dict(panel=group,arm=a,n=len(rr),known_endpoints=len(known),
                correct_subject=sum(r['iou_s']=='T' for r in known),correct_ref=sum(r['iou_r']=='T' for r in known),
                both=sum(r['iou_s']==r['iou_r']=='T' for r in known),both_seed=sum(r['seed_s']==r['seed_r']=='T' for r in known),
                returns_applied=sum(r['applied'] for r in rr),edited=len(edited),
                geo50_rescued=sum(r['final_iou']>.5>=r['raw_iou'] for r in rr),
                geo50_harmed=sum(r['raw_iou']>.5>=r['final_iou'] for r in rr),
                beyond_A256_and_same_verified_subject=sum(r['final_iou']>.5>=max(r['pool'],r['raw_iou']) and
                    all(r[k]=='T' for k in ('iou_raw','iou_final','seed_raw','seed_final')) for r in rr)))
    for (panel,rule,arm),rr in grouped(pairs,['panel','rule','arm']).items():
        summary.append(dict(panel=panel,rule=rule,arm=arm,pairs=len(rr),frames=len({r['frame'] for r in rr}),
            recordings=len({r['recording'] for r in rr}),**{key:sum(r[key] for r in rr) for key in
            ('both_subjects','both_refs','four','both_in_subject_pool','both_target_ranks','either_tie','both_targets_reversed','both_subjects_with_geometry','four_with_geometry')}))
    csv_file(OUT/(prefix+'metrics.csv'),metrics);csv_file(OUT/(prefix+'paired.csv'),changes)
    csv_file(OUT/(prefix+'identities.csv'),bound);csv_file(OUT/(prefix+'counterfactuals.csv'),summary)


def run(repository,symmetric=False):
    pred_dir=PRED
    base=read(PRED/'SEALED.json')
    assert base['protocol']==digest(SPEC) and all(base['code'][f]==digest(HERE/f) for f in CODE)
    assert base['observations']==digest(OBS/'SEALED.json')
    if symmetric:
        from symmetric_shape import SP, CFG, SF
        pred_dir=SP
        check=read(SP/'SEALED.json')
        assert check['protocol']==digest(CFG) and check['fields']==digest(SF/'SEALED.json')
        assert check['original']==digest(PRED/'SEALED.json')
    prefix='symmetric_' if symmetric else ''
    seal=read(pred_dir/'SEALED.json');rows=observations()
    assert not seal['labels_used'] and not seal['FINAL_access']
    assert len(rows)==seal['n']==len(seal['files'])
    scope=read(OUT/'panel.json')
    assert scope['labels']==digest(WORK/'labels.json') and scope['roster']==digest(WORK/'roster.json')
    labels={}
    for p,h in [(previous.OLD/'labels.json',read(previous.LAST/'results/panel.json')['labels']),
                (previous.NEW/'labels.json',read(previous.OUT/'panel.json')['labels']),
                (WORK/'labels.json',scope['labels'])]:
        assert digest(p)==h
        d=read(p);assert not (d.keys()&labels.keys());labels.update(d)
    hist=historical_records(repository);panels={g:read(PANELS[g]) for g in COHORTS}
    past=read(previous.PRED/'SEALED.json')
    old_idx=previous.ARMS.index('joint_raw')
    content={};cache={};outs=[];pair_data=defaultdict(dict);pair_rosters={};maxerr=0.
    same=0;target_dirs=set();facets=defaultdict(int)
    for index,(row,rp,rsha,_,_,fid,*_) in enumerate(rows,1):
        n=name(row);uid=row['uid'];assert digest(pred_dir/n)==seal['files'][n] and digest(rp)==rsha
        e=labels[uid];f=row['frame'];target=f['target'];source=row['source'];target_dirs.add((source,target))
        gt,s,r,fr,rec,view,original=metadata(row,e,panels,hist,cache)
        tid=e.get('subject',0); byname=e['group']
        group='natural' if byname in COHORTS else ('change_landmark' if byname in ('name','depth') else byname)
        with np.load(pred_dir/n) as output,np.load(rp) as native:
            raw=output['raw'];geo=output['final'];ref=output['ref'];seeds=native['seeds']
            if byname in COHORTS:
                case=panels[byname][e['case']]['case'];rel=f'{COHORTS[byname]}/{case:02d}_{source}.npz'
                p=ROOT/'crossplatform/round171/predictions'/rel
                assert digest(p)==hist[0][rel]
                with np.load(p) as old:seeds=old['seeds']
            np.testing.assert_equal(raw[0],native['raw'])
            if n in past['files']:
                assert digest(previous.PRED/n)==past['files'][n]
                k=ARMS.index('frozen_words')
                with np.load(previous.PRED/n) as old:
                    np.testing.assert_equal(raw[k],old['raw'][old_idx]);np.testing.assert_equal(ref[k],old['ref'][old_idx])
                    np.testing.assert_allclose(output['scores'][k],old['subject_scores'][old_idx],rtol=0,atol=1e-13)
                same+=1
            u,inverse=np.unique(np.vstack([raw,geo]),axis=0,return_inverse=True)
            a,err=score_boxes(u,gt[tid]); b,err2=score_boxes(np.array([sensor_frame_box(v,target,view) for v in u]),
                original if original is not None else sensor_frame_box(gt[tid],target,view))
            maxerr=max(maxerr,err,err2,float(np.max(abs(a-b))));assert maxerr<1.e-8
            np.testing.assert_equal(a[:,None]>[.25,.5],b[:,None]>[.25,.5])
            before,after=np.split(a[inverse],2);pool=best(native['a'],gt[tid])
            T=output['candidates'];R=output['ref_candidates']
            own=arrays_to_owners([raw,ref,geo,T,R],gt,seeds)
            if e.get('group')=='switch_sign':
                pj=projection(gt,target,view,f['calibration'],f['rgb'])
                content[uid]=(s,pj)
            if 'pair' in e:
                pair=e['pair'];h=e['half'];pk=(source,pair)
                signature=sha256(T.tobytes()+native['t_box'].tobytes()+native['t_p'].tobytes()).digest()
                assert pair_rosters.setdefault(pk,signature)==signature
            for k,arm in enumerate(ARMS):
                row_out=dict(panel=group,case=uid,source=source,target=target,frame=fr,recording=rec,
                    arm=arm,known=int(s is not None and r is not None),applied=int(output['applied'][k]),
                    raw_iou=before[k],final_iou=after[k],pool=pool,changed_geometry=int(np.max(abs(raw[k]-geo[k]))>1.e-8))
                for rule,short in [('unique_iou','iou'),('seed_ownership','seed')]:
                    a1,a2,a3,Ts,Rs=own[rule]
                    so,ro,fo=(int(x[k]) for x in (a1,a2,a3))
                    row_out.update({short+'_s':state(so,s),short+'_r':state(ro,r),
                                    short+'_raw':state(so,tid),short+'_final':state(fo,tid)})
                    if 'pair' in e:
                        scores=output['scores'][k]
                        ranked={int(i):float(scores[Ts==i].max()) for i in np.unique(Ts) if i>=0}
                        pair_data[group,pair,source,target,fr,rec,rule,arm][h]=(s,r,so,ro,fo,ranked,uid)
                outs.append(row_out)
            key_coverage=output['coverage']
            if fid not in facets or np.asarray(key_coverage)[:,0].sum()>0:facets[fid]=key_coverage.copy()
        if index%300==0:print('scored',index,len(rows),flush=True)
    pair_out=[]
    for (group,pair,src,target,fr,rec,rule,arm),halves in pair_data.items():
        assert set(halves)=={0,1}
        x,y=halves[0],halves[1];s0,r0,so0,ro0,f0,m0,id0=x;s1,r1,so1,ro1,f1,m1,id1=y
        assert s0!=s1 and ((r0==r1)==(group in ('new_side','switch_sign')))
        if group=='switch_sign':
            px=content[id0][1]
            assert (px[s0,0]-.5)*(px[s1,0]-.5)>0
            assert (px[s0,0]-px[r0,0])*(px[s1,0]-px[r0,0])<0
        available=all(i in v for i in (s0,s1) for v in (m0,m1))
        d0=m0[s0]-m0[s1] if available else 0
        d1=m1[s1]-m1[s0] if available else 0
        pair_out.append(dict(panel=group,pair=pair,frame=fr,recording=rec,source=src,target=target,rule=rule,arm=arm,
            both_subjects=int((so0,so1)==(s0,s1)),both_refs=int((ro0,ro1)==(r0,r1)),
            four=int((so0,ro0,so1,ro1)==(s0,r0,s1,r1)),both_in_subject_pool=int(available),
            both_target_ranks=int(available and min(d0,d1)>1e-12),either_tie=int(available and min(abs(d0),abs(d1))<=1e-12),
            both_targets_reversed=int(available and max(d0,d1)<-1e-12),
            both_subjects_with_geometry=int((f0,f1)==(s0,s1)),four_with_geometry=int((f0,ro0,f1,ro1)==(s0,r0,s1,r1))))
    assert len(target_dirs)==6 and same==len(previous.observed_roster())
    write_gz(OUT/(prefix+'task_rows.csv.gz'),outs);write_gz(OUT/(prefix+'endpoint_rows.csv.gz'),pair_out)
    summarize(outs,pair_out,prefix)
    coverage=np.sum(list(facets.values()),axis=(0,1))
    save(OUT/(prefix+'audit.json'),dict(n=len(rows),predictions=len(outs),old_identical_predictions_and_ranks=same,
        same_subject_pool_pairs=len(pair_rosters),distinct_frame_sources=len(facets),active_segment_statistics=coverage.tolist(),
        max_scorer_and_sensor_error=maxerr,prediction_seal=digest(pred_dir/'SEALED.json'),no_training=True,FINAL_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--symmetric',action='store_true')
    args=p.parse_args()
    run(args.archive,args.symmetric)
