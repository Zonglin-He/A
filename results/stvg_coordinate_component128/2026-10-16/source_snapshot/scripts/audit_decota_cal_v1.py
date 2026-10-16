"""Independent score, fold, risk selection, calibration and reconstruction audit."""
import collections
import itertools
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from scripts.decota_matrix_common_v1 import read,load,sha,write
from scripts.run_decota_cal_v1 import OUT,PRIOR,plan,WEIGHTS,CONFIGS


def metrics(boxes,gt,ids,indices):
    a=np.asarray(boxes,dtype=np.float64);b=np.asarray(gt['boxes'],dtype=np.float64)
    xy_a=np.c_[a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2]
    xy_b=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]
    overlap=np.maximum(0,np.minimum(xy_a[:,2:],xy_b[:,2:])-np.maximum(xy_a[:,:2],xy_b[:,:2]))
    area=overlap[:,0]*overlap[:,1]
    iou=area/np.maximum(a[:,2]*a[:,3]+b[:,2]*b[:,3]-area,1e-12)
    valid=np.asarray(gt['valid'],bool);ids=np.asarray(ids);g,h=gt['interval']
    l,r=ids[indices[0]],ids[indices[1]]+1
    included=(ids>=l)&(ids<r);union=(ids>=min(l,g))&(ids<max(r,h))
    inter=max(0,min(r,h)-max(l,g))
    return dict(vIoU_corrected=float(iou[included&valid].sum()/max(union.sum(),1)),
                sIoU=float(iou[valid].mean()),tIoU=float(inter/(max(r,h)-min(l,g))),
                temporal_recall=float(inter/(h-g)),temporal_precision=float(inter/(r-l)))


def run():
    from scripts.run_stvg_fullscale_v1 import plan as source_plan,label_payload
    p=plan();labels=label_payload(source_plan());done=read(OUT/'complete.json')
    assert done['rows_sha256']==sha(OUT/'rows.json') and done['summary_sha256']==sha(OUT/'summary.json')
    rows=read(OUT/'rows.json');summary=read(OUT/'summary.json');selections=read(OUT/'spatial_selection.json')
    lookup={r['key']:r for rr in p['cohorts'].values() for r in rr}
    counts=collections.Counter();maximum=collections.defaultdict(float)
    def close(a,b,kind,tol=1e-10):
        err=abs(a-b);maximum[kind]=max(maximum[kind],err);assert err<tol,(kind,a,b);counts[kind]+=1
    # Independently verify all source partitions, NLL fits and risk-utility choice.
    for c,rs in p['cohorts'].items():
        signal={}
        for r in rs:
            x=load(r['signal_path']);signal[r['key']]=x['variants']['full']['signals']
        for f in (OUT/'temporal'/c).glob('selection_*.json'):
            s=read(f);assert set(s['train_sources']).isdisjoint(s['eval_sources']);counts['outer_source_partition']+=1
            fits=list(s['calibrators'].items())+[(k,v) for inner in s['inner_calibrators'].values() for k,v in inner.items()]
            for kind,cal in fits:
                xx=[];yy=[];ww=[]
                for key in cal['train_keys']:
                    row=lookup[key];ids=np.asarray(row['input']['frame_ids'],float)
                    edges=np.r_[row['input']['start_frame'],(ids[:-1]+ids[1:])/2,row['input']['end_frame']]
                    weights=np.diff(edges);weights/=weights.sum()*cal['sources']
                    g,h=labels[key]['interval'];y=((ids>=g)&(ids<h)).astype(float)
                    x=logit(np.clip(np.asarray(signal[key][kind],float),1e-6,1-1e-6))
                    xx.extend(x);yy.extend(y);ww.extend(weights)
                x,y,w=map(np.asarray,(xx,yy,ww));z=cal['alpha']*x+cal['beta']
                close(float(np.sum(w*(np.logaddexp(0,z)-y*z))),cal['NLL_after'],'calibration_NLL')
                grad=np.array([np.sum(w*(expit(z)-y)*x),np.sum(w*(expit(z)-y))])
                # At a bound KKT uses the feasible-direction projected gradient.
                if cal['alpha']<=1.001e-4:grad[0]=min(grad[0],0)
                if cal['alpha']>=99.999:grad[0]=max(grad[0],0)
                if cal['beta']<=-29.999:grad[1]=min(grad[1],0)
                if cal['beta']>=29.999:grad[1]=max(grad[1],0)
                assert np.max(abs(grad))<1e-5,(kind,cal['train_keys'],grad)
                counts['calibration_KKT']+=1
            utilities=[]
            for i in range(len(CONFIGS)):
                tt=[t for t in s['trials'] if t['config_index']==i]
                for t in tt:
                    cal=s['inner_calibrators'][str(t['inner_fold'])][CONFIGS[i]['signal']]
                    assert t['source'] not in cal['train_sources'];counts['inner_source_partition']+=1
                d=np.array([t['delta'] for t in tt]);u=np.mean(np.where(d>=0,d,2*d))
                close(float(u),s['utilities'][i],'temporal_risk_utility');utilities.append(u)
            assert int(np.argmax(utilities))==s['selected_config_index'];counts['temporal_selection']+=1
    # Sealed predictions and provenance; no label-generated output is written.
    for r in rows:
        key=r['key'];c,idx=key.split(':');x=load(r['predictions_path']);assert sha(r['predictions_path'])==r['predictions_sha256']
        sp=load(x['spatial_source_path']);assert sha(x['spatial_source_path'])==x['spatial_sha256']
        temp=load(OUT/'temporal'/c/(idx+'.pt'));ids=sp['frame_ids'];src=lookup[key]
        assert not x['GT_online'] and not temp['GT_online'] and temp['no_op_exact']
        assert src['source'] not in temp['C']['calibration']['train_sources']
        z=temp['C'];counts['online_separation_and_noop']+=1
        if z['curve']:
            assert z['best_step']==min(range(len(z['curve'])),key=lambda j:z['curve'][j]['loss'])
            from vg_tta.decota_cal_v1 import state_hash
            assert state_hash(z['state'])==z['state_sha256'];counts['actual_best_state']+=1
            pos=set(z['audit']['positive']);neg=set(z['audit']['negative'])
            for ij in [z['indices'],z['no_parameter_indices']]:
                selected=set(range(ij[0],ij[1]+1));assert pos<=selected and not neg&selected
                counts['compatible_decode']+=1
        elif z['audit']['reason']=='no_positive':
            assert z['indices']==temp['native_indices'];counts['native_abstain']+=1
        for name,pred in x['predictions'].items():
            m=metrics(pred['boxes'],labels[key],ids,pred['indices'])
            for field,value in m.items():close(value,r['metrics'][name][field],'independent_metrics')
        for name,original in r['chosen'].items():
            for field in ['vIoU_corrected','sIoU','tIoU']:
                close(r['metrics'][name][field],r['metrics'][original][field],'matrix_mapping')
        for arm in ['B','C','F']:
            assert len(sp['positions'][arm])<=8
            lo,hi=sp['spans'][arm];assert all(lo<=i<=hi for i in sp['positions'][arm]);counts['eight_in_span']+=1
        base=np.asarray(temp['frozen_boxes'],float)
        for name,audit in x['audits'].items():
            if 'provenance' not in audit:continue
            bb=np.asarray(x['predictions'][name]['boxes']);ps=audit['pseudo']
            expected=base.copy();origin=['native']*len(ids)
            if len(ps)>=2:
                positions=[v['position'] for v in ps];times=np.asarray(ids)[positions];values=np.asarray([v['box'] for v in ps])
                use=np.arange(positions[0],positions[-1]+1)
                for col in range(4):expected[use,col]=np.interp(np.asarray(ids)[use],times,values[:,col])
                for t in use:origin[t]='interpolation'
            for v in ps:expected[v['position']]=v['box'];origin[v['position']]='observation'
            assert origin==audit['provenance'];counts['provenance']+=1
            close(float(np.max(abs(expected-bb))),0.,'absolute_interpolation',2e-6)
            if audit.get('reference_constrained'):
                assert audit['reference_node'] in audit['path'];counts['immutable_reference']+=1
    for c in p['cohorts']:
        for fold in [-1,0,1,2,3]:
            ss=selections[c+':'+str(fold)];rr=[r for r in rows if r['cohort']==c and r['source'] in ss['train_sources']]
            utilities=[]
            for i,w in enumerate(WEIGHTS):
                d=np.array([r['metrics'][f'B_reference_{w:g}']['vIoU_corrected']-r['metrics']['B']['vIoU_corrected'] for r in rr])
                u=float(np.where(d>=0,d,2*d).mean());close(u,ss['utility'][i],'spatial_risk_utility');utilities.append(u)
            assert WEIGHTS[int(np.argmax(utilities))]==ss['weight'];counts['spatial_selection']+=1
    for group,g in summary['groups'].items():
        c,split=group.split('/');rr=[r for r in rows if r['cohort']==c and r['split']==split]
        for name,m in g['methods'].items():
            for field,val in m['metrics'].items():
                by=collections.defaultdict(list)
                for r in rr:by[r['source']].append(r['metrics'][name][field])
                close(float(np.mean([np.mean(v) for v in by.values()])),val['mean'],'aggregation')
    write(OUT/'independent_audit.json',dict(measurement='valid',created=time.time(),checks=dict(counts),
        maximum_errors=dict(maximum),scope='Independent metrics, calibration KKT, folds, selection, constraints, interpolation, provenance; bootstrap not independently reimplemented.',
        source_checkpoint_inference_not_rerun=True,production_unchanged=True))
    print('AUDIT_VALID',dict(counts),flush=True)


if __name__=='__main__':run()
