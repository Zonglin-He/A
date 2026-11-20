"""Direction-macro task results, all paired threshold transitions and physical identity controls."""
from collections import defaultdict
import numpy as np
from shared_paths import ARMS, OUT
from score_roles import measure, record_intervals, csv_file


def grouped(rows,keys):
    d=defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return d


def groups(rows,kind):
    d={k[0]:v for k,v in grouped(rows,[kind]).items()}
    if kind=='group':
        d['ordinary_combined']=d['ordinary_uniform']+d['ordinary_relation']
        d['old_switch']=d['name']+d['depth']
    if 'new_side' in d:
        for exposure in ('new','seen'):
            d['new_side_'+exposure+'_recording']=[r for r in d['new_side'] if r['exposure']==exposure]
    return d


def paired_counts(mat,i,j):
    v={}
    for t in (25,50):
        a,b=mat[:,i]>t/100,mat[:,j]>t/100
        v['rescue'+str(t)]=int((a & ~b).sum());v['harm'+str(t)]=int((b & ~a).sum())
    return v


def tabulate(rows,pairs,contrast):
    metrics=[];paired=[];geo=[];ident=[];interval=[]
    key_controls={'joint_raw':('v1','previous_noun','independent','separate_worlds','separate_scales','wrong_reference','image_center'),
        'joint_shape':('v1','joint_raw','fused_boxes','ground_only','marginal_shape','v1_ground'),
        'v1_ground':('v1',)}
    for group,sub in groups(rows,'group').items():
        mats={};recs={};scores=[];all_mat=[]
        def describe(direction,aa,mat,values):
            n=len(aa)//len(ARMS)
            for i,arm in enumerate(ARMS):
                rr=[r for r in aa if r['arm']==arm]
                assert len(rr)==n
                header=dict(group=group,direction=direction,arm=arm,n=n)
                metrics.append(dict(**header,frames=len({r['frame'] for r in rr}),recordings=len({r['recording'] for r in rr}),
                   applied=sum(r['applied'] for r in rr),changed_raw=sum(r['changed_raw'] for r in rr),
                   acc25=float(values[i,0]),acc50=float(values[i,1]),miou=float(values[i,2])))
                for dep in key_controls.get(arm,('v1',)):
                    j=ARMS.index(dep)
                    paired.append(dict(**header,control=dep,delta25=values[i,0]-values[j,0],
                        delta50=values[i,1]-values[j,1],delta_miou=values[i,2]-values[j,2],**paired_counts(mat,i,j)))
                for tag in ('iou','seed'):
                    both=[r for r in rr if r['known']]
                    ident.append(dict(**header,rule=tag,known=len(both),subject=sum(r[tag+'_s']=='T' for r in both),
                        reference=sum(r[tag+'_r']=='T' for r in both),both=sum(r[tag+'_s']==r[tag+'_r']=='T' for r in both)))
                    hit=[r for r in rr if r[tag+'_raw']=='T']
                    changed=[r for r in hit if r['edited']]
                    geo.append(dict(**header,rule=tag,correct_raw=len(hit),edited=sum(r['edited'] for r in rr),
                        edited_correct=len(changed),rescue50=sum(r['iou']>.5>=r['raw_iou'] for r in changed),
                        harm50=sum(r['raw_iou']>.5>=r['iou'] for r in changed),
                        outside_a256=sum(r['beyond_a256'] and r[tag+'_final']=='T' for r in hit),
                        outside_every_pool=sum(r['beyond_all'] and r[tag+'_final']=='T' for r in hit)))
        for (src,target),aa in sorted(grouped(sub,['source','target']).items()):
            byarm={k[0]:sorted(v,key=lambda x:x['case']) for k,v in grouped(aa,['arm']).items()}
            assert len(byarm)==len(ARMS)
            assert all([r['case'] for r in x]==[r['case'] for r in byarm[ARMS[0]]] for x in byarm.values())
            mat=np.array([[r['iou'] for r in byarm[a]] for a in ARMS]).T
            vals=measure(mat);describe(src+'_to_'+target,aa,mat,vals)
            mats[src,target]=mat;recs[src,target]=dict(recording=np.array([r['recording'] for r in byarm[ARMS[0]]]))
            scores.append(vals);all_mat.append(mat)
        describe('macro',sub,np.vstack(all_mat),np.mean(scores,axis=0))
        for a,b in [('joint_raw','independent'),('joint_raw','wrong_reference'),('joint_raw','image_center'),
                     ('joint_raw','separate_worlds'),('joint_raw','separate_scales'),('joint_shape','joint_raw'),
                     ('joint_shape','ground_only'),('joint_shape','v1')]:
            lo,hi=record_intervals(recs,mats,ARMS.index(a),ARMS.index(b))
            interval.append(dict(group=group,arm=a,control=b,delta50_ci_low=lo,delta50_ci_high=hi))
    for n,data in [('metrics',metrics),('paired',paired),('identities',ident),('geometry',geo),('intervals',interval)]:
        csv_file(OUT/(n+'.csv'),data)
    for desc,table,keys in [('paired_identity',pairs,('both_subjects','both_refs','four','distinct')),
                            ('contrast',contrast,('available','both_correct','either_tie','both_reversed'))]:
        records=[]
        for group,part in groups(table,'panel').items():
            for (mode,arm),rr in grouped(part,['rule','arm']).items():
                records.append(dict(group=group,rule=mode,arm=arm,pairs=len(rr),frames=len({r['frame'] for r in rr}),
                    **{key:sum(r[key] for r in rr) for key in keys}))
        csv_file(OUT/(desc+'.csv'),records)
