"""Six-direction task pairs and two-caption physical-identity trials, including all losses."""
from collections import defaultdict
import numpy as np
from paths_witness import OUT
from predict_witness import ARMS
from endpoint_score import write_gz
from score_roles import csv_file,measure,record_intervals

def pair_table(rows,labels,endpoints):
    pairs={}
    for row in rows:
        lab=labels[row['case']]
        if 'pair' in lab:
            pairs[lab['pair'],row['source']]=(lab['group'],row['target'],lab['recording'])
    detailed=[]
    for (pair,src),(g,target,rec) in pairs.items():
        for arm in ARMS:
            for method in ('iou','seed'):
                x,y=(endpoints[pair,src,h,arm,method] for h in (0,1))
                def hits(u,truth):
                    return [int(u[i]==truth[3+i]) for i in (0,1)]
                observed=[hits(x,x),hits(y,y)];crossed=[hits(x,y),hits(y,x)]
                def fields(a):
                    return (int(a[0][0] and a[1][0]),int(a[0][1] and a[1][1]),int(all(a[0]+a[1])))
                s,r,b=fields(observed);cs,cr,cb=fields(crossed)
                detailed.append(dict(pair=pair,source=src,target=target,group=g,recording=rec,arm=arm,rule=method,
                    both_subjects=s,both_references=r,all_four=b,opposite_subjects=cs,opposite_refs=cr,opposite_all=cb,
                    resolved_subject_switch=int(x[0]>=0 and y[0]>=0 and x[0]!=y[0]),
                    both_final_subjects=int(x[2]==x[3] and y[2]==y[3]),both_correct_single_total=sum(sum(z) for z in observed)))
    write_gz(OUT/'pair_rows.csv.gz',detailed)
    totals=[]
    for g in ('name','depth','both_families'):
        for direction in ('drone','quad','all_sources'):
            for rule in ('iou','seed'):
                by={a:[r for r in detailed if (r['group']==g or g=='both_families') and (r['source']==direction or direction=='all_sources') and r['rule']==rule and r['arm']==a] for a in ARMS}
                if not by['v1']:continue
                for a,r in by.items():
                    base=dict(group=g,source=direction,rule=rule,arm=a,pair_directions=len(r),physical_frames=len({x['pair'] for x in r}),
                         recordings=len({x['recording'] for x in r}))
                    for k in ('both_subjects','both_references','all_four','opposite_subjects','opposite_refs','opposite_all',
                              'both_final_subjects','resolved_subject_switch'):
                        base[k]=sum(x[k] for x in r)
                    for control in ('v1','independent','side','single_cut','misbound'):
                        b=by[control]; assert [x['pair'] for x in b]==[x['pair'] for x in r]
                        for key in ('both_subjects','all_four'):
                            base[key+'_rescue_vs_'+control]=sum(x[key]>y[key] for x,y in zip(r,b))
                            base[key+'_harm_vs_'+control]=sum(x[key]<y[key] for x,y in zip(r,b))
                    totals.append(base)
    csv_file(OUT/'pair_identity.csv',totals)

def tables(rows,labels,endpoints):
    pair_table(rows,labels,endpoints)
    metrics,pairs,geometry,identity,intervals=[],[],[],[],[]
    for g in sorted({r['group'] for r in rows}):
        buckets=defaultdict(lambda:defaultdict(list))
        for r in rows:
            if r['group']==g:buckets[r['source'],r['target']][r['arm']].append(r)
        mats={};recs={};segments=[]
        for (src,target),by in buckets.items():
            assert all([r['case'] for r in by[a]]==[r['case'] for r in by['v1']] for a in ARMS)
            mat=np.array([[x['iou'] for x in by[a]] for a in ARMS]).T
            mats[src,target]=mat;recs[src,target]=dict(recording=np.array([x['recording'] for x in by['v1']]))
            segments.append((src+'_to_'+target,by,mat,measure(mat)))
        stacked={a:[r for item in buckets.values() for r in item[a]] for a in ARMS}
        segments.append(('macro',stacked,np.concatenate(list(mats.values())),np.mean([measure(v) for v in mats.values()],axis=0)))
        for direction,by,mat,scores in segments:
            for k,a in enumerate(ARMS):
                current=by[a]
                base=dict(group=g,direction=direction,arm=a,n=len(current))
                metrics.append(dict(**base,frames=len({r['frame'] for r in current}),recordings=len({r['recording'] for r in current}),
                              parsed=sum(r['parsed'] for r in current),applied=sum(r['applied'] for r in current),
                              acc25=scores[k,0],acc50=scores[k,1],miou=scores[k,2]))
                for rule in ('iou','seed'):
                    known=[x for x in current if x['known_ends']]
                    identity.append(dict(**base,rule=rule,known=len(known),covered=sum(x[rule+'_covered'] for x in known),
                        subject=sum(x[rule+'_subject']=='T' for x in known),reference=sum(x[rule+'_ref']=='T' for x in known),
                        both=sum(x[rule+'_subject']==x[rule+'_ref']=='T' for x in known)))
                    same=[x for x in current if x[rule+'_official']=='T']
                    geometry.append(dict(**base,rule=rule,correct_raw=len(same),edited=sum(x['edit'] for x in current),
                       affected_correct=sum(x['edit'] for x in same),rescue50=sum(x['iou']>.5>=x['raw_iou'] for x in same),
                       harm50=sum(x['raw_iou']>.5>=x['iou'] for x in same),
                       outside_a256=sum(x['iou']>.5>=x['base_pool'] and x[rule+'_official_final']=='T' for x in same),
                       outside_all=sum(x['iou']>.5>=x['all_pool'] and x[rule+'_official_final']=='T' for x in same)))
                for ctrl in ARMS:
                    if ctrl!='v1' and a not in ('witness_raw','witness_chord'):continue
                    j=ARMS.index(ctrl);rs=dict(**base,control=ctrl,
                       delta25=scores[k,0]-scores[j,0],delta50=scores[k,1]-scores[j,1],delta_miou=scores[k,2]-scores[j,2])
                    for val in (25,50):
                        post,pre=mat[:,k]>val/100,mat[:,j]>val/100
                        rs['rescue'+str(val)]=int(sum(post & ~pre));rs['harm'+str(val)]=int(sum(pre & ~post))
                    pairs.append(rs)
        for a,c in (('witness_chord','v1'),('witness_raw','independent'),('witness_raw','side'),
                    ('witness_raw','single_cut'),('witness_raw','misbound'),('witness_chord','witness_raw'),
                    ('witness_chord','marginal_chord'),('witness_chord','simple_mean')):
            lo,hi=record_intervals(recs,mats,ARMS.index(a),ARMS.index(c))
            intervals.append(dict(group=g,arm=a,control=c,lo_delta50_pp=lo,hi_delta50_pp=hi))
    for f,x in (('metrics',metrics),('paired',pairs),('geometry',geometry),('identity',identity),('intervals',intervals)):
        csv_file(OUT/(f+'.csv'),x)
