"""Report full paired denominators, recording bootstrap, and object-equal noun eligibility."""
from collections import defaultdict
import numpy as np
from common_lex import OUT, ARMS
from score_roles import measure, csv_file, record_intervals


def groupby(rows, keys):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def tables(rows, pairs, noun, oracle):
    task=[];paired=[];identity=[];shapes=[];intervals=[]
    composites={'ordinary_combined':{'ordinary_uniform','ordinary_relation'},'controlled':{'name','depth'}}
    bygroup=groupby(rows,['group'])
    for name,pieces in composites.items():
        bygroup[(name,)]=[x for x in rows if x['group'] in pieces]
    for (g,),group in bygroup.items():
        mats={};recs={};per_dir=[];ordered=[]
        for (src,target),observed in sorted(groupby(group,['source','target']).items()):
            cases=groupby(observed,['case'])
            by_arm=[[next(x for x in rr if x['arm']==arm) for rr in cases.values()] for arm in ARMS]
            mat=np.asarray([[x['iou'] for x in col] for col in by_arm]).T
            mats[src,target]=mat
            recs[src,target]=dict(recording=np.array([r['recording'] for r in by_arm[0]]))
            ordered.append(by_arm)
            per_dir.append((src+'_to_'+target,by_arm,mat,measure(mat)))
        overall=np.mean([d[3] for d in per_dir],axis=0)
        combined=[[x for b in ordered for x in b[i]] for i in range(len(ARMS))]
        for direction,by,mat,values in per_dir+[('macro',combined,np.concatenate(list(mats.values())),overall)]:
            for i,arm in enumerate(ARMS):
                sub=by[i]
                base=dict(group=g,direction=direction,arm=arm,n=len(sub))
                task.append(dict(**base,frames=len({x['frame'] for x in sub}),recordings=len({x['recording'] for x in sub}),
                    read=sum(x['applied'] for x in sub),subject_box_changes=sum(x['changed_identity_box'] for x in sub),
                    acc25=values[i,0],acc50=values[i,1],miou=values[i,2]))
                for tag in ('seed','iou'):
                    known=[x for x in sub if x['known_both']]
                    identity.append(dict(**base,rule=tag,known=len(known),subject=sum(x[tag+'_s']=='T' for x in known),
                        ref=sum(x[tag+'_r']=='T' for x in known),both=sum(x[tag+'_s']==x[tag+'_r']=='T' for x in known)))
                    correct=[x for x in sub if x[tag+'_official']=='T']
                    shapes.append(dict(**base,rule=tag,correct_raw=len(correct),with_donor=sum(x['donors']>0 for x in sub),
                        edited=sum(x['geometry_edit'] for x in sub),edited_correct=sum(x['geometry_edit'] for x in correct),
                        rescue50=sum(x['iou']>.5>=x['raw_iou'] for x in correct),
                        harm50=sum(x['raw_iou']>.5>=x['iou'] for x in correct),
                        outside_a256=sum(x['beyond_original'] and x[tag+'_final']=='T' for x in correct),
                        outside_every_pool=sum(x['beyond_every_pool'] and x[tag+'_final']=='T' for x in correct)))
                for j,ctrl in enumerate(ARMS):
                    if ctrl!='v1' and arm not in ('joint_raw','joint_face'):continue
                    v=dict(**base,control=ctrl,delta25=values[i,0]-values[j,0],delta50=values[i,1]-values[j,1],
                           delta_miou=values[i,2]-values[j,2])
                    for cut in (25,50):
                        old,new=mat[:,j]>cut/100,mat[:,i]>cut/100
                        v['rescue'+str(cut)]=int(np.count_nonzero(new & ~old))
                        v['harm'+str(cut)]=int(np.count_nonzero(old & ~new))
                    paired.append(v)
        for a,b in [('joint_raw','independent'),('joint_raw','previous_ordered'),('joint_raw','single_order'),
                    ('joint_raw','no_noun'),('joint_raw','wrong_reference'),('joint_raw','wrong_placement'),
                    ('joint_face','joint_raw'),('joint_face','symmetric'),('joint_face','blind_donors'),
                    ('joint_face','wrong_donors'),('joint_face','v1')]:
            lo,hi=record_intervals(recs,mats,ARMS.index(a),ARMS.index(b))
            intervals.append(dict(group=g,arm=a,control=b,lower_delta50_pp=lo,upper_delta50_pp=hi))
    for filename,tab in [('metrics',task),('paired',paired),('identity',identity),('geometry',shapes),('intervals',intervals)]:
        csv_file(OUT/(filename+'.csv'),tab)

    results=[];summ=[];upp=[]
    for (mode,a),v in groupby(pairs,['rule','arm']).items():
        results.append(dict(rule=mode,arm=a,frames=len({x['pair'] for x in v}),pair_directions=len(v),
          **{k:sum(x[k] for x in v) for k in ('both_subjects','both_refs','four','resolved_switch')}))
    for (mode,side,a),v in groupby(noun,['rule','role','arm']).items():
        summ.append(dict(rule=mode,role=side,arm=a,n=len(v),**{k:sum(x[k] for x in v) for k in (
            'available','retained','correct_classes','retained_classes','wrong_classes','accepted_wrong','unknown_selected')}))
    for (mode,),v in groupby(oracle,['rule']).items():
        pair=groupby(v,['source','pair'])
        assert all(len(x)==2 for x in pair.values())
        upp.append(dict(rule=mode,n=len(v),both_available=sum(x['available'] for x in v),
                    both_selected=sum(x['both'] for x in v),pair_n=len(pair),pair_available=sum(all(x['available'] for x in p) for p in pair.values()),
                    both_subjects=sum(all(x['s'] for x in p) for p in pair.values()),four=sum(all(x['both'] for x in p) for p in pair.values())))
    for f,v in [('paired_identity',results),('noun_eligibility',summ),('exact_noun_oracle',upp)]:
        csv_file(OUT/(f+'.csv'),v)
