"""Direction-macro, recording-paired, counterfactual and geometry outcomes."""
import numpy as np
from paths_tree import OUT,ARMS
from score_roles import csv_file,record_intervals
from tables_scene import grouped,paired_counts

def tables(rows,pairs):
    samples={k[0]:v for k,v in grouped(rows,['panel']).items()}
    samples['ordinary_combined']=samples['ordinary_uniform']+samples['ordinary_relation']
    metrics=[];comparisons=[];ident=[];pairsum=[];coverage=[]
    for panel,sub in samples.items():
        for setting,field in [('measured','iou'),('completed','final_iou')]:
            cache={};record={};vals=[]
            for (s,t),v in grouped(sub,['source','target']).items():
                by=[sorted([y for y in v if y['arm']==arm],key=lambda y:y['case']) for arm in ARMS]
                assert all([y['case'] for y in yy]==[y['case'] for y in by[0]] for yy in by)
                mat=np.array([[y[field] for y in yy] for yy in by]).T
                scores=100*np.stack([(mat>.25).mean(0),(mat>.5).mean(0),mat.mean(0)])
                vals.append(scores);cache[s,t]=mat
                record[s,t]=dict(recording=np.array([y['recording'] for y in by[0]]))
                for j,arm in enumerate(ARMS):
                    metrics.append(dict(panel=panel,geometry=setting,direction=s+'_to_'+t,arm=arm,n=len(mat),
                                        acc25=scores[0,j],acc50=scores[1,j],miou=scores[2,j]))
            allm=np.vstack(list(cache.values()));macro=np.mean(vals,axis=0)
            k=ARMS.index('shared_cut')
            for j,arm in enumerate(ARMS):
                metrics.append(dict(panel=panel,geometry=setting,direction='macro',arm=arm,n=len(allm),
                                    acc25=macro[0,j],acc50=macro[1,j],miou=macro[2,j]))
                ci=record_intervals(record,cache,k,j)
                for direction,m in [('all',allm)]+[(s+'_to_'+t,m) for (s,t),m in cache.items()]:
                    comparisons.append(dict(panel=panel,geometry=setting,direction=direction,control=arm,
                        n=len(m),better=int((m[:,k]>m[:,j]+1e-10).sum()),worse=int((m[:,k]<m[:,j]-1e-10).sum()),
                        **paired_counts(m,k,j),macro_delta50_low=ci[0] if direction=='all' else '',
                        macro_delta50_high=ci[1] if direction=='all' else ''))
        for (arm,),v in grouped(sub,['arm']).items():
            known=[y for y in v if y['known']]
            for setting,pre in [('measured',''),('completed','final_')]:
                for rule in ('iou','seed'):
                    s=rule+'_'+pre+'subject';r=rule+'_'+pre+'reference'
                    ident.append(dict(panel=panel,arm=arm,rule=rule,geometry=setting,n=len(v),known=len(known),
                        applied=sum(y['active'] for y in v),subject=sum(y[s]=='T' for y in known),
                        reference=sum(y[r]=='T' for y in known),both=sum(y[s]==y[r]=='T' for y in known),
                        unknown_subject=sum(y[s]=='U' for y in known),unknown_reference=sum(y[r]=='U' for y in known)))
            coverage.append(dict(panel=panel,arm=arm,n=len(v),applied=sum(y['active'] for y in v),
                edited=sum(y['geometry_changed'] for y in v),
                geometry_rescues=sum(y['final_iou']>.5>=y['iou'] for y in v),
                geometry_harms=sum(y['iou']>.5>=y['final_iou'] for y in v),
                **{k:sum(y[k] for y in v) for k in ('conditional_geo_rescue','conditional_geo_harm',
                    'a256_geometry_rescue','all_native_geometry_rescue','all_native_with_reference')}))
    for (panel,rule,arm),v in grouped(pairs,['panel','rule','arm']).items():
        keys=[k for k in v[0] if k.endswith(('_four','_two_subjects','_two_refs')) or k=='resolved_distinct_subjects']
        pairsum.append(dict(panel=panel,rule=rule,arm=arm,pairs=len(v),frames=len({r['frame'] for r in v}),
                            recordings=len({r['recording'] for r in v}),**{k:sum(y[k] for y in v) for k in keys}))
    for f,t in [('metrics',metrics),('paired',comparisons),('identities',ident),('geometry',coverage),('counterfactuals',pairsum)]:
        csv_file(OUT/(f+'.csv'),t)
