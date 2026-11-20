"""Post-seal overlap between grounded identity and geometric improvement on the same record."""
import pandas as pd
import numpy as np
from vote_paths import *
from shared_paths import anonymous
from score_roles import inventory
from endpoint_score import arrays_to_owners

def main():
    sealed=read(WORK/'SEALED.json')
    assert not sealed['labels_used'] and sealed['protocol']==digest(SPEC)
    cat=catalog();tab=ordinary()
    index={(r['source'],anonymous((r['target'],r['ordinal']))):r for r in cat['rows'] if r['scope']=='ordinary'}
    j=ARMS.index('association')
    geo=pd.concat([pd.read_csv(OUT/(s+'_geometry_rows.csv.gz'),dtype={'case':str,'recording':str,'frame':str}) for s in SOURCES])
    hits=geo.query("panel=='full' and arm=='association' and outside_a256==1")
    rows=[]
    for (src,case),grp in hits.groupby(['source','case']):
        row=index[src,case];f=row['frame'];info=cat['frames'][f];v=tab[row['target']][row['ordinal']]
        p=WORK/'predictions'/(row_id(row)+'.npz')
        assert digest(p)==sealed['files'][row_id(row)] and digest(info['lex'])==info['lh']
        with np.load(info['lex']) as lex,np.load(p) as z:
            pts=np.unique(lex['sample'].astype(float),axis=0)
            gt,_=inventory(v['metapath'],row['target'],v['suffix'],v['object_index'])
            for (geom,control),cc in grp.groupby(['geometry','control']):
                owners=arrays_to_owners([z['raw'],z[control],z[geom]],gt,pts)
                assert all(vv[1][j]==0 and vv[2][j]==0 for vv in owners.values())
                for x in cc.to_dict('records'):
                    out=dict(source=src,target=row['target'],case=case,frame=x['frame'],recording=x['recording'],
                        geometry=geom,control=x['control'],outside_all_native=x['outside_all_native'],
                        raw_subject_verified=int(all(vv[0][j]==0 for vv in owners.values())))
                    for name in ['tree','direct_votes','direct_null','independent','misbound_reference','inverse_association']:
                        i=ARMS.index(name)
                        out[name+'_different_node']=int(z['selected'][j,0]!=z['selected'][i,0])
                        # Both ownership rules must name the same other annotated object.
                        endpoints=[int(vv[0][i]) for vv in owners.values()]
                        out[name+'_wrong_resolved']=int(len(set(endpoints))==1 and endpoints[0]>0)
                        out[name+'_same_subject']=int(all(t==0 for t in endpoints))
                    rows.append(out)
    data=pd.DataFrame(rows)
    data.to_csv(OUT/'joint_geometry_rows.csv.gz',index=False)
    counts=data.drop(columns=['source','target','frame','recording']).groupby(['geometry','control'])
    d=counts.sum(numeric_only=True);d['events']=counts.size();d.to_csv(OUT/'joint_geometry.csv')
    ranks=pd.concat([pd.read_csv(OUT/(s+'_ranking_rows.csv.gz'),dtype={'case':str}) for s in SOURCES])
    ranks=ranks.pivot(index=['panel','ownership','case','source'],columns='arm',values='margin').reset_index()
    delta=ranks.association-ranks.tree
    ranks['margin_increased']=delta>1e-10;ranks['margin_decreased']=delta< -1e-10
    ranks['overtook']=(ranks.association>0)&(ranks.tree<=0)
    ranks['fell_behind']=(ranks.association<=0)&(ranks.tree>0)
    g=ranks.groupby(['panel','ownership']);d=g[['margin_increased','margin_decreased','overtook','fell_behind']].sum()
    d['records']=g.size();d.to_csv(OUT/'reference_margin_changes.csv')
if __name__=='__main__':main()

