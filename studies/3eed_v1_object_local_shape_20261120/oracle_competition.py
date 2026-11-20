"""Post-seal explanation of the gap between a two-name discrimination test and scene-wide decisions."""
from collections import defaultdict
import numpy as np
import pandas as pd
from local_paths import *
from oracle_local import DIAG, rows
from observe_witness import interpret
from role_cut import separate,medoid

def report():
    assert not read(WORK/'SEALED.json')['labels_used']
    original=pd.read_csv(OUT/'oracle_grounding_rows.csv.gz')
    labels=read(previous.EXTRA/'labels.json')
    texts={(v['source'],v['uid']):v['frame']['text'] for v in previous.noun_inputs()}
    pairs=defaultdict(list)
    for r in rows():
        e=labels[r['uid']];pairs[r['source'],e['pair']].append((r,e))
    out=[]
    for (src,pair),examples in pairs.items():
        assert len(examples)==2
        for r,e in examples:
            fid=r['frame'];fp=DIAG/src/(fid+'.npz')
            assert digest(fp)==read(DIAG/src/'SEALED.json')['files'][fid]
            with np.load(fp) as z:fields={a:z[a] for a in ('combined','shape_difference','misbound_crop','local_direct','scrambled_direct')}
            with np.load(DIAG/(fid+'_regions.npz')) as z:view=z['view']
            _,_,side,depth,near=interpret(texts[src,r['uid']],{})
            words=r['words'];n=len(view);eye=np.eye(n);valid=~eye.astype(bool);one=np.ones(n)
            two_subjects=[c['subject'] for _,c in examples]
            two_refs=[c['reference'] for _,c in examples]
            assert len(set(two_subjects+two_refs))==4
            for a,p in fields.items():
                joint,_=separate(valid,one,p,words,view,side,depth,near)
                for candidate_set in ('scene','two_references','two_subjects','two_each','known_reference'):
                    j=joint.copy()
                    if candidate_set in ('two_references','two_each'):j[:,~np.isin(np.arange(n),two_refs)]=0
                    if candidate_set in ('two_subjects','two_each'):j[~np.isin(np.arange(n),two_subjects)]=0
                    if candidate_set=='known_reference':j[:,np.arange(n)!=e['reference']]=0
                    ss,rr,_=medoid(j,eye,one,one,valid);sub=int(ss==e['subject']);ref=int(rr==e['reference'])
                    out.append(dict(source=src,pair=pair,frame=e['frame'],recording=e['recording'],half=e['half'],arm=a,
                        candidate_set=candidate_set,subject=sub,reference=ref,both=sub*ref))
    f=pd.DataFrame(out)
    k=['source','pair','half','arm']
    ref=original[original.arm.isin(f.arm.unique())]
    result=f[f.candidate_set=='scene'].merge(ref,on=k,validate='one_to_one')
    for c in ('subject','reference','both'):np.testing.assert_equal(result[c+'_x'].to_numpy(),result[c+'_y'].to_numpy())
    f.to_csv(OUT/'oracle_competition_rows.csv.gz',index=False)
    reduced=f.groupby(['source','pair','frame','recording','arm','candidate_set'])[['subject','reference','both']].min().reset_index()
    g=reduced.groupby(['arm','candidate_set']);out=g[['subject','reference','both']].sum();out['pairs']=g.size();out.to_csv(OUT/'oracle_competition.csv')

    forced=pd.read_csv(OUT/'oracle_assignment_rows.csv.gz');comparison=[]
    for task,df,label in [('four_endpoints',reduced[reduced.candidate_set=='scene'],'both'),('two_reference_names',forced,'correct')]:
        pivot=df.pivot(index=['frame','source','recording','pair'],columns='arm',values=label)
        for peer in ('combined','misbound_crop','local_direct','scrambled_direct'):
            d=pivot['shape_difference']-pivot[peer]
            by=d.groupby(level='recording').agg(['sum','size'])
            rng=np.random.default_rng(4907);rb=rng.integers(len(by),size=(5000,len(by)))
            boot=100*by['sum'].to_numpy()[rb].sum(1)/by['size'].to_numpy()[rb].sum(1)
            comparison.append(dict(task=task,control=peer,pairs=len(d),physical_frames=d.index.get_level_values('frame').nunique(),
                recordings=len(by),rescue=int((d==1).sum()),harm=int((d==-1).sum()),delta_pp=100*d.mean(),
                ci_lo=float(np.quantile(boot,.025)),ci_hi=float(np.quantile(boot,.975))))
    pd.DataFrame(comparison).to_csv(OUT/'oracle_paired.csv',index=False)
if __name__=='__main__':report()
