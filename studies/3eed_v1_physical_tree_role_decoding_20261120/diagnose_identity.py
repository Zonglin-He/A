"""Post-seal identity interventions on the physical inventory, never used as inference or tuning."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from paths_tree import *
from score_tree import labels
from evaluate_witness import metadata,PANELS,COHORTS,historical_records
from panel_witness import inventory as physical_inventory
from endpoint_score import arrays_to_owners,write_gz
from score_roles import state,csv_file
from tables_scene import grouped
from physical_nodes import overlap_prior
from physical_evidence import Returns
from role_cut import shared,structure,medoid

def run(repo):
    seal=read(PRED/'SEALED.json');assert not seal['labels_used'] and seal['pins']==base_pins()
    annotation=labels();hist=historical_records(repo);panels={g:read(PANELS[g]) for g in COHORTS}
    cache={};rows=[];pairs=defaultdict(dict)
    names=('deployed','correct_reference_only','category_only','both')
    object_pins=read(OBJECTS/'SEALED.json')['files']
    for row,rpath,rh,lp,lh,fid,word,*_ in observations():
        uid=row['uid'];e=annotation[uid];g=e['group'];target=row['frame']['target']
        if g not in COHORTS and 'pair' not in e:continue
        truth,s,r,fr,rec,view,_=metadata(row,e,panels,hist,cache)
        if s is None or r is None:continue
        group='natural' if g in COHORTS else ('change_landmark' if g in ('name','depth') else g)
        file=PRED/name(row);assert digest(file)==seal['files'][name(row)] and digest(rpath)==rh
        with np.load(file) as z,np.load(rpath) as role:
            i=ARMS.index('shared_cut');oc=arrays_to_owners([z['raw'][i:i+1],z['ref'][i:i+1]],np.asarray(truth),None)['unique_iou']
            plain=(int(oc[0][0]),int(oc[1][0]));choices={x:plain for x in names};na=nr=0
            if z['applied'][i]:
                p=OBJECTS/(fid+'.npz');assert digest(p)==object_pins[fid]
                with np.load(p) as d:data=dict(d)
                # The evaluation inventory can be in a different order for natural descriptions. Map by geometry.
                cls,boxes,conflict=physical_inventory(dict(metapath=row['frame']['calibration'],platform=target,suffix=view))
                owners=arrays_to_owners([data['boxes']],np.asarray(truth),None)['unique_iou'][0]
                clsids=arrays_to_owners([data['boxes']],boxes,None)['unique_iou'][0]
                na=int(np.any((owners==s)&np.isfinite(data['view'][:,0])))
                nr=int(np.any((owners==r)&np.isfinite(data['view'][:,0])))
                pts=Returns(data['xyz']);us=overlap_prior(data,role,'t',pts);ur=overlap_prior(data,role,'r',pts)
                _,K=structure(data['parent'],data['stop'],data['size']);valid=K==0
                for arm in names[1:]:
                    q=data['p'];prior=ur
                    if arm in ('category_only','both'):
                        q=np.zeros_like(q)
                        for v,ident in enumerate(clsids):
                            if ident>=0 and not conflict and str(cls[ident]) in CLASSES[:-1]:
                                q[v,CLASSES.index(str(cls[ident]))]=1.
                    if arm in ('correct_reference_only','both'):prior=ur*(owners==r)
                    j,_=shared(data['parent'],valid,data['stop'],q,word,data['view'],int(role['side']),
                               int(role['depth']),bool(role['immediate']),'physical')
                    if arm in ('correct_reference_only','both'):j*=owners[None,:]==r
                    si,ri,_=medoid(j,K,us,prior,valid)
                    if si<0 or ri<0:choices[arm]=(-1,-1);continue
                    if arm in ('correct_reference_only','both'):assert owners[ri]==r
                    choices[arm]=(int(owners[si]),int(owners[ri]) if ri>=0 else -1)
            for arm,(so,ro) in choices.items():
                out=dict(panel=group,case=uid,source=row['source'],target=target,frame=fr,recording=rec,arm=arm,
                         applied=int(z['applied'][i]),subject_available=na,ref_available=nr,subject=state(so,s),
                         reference=state(ro,r),both=int(so==s and ro==r))
                rows.append(out)
                if 'pair' in e:pairs[group,e['pair'],row['source'],arm][e['half']]=(so==s,ro==r,bool(na and nr))
    tape=[]
    for (panel,pair,src,arm),v in pairs.items():
        x,y=v[0],v[1]
        tape.append(dict(panel=panel,pair=pair,source=src,arm=arm,two_subjects=int(x[0] and y[0]),
                         two_refs=int(x[1] and y[1]),four=int(all((x[0],y[0],x[1],y[1]))),cover=int(x[2] and y[2])))
    summary=[];paired=[]
    for (panel,arm),v in grouped(rows,['panel','arm']).items():
        summary.append(dict(panel=panel,arm=arm,n=len(v),applied=sum(x['applied'] for x in v),
              subject=sum(x['subject']=='T' for x in v),reference=sum(x['reference']=='T' for x in v),
              both=sum(x['both'] for x in v),unknown_ref=sum(x['reference']=='U' for x in v)))
    for (panel,arm),v in grouped(tape,['panel','arm']).items():
        paired.append(dict(panel=panel,arm=arm,n=len(v),**{k:sum(x[k] for x in v) for k in ('two_subjects','two_refs','four','cover')}))
    csv_file(OUT/'offline_identity.csv',summary);csv_file(OUT/'offline_pairs.csv',paired)
    write_gz(OUT/'offline_identity_rows.csv.gz',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);run(p.parse_args().archive)
