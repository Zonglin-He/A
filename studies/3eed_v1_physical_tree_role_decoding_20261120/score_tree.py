"""Post-seal paired scoring. All examined labels and inventories are confined to this stage."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from paths_tree import *
from predict_tree import CODE
from evaluate_witness import metadata, PANELS, COHORTS, historical_records
from score_roles import score_boxes, state, sensor_frame_box, csv_file, many_iou
from endpoint_score import arrays_to_owners,write_gz
from score_swap import best
from score_lex import lexical_has_pass

def labels():
    entries={}
    for p,h in [(parent.previous.OLD/'labels.json',read(parent.previous.LAST/'results/panel.json')['labels']),
                (parent.previous.NEW/'labels.json',read(parent.previous.OUT/'panel.json')['labels']),
                (parent.WORK/'labels.json',read(parent.OUT/'panel.json')['labels'])]:
        assert digest(p)==h
        e=read(p);assert e.keys().isdisjoint(entries)
        entries.update(e)
    return entries

def run(repo):
    seal=read(PRED/'SEALED.json')
    assert seal['pins']==base_pins() and seal['protocol']==digest(SPEC) and not seal['labels_used']
    assert seal['objects']==digest(OBJECTS/'SEALED.json') and all(seal['code'][f]==digest(HERE/f) for f in CODE)
    examples=observations()
    assert len(examples)==seal['n']==len(seal['files'])
    entries=labels();panels={g:read(PANELS[g]) for g in COHORTS}
    hist=historical_records(repo);cache={}; rows=[]; pairs=defaultdict(dict)
    opportunities=[];maxerr=0.;check_lex=0; member={}
    for index,(row,rp,rh,lp,lh,fid,words,*_) in enumerate(examples,1):
        key=name(row);assert digest(PRED/key)==seal['files'][key] and digest(rp)==rh
        uid=row['uid'];f=row['frame'];e=entries[uid];g=e['group']
        group='natural' if g in COHORTS else ('change_landmark' if g in ('name','depth') else g)
        source=row['source'];target=f['target']
        gt,s,r,fr,rec,view,sensor=metadata(row,e,panels,hist,cache)
        truth=np.asarray(gt);tid=e.get('subject',0)
        with np.load(PRED/key) as z,np.load(rp) as native:
            raw=z['raw'];ref=z['ref'];final=z['final'];ref_final=z['ref_final']; seeds=native['seeds']
            if g in COHORTS:
                case=panels[g][e['case']]['case'];rel=f'{COHORTS[g]}/{case:02d}_{source}.npz'
                p=ROOT/'crossplatform/round171/predictions'/rel
                assert digest(p)==hist[0][rel]
                with np.load(p) as old:seeds=old['seeds']
            np.testing.assert_equal(raw[0],native['raw'])
            x,inv=np.unique(np.vstack([raw,final]),axis=0,return_inverse=True)
            v,err=score_boxes(x,truth[tid])
            alternate,err2=score_boxes(np.array([sensor_frame_box(b,target,view) for b in x]),
                sensor if sensor is not None else sensor_frame_box(truth[tid],target,view))
            maxerr=max(maxerr,err,err2,float(np.max(abs(v-alternate))))
            assert maxerr<1e-8 and np.array_equal(v[:,None]>[.25,.5],alternate[:,None]>[.25,.5])
            before,after=np.split(v[inv],2);pool=best(native['a'],truth[tid])
            strict=float(max([pool]+[best(native[k],truth[tid]) for k in ('t_box','r_box')
                                     if k in native and len(native[k])]))
            cr=z['physical_raw'];ct=z['physical_subject'];cl=z['physical_ref']
            raw_best=float(many_iou(cr,truth[tid]).max(initial=0))
            full_best=float(many_iou(ct,truth[tid]).max(initial=0))
            owns=arrays_to_owners([raw,ref,final,ref_final,cr,ct,cl],truth,seeds)
            extra=None
            # Only an already selected, newly passing box can be a rescue beyond all native candidates.
            if ((after>.5)&(before<=.5)&(strict<=.5)&(raw_best<=.5)&z['applied']).any():
                assert digest(lp)==lh
                with np.load(lp) as lex:
                    extra=lexical_has_pass(lex['native_boxes'],lex['sample'],truth[tid])
                    check_lex+=1
            if 'pair' in e:
                import hashlib
                signature=hashlib.sha256(cr.tobytes()).digest()
                assert member.setdefault((source,e['pair']),signature)==signature
            for i,arm in enumerate(ARMS):
                out=dict(panel=group,case=uid,source=source,target=target,frame=fr,recording=rec,
                    arm=arm,known=int(s is not None and r is not None),active=int(z['applied'][i]),
                    nodes=int(z['nodes']),visible=len(cr),iou=before[i],final_iou=after[i],a256=pool,
                    phrase_pool=strict,raw_inventory_best=raw_best,completed_inventory_best=full_best,
                    geometry_changed=int(np.max(np.abs(final[i]-raw[i]))>1e-8))
                for rule,ab in [('unique_iou','iou'),('seed_ownership','seed')]:
                    sr,rr,sf,rf,base,csubject,cref=owns[rule]
                    out.update({ab+'_subject':state(int(sr[i]),tid),ab+'_reference':state(int(rr[i]),r),
                        ab+'_final_subject':state(int(sf[i]),tid),ab+'_final_reference':state(int(rf[i]),r),
                        ab+'_in_subject_raw':int(tid in base),ab+'_in_reference_raw':int(r is not None and r in base),
                        ab+'_in_subject_final':int(tid in csubject),ab+'_in_reference_final':int(r is not None and r in cref)})
                    if 'pair' in e:
                        k=(group,e['pair'],source,target,fr,rec,rule,arm)
                        pairs[k][e['half']]=dict(raw=(sr[i]==s,rr[i]==r),final=(sf[i]==s,rf[i]==r),
                            available=(s in base,r in base),completed=(s in csubject,r in cref),
                            raw_s=int(sr[i]),true_s=int(s),raw_r=int(rr[i]),true_r=int(r))
                agreed=all(out[tag+'_subject']==out[tag+'_final_subject']=='T' for tag in ('iou','seed'))
                out['conditional_geo_rescue']=int(agreed and after[i]>.5>=before[i])
                out['conditional_geo_harm']=int(agreed and before[i]>.5>=after[i])
                out['a256_geometry_rescue']=int(out['conditional_geo_rescue'] and pool<=.5)
                out['all_native_geometry_rescue']=int(out['conditional_geo_rescue'] and strict<=.5 and raw_best<=.5 and extra is False)
                out['all_native_with_reference']=int(out['all_native_geometry_rescue'] and all(out[tag+'_reference']=='T' for tag in ('iou','seed')))
                rows.append(out)
                if out['conditional_geo_rescue'] or out['conditional_geo_harm']:
                    opportunities.append(out.copy())
        if index%250==0:print('scored',index,flush=True)
    pairrows=[]
    for (panel,pair,src,tgt,fr,rec,rule,arm),halves in pairs.items():
        x,y=halves[0],halves[1]
        assert x['true_s']!=y['true_s'] and set(halves)=={0,1}
        a=dict(panel=panel,pair=pair,source=src,target=tgt,frame=fr,recording=rec,rule=rule,arm=arm)
        for stage in ('raw','final','available','completed'):
            a[stage+'_two_subjects']=int(x[stage][0] and y[stage][0])
            a[stage+'_two_refs']=int(x[stage][1] and y[stage][1])
            a[stage+'_four']=int(all(x[stage]) and all(y[stage]))
        a['resolved_distinct_subjects']=int(min(x['raw_s'],y['raw_s'])>=0 and x['raw_s']!=y['raw_s'])
        pairrows.append(a)
    write_gz(OUT/'task_rows.csv.gz',rows);write_gz(OUT/'paired_endpoint_rows.csv.gz',pairrows)
    if opportunities:write_gz(OUT/'geometry_rows.csv.gz',opportunities)
    save(OUT/'audit.json',dict(records=len(examples),predictions=len(rows),paired_rows=len(pairrows),
        matched_common_subject_inventories=len(member),max_independent_scorer_error=maxerr,
        lexical_ceiling_full_checks=check_lex,seal=digest(PRED/'SEALED.json'),protocol=digest(SPEC),
        directions=len({(x['source'],x['target']) for x in rows}),source_updates=0,target_updates=0,FINAL_access=False))
    from tables_tree import tables
    tables(rows,pairrows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);run(p.parse_args().archive)
