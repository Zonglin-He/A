"""Independent loop-based endpoint, support, aggregation and ratio audit."""
import collections,json,sys,hashlib
from pathlib import Path
import numpy as np

def read(f):return json.loads(Path(f).read_text())

def audit(folder):
    folder=Path(folder).resolve();root=folder.parents[2]
    config=read(folder/'CONFIG.json');cnt=collections.Counter();maxerr=0.;eps=1e-12
    def equal(a,b):
        nonlocal maxerr
        if isinstance(a,dict):
            assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
            for k in a:equal(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        elif a is None or isinstance(a,(str,bool)):assert a==b,(a,b)
        else:
            err=abs(float(a)-float(b));assert err<1e-10,(a,b,err)
            maxerr=max(maxerr,err);cnt['scalar_checks']+=1
    for name,h in config['inputs'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    effective=read(folder/'EFFECTIVE_PINS.json')
    assert set(effective)==set(config['pins'])
    for name,h in effective.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            prior=read(root/'results/tastvg_temporal_boundary_support/2026-10-03'/split/ds/'ROWS.json')
            expert=[r for r in prior if r['expert_scheduled']];rows=read(folder/split/ds/'ROWS.json')
            assert len(rows)==len(expert)==(96 if split=='search' else 48)
            for old,r in zip(expert,rows):
                assert r['input_row_sha256']==hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                for k in ['dataset','split','source_id','condition','order','arrival','A_state_pre_sha256','A_state_post_sha256']:equal(r[k],old[k])
                equal(r['intervals'],old['intervals_normalized']);equal(r['candidate_v'],old['candidate_v']);equal(r['candidate_t'],old['candidate_t'])
                a=old['choices']['A8'];xy=r['intervals'];anchor=xy[a];width=anchor[1]-anchor[0];v=r['candidate_v'];seconds=r['duration_seconds']
                equal(r['anchor_index'],a);equal(r['anchor'],anchor);equal(seconds,old['duration_seconds'])
                best=[i for i,value in enumerate(v) if max(v)-value<=eps]
                distances=[abs(x-anchor[0])+abs(y-anchor[1]) for x,y in xy]
                equal(r['oracle_indices'],best)
                near=sorted(best,key=lambda i:(distances[i],i))[0];far=sorted(best,key=lambda i:(-distances[i],i))[0]
                equal(r['nearest_index'],near);equal(r['farthest_index'],far)
                kinds=[]
                for i in best:
                    ds1=xy[i][0]-anchor[0];de=xy[i][1]-anchor[1]
                    kind=('same' if abs(ds1)<=eps and abs(de)<=eps else 'start_only' if abs(de)<=eps else
                          'end_only' if abs(ds1)<=eps else 'expansion' if ds1<0<de else 'contraction' if de<0<ds1 else 'co_directional')
                    kinds.append(kind)
                equal(r['oracle_tie'],float(len(best)>1));equal(r['tie_geometric_ambiguity'],float(len(set(kinds))>1))
                for label,i in [('nearest',near),('farthest',far)]:
                    ds1=xy[i][0]-anchor[0];de=xy[i][1]-anchor[1];distance=abs(ds1)+abs(de)
                    centre=(xy[i][0]+xy[i][1]-anchor[0]-anchor[1])/2
                    length=(xy[i][1]-xy[i][0])-width
                    equal(distance,max(2*abs(centre),abs(length)))
                    intersection=max(0.,min(anchor[1],xy[i][1])-max(anchor[0],xy[i][0]))
                    union=max(anchor[1],xy[i][1])-min(anchor[0],xy[i][0])
                    dom='centre' if 2*abs(centre)>abs(length)+eps else 'extent' if abs(length)>2*abs(centre)+eps else 'equal'
                    z=dict(interval=xy[i],signed_start=ds1,signed_end=de,start_abs=abs(ds1),end_abs=abs(de),distance=distance,
                        start_seconds=abs(ds1)*seconds,end_seconds=abs(de)*seconds,distance_seconds=distance*seconds,
                        distance_anchor=distance/width,signed_centre=centre,signed_length=length,
                        centre_seconds=abs(centre)*seconds,length_seconds=abs(length)*seconds,
                        doubled_centre_anchor=2*abs(centre)/width,length_anchor=abs(length)/width,
                        overlap_tIoU=intersection/union,disjoint=intersection<=eps,category=kinds[best.index(i)],dominance=dom)
                    equal(r[label],z)
                o8=max(v[:8]);o32=max(v);gain=o32-o8
                for f,value in dict(A8_v=v[a],A8_t=r['candidate_t'][a],O8_v=o8,O32_v=o32,
                        capacity_gain=gain,selection_gap8=o8-v[a],selection_gap32=o32-v[a],gain_positive=float(gain>eps),
                        nearest_distance=distances[near],farthest_distance=distances[far],
                        nearest_distance_anchor=distances[near]/width,farthest_distance_anchor=distances[far]/width,
                        nearest_distance_seconds=distances[near]*seconds,farthest_distance_seconds=distances[far]*seconds,
                        start_abs=r['nearest']['start_abs'],end_abs=r['nearest']['end_abs'],start_seconds=r['nearest']['start_seconds'],
                        end_seconds=r['nearest']['end_seconds'],centre_seconds=r['nearest']['centre_seconds'],
                        length_seconds=r['nearest']['length_seconds'],overlap_tIoU=r['nearest']['overlap_tIoU'],
                        disjoint=float(r['nearest']['disjoint'])).items():equal(r[f],value)
                for kind in ['same','start_only','end_only','expansion','contraction','co_directional']:
                    equal(r['gain_'+kind],gain*float(r['nearest']['category']==kind))
                for kind in ['centre','extent','equal']:equal(r['gain_dominance_'+kind],gain*float(r['nearest']['dominance']==kind))
                for side,coord in [('start',0),('end',1)]:
                    kept=[j for j in range(32) if abs(xy[j][coord]-anchor[coord])<=eps]
                    equal(r['fixed_'+side+'_gain'],max(v[j] for j in kept)-v[a]);equal(r['fixed_'+side+'_count'],len(kept))
                for norm,radii in config['radii'].items():
                    last8=last32=-1.
                    for radius in radii:
                        tag='inf' if radius is None else str(radius).replace('.','p');p=norm+'_'+tag
                        bound=float('inf') if radius is None else radius*(width if norm=='anchor' else 1.)
                        keep=[j for j in range(32) if distances[j]<=bound+eps];k8=[j for j in keep if j<8]
                        l8=max(v[j] for j in k8);l32=max(v[j] for j in keep)
                        assert l8>=last8-eps and l32>=last32-eps and l32>=l8-eps;last8=l8;last32=l32
                        for f,value in dict(L8_gain=l8-v[a],L32_gain=l32-v[a],local_extra=l32-l8,
                            accessible_extra=max(0.,min(gain,l32-o8)),near_gain_mass=gain*float(distances[near]<=bound+eps),
                            far_gain_mass=gain*float(distances[far]<=bound+eps),count8=len(k8),count32=len(keep)).items():equal(r[p+'_'+f],value)
                        if radius is None:equal(l8,o8);equal(l32,o32)
                        cnt['local_support_checks']+=1
                assert gain>=-eps;cnt['expert_rows']+=1
            for group,summary in read(folder/split/ds/'SUMMARY.json').items():
                rr=[r for r in rows if (r['condition']=='clean')==(group=='clean')]
                fields=sorted(summary['metrics']);sources=sorted({r['source_id'] for r in rr});mat=[];orders=collections.defaultdict(list)
                for s in sources:
                    vectors=[]
                    for o in sorted({r['order'] for r in rr}):
                        cells=[r for r in rr if r['source_id']==s and r['order']==o]
                        if not cells:continue
                        conditions=sorted({r['condition'] for r in cells})
                        vec=np.mean([np.mean([[r[f] for f in fields] for r in cells if r['condition']==c],0) for c in conditions],0)
                        vectors.append(vec);orders[o].append(vec)
                    mat.append(np.mean(vectors,0))
                mat=np.array(mat);rng=np.random.default_rng(20261003)
                sampled=rng.integers(0,len(mat),(10000,len(mat)))
                boot=mat[sampled].mean(1);ci=np.percentile(boot,[2.5,97.5],0);at={f:i for i,f in enumerate(fields)}
                equal(summary['sources'],len(sources));equal(summary['cells'],len(rr))
                equal(summary['positive_gain_cells'],sum(r['capacity_gain']>eps for r in rr))
                equal(summary['positive_gain_sources'],sum(x>eps for x in mat[:,at['capacity_gain']]))
                for j,f in enumerate(fields):
                    x=mat[:,j];loo=(x.sum()-x)/(len(x)-1) if len(x)>1 else x
                    expected=dict(mean=float(x.mean()),ci95=ci[:,j].tolist(),
                        source_values={str(s):float(z) for s,z in zip(sources,x)},
                        order_values={o:float(np.mean(vectors,0)[j]) for o,vectors in orders.items()},
                        leave_one_out_range=[float(loo.min()),float(loo.max())],cell_mean=float(np.mean([r[f] for r in rr])))
                    equal(summary['metrics'][f],expected)
                bd=boot[:,at['capacity_gain']];valid=bd>eps;denom=mat[:,at['capacity_gain']].mean()
                for f,z in summary['capacity_gain_shares'].items():
                    equal(z['mean'],float(mat[:,at[f]].mean()/denom) if denom>eps else None)
                    vals=boot[valid,at[f]]/bd[valid]
                    equal(z['ci95'],np.percentile(vals,[2.5,97.5]).tolist() if len(vals) else None)
                    equal(z['bootstrap_zero_denominator_draws'],int((~valid).sum()));equal(z['bootstrap_valid_draws'],int(valid.sum()))
                    equal(z['interval_condition'],'positive_bootstrap_total_capacity_gain')
                    if len(mat)>1:
                        d=(mat[:,at['capacity_gain']].sum()-mat[:,at['capacity_gain']])/(len(mat)-1)
                        n=(mat[:,at[f]].sum()-mat[:,at[f]])/(len(mat)-1);k=d>eps
                        equal(z['leave_one_out_range'],[float((n[k]/d[k]).min()),float((n[k]/d[k]).max())] if k.any() else None)
                cnt['summary_groups']+=1
    equal(read(folder/'COVERAGE.json'),dict(arrivals=1152,experts=288,nonexpert_without_candidates=864,corrupt_experts=240,clean_experts=48))
    assert cnt['expert_rows']==288 and cnt['summary_groups']==8
    return dict(status='pass',checks=dict(cnt),max_abs_numeric_error=maxerr,
        independent_endpoint_support_and_source_bootstrap=True,private_annotations_opened=0,
        GPU_used=False,pins_and_predecessor_inputs_verified=True)

if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
