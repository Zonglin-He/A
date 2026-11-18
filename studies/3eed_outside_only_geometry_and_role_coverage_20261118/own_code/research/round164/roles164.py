"""Trace role coverage and rank on frozen boxes and test a matched distinct-pair readout."""
from collections import Counter
import sys

import numpy as np

from scope164 import HERE, SETTINGS, W162, WORK, atomic_json, read, review_rows, role_panel, sha, target_rows
from scope164 import verify_seal, write_seal
from geometry162 import effective_count, surface_cost, to_parameters
from mechanisms162 import annotation_inventory, read_evidence, seed_owners
from run162 import array_file
from score154 import many_iou
from score162 import put_table
from solver162 import Options, Relationships, build_roles, choose
from support162 import SceneSupport


def r88_candidates(ev):
    inside = (np.abs(ev.seeds[None]-ev.boxes[:,None,:3])<=ev.boxes[:,None,3:6]/2).all(-1)
    size = inside.sum(1)
    mass = ev.affinity @ inside.T
    native = mass / np.sqrt(np.maximum(size,1))
    for e, mask in enumerate(ev.role_mask):
        got = []
        if mask.any():
            for q in np.argsort(-native[e], kind='stable'):
                if size[q]<2:
                    continue
                if any((inside[q]&inside[j]).sum()/max(1,(inside[q]|inside[j]).sum())>.5 for j in got):
                    continue
                got.append(int(q))
                if len(got)==3:
                    break
        a = ev.hypotheses[e]
        np.testing.assert_array_equal(np.asarray(got),a[a>=0])
        if got:
            np.testing.assert_array_equal(ev.supports[e,a>=0],inside[got])
            raw = np.maximum(mass[e,got],1e-8)
            np.testing.assert_allclose(raw/raw.sum(),ev.route_weights[e,a>=0],atol=2e-7,rtol=2e-6)
    return inside, size>=2, native


def duplicate(inside, a, b):
    if a<0 or b<0:
        return False
    left, right = inside[a],inside[b]
    return a==b or (left&right).sum() > .5*(left|right).sum()


def route_pair(ev, inside, a, b):
    choices = [sorted([(float(w),int(q)) for q,w in zip(ev.hypotheses[r],ev.route_weights[r])
                       if q>=0 and w>0],key=lambda item:-item[0]) for r in (a,b)]
    first,second = [c[0][1] if c else -1 for c in choices]
    separate = [(np.log(s)+np.log(t),i,j) for s,i in choices[0] for t,j in choices[1] if not duplicate(inside,i,j)]
    if separate:
        _,i,j = max(separate,key=lambda item:item[0])
    elif choices[0] and (not choices[1] or choices[0][0][0]>=choices[1][0][0]):
        i,j = first,-1
    else:
        i,j = -1,second
    return (first,second),(i,j)


def lcib_catalog(ev, settings):
    scene = SceneSupport(ev)
    opts = Options(**settings['options'])
    dims, sigma = [np.asarray(settings[k]) for k in ('log_dimensions','log_scale')]
    roles = build_roles(scene, opts, dims, sigma)
    relations = Relationships(ev,roles,opts.relation_weight)
    selected = np.asarray(choose(scene,roles,relations),dtype=int)
    cap = max((len(r.hypotheses) for r in roles if r),default=0)
    geometry = np.zeros((len(roles),cap,7))
    exists = np.zeros(geometry.shape[:2],bool)
    rank = np.full(exists.shape,-np.inf)
    for i,r in enumerate(roles):
        if r is None:
            continue
        for j,region in enumerate(r.hypotheses):
            geometry[i,j] = region.box
            exists[i,j] = True
            weights = region.point_weights(r.state[:,j])
            if effective_count(weights)<6 or weights.sum()<.75:
                continue
            support = r.state[:scene.nseed,j]
            affinity = (support@ev.affinity[i])*scene.nseed/max(support.sum(),1e-9)
            cost = surface_cost(region.points,to_parameters(region.box),region.fitter.scale)
            rank[i,j] = .85*region.log_prior + .43*np.log(np.clip(affinity,.2,8.)) - 2.5*region.fitter.mean_cost(cost,weights)
    return geometry,exists,rank,selected


def predict():
    assert not (HERE/'ROLES_SEAL.json').exists()
    panel, files = role_panel()
    registry = target_rows()
    settings = read(SETTINGS)['sources']
    out = WORK/'roles'
    out.mkdir(parents=True,exist_ok=True)
    eligible, total = [],0
    for i in panel:
        path = W162/'DEV'/f'{i:05d}.npz'
        assert sha(path)==files[path.name]
        with np.load(path,allow_pickle=False) as f:
            ev = read_evidence(f)
        if not len(ev.edges):
            continue
        inside,valid,native = r88_candidates(ev)
        paired = np.asarray([route_pair(ev,inside,int(a),int(b)) for a,b,_ in ev.edges],dtype=int)
        geometry,exists,rank,selected = lcib_catalog(ev,settings[registry[i]['source']])
        array_file(out/path.name, edges=ev.edges, boxes=np.c_[ev.boxes,np.zeros(256)], seeds=ev.seeds,
                   inside=inside, valid=valid, native=native, choices=paired, routes=ev.hypotheses,
                   routed_weights=ev.route_weights, lcib_geometry=geometry, lcib_exists=exists,
                   lcib_rank=rank, lcib_selected=selected)
        eligible.append(i)
        total+=len(ev.edges)
    assert total==132
    write_seal('ROLES',out,eligible,edges=total,no_case_truth_in_prediction=True,exact_R88_route_replay=True,
               parent_dev_seal_sha256=sha(HERE.parent/'round162/DEV_SEAL.json'))
    print('Sealed fixed-bank predictions on',total,'edges in',len(eligible),'expressions',flush=True)


def box_owners(candidates, annotated):
    """Independent annotation-proxy assignment, with inherited R162 uniqueness margin."""
    matches = np.stack([many_iou(candidates,b) for b in annotated],axis=1)
    order = np.argsort(-matches,axis=1,kind='stable')
    row = np.arange(len(matches))
    best = matches[row,order[:,0]]
    runner_up = matches[row,order[:,1]] if len(annotated)>1 else np.zeros(len(matches))
    return np.where((best>=.20)&(best-runner_up>=.08),order[:,0],-1)


def seed_region_owners(inside, seed_id, n):
    groups = (seed_id[:,None]==np.arange(n)[None]).astype(int)
    mass = inside.astype(int)@groups
    top = np.argsort(-mass,axis=1,kind='stable')
    row = np.arange(len(mass))
    winner = mass[row,top[:,0]]
    second = mass[row,top[:,1]] if n>1 else np.zeros(len(mass))
    count = inside.sum(1)
    unknown = count-mass.sum(1)
    return np.where((count>0)&(winner>=.55*count)&(winner-np.maximum(unknown,second)>=.15*count),top[:,0],-1)


def rank_of(ids, scores, ownership, gold):
    ids = np.asarray(ids,dtype=int)
    order = ids[np.argsort(-scores[ids],kind='stable')]
    matches = np.flatnonzero(ownership[order]==gold)
    return int(matches[0]+1) if len(matches) else 0


def owns(owners, q):
    return int(owners[q]) if q>=0 else -1


def adjudicate(path, ref, judgment):
    i, edge_index = ref['panel_index'],ref['relation_index']
    row = target_rows()[i]['r']
    answer = dict(case=ref['case'],source=ref['source'],target=ref['target'],review=judgment['type'])
    with np.load(path,allow_pickle=False) as p:
        a,b,_ = p['edges'][edge_index]
        assert [a,b]==ref['role_indices']
        independent, distinct = p['choices'][edge_index]
        answer.update(independent_both_routed=bool((independent>=0).all()),distinct_both_routed=bool((distinct>=0).all()),
                      selected_pair_changed=not np.array_equal(independent,distinct),
                      independent_shared_support=bool(duplicate(p['inside'],*independent)))
        if judgment['type']!='verified_boxes':
            return answer
        _, gt = annotation_inventory(row['metapath'],ref['target'],int(row['suffix'] or 0),int(row['object_index']))
        ownership = box_owners(p['boxes'],gt)
        point_owner = seed_region_owners(p['inside'],seed_owners(p['seeds'],gt),len(gt))
        expected = [0 if r==0 else judgment['ref'] for r in (a,b)]
        valid = [[q for q in p['routes'][r] if q>=0 and ownership[q]==gold] for r,gold in zip((a,b),expected)]
        answer['gold_pair_distinct_feasible'] = any(not duplicate(p['inside'],x,y) for x in valid[0] for y in valid[1])
        own = [owns(ownership,q) for q in independent]
        answer['independent_shared_annotated_box'] = own[0]>=0 and own[0]==own[1]
        for j,(r,gold) in enumerate(zip((a,b),expected)):
            name = 'subject' if r==0 else 'reference'
            na, routes, eligible = p['native'][r],p['routes'][r],np.flatnonzero(p['valid'])
            wf = np.zeros(len(na))
            kept = routes>=0
            wf[routes[kept]] = p['routed_weights'][r,kept]
            native_rank = rank_of(eligible,na,ownership,gold)
            routed_rank = rank_of(routes[kept],wf,ownership,gold)
            lcib_boxes = p['lcib_geometry'][r]
            lcib_exists = p['lcib_exists'][r]
            lcib_owners = np.full(len(lcib_boxes),-1,dtype=int)
            lcib_owners[lcib_exists] = box_owners(lcib_boxes[lcib_exists],gt) if lcib_exists.any() else []
            answer.update({
                name+'_a256_has_box':bool((ownership==gold).any()),
                name+'_a256_eligible_native_rank':native_rank,
                name+'_r88_native_rank':rank_of(routes[kept],na,ownership,gold),
                name+'_r88_plain_rank':routed_rank,
                name+'_r88_seed_proxy_rank':rank_of(routes[kept],wf,point_owner,gold),
                name+'_lcib_hypothesis_has_box':bool((lcib_owners==gold).any()),
                name+'_lcib_eligible_rank':rank_of(np.flatnonzero(np.isfinite(p['lcib_rank'][r])),
                                                    p['lcib_rank'][r],lcib_owners,gold),
                name+'_independent_correct':owns(ownership,independent[j])==gold,
                name+'_distinct_correct':owns(ownership,distinct[j])==gold,
                name+'_independent_seed_correct':owns(point_owner,independent[j])==gold,
                name+'_distinct_seed_correct':owns(point_owner,distinct[j])==gold,
                name+'_lcib_initial_correct':owns(lcib_owners,p['lcib_selected'][r])==gold,
            })
    for method in ('independent','distinct','independent_seed','distinct_seed','lcib_initial'):
        answer[method+'_dual_correct'] = all(answer[role+'_'+method+'_correct'] for role in ('subject','reference'))
    return answer


def score():
    out = HERE/'results'
    files = verify_seal('ROLES',WORK/'roles')
    panel = []
    manifest = target_rows()
    for filename in files:
        i=int(filename.split('.')[0])
        scope = manifest[i]
        with np.load(WORK/'roles'/filename,allow_pickle=False) as f:
            inside = f['inside']
            for a,b in f['choices']:
                panel.append(dict(edge=len(panel)+1,source=scope['source'],target=scope['target'],
                                  route_both_known=bool((a>=0).all()),distinct_both_known=bool((b>=0).all()),
                                  selection_changed=not np.array_equal(a,b),independent_shared_support=bool(duplicate(inside,*a))))
    assert len(panel)==132
    put_table(out/'role_panel_availability.csv',panel)
    reviews = []
    for r,d in review_rows():
        filename = f"{r['panel_index']:05d}.npz"
        assert filename in files
        reviews.append(adjudicate(WORK/'roles'/filename,r,d))
    full = [r for r in reviews if r['review']=='verified_boxes']
    assert len(full)==14 and len(reviews)==33
    fields = list(dict.fromkeys(k for r in reviews for k in r))
    put_table(out/'role_review.csv',[{k:r.get(k,'') for k in fields} for r in reviews])
    categories = ('no_A256_candidate','no_R88_eligible_candidate','dropped_by_R88','routed_but_ranked_below_top1','r88_rank1')
    stages = {}
    for role in ('subject','reference'):
        count = Counter()
        for r in full:
            geometry,pre,post = [r[role+'_'+k] for k in ('a256_has_box','a256_eligible_native_rank','r88_plain_rank')]
            step = 0 if not geometry else 1 if not pre else 2 if not post else 3 if post>1 else 4
            count[categories[step]]+=1
        stages[role] = {**{k:count[k] for k in categories},
            'dropped_native_top10':sum(0<r[role+'_a256_eligible_native_rank']<=10 and not r[role+'_r88_plain_rank'] for r in full),
            'lcib_full_coverage':sum(r[role+'_lcib_hypothesis_has_box'] for r in full),
            'lcib_eligible_coverage':sum(r[role+'_lcib_eligible_rank']>0 for r in full)}
    metrics = dict(all_panel_edges=len(panel),selected_review_edges=len(reviews),complete_labels=len(full),
        availability_changes=sum(r['selection_changed'] for r in panel),
        role_breakdown=stages,pair_coverage={k:sum(all(r[role+'_'+status] for role in ('subject','reference')) for r in full)
            for k,status in (('A256','a256_has_box'),('R88_eligible','a256_eligible_native_rank'),('R88','r88_plain_rank'),
                             ('R88_seed_proxy','r88_seed_proxy_rank'),('LCIB_initial','lcib_eligible_rank'))},
        dual_correct={k:sum(r[k+'_dual_correct'] for r in full)
            for k in ('independent','distinct','independent_seed','distinct_seed','lcib_initial')},
        selection_changes_complete=sum(r['selected_pair_changed'] for r in full),
        gold_pair_distinct_feasible=sum(r['gold_pair_distinct_feasible'] for r in full),
        independent_shared_annotated_box=sum(r['independent_shared_annotated_box'] for r in full),
        distinct_gains=sum(r['distinct_dual_correct'] and not r['independent_dual_correct'] for r in full),
        distinct_losses=sum(not r['distinct_dual_correct'] and r['independent_dual_correct'] for r in full),
        prediction_seal_sha256=sha(HERE/'ROLES_SEAL.json'), FINAL_access=False)
    atomic_json(out/'ROLE_SUMMARY.json',metrics)
    print(metrics,flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['score']:
        score()
    else:
        predict()
