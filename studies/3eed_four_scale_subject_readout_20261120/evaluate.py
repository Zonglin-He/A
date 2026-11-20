"""Pair sealed boxes on every ordinary row and score both roles on controlled noun swaps."""
import csv
import gzip
from collections import Counter, defaultdict

import numpy as np

from replay import BASE, CFG, OUT, PRIVATE, catalog, digest, pins, prior, read, row_id, save
from endpoint_score import arrays_to_owners, write_gz
from scope183 import ordinary
from score_components import recording_interval, summarize
from score_roles import csv_file, score_boxes, sensor_frame_box, sensor_truth
from shared_paths import anonymous

ARMS = ('previous_shared','subject_only_shared','independent')


def evaluate():
    seal = read(PRIVATE/'SEAL.json')
    assert not seal['labels_used'] and seal['code']==pins() and seal['decisions']==digest(PRIVATE/'decisions.npz')
    past = BASE/'results/compression_rows.csv.gz'
    assert digest(past)==CFG['base_paired_rows_sha256']
    labels_path = prior.EXTRA/'labels.json'
    assert digest(labels_path)==read(prior.old.OUT/'noun_panel.json')['labels']
    labels = read(labels_path)
    table = ordinary()
    effects = {}
    identities = defaultdict(list)
    noun_rows = []
    work = defaultdict(Counter)
    max_iou_error = 0.
    with np.load(PRIVATE/'decisions.npz') as sealed:
        keys, boxes, refs, details = [sealed[k] for k in ('keys','subjects','refs','details')]
    index = {k:i for i,k in enumerate(keys)}
    for row in catalog()['rows']:
        scope = row['scope']
        if scope not in ('ordinary','noun'):
            continue
        k = index.pop(row_id(row))
        s,r = boxes[k],refs[k]
        n,c,ordered,si,ri,alt_s,alt_r = map(int,details[k])
        cs = int(not np.array_equal(s[0],s[1],equal_nan=True))
        cr = int(not np.array_equal(r[0],r[1],equal_nan=True))
        for meter in (work[scope], work[scope+('_ordered' if ordered else '_simple')]):
            meter.update(rows=1, eager_boxes=n*c, deployed_boxes=int(si>=0), shared_emissions=int(si>=0),
                         shared_reference_argmax_skipped=int(si>=0), independent_emissions=int(alt_s>=0),
                         subject_box_changes=cs, reference_box_changes=cr,
                         subject_node_changes=int(si!=alt_s),reference_node_changes=int(ri!=alt_r))
        if scope=='ordinary':
            tgt,ix = row['target'],row['ordinal']
            record = table[tgt][ix]
            truth = sensor_truth(tgt,record['metapath'],record['caption'],record['suffix'])
            predicted = np.stack([sensor_frame_box(b,tgt,record['suffix']) for b in s])
            ious,_ = score_boxes(predicted,truth)
            effects[row['source'],anonymous((tgt,ix))] = (*ious,cs,cr,int(si>=0),ordered)
        else:
            e = labels[row['uid']]
            with np.load(row['prior']) as old:
                owners = arrays_to_owners([s,r],np.array(e['truth']),old['seeds'])
            for rule,(sub,ref) in owners.items():
                sub_hit,ref_hit = sub==e['subject'],ref==e['reference']
                identities[rule,row['source'],e['pair']].append(np.stack([sub_hit,ref_hit]))
                noun_rows.append(dict(case=row_id(row),source=row['source'],pair=anonymous(('noun_pair',e['pair'])),
                         rule=rule,subject_box_changed=cs,reference_box_changed=cr,
                         shared_subject_hit=int(sub_hit[0]),independent_subject_hit=int(sub_hit[1]),
                         shared_reference_hit=int(ref_hit[0]),independent_reference_hit=int(ref_hit[1])))
    assert not index
    paired, directions, metrics = [],defaultdict(list),[]
    with gzip.open(past,'rt') as f:
        for record in csv.DictReader(f):
            before = float(record[CFG['base_arm']])
            loc = (record['source'],record['case'])
            active = int(record['active'])
            assert bool(active)==(loc in effects)
            shared,other,cs,cr,emit,order = effects.pop(loc,(before,before,0,0,0,0))
            max_iou_error = max(max_iou_error,abs(before-shared))
            paired.append({**{k:record[k] for k in ('source','target','recording','case','active')},
                           'shared_emission':emit,'ordered':order,**dict(zip(ARMS,[before,shared,other])),
                           'independent_subject_box_changed':cs,'independent_reference_box_changed':cr})
            directions[record['source'],record['target']].append((record['recording'],[before,shared,other],cs,cr))
    assert not effects and len(paired)==CFG['scopes']['ordinary'] and max_iou_error<1e-10
    def emit(direction,n,m,h,changes,ci=None):
        for j,arm in enumerate(ARMS):
            metrics.append(dict(direction=direction,n=n,arm=arm,acc25=m[0,j],acc50=m[1,j],miou=m[2,j],
                        delta25_pp=m[0,j]-m[0,0],delta50_pp=m[1,j]-m[1,0],delta_miou_pp=m[2,j]-m[2,0],
                        rescue25=h[0,j],harm25=h[1,j],rescue50=h[2,j],harm50=h[3,j],
                        iou_improved=h[4,j],iou_worsened=h[5,j],subject_boxes_changed=changes[0] if j==2 else 0,
                        reference_boxes_changed=changes[1] if j==2 else 0,
                        ci50_low=ci[0,j] if ci is not None else '',ci50_high=ci[1,j] if ci is not None else ''))
    panels = {}
    stats = []
    for direction,rows in sorted(directions.items()):
        rec,arr,cs,cr = zip(*rows)
        arr = np.asarray(arr)
        m,h = summarize(arr,np.zeros_like(arr,dtype=int))
        counts = np.array([sum(cs),sum(cr)])
        emit('_to_'.join(direction),len(arr),m,h,counts)
        stats.append((m,h,counts))
        panels[direction] = (arr,np.array(rec))
    m,h,c = zip(*stats)
    emit('macro',len(paired),np.mean(m,axis=0),np.sum(h,axis=0),np.sum(c,axis=0),recording_interval(panels))
    ident = {}
    for rule in ('unique_iou','seed_ownership'):
        blocks = [v for (name,src,pair),v in identities.items() if name==rule]
        assert len(blocks)==64 and all(len(v)==2 for v in blocks)
        values = np.array(blocks)  # source-pair, sentence, endpoint, arm
        both_sentences = values.all(1)
        ident[rule] = {arm:dict(subject_rows=int(values[:,:,0,i].sum()),reference_rows=int(values[:,:,1,i].sum()),
                          both_roles_same_sentence=int(values[:,:,:,i].all(2).sum()),
                          both_subjects=int(both_sentences[:,0,i].sum()),both_references=int(both_sentences[:,1,i].sum()),
                          all_four=int(both_sentences[:,:,i].all(1).sum())) for i,arm in enumerate(ARMS[1:])}
    OUT.mkdir(exist_ok=True)
    write_gz(OUT/'ordinary_rows.csv.gz',paired)
    write_gz(OUT/'noun_rows.csv.gz',noun_rows)
    csv_file(OUT/'metrics.csv',metrics)
    save(OUT/'verification.json',dict(seal_sha256=digest(PRIVATE/'SEAL.json'),historical_seal=CFG['compact_prediction_seal'],
         decision_rows=len(keys),frame_source_pairs=seal['frame_sources'],both_archived_endpoints_exact=2*len(keys),
         cached_scopes=work,noun_rows=128,noun_source_pairs=64,noun_identity=ident,ordinary_rows=len(paired),
         ordinary_fallback_not_rerun=len(paired)-work['ordinary']['rows'],max_baseline_iou_error=max_iou_error,
         coordinate_equality='seven values, exact including missing historical reference',FINAL_access=False))


if __name__=='__main__':
    evaluate()
