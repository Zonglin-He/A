"""Score locked height-transfer outputs and audit the clipping constraints."""
import ast
from collections import Counter,defaultdict
import csv
import re

import numpy as np

from download import STUDY,read_json,sha256,write_json
from height_geometry import EPS
from height_measurements import TOKEN, metric
from height_transfer import COLORS, CONFIG, EVIDENCE_ARMS, RUN, SOURCE, index_rows, source_files
from score import jsonl


HEIGHT_ARMS = ('prior','alternate_prior','visible_raw','initial_raw','clip_raw',
               'visible_with_fallback','initial_with_fallback','point_only_with_fallback','clip_with_fallback')
PAIRS = (
    ('native','prior'),('native','clip_with_fallback'),('initial_with_fallback','clip_with_fallback'),
    ('point_only_with_fallback','clip_with_fallback'),('visible_with_fallback','clip_with_fallback'),
    ('prior','clip_with_fallback'),('native','prior_number_prompt'),('native','clipped_number_prompt'),
    ('prior_number_prompt','clipped_number_prompt'),('native','clipped_state_prompt'),('prior_number_prompt','clipped_state_prompt'),
    ('initial_state_prompt','clipped_state_prompt'),('clipped_number_prompt','clipped_state_prompt')
)


def validate_seals():
    seal=read_json(RUN/'prediction_seal.json')
    assert not seal['labels'] and seal['training_steps']==0
    for f,digest in seal['files'].items():
        assert sha256(RUN/f)==digest
    g=read_json(RUN/'geometry_seal.json')
    assert not g['label_access'] and g['inputs']==sha256(RUN/'input_seal.json')
    assert sha256(RUN/'geometry_stage.jsonl')==g['geometry']
    assert sha256(RUN/'pointmap_manifest.json')==g['pointmaps']
    for name,digest in g['code'].items():
        assert sha256(STUDY/name)==digest
    for key,digest in read_json(RUN/'pointmap_manifest.json').items():
        assert re.fullmatch('[0-9a-f]{64}',key) and sha256(RUN/'pointmaps'/f'{key}.npy')==digest
    i=read_json(RUN/'input_seal.json')
    assert not i['reference_answer_access']
    assert i['protocol']==sha256(CONFIG) and i['inputs']==sha256(RUN/'inputs.jsonl')
    assert i['sources']==sha256(SOURCE/'source.json')
    return seal


def references(roster):
    # This is the first stage that opens the serialized reference responses.
    import pyarrow.parquet as pq
    gold={}
    parity=Counter()
    for path in source_files():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['id','qa_info','conversations']):
            for r in batch.to_pylist():
                q=r['id']
                if q not in roster:
                    continue
                qa=ast.literal_eval(r['qa_info'])
                conversation=ast.literal_eval(r['conversations'])
                text=conversation[0]['value']
                for name,color in zip(qa['class'],COLORS):
                    text=text.replace('<mask>',f'the {name} (highlighted by a {color} box)',1)
                same=text==roster[q]['prompt']
                words=text.split()==roster[q]['prompt'].split()
                parity['character_exact']+=same
                parity['same_after_whitespace_normalization']+=words
                num=metric(conversation[1]['value'],bare_meters=False)
                assert num is not None, ('reference parse failure',q)
                gold[q]=(num,conversation[1]['value'])
    assert gold.keys()==roster.keys()
    return gold,dict(parity)


def place(value,native):
    return native if value is None else value


def measurements(row):
    g=row['geometry']
    native=metric(row['native'])
    present=dict(native=native,**{name:metric(value['response']) for name,value in row['evidence'].items()})
    if row['category']=='height_data':
        present.update(prior=row['prior_m'],alternate_prior=row['alternate_prior_m'],
                       visible_raw=g.get('visible_height_m'),initial_raw=g.get('initial_height_m'),clip_raw=g.get('height_m'),
                       visible_with_fallback=place(g.get('visible_height_m'),native),
                       initial_with_fallback=place(g.get('initial_height_m'),native),
                       point_only_with_fallback=place(g.get('no_ray_height_m'),native),
                       clip_with_fallback=place(g.get('height_m'),native))
    return present


def explicit_dimension(response, category, object_reference):
    """Find an explicitly attributed dimension in a previously unparseable reply."""
    dimension = 'height' if category=='height_data' else 'width'
    subject = object_reference.split(' (highlighted by',1)[0].removeprefix('the ')
    named = rf'{dimension}\s+(?:of|for)\s+(?:the\s+)?{re.escape(subject)}(?:\s+\(highlighted by a (?:red|blue|green) box\))?'
    generic = rf'(?:estimated complete|complete|full)\s+{dimension}'
    field = re.compile(rf'^\s*(?:the\s+)?(?:object\s+)?(?:{named}|{generic})\s*(?::|=|\bis\b)\s*(.*)$',re.I)
    lines = response.splitlines()
    measurements = []
    for i,line in enumerate(lines):
        match = field.match(line)
        if match:
            raw = match[1] or (lines[i+1] if i+1<len(lines) else '')
            value = metric(raw)
            if value is not None:
                measurements.append(value)
    if measurements and max(measurements)-min(measurements)<=max(.005,.02*measurements[0]):
        return measurements[0]
    return None


def response_diagnostic(roster, predictions):
    values = {}
    counts = Counter()
    for q,row in predictions.items():
        for arm in ('initial_state_prompt','clipped_state_prompt'):
            view = row['evidence'][arm]
            text = view['response']
            value = metric(text)
            if value is None and view['used']:
                units=list(TOKEN.finditer(text.casefold()))
                bucket = 'multiple_unit_numbers' if len(units)>1 else 'one_unit_number' if units else 'no_unit_number'
                counts[row['category'],arm,'unparseable_'+bucket] += 1
                value = explicit_dimension(text,row['category'],roster[q]['object'])
                counts[row['category'],arm,'explicit_dimension_salvaged'] += value is not None
            values[q,arm] = value
    return values, counts


def check_box(g,prior):
    if g['status'] not in ('clipped','unchanged'):
        assert g.get('height_m')==g.get('initial_height_m')
        return
    first=np.array(g['initial_bounds'])
    final=np.array(g['final_bounds'])
    assert np.array_equal(first[0],final[0]) and np.array_equal(first[:,0:2],final[:,0:2])
    assert final[1,2]<=first[1,2]+EPS and final[1,2]>=final[0,2]
    assert abs(g['height_m']-(final[1,2]-final[0,2]))<EPS
    assert g['height_m']>=g['visible_height_m']-EPS
    expected=min(g['initial_height_m'],max(prior,g['fit_selection_required_height_m'],
                                           g['selection_entry_required_height_m']))
    assert abs(expected-g['height_m'])<EPS
    assert g['selection']['incumbent_hits']>0
    assert g['selection']['hit_changes']==g['selection']['common_depth_changes']==0


def score(value,reference):
    error=abs(value/reference-1) if value is not None else None
    return dict(value_m=value,absolute_relative_error=error,success=error is not None and error<=.25+1e-12)


def basic(rows):
    parsed=[r['absolute_relative_error'] for r in rows if r['absolute_relative_error'] is not None]
    good=sum(r['success'] for r in rows)
    return dict(n=len(rows),parsed=len(parsed),success=good,success_pct=100*good/len(rows),
                mean_are=float(np.mean(parsed)) if parsed else None,median_are=float(np.median(parsed)) if parsed else None)


def paired(rows,left,right):
    l=[r['scores'][left] for r in rows]
    r=[r['scores'][right] for r in rows]
    delta=np.array([int(b['success'])-int(a['success']) for a,b in zip(l,r,strict=True)])
    clusters=defaultdict(list)
    for j,row in enumerate(rows):
        clusters[row['image']].append(j)
    weight=np.array([len(v) for v in clusters.values()])
    changes=np.array([delta[v].sum() for v in clusters.values()])
    ids=np.random.default_rng(20261122).integers(len(weight),size=(4000,len(weight)))
    ci=np.percentile(100*changes[ids].sum(axis=1)/weight[ids].sum(axis=1),[2.5,97.5])
    are=np.array([(a['absolute_relative_error'],b['absolute_relative_error']) for a,b in zip(l,r,strict=True)
        if a['absolute_relative_error'] is not None and b['absolute_relative_error'] is not None]).reshape(-1,2)
    return dict(left=left,right=right,n=len(rows),left_success=sum(x['success'] for x in l),
        right_success=sum(x['success'] for x in r),rescue=int((delta==1).sum()),harm=int((delta==-1).sum()),
        delta_pp=float(100*delta.mean()),image_clusters=len(weight),ci95_pp=ci.tolist(),common_parsed=len(are),
        common_are_before=float(are[:,0].mean()) if len(are) else None,
        common_are_after=float(are[:,1].mean()) if len(are) else None)


def ray_summary(rows,key):
    available=[r['geometry'][key] for r in rows if key in r['geometry']]
    sums={name:sum(v[name] for v in available) for name in ('total','incumbent_hits','hit_changes','common_depth_changes')}
    return dict(objects=len(available),**sums,max_entry_delta_m=max((v['max_common_entry_delta_m'] for v in available),default=0.0))


def audit():
    seal=validate_seals()
    roster=index_rows(RUN/'inputs.jsonl')
    predictions=index_rows(RUN/'predictions.jsonl')
    original=index_rows(RUN/'geometry_stage.jsonl')
    assert roster.keys()==predictions.keys()==original.keys()
    assert len(roster)==seal['rows']
    for q,row in predictions.items():
        assert {k:v for k,v in row.items() if k!='evidence'}==original[q]
        assert metric(row['prior'],bare_meters=True)==row['prior_m']
        assert metric(row['alternate_prior'],bare_meters=True)==row['alternate_prior_m']
        check_box(row['geometry'],row['prior_m'])
        evidence=list(row['evidence'].values())
        for j,a in enumerate(evidence):
            if not a['used']:
                assert a['response']==row['native']
            for b in evidence[:j]:
                if a['prompt_sha256']==b['prompt_sha256']:
                    assert a['response']==b['response']
    explicit, formatting_counts = response_diagnostic(roster,predictions)
    truth,prompt_match=references(roster)
    scored=[]
    for q,row in predictions.items():
        ref,gold_text=truth[q]
        scored.append(dict(**row,ground_truth_m=ref,ground_truth_text=gold_text,
                           scores={arm:score(v,ref) for arm,v in measurements(row).items()}))
    jsonl(RUN/'scored_rows.jsonl',scored)
    results={}
    paired_rows=[]
    for kind in ('height_data','width_data'):
        rows=[r for r in scored if r['category']==kind]
        fit=[r for r in rows if 'initial_height_m' in r['geometry']]
        ray=[r for r in rows if r['geometry']['status'] in ('clipped','unchanged')]
        clips=[r for r in ray if r['geometry']['status']=='clipped']
        scope=dict(all=rows,initial_available=fit,ray_eligible=ray,clipped=clips)
        for domain in sorted({r['source'] for r in rows}):
            scope['dataset_'+domain]=[r for r in rows if r['source']==domain]
        summary={k:{arm:basic([v['scores'][arm] for v in subset]) for arm in subset[0]['scores']}
                 for k,subset in scope.items() if subset}
        for k,subset in scope.items():
            if subset and not k.startswith('dataset_'):
                paired_rows.extend(dict(category=kind,cohort=k,**paired(subset,a,b))
                    for a,b in PAIRS if a in subset[0]['scores'] and b in subset[0]['scores'])
        stability=[abs(v['prior_m']-v['alternate_prior_m'])/v['prior_m'] for v in rows
                   if v['prior_m'] is not None and v['alternate_prior_m'] is not None]
        point_slack=[g['geometry']['raw_top_slack_m'] for g in fit]
        initial_over_visible=[r['geometry']['initial_height_m']/r['geometry']['visible_height_m']
                              for r in fit if r['geometry']['visible_height_m']>1e-8]
        trigger={
            'total':len(rows),'initial_available':len(fit),'ray_eligible':len(ray),'clip_changes':len(clips),
            'nonzero_raw_top_slack':sum(v>1e-6 for v in point_slack),
            'raw_top_slack_median_m':float(np.median(point_slack)) if point_slack else None,
            'median_initial_over_visible':float(np.median(initial_over_visible)) if initial_over_visible else None,
            'clip_median_removed_height_m':float(np.median([p['geometry']['initial_height_m']-
                p['geometry']['height_m'] for p in clips])) if clips else None,
            'clip_equal_prior':sum(abs(r['geometry']['height_m']-r['prior_m'])<=1e-6 for r in clips),
            'clip_point_only_different':sum(abs(r['geometry']['height_m']-r['geometry']['no_ray_height_m'])>1e-6 for r in ray),
            'has_top_entry':sum(r['geometry']['selection_entry_on_top']>0 for r in ray),
            'prior_at_least_initial':sum(r['prior_m']>=r['geometry']['initial_height_m']-1e-6 for r in ray),
            'status':dict(Counter(r['geometry']['status'] for r in rows)),
        }
        source_detail = {}
        if kind=='height_data' and ray:
            geometry_arms=('prior','visible_raw','initial_raw','clip_raw')
            source_detail=dict(
                median_bottom_extension_m=float(np.median([r['geometry']['initial_height_m']-
                    r['geometry']['visible_height_m']-r['geometry']['raw_top_slack_m'] for r in ray])),
                median_value_over_reference={arm:float(np.median([r['scores'][arm]['value_m']/r['ground_truth_m']
                    for r in ray])) for arm in geometry_arms},
                overestimate_more_than_25_pct={arm:sum(r['scores'][arm]['value_m']>1.25*r['ground_truth_m']
                    for r in ray) for arm in geometry_arms})
        results[kind]=dict(metrics=summary,triggers=trigger,eligible_geometry_bias=source_detail,
            response_format_sensitivity={arm:dict(
                unparseable_used={label:count for (cat,a,label),count in formatting_counts.items() if cat==kind and a==arm},
                explicit_dimension_metrics=basic([score(explicit[r['id'],arm],r['ground_truth_m']) for r in rows]))
                for arm in ('initial_state_prompt','clipped_state_prompt')},
            stability=dict(primary_count=sum(r['prior_m'] is not None for r in rows),
                      alternate_count=sum(r['alternate_prior_m'] is not None for r in rows),common=len(stability),
                      agree_within_25_pct=sum(x<=.25 for x in stability),
                      median_relative_delta=float(np.median(stability)) if stability else None),
            selection=ray_summary(ray,'selection'),audit=ray_summary(ray,'audit'),
            selection_on_changed=ray_summary(clips,'selection'),audit_on_changed=ray_summary(clips,'audit'),
            point_only_selection=ray_summary(ray,'point_only_selection'),
            prompt_support={a:sum(r['evidence'][a]['used'] for r in rows) for a in EVIDENCE_ARMS},
            state_prompts_different=sum(r['evidence']['initial_state_prompt']['prompt_sha256']!=
                                        r['evidence']['clipped_state_prompt']['prompt_sha256'] for r in rows),
            state_answers_different=sum(r['evidence']['initial_state_prompt']['response']!=
                                        r['evidence']['clipped_state_prompt']['response'] for r in rows),
            number_copy_counts={
                'prior_number_prompt':sum(r['prior_m'] is not None and
                    r['scores']['prior_number_prompt']['value_m'] is not None and
                    abs(r['scores']['prior_number_prompt']['value_m']-r['prior_m'])<=.002 for r in rows),
                'clipped_state_prompt':sum(r['geometry'].get('height_m') is not None and
                    r['scores']['clipped_state_prompt']['value_m'] is not None and
                    abs(r['scores']['clipped_state_prompt']['value_m']-r['geometry']['height_m'])<=.002 for r in rows)})
    with (RUN/'paired.csv').open('w',newline='') as out:
        csv_writer=csv.DictWriter(out,fieldnames=list(paired_rows[0]),lineterminator='\n')
        csv_writer.writeheader()
        csv_writer.writerows(paired_rows)
    write_json(RUN/'summary.json',dict(scores=results,paired=paired_rows,official_question_parity=prompt_match,
        prediction_seal=sha256(RUN/'prediction_seal.json'),scored_rows_sha256=sha256(RUN/'scored_rows.jsonl'),
        model_runtime=read_json(RUN/'runtime.json'),scorer='±25% full-cohort. SI/imperial parsing without GPT-4.'))
    for kind,res in results.items():
        print(kind,res['triggers'],{k:(v['success'],v['parsed'],v['n']) for k,v in res['metrics']['all'].items()},flush=True)
    print('official prompt agreement',prompt_match,flush=True)


if __name__=='__main__':
    audit()
