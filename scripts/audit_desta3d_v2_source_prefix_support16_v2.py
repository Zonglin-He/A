"""CPU-only, all fixed16 saved source supports. No model or video is loaded."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.ptd_8b_teacher_feasibility_v1 import append_targets
from vg_tta.desta3d_v2_source import split_source_loss_masks

BASE = ROOT / 'artifacts/desta3d_v2/tta_v2'
RUN = BASE / 'source_fp32_head_control_v1'
OUT = BASE / 'source_prefix_support16_v2'
PTD = ROOT / 'external/ParallelTubeDecoding/src'
TOKENIZER = ROOT / 'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B'
ARMS = ('no_update', 'supervised', 'supervised_fp32')
TOL = 2e-6
RESERVE = 8 * 2**30
CAP = 20_000_000
SPAN = re.compile(r'<\|time_start\|><t(\d+)><t(\d+)><\|time_end\|>')
ROW = re.compile(r'<t(\d+)><\|box_start\|>(<\d+><\d+><\d+><\d+>)<\|box_end\|>')
REF = re.compile(r'^<\|object_ref_start\|>.*?<\|object_ref_end\|>', re.S)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def support(text, nframes, require_boxes):
    ref = REF.match(text)
    spans = SPAN.findall(text)
    assert ref and len(spans) == 1, 'semantic/time grammar'
    interval = list(map(int, spans[0]))
    assert 1 <= interval[0] <= interval[1] <= nframes, 'time bounds'
    rows = ROW.findall(text)
    positions = [int(x[0]) for x in rows]
    coords = [[int(x) for x in re.findall(r'\d+', row[1])] for row in rows]
    if require_boxes:
        assert positions == list(range(interval[0], interval[1]+1)), 'box support gap'
        assert len(rows) == text.count('<|box_start|>'), 'unparsed row'
    return {'reference': ref.group(), 'time_1based': interval,
            'box_positions_1based': positions, 'coordinate_token_values': coords}


def closed_counts(reference_tokens, boxes):
    assert reference_tokens >= 2 and boxes > 0
    return {'event_ntp': reference_tokens+4,
            'event_mtp': 6*math.ceil(reference_tokens/6)+6,
            'spatial_ntp': 7*boxes, 'spatial_mtp': 6*boxes,
            'event_endpoint_targets': 4, 'spatial_coordinate_targets': 8*boxes,
            'spatial_time_anchor_targets': boxes, 'spatial_box_delimiter_targets': 4*boxes}


def weighted(info):
    assert info['ntp_count']+info['mtp_count'] == info['tokens']
    terms = {k: info[k+'_loss']*info[k+'_count']/info['tokens'] for k in ('ntp', 'mtp')}
    value = math.fsum(terms.values())
    assert abs(value-info['ce']) <= TOL, (value, info['ce'])
    return {'contributions': terms, 'CE_reconstructed': value,
            'absolute_error': abs(value-info['ce'])}


def controls():
    assert closed_counts(14, 3) == {'event_ntp':18,'event_mtp':24,'spatial_ntp':21,
        'spatial_mtp':18,'event_endpoint_targets':4,'spatial_coordinate_targets':24,
        'spatial_time_anchor_targets':3,'spatial_box_delimiter_targets':12}
    assert closed_counts(7, 1)['event_mtp'] == 18
    assert closed_counts(12, 5)['event_mtp'] == 18
    good = '<|object_ref_start|>a<|object_ref_end|>\n<|time_start|><t2><t3><|time_end|>\n<t2><|box_start|><0><1><2><3><|box_end|>\n<t3><|box_start|><0><0><0><0><|box_end|>'
    assert support(good, 3, True)['box_positions_1based'] == [2, 3]
    # Native zero-area boxes remain legitimate support records.
    for bad in [good.replace('<t3><|box_start|>', '<t2><|box_start|>'),
                good.replace('<t2><t3>', '<t3><t2>')]:
        try:
            support(bad, 3, True)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid support accepted')
    z = weighted({'tokens':8,'ntp_count':2,'mtp_count':6,'ntp_loss':1.,'mtp_loss':3.,'ce':2.5})
    assert z['CE_reconstructed'] == 2.5 and z['CE_reconstructed'] != (1.+3.)/2
    interval = {'begin_fid': 20, 'end_fid': 40}
    assert [f for f in [19,20,39,40] if interval['begin_fid'] <= f < interval['end_fid']] == [20,39]
    assert not torch.cuda.is_initialized()
    return {'status':'passed','controls':['closed token formulas and padding','grammar/zero-box retention/gap rejection','weighted NTP/MTP denominator','actual interval dictionary and half-open boundaries'],
            'new_model_calls':0,'CUDA_initialized':False}


def inputs():
    files = [Path(__file__), ROOT/'protocols/desta3d_v2_source_prefix_support16_v2.md',
             ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py', ROOT/'scripts/desta3d_source_fit_v1.py',
             ROOT/'vg_tta/desta3d_v2_source.py', ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',
             PTD/'dataset/sft_dataset.py', PTD/'model/ptd_generation.py', PTD/'model/ptd_tokens.py',
             ROOT/'methods/CURRENT_METHOD.json', TOKENIZER/'tokenizer.json', TOKENIZER/'tokenizer_config.json',
             RUN/'INPUTS.json',RUN/'SOURCE_RECORDS.json',RUN/'CONFIG.json',RUN/'LOCK.json',
             RUN/'ALL_PREDICTIONS_SEAL.json',RUN/'COMPLETE.json',RUN/'independent_readback_v2/REPORT.json',
             BASE/'source_input_vjp_v1/PREFIX_SUPPORT_READBACK.json', BASE/'source_cast_probe_v1/RAW_ENDPOINTS.pt']
    for i in range(16):
        ep = RUN/'episodes'/f'{i:02}'
        files += [ep/'INPUT_IDENTITY.json', ep/'OLD_SUPERVISED_DUAL_CE.json', ep/'supervised_fp32/SUMMARY.json']
        files += [ep/(a+'.pt') for a in ARMS]
    return files


def register():
    assert not OUT.exists(), 'use a new isolated version after a failure'
    assert shutil.disk_usage(ROOT).free >= RESERVE+CAP
    save('CPU_PREFLIGHT.json', controls())
    save('REGISTRATION.json', {'time':time.time(),'status':'registered_CPU_only',
        'queries':16,'parents':16,'states':list(ARMS),'new_GPU_calls':0,'new_optimizer_steps':0,
        'source_records':'exact existing16 only','target_data':False,'new_predictions':0,
        'source_label_side_reconstruction':'masked dummy prompt, not actual full multimodal support',
        'CE_aggregation_abs_tolerance':TOL,'storage_cap_bytes':CAP,'disk_reserve_bytes':RESERVE,
        'cumulative_GPU_seconds_unchanged':38820.55116519004,'cumulative_cap_seconds':None})
    save('LOCK.json', {'pins':{str(p):sha(p) for p in inputs()}})
    print('CPU registration completed', flush=True)


def reconstruct(tokenizer, rec, identity):
    tokens = tokenizer.encode(rec['response']+'<|im_end|>\n', add_special_tokens=False)
    prompt = tokenizer.encode('assistant', add_special_tokens=False)
    ids = torch.tensor(prompt+tokens, dtype=torch.long)
    data = {'input_ids':ids,'labels':torch.tensor([-100]*len(prompt)+tokens),
            'mm_token_type_ids':torch.zeros_like(ids),
            'video_grid_thw':torch.tensor(identity['preprocess']['grid'])}
    data = append_targets(SimpleNamespace(tokenizer=tokenizer), data, len(prompt))
    masks = split_source_loss_masks(data, tokenizer)
    kinds = Counter()
    target_rows = []
    for p, category, is_ntp in zip(masks['target_positions'].tolist(), masks['target_categories'], masks['ce_masks']['ntp'].tolist()):
        token = tokenizer.decode([int(data['labels'][p])], skip_special_tokens=False)
        if category == 'semantic':
            family = 'semantic_null' if token == '<null>' else 'semantic_marker' if token in ('<|object_ref_start|>','<|object_ref_end|>') else 'semantic_text'
        elif category == 'time':
            family = 'endpoint' if re.fullmatch(r'<t\d+>', token) else 'time_null' if token == '<null>' else 'time_marker'
        elif category == 'box':
            family = 'coordinate' if re.fullmatch(r'<\d+>', token) else 'box_time_anchor' if re.fullmatch(r'<t\d+>', token) else 'box_marker'
        else:
            family = 'excluded_structural'
        mode = 'NTP' if is_ntp else 'MTP'
        kinds[(mode, family)] += 1
        target_rows.append({'target_position_in_dummy_context':p,'token_id':int(data['labels'][p]),
            'token':token,'source_span':category,'mode':mode,'family':family})
    return masks['counts'], {m: {f:v for (mm,f),v in kinds.items() if mm==m} for m in ('NTP','MTP')}, target_rows


def audit():
    tick = time.monotonic()
    assert not (OUT/'REPORT.json').exists()
    assert shutil.disk_usage(ROOT).free >= RESERVE+CAP
    for p,h in read(OUT/'LOCK.json')['pins'].items():
        assert sha(p) == h, p
    seal = read(RUN/'ALL_PREDICTIONS_SEAL.json')
    for p,h in seal['pins'].items():
        assert sha(p) == h, p
    # Existing source labels/roster were pinned before the original GPU run.
    oldpins = read(RUN/'LOCK.json')['pins']
    for f in ('SOURCE_RECORDS.json','INPUTS.json'):
        assert oldpins[str(RUN/f)] == sha(RUN/f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    rows = read(RUN/'INPUTS.json')
    records = {r['key']:r for r in read(RUN/'SOURCE_RECORDS.json')}
    assert len(rows)==len(records)==len({x['source'] for x in rows})==16
    report_old = read(RUN/'independent_readback_v2/REPORT.json')
    metrics = {a:{q['key']:q['metrics'] for q in report_old['arms'][a]['query_rows']} for a in ARMS}
    cases = []
    ce_checks = 0
    for i,row in enumerate(rows):
        ep = RUN/'episodes'/f'{i:02}'
        rec = records[row['key']]
        identity = read(ep/'INPUT_IDENTITY.json')
        frames = rec['frame_ids']
        assert row['source']==rec['source']==identity['source']
        assert frames==row['input']['frame_ids']==identity['frame_ids']
        gt = support(rec['response'], len(frames), True)
        assert gt['box_positions_1based']==rec['response_box_positions_1based']
        active = [j+1 for j,v in enumerate(rec['event_active']) if v]
        assert active == gt['box_positions_1based']
        assert all(rec['event_interval']['begin_fid'] <= frames[j-1] < rec['event_interval']['end_fid'] for j in active)
        counts, families, targets = reconstruct(tokenizer, rec, identity)
        formula = closed_counts(len(tokenizer.encode(gt['reference'],add_special_tokens=False)), len(active))
        for b in ('event','spatial'):
            assert counts[b]==identity['task_token_counts'][b]
            for m in ('ntp','mtp'):
                assert counts[b+'_'+m] == formula[b+'_'+m]
        assert sum(families[m].get('endpoint',0) for m in families)==formula['event_endpoint_targets']
        assert sum(families[m].get('coordinate',0) for m in families)==formula['spatial_coordinate_targets']
        native = {}
        for arm in ARMS:
            p = load(ep/(arm+'.pt'))
            assert p['key']==rec['key'] and p['frame_ids']==frames and p['preprocess']==identity['preprocess']
            assert p['format_ok'], 'all48 were format valid in the existing seal; unexpected change'
            ev = support(p['event_completion'], len(frames), False)
            sp = support(p['spatial_completion'], len(frames), True)
            assert ev['reference']==sp['reference'] and ev['time_1based']==sp['time_1based']
            assert sp['box_positions_1based']==[j+1 for j in p['positions']]
            assert ev['time_1based']==[j+1 for j in p['interval']]
            assert tokenizer.encode(ev['reference'],add_special_tokens=False)==p['readout']['spatial_reference_token_ids']
            raw = p['readout']['raw_blocks'].tolist()
            assert len(raw)==len(p['positions'])==len(p['geometry_valid'])
            for block, coords in zip(raw, sp['coordinate_token_values']):
                assert block==tokenizer.encode('<|box_start|>'+''.join(f'<{v}>' for v in coords)+'<|box_end|>',add_special_tokens=False)
            gtset,nset = set(active), set(sp['box_positions_1based'])
            native[arm] = {'reference':ev['reference'],'reference_exact_GT':ev['reference']==gt['reference'],
                'time_1based':ev['time_1based'],'time_exact_GT':ev['time_1based']==gt['time_1based'],
                'positions_1based':sp['box_positions_1based'],
                'extra_positions_1based':sorted(nset-gtset),'missing_positions_1based':sorted(gtset-nset),
                'extra_physical_frames':[frames[j-1] for j in sorted(nset-gtset)],
                'missing_physical_frames':[frames[j-1] for j in sorted(gtset-nset)],
                'intersection_count':len(gtset & nset),'GT_count':len(gtset),'native_count':len(nset),
                'GT_frame_support_recall':len(gtset & nset)/len(gtset),
                'valid_geometry_count':int(p['geometry_valid'].sum()),'invalid_geometry_count':int((~p['geometry_valid']).sum()),
                'spatial_prefix_length':p['readout']['spatial_prefix_length'],
                'existing_vIoU':metrics[arm][row['key']]['vIoU'],
                'existing_delta_B1_pp':100*(metrics[arm][row['key']]['vIoU']-metrics['no_update'][row['key']]['vIoU'])}
        summary = read(ep/'supervised_fp32/SUMMARY.json')
        old = read(ep/'OLD_SUPERVISED_DUAL_CE.json')
        ce = {}
        for dtype in ('BF16','FP32'):
            snapshots = {f'new_before_step{j}':summary['history'][j-1][dtype] for j in (1,2,3)}
            snapshots.update(new_final=summary[dtype+'_after'],old_final=old[dtype])
            ce[dtype] = {}
            for state,values in snapshots.items():
                ce[dtype][state] = {}
                for branch in ('event','spatial'):
                    v = values[branch]
                    assert v['tokens']==counts[branch]
                    assert all(v[m+'_count']==counts[branch+'_'+m] for m in ('ntp','mtp'))
                    ce[dtype][state][branch] = {'saved':{k:v[k] for k in ('ce','tokens','ntp_loss','mtp_loss','ntp_count','mtp_count')},**weighted(v)}
                    ce_checks += 1
                assert abs(values['total']-sum(values[b]['ce'] for b in ('event','spatial'))) <= TOL
            for final in ('new_final','old_final'):
                ce[dtype][final+'_change'] = {b:{
                    'CE':ce[dtype][final][b]['saved']['ce']-ce[dtype]['new_before_step1'][b]['saved']['ce'],
                    **{m+'_weighted_contribution':ce[dtype][final][b]['contributions'][m]-ce[dtype]['new_before_step1'][b]['contributions'][m] for m in ('ntp','mtp')}
                    } for b in ('event','spatial')}
        cases.append({'key':row['key'],'source':row['source'],'frames':len(frames),'GT':gt,
            'source_GT_physical_interval_half_open':rec['event_interval'],
            'actual_training_support_hashes':identity['support_sha'],
            'evidence_level':'label-side reconstruction + actual saved counts/hashes; full actual tensor only separately available for28199',
            'counts':counts,'independent_formula':formula,'token_families':families,
            'endpoint_fraction_of_event_CE_denominator':4/counts['event'],
            'native':native,'CE':ce,'label_side_all_supervised_targets':targets})
    prior = read(BASE/'source_input_vjp_v1/PREFIX_SUPPORT_READBACK.json')
    assert prior['GT_record_response_equals_actual_training_prefix']
    # One preserved non-dummy actual support proves the label-side reconstruction for that query.
    actual = load(BASE/'source_cast_probe_v1/RAW_ENDPOINTS.pt')['task_support']
    am = split_source_loss_masks(actual, tokenizer)
    assert am['counts']==cases[0]['counts']
    plen=int(actual['ptd_prefix_lengths'].item())
    start=int(torch.nonzero(actual['labels'][0]!=-100).flatten()[0])
    assert tokenizer.decode(actual['input_ids'][0,start:plen],skip_special_tokens=False)==records[rows[0]['key']]['response']+'<|im_end|>\n'
    maxerr=max(c['CE'][d][s][b]['absolute_error'] for c in cases for d in ('BF16','FP32')
               for s in ('new_before_step1','new_before_step2','new_before_step3','new_final','old_final') for b in ('event','spatial'))
    aggregate = {'by_arm':{a:{'semantic_reference_exact_GT':sum(c['native'][a]['reference_exact_GT'] for c in cases),
         'time_and_frame_support_exact_GT':sum(c['native'][a]['time_exact_GT'] for c in cases),
         'has_extra_frame_support':sum(bool(c['native'][a]['extra_positions_1based']) for c in cases),
         'has_missing_GT_frames':sum(bool(c['native'][a]['missing_positions_1based']) for c in cases),
         'total_native_frames':sum(c['native'][a]['native_count'] for c in cases),
         'invalid_geometry_frames':sum(c['native'][a]['invalid_geometry_count'] for c in cases)} for a in ARMS},
         'GT_box_rows':sum(c['native']['no_update']['GT_count'] for c in cases),
         'endpoint_event_denominator_fraction_range':[min(c['endpoint_fraction_of_event_CE_denominator'] for c in cases),max(c['endpoint_fraction_of_event_CE_denominator'] for c in cases)],
         'spatial_coordinate_denominator_fraction':8/13,'CE_reaggregation_checks':ce_checks,'max_CE_reaggregation_error':maxerr}
    assert not torch.cuda.is_initialized()
    result = {'status':'completed_CPU_audit','time':time.time(),'queries':16,'parents':16,'native_states':48,
        'new_model_calls':0,'new_GPU_seconds':0,'new_optimizer_steps':0,'new_predictions':0,'target_data':False,
        'source_labels':'same16 already exposed training records only',
        'aggregate':aggregate,'cases':cases,'actual_full_support_available':{'query28199':True,'other15':False},
        'limitations':['dummy prompt is not multimodal prefix replay; actual masks only have saved hashes for15',
            'NTP/MTP means do not identify endpoint/coordinate CE; per-token losses absent for this all16 comparison',
            'support mismatch is not a demonstrated cause or aligned-objective benefit; existing metrics only'],
        'CPU_elapsed_seconds':time.monotonic()-tick,'CUDA_initialized':False}
    save('REPORT.json',result)
    save('CPU_RECEIPT.json',{'elapsed_seconds':result['CPU_elapsed_seconds'],'GPU_seconds':0,
        'cumulative_GPU_seconds_unchanged':38820.55116519004,'cap':None,'free_disk_bytes':shutil.disk_usage(ROOT).free})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=CAP
    assert shutil.disk_usage(ROOT).free>=RESERVE
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['controls','register','audit'])
    action=parser.parse_args().action
    try:
        if action=='controls': print(json.dumps(controls()))
        else: globals()[action]()
    except BaseException as exc:
        if OUT.exists() and not (OUT/'FAILURE.json').exists():
            save('FAILURE.json',{'time':time.time(),'action':action,'error':repr(exc),'traceback':traceback.format_exc(),'GPU_seconds':0})
        raise
