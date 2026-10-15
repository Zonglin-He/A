"""Four matched reference/update cells with real Source-reset full-model readouts."""
import collections
import copy
import gc
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import *


def signature(anchors, aware):
    fields = ['position', 'physical_frame_id', 'box_cxcywh', 'weight']
    if aware:
        fields += ['evidence_gate']
    return [{k: a[k] for k in fields} for a in anchors]


def main():
    import torch
    import numpy as np
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source_helper
    from scripts.run_stvg_table1_student004 import stable_fit_key
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for, frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward, transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_weightfree_coordinate_fit_v1 import fit_validated
    from scripts.stvg_ecld_spatial_fit_v1 import fit_evidence_aware
    verify_runtime()
    sys.addaudithook(guard)
    refs = {x['case_id']: x for x in read(BASE / 'REFERENCE_LOCK.json')['records']}
    design = read(BASE / 'DESIGN_LOCK.json')
    parent_barrier = read(PARENT / 'STUDENT_GLOBAL_BARRIER.json')
    parent_files = {x['path']: x for x in parent_barrier['files']}
    install_clean_loader()
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = source = None
    files = []
    start = time.perf_counter()
    for rec in roster():
        cid = rec['probe_alias']
        dest = BASE / 'student' / (cid + '.pt')
        receipt = dest.with_suffix('.receipt.json')
        if dest.exists() and receipt.exists():
            previous = read(receipt)
            assert previous['sha256'] == sha(dest)
            assert previous['reference_lock_sha256'] == sha(BASE / 'REFERENCE_LOCK.json')
            files.append(previous['file'])
            continue
        assert not dest.exists() and not receipt.exists()
        case, source_path = load_case(rec)
        reference = refs[cid]
        parent_path = PARENT / 'student' / (cid + '.pt')
        assert sha(parent_path) == parent_files[str(parent_path.relative_to(ROOT))]['sha256']
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        assert parent['source_case_sha256'] == case['source_case_sha256'] and not parent['GT_read']
        temporal = torch.load(PARENT / 'temporal' / (cid + '.pt'), map_location='cpu', weights_only=False)
        interval = temporal['selected_interval']
        assert parent['arms']['T']['interval'] == interval
        arms = dict(F=native_arm(case), T=native_arm(case, interval),
                    S_historical=copy.deepcopy(parent['arms']['S_new_crop']),
                    TS_historical=copy.deepcopy(parent['arms']['TS_new_crop']))
        fits, readouts, receipts = {}, {}, {}
        ids = list(map(int, case['frame_ids']))
        expected = case['source_model_state_sha256']
        pixel = case['pixel_sha256']
        capture_s = 0.
        if any(reference[v]['anchors'] for v in ['A', 'B', 'C', 'D']):
            transient = dict(case)
            transient['source'] = source_helper.SOURCE_MODEL_NAME[rec['dataset']]
            transient['_source_case_sha256'] = case['source_case_sha256']
            _, row = source_helper._validate_roster_record(rec)
            transient['parent'] = row['source']
            def catalog(record):
                assert record['case_id'] == cid
                assert record['source_case_sha256'] == case['source_case_sha256']
                return transient, {}
            h.catalog = catalog
            rr = dict(case_id=cid, dataset=rec['dataset'], source_case_sha256=case['source_case_sha256'])
            model, source, _, cat, raw, frames, ids, records, batch, base, native, expected, pixel, capture_s = h._capture_source(
                rr, model, source, torch, collections, model_for, frames_for, raw_row,
                frozen_forward, prediction, state_hash)
            assert list(ids) == list(map(int, case['frame_ids']))
            assert pixel == case['pixel_sha256'] and expected == case['source_model_state_sha256']
            assert np.array_equal(arms['F']['boxes'], h._xyxy(native['boxes']))
            assert h._same_logits(native['logits'], case['zero']['logits'])
            initial = {k: v.clone() for k, v in base.initial.items()}
        for variant in ['A', 'B', 'C', 'D']:
            aware = variant in ['C', 'D']
            cfg = dict(design['spatial_configs'][rec['dataset']])
            aa = reference[variant]['anchors']
            if not aa:
                arms['S_' + variant] = copy.deepcopy(arms['F'])
                arms['TS_' + variant] = copy.deepcopy(arms['T'])
                receipts[variant] = dict(actual_fit_count=0, status='no_reference_no_update')
                continue
            reused = next((v for v in fits if (v in ['C', 'D']) == aware
                and signature(reference[v]['anchors'], aware) == signature(aa, aware)), None)
            if reused is not None:
                fits[variant] = copy.deepcopy(fits[reused])
                readouts[variant] = copy.deepcopy(readouts[reused])
                arms['S_' + variant] = copy.deepcopy(arms['S_' + reused])
                arms['TS_' + variant] = copy.deepcopy(arms['TS_' + reused])
                receipts[variant] = dict(actual_fit_count=0, status='exact_same_fitter_reference_config_seed_reuse',
                                         reused_variant=reused)
                continue
            anchors = h._anchors(reference[variant], ids, torch)
            for a, original in zip(anchors, aa):
                a['evidence_gate'] = bool(original['evidence_gate'])
            partial = BASE / 'partial' / cid / variant
            partial.mkdir(parents=True, exist_ok=False)
            def checkpoint(payload):
                save_pt(partial / f"step{payload['next_step']}.pt", payload)
            fitter = fit_evidence_aware if aware else fit_validated
            fitted = fitter(base, {k: v.clone() for k, v in initial.items()}, anchors, ids,
                stable_fit_key(rec['dataset'], rec['query_key_sha256']), cfg, validate=True,
                reference_sha256=sha(BASE / 'REFERENCE_LOCK.json'), checkpoint_callback=checkpoint)
            out = h._readout(model, batch, ids, records, fitted, raw['parses']['subject'], core, torch,
                transient_native_masks=transient_native_masks, frozen_visual_host=frozen_visual_host,
                frozen_attention_storage=frozen_attention_storage, inactive_modules_on_host=inactive_modules_on_host,
                query_subject=query_subject)
            assert h._same_logits(out['logits'], native['logits'])
            assert state_hash(model.state_dict()) == expected
            assert torch.equal(out['boxes'].cpu(), fitted['final_cached_decoder_boxes'].cpu())
            arms['S_' + variant] = dict(boxes=h._xyxy(out['boxes']), interval=list(case['Native_interval']))
            arms['TS_' + variant] = dict(boxes=h._xyxy(out['boxes']), interval=list(interval))
            fits[variant], readouts[variant] = fitted, out
            receipts[variant] = dict(actual_fit_count=1, status='actual_fit_full_redecode',
                accepted_steps=fitted['optimizer_steps'], config=cfg, evidence_aware=aware)
        payload = dict(case_id=cid, dataset=rec['dataset'], parent_sha256=rec['parent_sha256'],
            arms=arms, fits=fits, readouts=readouts, receipts=receipts, reference_records=reference,
            frame_ids=list(ids), pixel_sha256=pixel, source_case_sha256=case['source_case_sha256'],
            source_model_state_sha256=expected, source_capture_seconds=capture_s,
            source_file_sha256=sha(source_path), historical_prediction_sha256=sha(parent_path),
            temporal_file_sha256=sha(PARENT / 'temporal' / (cid + '.pt')), source_restored=True, GT_read=False)
        item = save_pt(dest, payload)
        item['actual_fit_count'] = sum(v['actual_fit_count'] for v in receipts.values())
        write(receipt, dict(status='complete', sha256=sha(dest),
            reference_lock_sha256=sha(BASE / 'REFERENCE_LOCK.json'), file=item, GT_read=False))
        files.append(item)
        write(BASE / 'STUDENT_STATUS.json', dict(status='running', done=len(files), total=32,
              actual_new_fit_count=sum(x['actual_fit_count'] for x in files), GT_read=False), mutable=True)
        print('ecld student', cid, item['actual_fit_count'], flush=True)
        del payload
        gc.collect()
        torch.cuda.empty_cache()
    write(BASE / 'STUDENT_GLOBAL_BARRIER.json', dict(status='complete', count=len(files), files=files,
        wall_seconds=time.perf_counter()-start, GT_read=False))


if __name__ == '__main__':
    main()
