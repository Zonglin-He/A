"""Read sealed expert JSON metadata only; no media, labels, predictions or GPU.

This describes sampling/support, not expert correctness or native task utility.
The output is a new diagnostic file and never changes the live experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.fmean(values), max=max(values))


def audit(directory):
    started = time.monotonic()
    seal_path = directory / 'EXPERT_EVIDENCE_SEAL.json'
    seal = json.loads(seal_path.read_text())['files']
    roster_path = directory / 'DEV64.json'
    registration = json.loads((directory / 'REGISTRATION.json').read_text())
    assert sha(roster_path) == registration['pins'][str(roster_path)]
    roster = json.loads(roster_path.read_text())
    rows = []
    consumed = {str(roster_path): sha(roster_path), str(seal_path): sha(seal_path)}
    for index, source in enumerate(roster):
        evidence = {}
        for branch in ('temporal', 'spatial'):
            path = directory / 'experts' / branch / f'{index:02}' / 'EVIDENCE.json'
            digest = sha(path)
            assert digest == seal[str(path.relative_to(directory))]
            consumed[str(path)] = digest
            evidence[branch] = json.loads(path.read_text())
        temporal, spatial = evidence['temporal'], evidence['spatial']
        ids, fps = source['input']['frame_ids'], source['input']['fps']
        assert temporal['frame_ids'] == spatial['frame_ids'] == ids
        assert temporal['key'] == spatial['key'] == source['key']
        assert temporal['pixel_sha256'] == spatial['pixel_sha256']
        assert len(ids) > 1 and all(b > a for a, b in zip(ids, ids[1:]))
        slots, picks = temporal['slots_physical'], temporal['nearest_observation_indices']
        # Independent Python nearest-frame and tie-breaking reconstruction.
        assert picks == [min(range(len(ids)), key=lambda i: (abs(ids[i]-s), i)) for s in slots]
        assert len(slots) == max(1, int(temporal['duration']*2))
        gaps = [(b-a)/fps for a, b in zip(ids, ids[1:])]
        anchor, interval = spatial['anchor'], temporal['interval_physical']
        anchor_frame = ids[anchor['position']] if anchor else None
        inside = (interval[0] <= anchor_frame <= interval[1]
                  if anchor_frame is not None and interval is not None else None)
        selected = temporal['selected']
        rows.append(dict(index=index, observations=len(ids), slots=len(slots),
            unique_slot_observations=len(set(picks)), repeated_slots=len(picks)-len(set(picks)),
            repeat_fraction=1-len(set(picks))/len(picks),
            observation_gap_median_seconds=statistics.median(gaps),
            observation_gap_max_seconds=max(gaps),
            nearest_slot_max_offset_seconds=max(abs(ids[i]-s)/fps for i, s in zip(picks, slots)),
            anchor_inside_predicted_interval=inside,
            temporal_selected_score=temporal['scores'][selected] if selected is not None else None,
            spatial_anchor_score=anchor['score'] if anchor else None))
    summary = dict(queries=len(rows), parents=len({s['source'] for s in roster}),
        observations=describe([r['observations'] for r in rows]),
        temporal_slots=describe([r['slots'] for r in rows]),
        queries_with_repeated_slots=sum(r['repeated_slots'] > 0 for r in rows),
        repeat_fraction=describe([r['repeat_fraction'] for r in rows]),
        repeated_slots_total=sum(r['repeated_slots'] for r in rows),
        slots_total=sum(r['slots'] for r in rows),
        observation_gap_median_seconds=describe([r['observation_gap_median_seconds'] for r in rows]),
        observation_gap_max_seconds=describe([r['observation_gap_max_seconds'] for r in rows]),
        nearest_slot_max_offset_seconds=describe([r['nearest_slot_max_offset_seconds'] for r in rows]),
        anchors_inside_predicted_interval=sum(r['anchor_inside_predicted_interval'] is True for r in rows),
        anchors_outside_predicted_interval=sum(r['anchor_inside_predicted_interval'] is False for r in rows),
        anchor_interval_unavailable=sum(r['anchor_inside_predicted_interval'] is None for r in rows))
    return dict(status='sealed_metadata_readback', time=datetime.now(timezone.utc).isoformat(),
                CPU_seconds=time.monotonic()-started, new_GPU_seconds=0,
                GT_read=False, media_decoded=False, native_predictions_read=False,
                consumed_file_hashes=consumed, summary=summary, rows=rows,
                limitation='Outside predicted interval is not identity error; repeats/gaps do not establish task harm. Scores are not calibrated correctness probabilities.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    if args.output.exists():
        raise FileExistsError('Keep the previous diagnostic; use a new output path.')
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
