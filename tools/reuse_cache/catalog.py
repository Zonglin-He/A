"""Index existing private evidence without copying or changing sealed assets.

Registration checks small receipts and payload existence/size. Payload hashes
are verified lazily before every reuse. Registration is not full validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from .store import sha, digest


class Catalog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS entries (path TEXT PRIMARY KEY, '
                        'backbone TEXT, kind TEXT, dataset TEXT, identity TEXT, record TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS lookup ON entries(backbone,kind,dataset,identity)')

    def add(self, record):
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
        old = self.db.execute('SELECT record FROM entries WHERE path=?', (record['path'],)).fetchone()
        if old:
            if old[0] != encoded:
                raise ValueError('Registered immutable evidence changed: ' + record['path'])
            return
        self.db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?)',
                        tuple(record[k] for k in ['path', 'backbone', 'kind', 'dataset', 'identity']) + (encoded,))

    def find(self, backbone, kind, dataset, identity):
        rows = self.db.execute('SELECT record FROM entries WHERE backbone=? AND kind=? '
                               'AND dataset=? AND identity=?', (backbone, kind, dataset, identity))
        return [json.loads(row[0]) for row in rows]

    @staticmethod
    def verified_path(record):
        p = Path(record['path'])
        if sha(record['receipt']) != record['receipt_sha256'] or sha(p) != record['sha256']:
            raise ValueError('Receipt or payload changed; do not reuse ' + str(p))
        return p

    def counts(self):
        return [dict(backbone=b, kind=k, entries=n, logical_bytes=size)
                for b, k, n, size in self.db.execute(
                    "SELECT backbone,kind,COUNT(*),SUM(json_extract(record,'$.bytes')) "
                    'FROM entries GROUP BY backbone,kind')]

    def teacher(self, backbone, frames, caption, fingerprint):
        """Cache-only bridge; a miss never starts an expert or changes positions."""
        import torch
        context = dict(RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                       shape=list(frames.shape), caption=caption,
                       teacher_fingerprint=fingerprint)
        key = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rows = self.find(backbone, 'teacher', 'content_bound', key)
        if not rows:
            raise KeyError('Exact teacher context is not cached')
        record = rows[0]
        path = self.verified_path(record)
        receipt = json.loads(Path(record['receipt']).read_text())
        if receipt['context'] != context:
            raise ValueError('Teacher input context changed')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['RGB_sha256'] != context['RGB_sha256']:
            raise ValueError('Teacher RGB binding changed')
        return value, dict(cache_hit=True, cache_key=key, new_teacher_calls=0,
                           payload_sha256=record['sha256'])


def refresh(root, dbpath):
    root = Path(root).resolve()
    cat = Catalog(dbpath)
    skipped = 0
    for family, backbone in [('tubedetr_boxfixed_development', 'tubedetr'),
                             ('llava_st_development', 'llava_st')]:
        base = root / 'outputs' / family
        plan_path = root / 'results' / family / 'plan.json'
        plan_hash = sha(plan_path)
        for ds in ['vidstg', 'hc2']:
            folder = base / ds / 'captures'
            seal = json.loads((folder / 'sealed.json').read_text())
            if seal['status'] != 'complete' or seal['queries'] != 128 or seal['plan_sha256'] != plan_hash:
                raise ValueError('Capture seal/plan mismatch')
            for q, e in seal['files'].items():
                payload, receipt = folder / e['path'], folder / e['receipt_path']
                if sha(receipt) != e['receipt_sha256']:
                    raise ValueError('Capture receipt changed')
                r = json.loads(receipt.read_text())
                if r['payload_sha256'] != e['sha256'] or r['plan_sha256'] != plan_hash:
                    raise ValueError('Capture payload binding changed')
                cat.add(dict(path=str(payload.resolve()), receipt=str(receipt.resolve()),
                             receipt_sha256=e['receipt_sha256'], sha256=e['sha256'],
                             bytes=payload.stat().st_size, backbone=backbone, kind='spatial_capture',
                             dataset=ds, identity=r['request_sha256'], plan_sha256=plan_hash,
                             validation='receipt_and_seal_checked_payload_hash_on_reuse',
                             capabilities=['Source', 'native_interval', 'spatial_features', 'eight_observations'],
                             missing=['new_temporal_prior', 'new_pretext_grid']))
            # Full predictions retain useful small native outputs even when dense
            # features were not saved. Live per-query receipts do not imply a
            # complete domain or permission to score a partial population.
            full = base / ds / 'full'
            full_seal = json.loads((full / 'sealed.json').read_text()) if (full / 'sealed.json').exists() else None
            for receipt in sorted(full.glob('*.json')):
                if receipt.name in {'sealed.json', 'metrics.json', 'details_private.json'}:
                    continue
                try:
                    r = json.loads(receipt.read_text())
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(r, dict) or 'payload_sha256' not in r:
                    continue
                if r['plan_sha256'] != plan_hash:
                    raise ValueError('Full prediction belongs to a different plan')
                payload = receipt.with_suffix('.npz')
                receipt_hash = sha(receipt)
                if full_seal is not None:
                    e = full_seal['files'][receipt.stem]
                    if e['sha256'] != r['payload_sha256'] or e['receipt_sha256'] != receipt_hash:
                        raise ValueError('Full receipt differs from complete seal')
                cat.add(dict(path=str(payload.resolve()), receipt=str(receipt.resolve()),
                             receipt_sha256=receipt_hash, sha256=r['payload_sha256'],
                             bytes=payload.stat().st_size, backbone=backbone, kind='native_prediction',
                             dataset=ds, identity=r['request_sha256'], plan_sha256=plan_hash,
                             validation='per_query_receipt_checked_payload_hash_on_reuse',
                             capabilities=['Source', 'native_interval', 'positions8'],
                             missing=['spatial_features', 'new_temporal_prior', 'new_pretext_grid']))
        # Teacher keys include exact RGB, caption, shape and teacher fingerprint.
        teacher_root = base / 'teacher_cache'
        for receipt in sorted(teacher_root.glob('*.json')):
            try:
                r = json.loads(receipt.read_text())
            except json.JSONDecodeError:
                skipped += 1
                continue  # A live writer has not committed this receipt yet.
            context = r['context']
            key = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if key != receipt.stem:
                raise ValueError('Teacher key/context mismatch')
            payload = receipt.with_suffix('.pt')
            cat.add(dict(path=str(payload.resolve()), receipt=str(receipt.resolve()),
                         receipt_sha256=sha(receipt), sha256=r['sha256'], bytes=payload.stat().st_size,
                         backbone=backbone, kind='teacher', dataset='content_bound', identity=key,
                         teacher_fingerprint=context['teacher_fingerprint'],
                         validation='context_key_checked_payload_hash_on_reuse',
                         capabilities=['native_boxes', 'native_masks', 'selected_query'],
                         missing=['new_positions_if_eight_RGB_differ']))
    cat.db.commit()
    result = dict(status='registered_for_lazy_verified_reuse', groups=cat.counts(),
                  skipped_incomplete_receipts=skipped, payload_bytes_copied=0,
                  GPU_calls=0, labels_read=False, old_assets_modified=False,
                  not_a_full_payload_hash_audit=True)
    cat.db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--database', type=Path, default=Path('runtime/reuse_cache/catalog.sqlite'))
    args = parser.parse_args()
    print(json.dumps(refresh(args.root, args.database), indent=2))
