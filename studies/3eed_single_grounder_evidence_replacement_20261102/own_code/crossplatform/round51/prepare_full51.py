"""Unexecuted R51 full-only I/O adapter for immutable prepare51.py.

Same geometry, attributes, source and graph logic. Only the growing per-file
ledger changes from repeated full JSON rewrites to fsync'ed append journal,
followed by one identical-schema JSON aggregate before the original sealing.
No import-time data access and no new execution/experiment grant is created.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare51

ORIGINAL_PREPARE_SHA256 = '932c57f44cecb49d1205c044eba2fcf05a7b627219c4509d7b505b0be94afe13'


class AppendReadLedger:
    active = []

    def __init__(self, out):
        self.path = Path(out) / 'preparation_IO.json'
        self.journal_path = Path(out) / 'preparation_IO.jsonl'
        self.rows = []
        self.aggregated = False
        if self.path.exists() or self.journal_path.exists():
            raise FileExistsError('No ledger resume or overwrite')
        self.active.append(self)

    def bind(self, path, role, expected=None, data_id=None, selected_fields=None):
        p = Path(path).resolve()
        actual = prepare51.sha(p)
        item = {'role': role, 'path_PRIVATE': str(p), 'sha256': actual,
                'data_id_PRIVATE': data_id, 'selected_fields': selected_fields,
                'bytes_hashed': p.stat().st_size, 'status': 'HASH_BOUND_BEFORE_PARSE'}
        # Persist before the caller parses/opens the actual selected data. A
        # previously started failure is never retried or silently repaired.
        with self.journal_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(item)
        if expected is not None and actual != expected:
            raise ValueError('Input hash drift: ' + role)
        return actual

    def aggregate(self, original_dump):
        if self.aggregated:
            return
        original_dump(self.path, {'schema': 'R51_raw_safe_preparation_IO_v1',
                                  'files': self.rows,
                                  'metadata_file_hash_is_not_annotation_parsing': True})
        self.aggregated = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--authorization', required=True)
    args = ap.parse_args()
    plan = prepare51.read(args.plan)
    auth = prepare51.read(args.authorization)
    if auth.get('state') != 'ROOT_AUTHORIZED_R51_RAW_SAFE_PREPARATION' or auth.get('plan_sha256') != prepare51.sha(args.plan):
        raise ValueError('Independent full-stage root grant required')
    if plan.get('n') != 5912 or plan.get('append_only_preparation_IO') is not True:
        raise ValueError('This adapter is only for separately locked full5912 append-I/O')
    if prepare51.sha(Path(prepare51.__file__)) != ORIGINAL_PREPARE_SHA256:
        raise ValueError('The immutable original preparation implementation drifted')
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if plan['programs_sha256'].get(relative) != prepare51.sha(__file__):
        raise ValueError('Full I/O adapter must be explicitly program-pinned')
    original_dump = prepare51.dump
    output_root = prepare51.path_in_workspace(plan['output_root'])

    def dump_with_ledger_finalize(p, value):
        p = Path(p)
        # COUNTS precedes the original COSTS and file-map construction, so both
        # aggregate and journal are included in the unchanged final seal map.
        # Also retain a complete aggregate on failure, without faking a seal.
        if p.parent == output_root and p.name in ('COUNTS.json', 'FAILED.json'):
            for ledger in AppendReadLedger.active:
                ledger.aggregate(original_dump)
        original_dump(p, value)

    prepare51.ReadLedger = AppendReadLedger
    prepare51.dump = dump_with_ledger_finalize
    # The immutable worker still checks all source/graph/program/input hashes,
    # cohort identity, output boundary and non-overwrite before any sample I/O.
    prepare51.main()


if __name__ == '__main__':
    main()
