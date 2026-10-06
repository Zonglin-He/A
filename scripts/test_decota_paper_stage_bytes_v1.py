"""Synthetic opaque-byte corruption tests; no experiment/model/GT access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_decota_paper_stage_bytes_v1 import check_barrier, check_payload


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.p = self.base / 'opaque.npz'
        self.p.write_bytes(b'not-decoded-or-parsed')
        self.digest = hashlib.sha256(self.p.read_bytes()).hexdigest()
        self.rc = dict(GT_read=False, runtime_lock_sha256='pin', sha256=self.digest,
                       bytes=self.p.stat().st_size, time=1.)
        self.save()

    def save(self):
        self.p.with_suffix('.json').write_text(json.dumps(self.rc))

    def check(self):
        return check_payload(self.base, 'opaque.npz', self.digest, 'pin', 2.)

    def test_valid_opaque_bytes_do_not_require_numpy_or_payload_parsing(self):
        self.assertEqual(self.check(), len(b'not-decoded-or-parsed'))

    def test_changed_payload_is_rejected(self):
        self.p.write_bytes(b'mutated')
        with self.assertRaises(AssertionError): self.check()

    def test_changed_receipt_pin_size_GT_and_late_time_are_rejected(self):
        for field, value in [('runtime_lock_sha256', 'other'), ('bytes', 0),
                             ('GT_read', True), ('time', 3.)]:
            with self.subTest(field=field):
                old = self.rc[field]; self.rc[field] = value; self.save()
                with self.assertRaises(AssertionError): self.check()
                self.rc[field] = old; self.save()

    def test_missing_coverage_and_path_escape_are_rejected(self):
        b = dict(status='sealed', GT_read=False, time=2., files={'opaque.npz': self.digest})
        (self.base / 'barrier.json').write_text(json.dumps(b))
        with self.assertRaises(AssertionError):
            check_barrier(self.base, 'barrier.json', ['opaque.npz', 'missing.npz'], 'pin')
        with self.assertRaises(AssertionError):
            check_payload(self.base, '../outside.npz', self.digest, 'pin', 2.)


if __name__ == '__main__': unittest.main()
