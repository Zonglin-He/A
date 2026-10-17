"""Regression: the unchanged real Student gate rejects the old manifest and accepts the pin repair."""
from pathlib import Path
import copy,json,tempfile,unittest
from unittest.mock import patch
from scripts import stvg_qwen35_geometry_adapters_v2 as a
from scripts import stvg_qwen35_geometry_run_v2 as run
from scripts import run_stvg_soft_ep_coordinate_trial003 as worker

class ResumeContracts(unittest.TestCase):
    def test_real_student_gate_old_failure_and_corrected_manifest(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'artifacts',prefix='qwen_manifest_cpu_') as tmp:
            t=Path(tmp);locks=[]
            for name in ['DESIGN_LOCK.json','ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json']:
                p=t/name;p.write_text('{}');locks.append(p)
            old=dict(status='locked',trial_id='G32',pins={str(p.relative_to(a.ROOT)):a.sha(p) for p in locks},protected={},GT_read=False)
            old['pins']['scripts/run_stvg_soft_ep_coordinate_trial003.py']=a.sha(a.ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py')
            a.write(t/'RUNTIME.json',old)
            with self.assertRaisesRegex(RuntimeError,'must pin this worker'):
                worker._assert_runtime(t,'G32',*locks)
            saved=copy.deepcopy(old);fixed=a.bind_required_student_code(old)
            self.assertEqual(old,saved)
            a.write(t/'RUNTIME.json',fixed,mutable=True)
            actual,_=worker._assert_runtime(t,'G32',*locks)
            self.assertEqual(actual,fixed)
            self.assertEqual(set(fixed['pins'])-set(old['pins']),set(a.REQUIRED_STUDENT_CODE))

    def test_conflicting_implementation_hash_is_not_silently_rebound(self):
        old=dict(pins={a.REQUIRED_STUDENT_CODE[0]:'0'*64})
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):a.bind_required_student_code(old)

    def test_resume002_runtime_and_deadline_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'BASE',Path(tmp)),patch.object(run,'ROOT',Path(tmp)):
            p=Path(tmp);(p/'bound.py').write_text('locked')
            (p/'EXECUTION_RUNTIME002.json').write_text(json.dumps(dict(pins={'bound.py':run.sha(p/'bound.py')})))
            run.verify_run();(p/'bound.py').write_text('changed')
            with self.assertRaises(AssertionError):run.verify_run()
            (p/'EXECUTION_RUNTIME002.json').write_text(json.dumps(dict(pins={})))
            (p/'ACTIVE_EXECUTION.json').write_text(json.dumps(dict(deadline_epoch=1)))
            with self.assertRaisesRegex(AssertionError,'wall budget'):run.verify_run()

if __name__=='__main__':unittest.main()
