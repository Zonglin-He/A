"""Pin the reviewed finite frozen feature capture and resource supervisor."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.capture_stvg_short_window_features_v1 import REQUIRED_CAPTURE_PINS,_preflight
B=R/'artifacts/stvg_short_event_tracklets_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 assert sha(R/'scripts/capture_stvg_short_window_features_v1.py')=='5aa3fc5bab663644d6cdca8824a63a523c29f77379e85ad6bc88bb2230d6da0b';assert sha(B/'CPU_FEATURE_CAPTURE_CONTRACTS.json')=='960920aafeeb114548a4b01c1089a8aad49358f8203977b867b1af19069f3c44'
 paths=set(REQUIRED_CAPTURE_PINS)|{'scripts/capture_stvg_short_window_features_v1.py','scripts/launch_stvg_short_window_v1.py','scripts/lock_stvg_short_window_feature_v1.py','protocols/stvg_short_event_tracklets_execution_v1.md','artifacts/stvg_short_event_tracklets_v1/CPU_FEATURE_CAPTURE_CONTRACTS.json','artifacts/stvg_short_event_tracklets_v1/PREPARED_MANIFEST.json'}
 old=json.loads((R/'artifacts/stvg_motion_guided_instance_opd_v1/METHOD_STUDENT_RUNTIME002.json').read_text());paths|=set(old['pins']);pins={p:sha(R/p)for p in sorted(paths)};protected=old['protected'];assert all(sha(R/p)==h for p,h in protected.items())
 lock=dict(status='locked',scope='8 old sources frozen feature capture; no fitting or GT',GT_read=False,manifest_sha256=sha(B/'PREPARED_MANIFEST.json'),pins=pins,protected=protected,time=time.time());save(B/'FEATURE_CAPTURE_LOCK.json',lock)
 save(B/'FEATURE_RUNTIME.json',dict(status='locked',GT_read=False,manifest_sha256=lock['manifest_sha256'],lock_sha256=sha(B/'FEATURE_CAPTURE_LOCK.json'),pins=pins,protected=protected,time=time.time()))
 _preflight();print('feature lock/root preflight pass')
if __name__=='__main__':main()
