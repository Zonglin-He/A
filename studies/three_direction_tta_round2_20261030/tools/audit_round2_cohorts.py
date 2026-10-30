"""Independent pre-score cohort/input audit; no annotation numeric targets read."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def main():
    av=ROOT/'avis/round2/cohorts';lock=json.loads((av/'COHORT_LOCK.json').read_text())
    d=json.loads((av/'dev.json').read_text());c=json.loads((av/'confirmation.json').read_text())
    assert digest(av/'dev.json')==lock['dev_manifest_sha256']
    assert digest(av/'confirmation.json')==lock['confirmation_manifest_sha256']
    old=set(lock['excluded_old8']);ids=[];frame_counts=[]
    available=[x.name for x in (ROOT/'avis/datasets/test/JPEGImages').iterdir() if x.is_dir() and x.name not in old]
    ranked=sorted(available,key=lambda x:hashlib.sha256((d['salt']+'\0'+x).encode()).hexdigest())
    for cohort,expected in [(d,ranked[:8]),(c,ranked[8:20])]:
        assert [x['video_id'] for x in cohort['videos']]==expected
        for row in cohort['videos']:
            assert row['video_id'] not in old
            assert row['length']==row['original_length']==len(row['image_paths'])
            assert row['frame_indices']==list(range(row['length']))
            assert row['source_frame_numbers']==list(range(1,row['length']+1))
            numbers=[int(Path(x).stem.rsplit('_',1)[1]) for x in row['image_paths']]
            assert numbers==row['source_frame_numbers']
            features=np.load(row['audio_path'],allow_pickle=False)
            assert features.shape==(row['length'],128) and np.isfinite(features).all()
            assert all(Path(x).is_file() for x in row['image_paths'])
        ids.append({x['video_id'] for x in cohort['videos']});frame_counts.append(sum(x['length'] for x in cohort['videos']))
    assert not ids[0]&ids[1]
    cp=ROOT/'crossplatform/round2';cl=json.loads((cp/'COHORT_PATH_LOCK.json').read_text())
    cd=cl['cohorts']['dev'];cc=cl['cohorts']['confirm'];allrows=cd+cc
    assert len(cd)==18 and len(cc)==36 and len({r['id'] for r in allrows})==54
    ds={(r['platform'],r['scene']) for r in cd};cs={(r['platform'],r['scene']) for r in cc};assert not ds&cs
    numeric_errors=[]
    for split,records in [('dev',cd),('confirm',cc)]:
        inp=json.loads((cp/f'inputs/{split}_manifest.json').read_text());assert [x['id'] for x in inp]==[x['id'] for x in records]
        for row,original in zip(inp,records):
            assert 'meta_path' not in row and not any('bbox' in k or 'target' in k for k in row)
            file=Path(original['lidar_path']);raw=np.load(file,allow_pickle=False) if file.suffix=='.npy' else np.fromfile(file,np.float32).reshape(-1,4)
            indices=np.random.default_rng(row['seed']).choice(len(raw),16384,replace=len(raw)<16384)
            xyz=raw[indices,:3].copy()
            if row['platform']=='drone':xyz[:,2]+=1.8
            if row['platform']=='waymo':
                view=int(file.parent.name.split('_')[-1]);rot=Rotation.from_euler('z',[0,-45,45,-90,90][view],degrees=True).as_matrix();xyz=xyz@rot.T
            expected=xyz.astype(np.float32);given=np.load(row['point_path'],allow_pickle=False)
            assert given.shape==(16384,3) and np.isfinite(given).all()
            assert digest(row['point_path'])==row['point_sha256']
            # Independent scipy/quaternion vs released trigonometric rotations can
            # differ near algebraic zero. Input-file hashes themselves stay exact.
            error=float(np.max(abs(expected-given)));numeric_errors.append(error);assert error<1e-6,(row['id'],error)
    result={'scope':'Independent pre-score input/cohort checks only; no numeric target arrays or boxes read',
            'AVIS':{'dev_videos':8,'confirmation_videos':12,'frames_per_split':frame_counts,'hash_selection_replayed':True,'old8_and_new_splits_video_disjoint':True,'numeric_frame_order_and_finite_audio_features':True},
            '3EED':{'dev':18,'confirm':36,'dev_confirm_scene_disjoint':True,'dev_scene_clusters':len(ds),'confirm_scene_clusters':len(cs),'independent_sampling_transform_max_error':max(numeric_errors),'independent_transform_absolute_tolerance_meters':1e-6,'exact_input_file_hashes_checked':True,'target_free_input_schema_checked':True,'history':'Drone/Quad scenes previously exposed; Waymo new scene sets per locked cohort. New frames alone are not new scenes.'}}
    output=ROOT/'research/round2/COHORT_INDEPENDENT_AUDIT.json'
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
