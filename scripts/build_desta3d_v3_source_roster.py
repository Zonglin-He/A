"""Seal full source row/label contracts after media QA, no model or scoring."""
import collections,json,math,os,shutil,sqlite3,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_desta3d_v3_source import read,put,sha,OUT as INTAKE
from vg_tta.desta3d_v3_data import SourcePool,balanced_epoch
OUT=ROOT/'artifacts/desta3d_v3/full_source_roster_v1'
def main():
    assert not (OUT/'STARTED.json').exists()
    qa=ROOT/'artifacts/desta3d_v3/source_media_audit_v1/COMPLETE.json'
    repair=ROOT/'artifacts/desta3d_v3/hc_media_repair_v1/COMPLETE.json'
    assert qa.exists() and repair.exists()
    put(OUT/'REGISTRATION.json',dict(time=time.time(),GPU_seconds=0,source_labels='official source-train only, including internal held-out source rows; no target outcome access',
        frame_grid='official 2fps evenly spaced physical crop capped32; GT on identical frame IDs; do not shift missing event edge boxes',
        failure_rule='unavailable/mismatched media retains full key/reason in quarantine; no query replacement; CE-ineligible rows retained as evidence-only',
        training_does_not_begin_until_full_manifest_is_sealed=True,
        pins={str(p):sha(p) for p in [Path(__file__),ROOT/'vg_tta/desta3d_v3_data.py',INTAKE/'MANIFEST.json',qa,repair]}))
    put(OUT/'STARTED.json',dict(time=time.time(),pid=os.getpid()));start=time.monotonic()
    pool=SourcePool();dbpath=OUT/'SOURCE.sqlite';db=sqlite3.connect(dbpath)
    db.execute('CREATE TABLE examples (key TEXT PRIMARY KEY, split TEXT, domain TEXT, parent TEXT, row_json TEXT, labels_json TEXT, provenance_json TEXT)')
    accepted=[];quarantine=[];counts=collections.Counter();seen_hashes={};cecounts=collections.Counter()
    try:
        for i,r in enumerate(pool.rows):
            assert shutil.disk_usage(ROOT).free>8*2**30
            try:
                row,record=pool.example(r,labels=True,materialize=False)
                # Source labels are a separately stored sidecar. Validation
                # execution never fetches labels_json before prediction seal.
                partition=(r['domain'],r['split']);h=row['input']['video_sha256']
                if h in seen_hashes and seen_hashes[h][1]!=r['split']:raise SystemExit('identical media bytes cross source split; stop full intake, never select one side')
                seen_hashes[h]=partition
                db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?)',(r['key'],r['split'],r['domain'],r['parent'],
                    json.dumps(row),json.dumps(record),json.dumps(r)))
                accepted.append({k:r[k] for k in ['key','split','domain','parent']})
                counts[r['domain']+':'+r['split']]+=1;cecounts[r['domain']+':'+r['split']+':'+str(record['response_eligible'])]+=1
            except Exception as e:quarantine.append(dict(key=r['key'],domain=r['domain'],parent=r['parent'],split=r['split'],error=repr(e)))
            if (i+1)%1000==0:db.commit();print('SOURCE_ROWS',i+1,len(pool.rows),'quarantine',len(quarantine),flush=True)
        db.commit();db.close()
        assert len(accepted)+len(quarantine)==len(pool.rows)
        for d in ['Vid','HC1']:
            tr={r['parent'] for r in accepted if r['domain']==d and r['split']=='train'};val={r['parent'] for r in accepted if r['domain']==d and r['split']=='validation'}
            assert tr and val and not tr&val
        tr=[r for r in accepted if r['split']=='train'];order=balanced_epoch(tr,0)
        put(OUT/'INPUTS.json',accepted);put(OUT/'QUARANTINE.json',quarantine)
        put(OUT/'SUMMARY.json',dict(official_intended_queries=len(pool.rows),accepted_queries=len(accepted),quarantined_queries=len(quarantine),
            counts=dict(counts),CE_counts=dict(cecounts),epoch_queries=len(order),updates_per_epoch=math.ceil(len(order)/4),max_epochs=5,
            total_horizon=5*math.ceil(len(order)/4),warmup_updates=math.ceil(.05*5*math.ceil(len(order)/4)),
            unique_training_parents={d:len({r['parent'] for r in tr if r['domain']==d}) for d in ['Vid','HC1']},
            quarantine_is_not_an_effect_based_selection=True,validation_GT_preparation='labels stored separately for later sealed scoring; no predictions or metrics read'))
        put(OUT/'COMPLETE.json',dict(time=time.time(),CPU_wall_seconds=time.monotonic()-start,GPU_seconds=0,
            pins={str(p):sha(p) for p in [dbpath,OUT/'INPUTS.json',OUT/'QUARANTINE.json',OUT/'SUMMARY.json',OUT/'REGISTRATION.json']},
            status='source_contracts_complete_not_full_pixel_decode_or_model_validation'))
    except BaseException:
        db.close();put(OUT/'FAILURE.json',dict(error=traceback.format_exc(),CPU_wall_seconds=time.monotonic()-start,GPU_seconds=0));raise
if __name__=='__main__':main()
