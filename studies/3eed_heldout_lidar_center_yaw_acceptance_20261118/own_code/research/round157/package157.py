"""Independent scalar reconstruction and allowlisted R157 public export."""
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

from score157 import (BASES, DIAG_BOOL, DIAG_NUM, DIAG_SCORES, DIAG_TEXT, GROUPS, HERE, LOCAL, META,
                      N, NAMES, ORACLES, POLICIES, ROOT)

sys.path.insert(0, str(ROOT / "research/round154"))
from package154 import DIRECTIONS, confidential_literals, dump, scan_release, sha, transport_csv

STUDY = "studies/3eed_heldout_lidar_center_yaw_acceptance_20261118"
ROWS = (*META, "macro_weight", "Frozen_A", "A", "D", "H", *DIAG_BOOL, *DIAG_NUM, *DIAG_TEXT,
        *DIAG_SCORES, *POLICIES, *ORACLES)
AGG = ("mIoU","Acc25","Acc50")
PAIRED = ("scope","arm","control","full_6way_weight","n","selected_H","local_gain_pp","local_harm_pp",
          "local_net_pp","global_gain_pp","global_harm_pp","global_net_pp","better","worse","same",
          "rescued25","harmed25","rescued50","harmed50")
COUNT = ("scope","n","origin_known",*(s for s in DIAG_BOOL if s.startswith(("eligible_","choose_"))),
         *("mean_" + s for s in DIAG_SCORES), *("median_" + s for s in DIAG_NUM))
CSV = {"all_rows.csv":ROWS,
       "direction.csv":("source","target","subset","aggregation","arm","n","weight_sum",*AGG),
       "scene.csv":("source","target","scene_cluster","aggregation","arm","n",*AGG),
       "subset.csv":("aggregation","group","subset","arm","n",*AGG),
       "paired.csv":PAIRED, "gate_counts.csv":COUNT}
SYMBOL = {"driver","arm","control","scope","source","target","subset","aggregation","group",*DIAG_TEXT}
OPTIONAL = {*DIAG_SCORES,"selected_H",*("mean_"+s for s in DIAG_SCORES)}
FLAGS = {"accepted","moved","has_B",*DIAG_BOOL}


def loadjson(p):
    return json.loads(p.read_bytes())


def copy(src, dst):
    assert src.is_file() and not src.is_symlink()
    dst.parent.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(src,dst)


def sanitized_table(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert len(reader.fieldnames) == len(fields) and set(reader.fieldnames) == set(fields), path
        rows = list(reader)
    assert rows
    for row in rows:
        assert None not in row
        for k,v in row.items():
            if k in SYMBOL:
                assert re.fullmatch(r"[A-Za-z0-9_]+", v), (path.name,k)
            elif k in FLAGS and path.name != "gate_counts.csv":
                assert v in ("True","False"), (path.name,k)
            elif k in OPTIONAL and v=="":
                pass
            else:
                assert math.isfinite(float(v)), (path.name,k)
    return rows


def audit_summary(rows, summary):
    assert len(rows)==N and [int(r["ordinal"]) for r in rows]==list(range(1,N+1))
    by_dir = {(s,t):[r for r in rows if (r["source"],r["target"])==(s,t)] for s,t in DIRECTIONS}
    assert all(by_dir.values()) and sum(map(len,by_dir.values()))==N
    byarm = {r["arm"]:r for r in summary if r["aggregation"]=="design_weighted" and r["group"]=="six_equal"
             and r["subset"]=="all"}
    assert set(byarm)==set(NAMES)
    error = 0.
    for k in NAMES:
        vals = []
        for rr in by_dir.values():
            y = np.array([float(r[k]) for r in rr])
            assert np.all((0<=y)&(y<=1))
            w = np.array([float(r["sampling_weight"]) for r in rr])
            assert np.all(w>0)
            vals.append([100*np.average(v,weights=w) for v in (y,y>.25,y>.50)])
        error=max(error,*[abs(v-byarm[k][name]) for name,v in zip(AGG,np.mean(vals,axis=0))])
    for r in rows:
        a,d,h = (float(r[arm]) for arm in ("A","D","H"))
        assert float(r["oracle_AH"])==max(a,h) and float(r["oracle_AD"])==max(a,d)
        assert all(float(r[arm])==(h if r["choose_"+arm]=="True" else a) for arm in POLICIES)
    assert error<1e-9
    return error


def export():
    dest=HERE/"publication/public_export_attempt001"
    if dest.exists():
        raise FileExistsError(dest)
    pred=loadjson(LOCAL/"PREDICTION_SEAL.json")
    audit=loadjson(HERE/"results/AUDIT.json")
    assert audit["status"]=="PASS" and audit["records"]==N==pred["n"]
    assert pred["prediction_sha256"]==audit["prediction_sha256"]==sha(LOCAL/"PREDICTIONS_PRIVATE.npz")
    assert pred["design_sha256"]==audit["design_sha256"]==sha(HERE/"experiment/DESIGN_LOCK.json")
    assert pred["GT_read"] is False and pred["FINAL_access"] is False and not audit["target_final_access"]
    dest.parent.mkdir(parents=True,exist_ok=True)
    build=Path(tempfile.mkdtemp(prefix=".build_",dir=dest.parent))
    try:
        src=HERE/"results"
        summary=loadjson(src/"summary.json")
        splits=[]
        check_rows=None
        for filename,columns in CSV.items():
            rr=sanitized_table(src/filename,columns)
            if filename=="all_rows.csv":
                check_rows=rr
            splits.append(transport_csv(build,"results/"+filename,rr,columns))
        err=audit_summary(check_rows,summary)
        dcheck=loadjson(src/"selector_discrimination.json")
        econ=loadjson(src/"signal_evidence.json")
        assert {r["mode"] for r in dcheck} == set(GROUPS) == set(econ)
        for row in dcheck:
            b=econ[row["mode"]]
            assert abs(b["net_pp"]-row["actual_pp"])<1e-9
            assert abs(b["expected_random_net_pp_same_counts_by_stratum"]-row["matched_random_expected_pp"])<1e-9
        for n in ("summary.json","AUDIT.json","signal_evidence.json","selector_discrimination.json"):
            copy(src/n,build/"results"/n)
        for n in ("DESIGN_LOCK.json","CODE_LOCK.json","TRANSFER_NOTES.md"):
            copy(HERE/"experiment"/n,build/"config"/n)
        copy(HERE/"ORIGIN_EVIDENCE.md",build/"config/ORIGIN_EVIDENCE.md")
        safe = {k:pred[k] for k in ("n","prediction_sha256","design_sha256","GT_read","FINAL_access")}
        dump(build/"PREDICTION_SEAL.json",safe)
        for n in ("predict157.py","heldout157.py","calibration157.py","score157.py","discrimination157.py","package157.py"):
            copy(HERE/n,build/"own_code/research/round157"/n)
        decision=loadjson(HERE/"DECISION.json")
        assert decision["TTA_executed"] is False or decision["TTA_executed"] is True
        if decision["TTA_executed"]:
            export_tta(build, splits)
        dump(build/"DECISION.json",decision)
        copy(HERE/"REPORT_ZH.md",build/"REPORT_ZH.md")
        (build/"REPORT.md").write_text("# R157 held-out LiDAR centre and yaw acceptance\n\n[中文报告与结果](REPORT_ZH.md)\n")
        dump(build/"TRANSPORT_SPLITS.json",splits)
        dump(build/"PUBLIC_EXPORT_AUDIT.json",dict(status="PASS",ordinary_records=N,
             signed_row_outcomes_preserved=True, independent_macro_replay_max_error=err,
             images_coordinates_features_raw_ids_or_origins_exported=False))
        files=[dict(path=p.relative_to(build).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(build.rglob("*")) if p.is_file()]
        dump(build/"MANIFEST.json",{"prefix":STUDY,"files":files})
        scan_release(build,confidential_literals())
        os.rename(build,dest)
    except BaseException:
        shutil.rmtree(build,ignore_errors=True)
        raise
    print("R157_PUBLIC_EXPORT_PASS", len(files)+1,sha(dest/"MANIFEST.json"),flush=True)


def export_tta(build, splits):
    from score_tta157 import ARMS, DIAGS, ALL
    src = HERE / "tta_results"
    seal = loadjson(LOCAL / "TTA_SEAL.json")
    audit = loadjson(src / "AUDIT.json")
    assert audit["status"] == seal["status"] == "PASS" and audit["n"] == seal["n"] == N
    assert audit["prediction_sha256"] == seal["prediction_sha256"] == sha(LOCAL / "TTA_PRIVATE.npz")
    assert audit["direct_prediction_sha256"] == loadjson(LOCAL / "PREDICTION_SEAL.json")["prediction_sha256"]
    assert audit["code_sha256"] == sha(HERE / "score_tta157.py")
    dump(build / "TTA_SEAL.json", {k:seal[k] for k in ("status","n","driver","prediction_sha256",
                                                        "direct_prediction_sha256","design_sha256","max_score_replay_error",
                                                        "max_native_center_replay_error_m","max_parameter_vs_box_output_difference_m",
                                                        "parameter_count_per_input","nonzero_steps","torch_autograd_single_step",
                                                        "GT_read","source_training")})
    columns = (*META, "macro_weight", "accepted", "moved", "driver", *ALL, *DIAGS)
    records = None
    for name, fields in {"all_rows.csv":columns, "direction.csv":CSV["direction.csv"], "scene.csv":CSV["scene.csv"],
                         "subset.csv":CSV["subset.csv"], "paired.csv":CSV["paired.csv"]}.items():
        rr = sanitized_table(src/name, fields)
        splits.append(transport_csv(build, "tta_results/"+name,rr,fields))
        if name == "all_rows.csv":
            records = rr
    assert [int(r["ordinal"]) for r in records] == list(range(1,N+1))
    assert sum(r["accepted"]=="True" for r in records)==audit["accepted"] and sum(r["moved"]=="True" for r in records)==audit["moved"]
    check = {r["arm"]:r for r in loadjson(src/"summary.json") if r["group"]=="six_equal"
             and r["aggregation"]=="design_weighted"}
    for name in ALL:
        assert abs(100*math.fsum(float(r["macro_weight"])*float(r[name]) for r in records)-check[name]["mIoU"]) < 1e-8
    for name in ("AUDIT.json","summary.json"):
        copy(src/name,build/"tta_results"/name)
    for name in ("TTA_CODE_LOCK.json","TTA_RELEASE.json"):
        copy(HERE/"experiment"/name,build/"config"/name)
    for name in ("tta157.py","score_tta157.py"):
        copy(HERE/name,build/"own_code/research/round157"/name)


if __name__=="__main__":
    export()
