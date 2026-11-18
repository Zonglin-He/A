"""Publish anonymous geometry metrics and first-party implementation after scoring."""
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research/round154"))
from package154 import confidential_literals, dump, scan_release, sha, transport_csv

STUDY = "studies/3eed_fixed_winner_surface_yaw_20261118"


def main():
    dest = HERE / "publication" / "public_export_attempt001"
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".build_", dir=dest.parent))
    results = HERE / "results"
    audit = json.loads((results / "AUDIT.json").read_bytes())
    verified = json.loads((HERE / "PREDICTION_AUDIT.json").read_bytes())
    assert audit["status"] == verified["status"] == "PASS"
    assert audit["n"] == 1798 and audit["protected_frames_opened"] == 0
    assert audit["prediction_sha256"] == verified["prediction_sha256"]
    public_verify = {k:v for k,v in verified.items()
                     if isinstance(v,(int,float,bool)) or k in ("status","prediction_sha256","design_sha256")}
    dump(build / "PREDICTION_AUDIT.json", public_verify)
    sources = ["predict155.py", "audit155.py", "score155.py", "describe155.py", "package155.py"]
    for name in sources:
        out = build / "own_code/research/round155" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, out)
    for rel in ("crossplatform/surface_dev45/refine.py", "crossplatform/replication36_surface/common.py",
                "crossplatform/geometry_refine.py"):
        out = build / "own_code" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, out)
    seal = json.loads((ROOT / "crossplatform/round155/SEAL.json").read_bytes())
    dump(build / "CAPTURE_CHECKS.json", {k:seal[k] for k in
         ("n","sha256","design_sha256","original_xyz_hash_matches","original_indices_hash_matches",
          "original_winners_checked","max_seed_reconstruction_error_m","new_network_forwards",
          "source_training","GT_read","FINAL_access","seconds")})
    for name in ("DESIGN_LOCK.json","CODE_LOCK.json"):
        out = build / "config" / name
        out.parent.mkdir(exist_ok=True)
        shutil.copyfile(HERE / "experiment" / name, out)
    prefix = build / "results"
    prefix.mkdir()
    for name in ("task_summary.json","AUDIT.json"):
        shutil.copyfile(results/name,prefix/name)
    d = next(path for path in (HERE / "ORIENTATION_DIAGNOSTICS.json", results / "ORIENTATION_DIAGNOSTICS.json") if path.is_file())
    shutil.copyfile(d, prefix / d.name)
    tables = []
    for path in sorted(results.glob("*.csv")):
        assert path.name in {"task_all_rows.csv","task_direction.csv","task_scene.csv","task_subset.csv","paired.csv"}
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            cols = reader.fieldnames
            rows = list(reader)
        assert cols and rows
        if path.name == "task_all_rows.csv":
            safe_cols = {"ordinal","source","target","has_B","sampling_weight","scene_cluster","frame_cluster",
                         "point_content_cluster","eligible_joint_zero","eligible_R145","frozen_A","geometry_zero","geometry_TTA"}
            safe_cols.update(f"{s}_{a}" for s in ("joint_zero","R145") for a in ("axis","pca","surface","sign_flip"))
            assert set(cols) == safe_cols and len(rows) == 1798
        tables.append(transport_csv(build,"results/"+path.name,rows,cols))
    dump(build / "TRANSPORT_SPLITS.json", tables)
    shutil.copyfile(HERE / "REPORT_ZH.md",build / "REPORT_ZH.md")
    (build / "REPORT.md").write_text("# R155 fixed-winner LiDAR yaw\n\n[中文报告与结果](REPORT_ZH.md)\n")
    records = [dict(path=f.relative_to(build).as_posix(), sha256=sha(f),bytes=f.stat().st_size)
               for f in sorted(build.rglob("*")) if f.is_file()]
    dump(build / "MANIFEST.json", {"prefix":STUDY,"files":records})
    scan_release(build, confidential_literals())
    os.rename(build, dest)
    print("R155_PUBLIC_EXPORT_PASS",len(records)+1,flush=True)


if __name__ == "__main__":
    main()
