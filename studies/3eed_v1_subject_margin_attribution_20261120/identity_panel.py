"""Attribution on sealed natural descriptions with uniquely assignable inventory identities."""
import argparse
import csv
from functools import lru_cache
import gzip
from pathlib import Path
import sys

import numpy as np
import torch

from attribute import HERE, ROOT, OUT, PRIVATE, can_win_by_scaling, write_csv
from scope183 import PLATFORMS, atomic, read, digest

OLD = HERE.parent / "3eed_v1_reciprocal_physical_roles_20261120"
sys.path.insert(0, str(OLD))
from capture_roles import frame, Grounder

NAMES = ("natural170","natural171")


def inputs():
    cfg = read(HERE/"identity_protocol.json")
    for path, pin in cfg["historical_inputs"].items():
        assert digest(ROOT/path) == pin
    return [(panel, read(ROOT / "crossplatform" / ("round"+panel[-3:]) / "INPUT_PRIVATE.json")) for panel in NAMES]


@torch.inference_mode()
def capture_panel():
    populations=inputs()
    old=read(OLD/"run_PRIVATE/SEAL.json")
    fields={key:[] for key in ("panel","case","source","prior","final","active")}
    torch.set_num_threads(1)
    for source in PLATFORMS:
        model=Grounder(source)
        model._frontend.parser=lru_cache(maxsize=128)(model._frontend.parser)
        for panel, records in populations:
            for case, record in enumerate(records):
                obs=frame(record["row"])
                if obs["target"]==source:
                    continue
                _,ev=model._read_observation(obs)
                v=model._decoder(ev)
                scores=v["subject_logits"][0].double().cpu().numpy()
                name=f"{panel}_{case:03d}_{source}.npz"
                saved=OLD/"run_PRIVATE"/name
                assert digest(saved)==old["files"][name]
                with np.load(saved,allow_pickle=False) as previous:
                    np.testing.assert_array_equal(ev["subject_boxes"][0].cpu().numpy(),previous["slate"])
                    assert int(v["choices"][0,0])==previous["choices"][0]
                    before=previous["scores"][0]
                    if previous["active"]:
                        before=np.log(before)
                    np.testing.assert_allclose(scores-scores.max(),before-before.max(),atol=1e-10,rtol=0.)
                data=dict(panel=panel,case=case,source=source,prior=ev["base_logit"][0].double().cpu().numpy(),
                          final=scores,active=bool(ev["has_reference"][0]))
                for key, value in data.items():
                    fields[key].append(value)
        del model
        torch.cuda.empty_cache()
    assert len(fields["case"])==168
    path=PRIVATE/"identity_predictions.npz"
    np.savez_compressed(path,**{k:np.stack(v) for k,v in fields.items()})
    atomic(PRIVATE/"identity_seal.json",dict(n=168,sha256=digest(path),protocol=digest(HERE/"identity_protocol.json"),
           matching_original_four_scores_and_winners=168,labels_used=False,FINAL_access=False))


def score_panel():
    from score_roles import owners, inventory, anon
    populations=dict(inputs())
    seal=read(PRIVATE/"identity_seal.json")
    file=PRIVATE/"identity_predictions.npz"
    assert seal["sha256"]==digest(file) and not seal["labels_used"] and not seal["FINAL_access"]
    with np.load(file,allow_pickle=False) as z:
        data=dict(z)
    with gzip.open(OLD/"results/diagnostic_rows.csv.gz","rt") as src:
        previous={(r["group"],r["case"],r["source"]):r for r in csv.DictReader(src) if r["group"] in NAMES}
    rows=[]
    for n,(panel,case,source) in enumerate(zip(data["panel"],data["case"],data["source"])):
        row=populations[panel][int(case)]["row"]
        target=row["platform"]
        boxes,_=inventory(row["metapath"],target,int(row["suffix"] or 0),row["object_index"])
        with np.load(OLD/"run_PRIVATE"/f"{panel}_{case:03d}_{source}.npz",allow_pickle=False) as z:
            assignment=np.array(owners(z["slate"],boxes))
        good=assignment==0
        a,final=data["prior"][n],data["final"][n]
        current=assignment[final.argmax()]
        eligible=current>0 and good.any()
        reference=bool(data["active"][n])
        anon_case=anon(f"{panel}|{case}")
        assert int(eligible)==int(previous[panel,anon_case,source]["wrong_with_correct_slot"])
        margin=float(a.max()-a[good].max()) if eligible else None
        limit=1.5+int(reference)
        rows.append(dict(panel=panel,row=anon_case,source=source,target=target,
                         resolved_wrong_with_target_slot=int(eligible),active_reference=int(reference),
                         gap=margin,cap=limit,blocked=int(margin>=limit) if eligible else None,
                         scale_can_recover=int(can_win_by_scaling(a,final-a,good)) if eligible else None))
    write_csv(OUT/"natural_identity_rows.csv.gz",rows)
    summary=[]
    for nm in (*NAMES,"both"):
        group=[r for r in rows if nm=="both" or r["panel"]==nm]
        cases=[r for r in group if r["resolved_wrong_with_target_slot"]]
        summary.append(dict(panel=nm,n=len(group),resolved_errors_with_target_slot=len(cases),
                  blocked=sum(r["blocked"] for r in cases), within_bound=sum(not r["blocked"] for r in cases),
                  scale_can_recover=sum(r["scale_can_recover"] for r in cases)))
    assert [r["resolved_errors_with_target_slot"] for r in summary]==[7,6,13]
    write_csv(OUT/"natural_identity.csv",summary)
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["capture","score"])
    phase=parser.parse_args().phase
    capture_panel() if phase=="capture" else score_panel()
