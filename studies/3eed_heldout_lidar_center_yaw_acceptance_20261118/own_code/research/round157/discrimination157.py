"""Post-seal discrimination of a paired score given eligibility and the same switch budget."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    results=HERE/"results"
    audit=json.loads((results/"AUDIT.json").read_text())
    assert audit["status"]=="PASS"
    with (results/"all_rows.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1798
    output=[]
    for mode in ("ray","surface","combined"):
        strata=defaultdict(list)
        for r in rows:
            if r["eligible_"+mode]=="True":
                strata[(r["source"],r["target"],r["has_B"],r["origin_ok"])].append(r)
        totals=defaultdict(float)
        count=0
        for rr in strata.values():
            n=len(rr)
            k=sum(r["choose_"+mode]=="True" for r in rr)
            selected=[r for r in rr if r["choose_"+mode]=="True"]
            count+=k
            def effect(r):
                return 100 * float(r["macro_weight"]) * (float(r["H"])-float(r["A"]))
            vals=[effect(r) for r in rr]
            mean=math.fsum(vals)/n
            totals["expected_equal_budget_pp"]+=k*mean
            totals["all_eligible_H_pp"]+=math.fsum(vals)
            totals["expected_positive_pp"]+=k/n*math.fsum(max(v,0) for v in vals)
            totals["expected_negative_pp"]+=k/n*math.fsum(min(v,0) for v in vals)
            totals["actual_net_pp"]+=math.fsum(effect(r) for r in selected)
            if n>1:
                totals["null_variance"]+=k*(1-k/n)*math.fsum((v-mean)**2 for v in vals)/(n-1)
        sd=math.sqrt(totals.pop("null_variance",0))
        output.append(dict(mode=mode,n=len(rows),eligible=sum(map(len,strata.values())),accepted=count,
                           eligible_switch_all_pp=totals["all_eligible_H_pp"],
                           matched_random_expected_pp=totals["expected_equal_budget_pp"],
                           matched_random_expected_gains_pp=totals["expected_positive_pp"],
                           matched_random_expected_harms_pp=totals["expected_negative_pp"],
                           randomization_sd_pp=sd,actual_pp=totals["actual_net_pp"],
                           incremental_over_expected_pp=totals["actual_net_pp"]-totals["expected_equal_budget_pp"]))
    (results/"selector_discrimination.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
