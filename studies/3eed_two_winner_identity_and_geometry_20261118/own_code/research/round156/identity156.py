"""Reconcile sealed, masked R156 image reads and estimate shares of positive oracle gain."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRIVATE = HERE.parents[1] / "crossplatform/round156/atlas_PRIVATE"
CATEGORIES = ("SAME_TARGET_GEOMETRY", "DIFFERENT_TARGET_RESCUE", "OTHER", "UNKNOWN")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(x, y):
    return x if x == y else "UNKNOWN"


def wilson(successes, n):
    """Marginal 95% interval for an IID PPS draw from this fixed cohort."""
    p = successes / n
    z = NormalDist().inv_cdf(.975)
    denominator = 1 + z*z/n
    midpoint = (p + z*z/(2*n)) / denominator
    halfwidth = z * (p*(1-p)/n + z*z/(4*n*n))**.5 / denominator
    return [max(0, midpoint - halfwidth), min(1, midpoint + halfwidth)]


def main():
    audit = json.loads((RESULTS / "PAIR_SCORE_AUDIT.json").read_text())
    assert audit["status"] == "PASS" and audit["n"] == 1798
    files = [HERE/"reader1_PRIVATE.csv", HERE/"reader2_PRIVATE.csv"]
    seals = [digest(p) for p in files]
    assert seals == [
        "9bdd542922f5ea3fb98616d3f592b7706823cdbdfc118ade8d09e485c7682634",
        "85ded02802d23520ef8ad2b3b3ffa76b3dcc854b4a7c533b786a5e0f66b772ad",
    ]
    first, second = [{r["case"]: r for r in read_csv(p)} for p in files]
    sample = json.loads((PRIVATE / "MAPPING_PRIVATE.json").read_text())
    pairrows = {int(r["ordinal"]): r for r in read_csv(RESULTS / "pair_rows.csv")}
    assert len(sample) == len(first) == len(second) == len({r["ordinal"] for r in sample})
    assert set(first) == set(second) == {m["case"] for m in sample}
    total_mass = sum(float(r["D_win_gain_pp"]) for r in pairrows.values())
    output = []
    for m in sample:
        case, ordinal = m["case"], m["ordinal"]
        r = pairrows[ordinal]
        a, d = float(r["A_pca"]), float(r["D_pca"])
        assert m["positive_draws"] + m["negative_draws"] > 0
        assert bool(m["positive_draws"]) == (d > a)
        assert bool(m["negative_draws"]) == (d < a)
        f, s = first[case], second[case]
        fc, sc = f["category"], s["classification"]
        assert fc in CATEGORIES and sc in CATEGORIES
        output.append(dict(
            case=case, ordinal=ordinal, source=r["source"], target=r["target"],
            positive_draws=m["positive_draws"], negative_draws=m["negative_draws"],
            A_pca=a, D_pca=d, D_minus_A_pp=100*(d-a),
            D_gain_macro_contribution_pp=100*(d-a)*float(r["macro_weight"]),
            reader1_category=fc, reader2_category=sc, consensus_category=reconcile(fc,sc),
            pair_consensus=reconcile(f["pair"],s["pair"]), A_consensus=reconcile(f["A"],s["A"]),
            D_consensus=reconcile(f["D"],s["D"]),
        ))
    assert sum(r["positive_draws"] for r in output) == 36
    assert sum(r["negative_draws"] for r in output) == 6
    with (RESULTS / "identity_rows.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    summary = dict(
        status="PASS",
        positive_gain_oracle_mass_pp=total_mass,
        reader_seals=seals,
        paired_reader_classification_disagreements=sum(r["reader1_category"] != r["reader2_category"] for r in output),
        sampling="36 draws IID with replacement proportional to positive six-direction weighted D-minus-A gain; 6 negative controls drawn without replacement proportional to magnitude. Repeated draws reuse the same masked interpretation.",
        uncertainty="Category-specific marginal 95% Wilson intervals quantify draw uncertainty conditional on the fixed development observations and these readings. They do not quantify reader error, sampling of scenes, or out-of-domain performance. UNKNOWN covers ambiguous and discordant reads.",
        review_boundary="Two independent masked reads by AI agents of the same calibrated imagery, text, annotated referent and point cloud. There is no independent human instance ground truth for the predictions.",
    )
    for name, field in (("positive","positive_draws"),("negative","negative_draws")):
        subset = [r for r in output if r[field]]
        n = sum(r[field] for r in subset)
        bycat = {}
        for c in CATEGORIES:
            rows = [r for r in subset if r["consensus_category"] == c]
            k = sum(r[field] for r in rows)
            stats = dict(draws=k, unique_records=len(rows))
            if name == "positive":
                stats.update(gain_mass_share=k/n, gain_mass_share_wilson95=wilson(k,n),
                             estimated_oracle_mass_pp=total_mass*k/n)
            bycat[c] = stats
        direction_records = [pairrows[r["ordinal"]] for r in subset]
        # Frame and content cluster IDs are already anonymized group keys on the entire cohort.
        summary[name] = dict(
            draws=n,unique_direction_records=len(subset),
            physical_frames=len({r["frame_cluster"] for r in direction_records}),
            scene_clusters=len({r["scene_cluster"] for r in direction_records}),
            agreed_A_background_draws=sum(r[field] for r in subset if r["A_consensus"]=="BACKGROUND"),
            classification_agreement_draws=sum(r[field] for r in subset if r["reader1_category"]==r["reader2_category"]),
            categories=bycat,
        )
    geometry = summary["positive"]["categories"]["SAME_TARGET_GEOMETRY"]["draws"]
    identity = summary["positive"]["categories"]["DIFFERENT_TARGET_RESCUE"]["draws"]
    branch="CENTER" if geometry > 18 else "LANGUAGE" if identity > 18 else "UNRESOLVED"
    summary["prelocked_branch"]=branch
    (RESULTS/"IDENTITY_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
