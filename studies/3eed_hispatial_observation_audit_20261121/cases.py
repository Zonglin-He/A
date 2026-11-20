"""Select representative paired cases from predetermined research strata."""
import json
import textwrap

from PIL import Image,ImageDraw,ImageFont

from common import OUT,PRIVATE,key,read,write
from summarize import read_csv


def run():
    table=read_csv("qa_rows.csv")
    native={r["qid"]:r for r in table if r["arm"]=="moge"}
    rows=[r for r in table if r["arm"]=="prior_depth" and r["mode"]=="language"]
    strata=[
      ("L3 geometry helped, answer harmed",lambda a,b:a["level"]=="L3" and a["all_objects_depth_improved"]=="1" and a["correct"]=="0" and b["correct"]=="1"),
      ("L3 geometry helped, answer rescued",lambda a,b:a["level"]=="L3" and a["all_objects_depth_improved"]=="1" and a["correct"]=="1" and b["correct"]=="0"),
      ("L2 geometry helped, answer harmed",lambda a,b:a["level"]=="L2" and a["all_objects_depth_improved"]=="1" and a["correct"]=="0" and b["correct"]=="1"),
      ("L1 object distance rescued",lambda a,b:a["family"]=="distance" and a["correct"]=="1" and b["correct"]=="0"),
      ("L0 depth estimate rescued",lambda a,b:a["family"]=="pixel_xyz" and a["correct"]=="1" and b["correct"]=="0"),
    ]
    frames={r["uid"]:r for r in read(PRIVATE/"frames.json")}
    questions={r["qid"]:r for r in read(PRIVATE/"questions.json")}
    answers=read(PRIVATE/"answers.json")
    chosen=[]
    selections=[]
    for label,condition in strata:
        candidates=[a for a in rows if condition(a,native[a["qid"]])]
        candidates.sort(key=lambda x:key("case",label,x["qid"]))
        candidate=next((c for c in candidates if c["uid"] not in {r["uid"] for r in chosen}),None)
        selected=dict(stratum=label,eligible=len(candidates))
        if candidate is not None:
            a=candidate;b=native[a["qid"]]
            selected.update(uid=a["uid"],qid=a["qid"],platform=a["target"],task=a["task"],
                n_on_named_object_min=int(a["min_object_holdout"]),original_correct=int(b["correct"]),
                corrected_correct=int(a["correct"]),depth_mae_original_m=float(b["global_depth_mae"]),
                depth_mae_corrected_m=float(a["global_depth_mae"]),
                object_depth_mae_original_m=float(b["object_depth_mae"]) if b["object_depth_mae"] else None,
                object_depth_mae_corrected_m=float(a["object_depth_mae"]) if a["object_depth_mae"] else None)
            chosen.append(selected)
        selections.append(selected)
    write(OUT/"examples.json",selections)
    if not chosen:
        return
    canvas=Image.new("RGB",(1140,324*len(chosen)),"#171a20")
    font=ImageFont.truetype("DejaVuSans.ttf",15)
    for i,v in enumerate(chosen):
        f=frames[v["uid"]]
        with Image.open(f["rgb"]) as img:
            im=img.convert("RGB")
        im.thumbnail((550,306));canvas.paste(im,(1,i*324+5))
        block=ImageDraw.Draw(canvas)
        records="\n".join(textwrap.wrap(questions[v["qid"]]["prompt"],65))
        truth=answers[v["qid"]]["value"]
        raw=read(PRIVATE/"answers_frozen"/(v["uid"]+".json"))["answers"]
        outputs={x["arm"]:x["answer"] for x in raw if x["qid"]==v["qid"]}
        lines=[v["stratum"],records,"",f"MoGe: {outputs['moge'][:96]}",
               f"Prior Depth Anything: {outputs['prior_depth'][:96]}",
               f"Gold: {truth}",f"Holdout z MAE: {v['depth_mae_original_m']:.2f} to {v['depth_mae_corrected_m']:.2f}m"]
        block.multiline_text((570,i*324+14),"\n".join(lines),font=font,fill="white",spacing=5)
    canvas.save(PRIVATE/"review"/"paired_results.jpg",quality=91)


if __name__=="__main__":
    run()
