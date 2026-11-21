"""Create a private RGB review of the 23 opportunities and all threshold rescues/harms."""
import csv
from html import escape
from PIL import Image
from endpoints import (OUT, PRIVATE, ancestry, canvas, digest, read)


def gallery():
    receipt=read(OUT/"evaluation_receipt.json")
    assert digest(OUT/"rows.csv")==receipt["files"]["rows"]
    with (OUT/"rows.csv").open() as f:
        rows=list(csv.DictReader(f))
    panel,_,_,_=ancestry()
    frame={r["uid"]:r for r in panel}
    obs=read(PRIVATE/"observations.json")
    dest=PRIVATE/"cases"
    dest.mkdir(exist_ok=True,mode=0o700)
    reviewed=[]
    refs=("H","span","mismatch","previous_B")
    title={"H":"原框","span":"只给跨度","mismatch":"错配","previous_B":"上轮跨度"}
    for r in rows:
        uid=r["uid"]; e=float(r["iou_endpoint"])
        changes=[("救回" if e>.5 else "误改")+"（对"+title[a]+"）" for a in refs if
                 (e>.5)!=(float(r["iou_"+a])>.5)]
        if r["opportunity50"]=="True":
            changes.insert(0,"23 帧机会")
        if not changes:
            continue
        o=obs[uid]; f=frame[uid]
        marks=""
        if "views" in o:
            assert digest(f["rgb"])==f["rgb_sha"]
            with Image.open(f["rgb"]) as img:
                original=img.convert("RGB")
            for k,desc in (("correct","实测端点"),("wrong","同帧错配端点")):
                if k not in o and k=="wrong":
                    continue
                sheet=canvas(original,o["views"],o.get(k))
                name=uid+"_"+k+".jpg"
                sheet.save(dest/name,quality=90)
                caption=desc if k in o else "端点不可用，原帧与目标裁剪"
                marks+=f'<figure><img loading="lazy" src="{name}"><figcaption>{caption}</figcaption></figure>'
        vals="　".join(f"{title.get(k,'实测端点')} {float(r['iou_'+k]):.3f}" for k in ("H","span","endpoint","mismatch"))
        reviewed.append(f'<section><h2>{escape(r["target"])} · {uid}</h2><p>{"　".join(changes)}</p>'
                        f'<p>IoU　{vals}</p><p>投影 {r["projection_status"]}　两种编码 {r["decision_endpoint"]}　'
                        f'最终选择 {r["label_endpoint"]}</p><div>{marks}</div></section>')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>端点与轮廓逐帧核验</title>
<style>body{font:16px system-ui,sans-serif;max-width:1320px;margin:2rem auto;padding:0 1rem;color:#18262c}
h1{font-size:1.6rem}h2{font-size:1.1rem}section{border-top:1px solid #b9c4c8;padding-top:1rem;margin:2rem 0}
section div{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:.8rem}
figure{margin:0}img{width:100%;height:auto}figcaption{color:#435862}p{margin:.5rem 0}</style>
<h1>端点与轮廓逐帧核验</h1><p>每幅图左侧为全景，右侧为冻结目标裁剪。青色为负端，橙色为正端。表中为双编码一致后的结果。</p>'''
    (dest/"index.html").write_text(html+"\n".join(reviewed)+"</html>")
    print("Private review pages:",len(reviewed),"frames",flush=True)


if __name__=="__main__":
    gallery()
