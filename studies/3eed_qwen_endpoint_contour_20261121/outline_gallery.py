"""Render private evidence for every oracle opportunity and every actual control crossover."""
from html import escape
import numpy as np
from PIL import Image,ImageDraw
from endpoints import OUT, PRIVATE, ancestry, digest, read
from outline_geometry import raw_paths, silhouette
from verify_outline import table


def draw_crop(base,rect,curves,polys):
    im=base.crop(rect)
    w,h=im.size
    size=(max(1,round(500*w/max(w,h))),max(1,round(500*h/max(w,h))))
    im=im.resize(size,Image.Resampling.BICUBIC); pen=ImageDraw.Draw(im)
    def xy(p):
        return tuple((np.array(p)-rect[:2])*size/np.array([w,h]))
    for ring,col in zip(polys,("#71ff71","#b794ff")):
        if ring is not None:
            points=[xy(p) for p in ring]
            pen.line(points+[points[0]],fill=col,width=2)
    for path,col in zip(curves,("#00eaff","#ff7200")):
        if not np.isfinite(path).all():
            continue
        points=[xy(p) for p in path]
        pen.line(points,fill=col,width=2)
        x,y=points[0]
        pen.ellipse((x-4,y-4,x+4,y+4),outline=col,width=2)
    return im


def run():
    assert digest(OUT/"outline_rows.csv")==read(OUT/"outline_evaluation_receipt.json")["files"]["outline_rows"]
    data=table("outline_rows"); panel,_,prep,_=ancestry(); frames={r["uid"]:r for r in panel}
    obs=read(PRIVATE/"observations.json")
    directory=PRIVATE/"outline_cases"; directory.mkdir(mode=0o700,exist_ok=True)
    parts=[]; n=0
    for r in data:
        arms=("H","span","endpoint","outline","outline_wrong","outline_rect","outline_single","outline_wrong_single","outline_censored","outline_wrong_censored","outline_rect_censored")
        passed={float(r["iou_"+a])>.5 for a in arms}
        if len(passed)==1 and r["opportunity50"]!="True":
            continue
        uid=r["uid"]; f=frames[uid]; o=obs[uid]; n+=1
        with Image.open(f["rgb"]) as img:
            base=img.convert("RGB")
        full=base.copy(); full.thumbnail((560,460)); full.save(directory/(uid+"-full.jpg"))
        chunks=[f'<figure><img src="{uid}-full.jpg" loading="lazy"><figcaption>完整观测</figcaption></figure>']
        if r["outline_eligible"]=="True":
            rect=o["views"][1]["rect"];raw=read(PRIVATE/"outline_replies"/(uid+".json"))
            polys=[silhouette(a,rect,i==1) for i,a in enumerate(raw["answers"])]
            curves=raw_paths(f,o,prep[uid]["geometry"])
            delta=np.subtract(o["wrong"],o["correct"])[0]
            for k,shift in (("正确",0),("错配",delta)):
                picture=draw_crop(base,rect,curves+shift,polys)
                name=uid+("-true.jpg" if k=="正确" else "-mismatch.jpg")
                picture.save(directory/name)
                chunks.append(f'<figure><img src="{name}" loading="lazy"><figcaption>{k}的端点与延伸路径</figcaption></figure>')
        vals="　".join(f'{label} {float(r["iou_"+arm]):.3f}' for arm,label in
                      (("H","原框"),("outline","轮廓"),("outline_wrong","错配"),("outline_rect","外接矩形"),("outline_censored","边界可判定")))
        parts.append(f'<section><h2>{r["target"]} · {uid}</h2><p>{escape(f["text"])}</p><p>{vals}</p>'
                     f'<p>23 帧机会　{r["opportunity50"]}　轮廓读出　{r["status_outline"]}　{r["action_outline"]}</p>'
                     f'<div>{"".join(chunks)}</div></section>')
    header='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>轮廓几何逐帧核验</title>
<style>body{font:16px system-ui;max-width:1420px;margin:2em auto;padding:0 1em}section{border-top:1px solid #ccc;margin:2em 0}
section div{display:flex;gap:1em;flex-wrap:wrap}figure{margin:0}img{max-width:420px;height:auto;max-height:360px}h2{font-size:1.1em}</style>
<h1>轮廓几何逐帧核验</h1><p>绿、紫线分别为原裁剪和镜像裁剪读出的轮廓，均画回原图。青、橙圆圈分别为实测负端、正端，线段为沿原轴外推的投影。完整保留 23 个机会及这些对照产生的所有 Acc@50 切换。</p>'''
    (directory/"index.html").write_text(header+"\n".join(parts)+"</html>")
    print("Private outline gallery frames",n)


if __name__=="__main__":
    run()
