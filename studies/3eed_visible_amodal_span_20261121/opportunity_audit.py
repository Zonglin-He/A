"""Post-seal case/control visual audit. The reader is not shown group or outcome."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from PIL import Image,ImageDraw
from common import OUT, PRIVATE, crop_photo, digest, inputs, read, write
from visual_audit import key

def prepare():
    panel,frozen=inputs()
    assert read(OUT/'prediction_seal.json')['private_sha']==digest(PRIVATE/'predictions.json')
    estimates=read(PRIVATE/'predictions.json')
    assert read(OUT/'oracle_seal.json')['private_sha']==digest(PRIVATE/'oracle.json')
    oracle=read(PRIVATE/'oracle.json')
    opportunities=[r for r in panel if frozen[r['uid']]['selected'] and oracle[r['uid']]['recoverable50']]
    cases={r['uid'] for r in opportunities}
    used=set(cases)
    for r in sorted(opportunities,key=lambda x:key('case',x['uid'])):
        options=[v for v in panel if v['uid'] not in used and frozen[v['uid']]['selected'] and v['target']==r['target']]
        control=min(options,key=lambda v:(r['recording']!=v['recording'],r['panel']!=v['panel'],key('control',r['uid'],v['uid'])))
        used.add(control['uid'])
    used.update(uid for uid,e in estimates.items() if any(abs(a)>1.e-7 for a in e['offsets'].values()))
    rows=sorted((r for r in panel if r['uid'] in used),key=lambda r:key('audit_order',r['uid']))
    roster={}
    for start in range(0,len(rows),6):
        sheet=Image.new('RGB',(960,480),(44,44,44)); c=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+6]):
            uid=r['uid']; img,rect=crop_photo(r,frozen[uid])
            w,h=img.size; factor=min(290/w,194/h)
            thumb=img.resize((max(1,round(w*factor)),max(1,round(h*factor))),Image.Resampling.BICUBIC)
            x=320*(j%3)+15+(290-thumb.width)//2
            y=240*(j//3)+28+(194-thumb.height)//2
            sheet.paste(thumb,(x,y))
            code=f'B{start+j+1:02d}'
            c.text((320*(j%3)+7,240*(j//3)+4),f'{code}  {w} x {h}',fill='white')
            roster[code]=dict(uid=uid,crop=rect,display=[x,y,thumb.width,thumb.height])
        sheet.save(PRIVATE/f'mixed_{start//6+1:02d}.png')
    write(PRIVATE/'mixed_roster.json',roster)
    print('Blinded crops',len(rows),'sheets',(len(rows)+5)//6)

def geometry():
    from predict_span import process
    panel,frozen=inputs()
    roster=read(PRIVATE/'mixed_roster.json'); marks=read(PRIVATE/'mixed_marks.json')
    assert marks.keys()==roster.keys()
    by={r['uid']:r for r in panel}
    jobs=[]
    for k,e in roster.items():
        uid=e['uid']
        # The task side of the diagnostic sees the marked RGB locations only.
        reply=dict(crop=e['crop'],parsed=marks[k])
        jobs.append((by[uid],frozen[uid],reply,None))
    with ProcessPoolExecutor(max_workers=5) as pool:
        results={e['uid']:v for e,v in zip(roster.values(),pool.map(process,jobs),strict=True)}
    write(PRIVATE/'mixed_geometry.json',results)
    write(OUT/'mixed_review_seal.json',dict(frames=len(results),private_sha=digest(PRIVATE/'mixed_geometry.json'),
          marks=digest(PRIVATE/'mixed_marks.json'),roster=digest(PRIVATE/'mixed_roster.json'),
          primary_seal=digest(OUT/'prediction_seal.json'),code=digest(__file__),
          reader='assistant blind to case, frozen outcome, Qwen replies, depth and GT',
          postseal_purpose='descriptive case-control audit and landmark-assisted diagnostic only'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('prepare','geometry'))
    {'prepare':prepare,'geometry':geometry}[ap.parse_args().action]()
