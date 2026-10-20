"""Prepare official evaluation members and labels offline; never load a model.

Use the pinned TA-STVG data.tar annotation JSON, never raw HC val_v2 as a
substitute. Outputs must be new files. Inference only receives the resulting
label-free manifest; the separate labels are opened by the sealed scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path


def normalized(text):return ' '.join(text.strip().lower().split())


def prepare(dataset,annotations,base):
    result=[];labels=[];seen=set()
    if dataset not in ['hc2','vidstg']:raise ValueError('Unsupported dataset')
    for record in base:
        if record['dataset']!=dataset:raise ValueError('Mixed dataset')
        meta=record['metadata']
        key=Path(meta['video_path']).name if dataset=='hc2'else str(meta['index'])
        if key not in annotations:
            if dataset=='hc2':continue  # Membership only; never a GT score filter.
            raise ValueError('Missing official VidSTG member')
        if key in seen:raise ValueError('Repeated official member')
        item=annotations[key];seen.add(key)
        caption=item['caption']if dataset=='hc2'else item['sentence']['description']
        if normalized(meta['caption'])!=normalized(caption):raise ValueError('Official query text differs')
        if (meta['width'],meta['height'])!=(item['width'],item['height']):raise ValueError('Official input dimensions differ')
        if dataset=='hc2'and meta['frame_count']!=item['img_num']:raise ValueError('Official HC frame count differs')
        if dataset=='hc2':
            begin=item['st_frame']-1;end=begin+len(item['bbox'])-1
            boxes={str(begin+i):[x,y,min(x+w,meta['width']),min(y+h,meta['height'])]for i,(x,y,w,h)in enumerate(item['bbox'])}
        else:
            if Path(item['vid']).stem!=Path(str(meta['source'])).stem:raise ValueError('Official parent video differs')
            if meta['start_frame']!=item['used_segment']['begin_fid']or meta['end_frame']!=item['used_segment']['end_fid']+1:raise ValueError('Official input window differs')
            if item['frame_count']!=meta['end_frame']-meta['start_frame']or meta['frame_count']<meta['end_frame']:raise ValueError('Official segment length differs')
            begin=item['ori_temp_gt']['begin_fid'];end=min(item['ori_temp_gt']['end_fid'],item['used_segment']['end_fid'])
            boxes={str(begin+i):[b['xmin'],b['ymin'],b['xmax'],b['ymax']]for i,b in enumerate(item['target_bboxs'])}
        if not 0<=begin<end or len(boxes)!=end-begin+1:raise ValueError('Invalid official trajectory span')
        result.append(record);labels.append(dict(id=record['id'],dataset=dataset,parent=meta['source'],interval=[begin,end],boxes=boxes))
    if seen!=set(annotations):raise ValueError('Base manifest does not cover the complete official evaluation')
    if len({r['id']for r in result})!=len(result):raise ValueError('Repeated query ID')
    return result,labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['hc2','vidstg'])
    for key in ['annotations','base-manifest','manifest','labels']:p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    if a.manifest.exists()or a.labels.exists():raise FileExistsError('Do not overwrite a frozen manifest or labels')
    annotations=json.loads(a.annotations.read_text());base=[json.loads(x)for x in a.base_manifest.read_text().splitlines()if x.strip()]
    rows,labels=prepare(a.dataset,annotations,base)
    for path,data in [(a.manifest,rows),(a.labels,labels)]:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n'for x in data))
    print(json.dumps(dict(status='prepared_offline_only',queries=len(rows),annotations_sha256=hashlib.sha256(a.annotations.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),model_calls=0)))

if __name__=='__main__':main()
