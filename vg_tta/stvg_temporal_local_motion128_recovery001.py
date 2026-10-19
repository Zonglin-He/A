"""Per-node short motion clips, original global appearance/text and time support."""
import hashlib,subprocess
import numpy as np
import torch
import torch.nn.functional as F


def clip_indices(ids,total):
    ids=np.asarray(ids,dtype=np.int64);assert len(ids)==65 and np.all(np.diff(ids)>0)and ids[0]>=0 and ids[-1]<total
    delta=np.arange(-3,5,dtype=np.int64)
    dense=np.clip(ids[:,None]+2*delta,ids[0],ids[-1]);sparse=np.empty_like(dense)
    for off in (0,1):
        seq=ids[off::2];sparse[off::2]=seq[np.clip(np.arange(len(seq))[:,None]+delta,0,len(seq)-1)]
    assert np.array_equal(dense[:,3],ids)and np.array_equal(sparse[:,3],ids)
    return {'DenseLocal':dense,'SparseLocal':sparse}


def decode_all(meta,root):
    from methods.stvg_referdino_opd_v1.runtime import sha
    exe=root/'.conda/tubedetr/bin/ffmpeg';assert sha(exe)=='bfec0e82c8497b0b979b1a7812ba1101da3646e8a3330bcb5537badb840a527e'
    assert sha(meta['video_path'])==meta['video_sha256']
    p=subprocess.run([str(exe),'-v','error','-i',meta['video_path'],'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    a=np.frombuffer(p.stdout,np.uint8).reshape(-1,meta['height'],meta['width'],3)
    assert len(a)>meta['frame_ids'][-1];return a


def normalize(frames,metadata):
    # Same per-frame make_batch formula; all resizing remains CPU and full image.
    width=min(int(224*metadata['width']/metadata['height']),int(224*1.4));out=[]
    for j in range(0,len(frames),32):
        x=torch.from_numpy(frames[j:j+32].copy()).permute(0,3,1,2).float()/255.
        x=F.interpolate(x,size=(224,width),mode='bilinear',align_corners=False,antialias=True)
        out.append((x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None])
    return torch.cat(out)


def properties():
    ids=np.rint(np.linspace(0,500,65)).astype(int);v=clip_indices(ids,501);checks=0
    for t in range(65):
        seq=ids[t%2::2].tolist();ix=seq.index(int(ids[t]))
        for j in range(8):
            assert v['DenseLocal'][t,j]==min(500,max(0,int(ids[t])+2*(j-3)))
            assert v['SparseLocal'][t,j]==seq[min(len(seq)-1,max(0,ix+j-3))];checks+=2
    return dict(status='pass',independent_index_checks=checks,center_exact=True,boundary_clamp_not_reorder=True,model_GT_run=False)
