"""Bounded CPU JPEG cache with exact original SAM2 async normalization."""
from collections import OrderedDict
import operator
from contextlib import contextmanager

class BoundedVideoFrameLoader:
    def __init__(self,img_paths,image_size,offload_video_to_cpu,img_mean,img_std,compute_device):
        if not offload_video_to_cpu:raise RuntimeError('bounded dispatcher only supports original CPU-video path')
        if not img_paths:raise RuntimeError('empty original frame sequence')
        self.img_paths=tuple(img_paths);self.image_size=image_size
        self.offload_video_to_cpu=offload_video_to_cpu;self.img_mean=img_mean;self.img_std=img_std;self.compute_device=compute_device
        self.images=OrderedDict();self.max_cached_frames=2;self.peak_cached_frames=0
        self.video_height=None;self.video_width=None;self.exception=None;self.thread=None
        self.__getitem__(0)
    def __len__(self):return len(self.img_paths)
    def __getitem__(self,index):
        index=operator.index(index)
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        if index in self.images:
            self.images.move_to_end(index);return self.images[index]
        from sam2.utils.misc import _load_img_as_tensor
        image,height,width=_load_img_as_tensor(self.img_paths[index],self.image_size)
        image-=self.img_mean;image/=self.img_std
        self.video_height=height;self.video_width=width
        self.images[index]=image
        while len(self.images)>self.max_cached_frames:self.images.popitem(last=False)
        self.peak_cached_frames=max(self.peak_cached_frames,len(self.images))
        return image

@contextmanager
def bounded_frame_loading():
    import sam2.utils.misc as misc
    original=misc.AsyncVideoFrameLoader;misc.AsyncVideoFrameLoader=BoundedVideoFrameLoader
    try:yield
    finally:misc.AsyncVideoFrameLoader=original
