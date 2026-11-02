"""R50 image-only adapter, never raw XYZ or whole-scene reconstruction.

Import is inert. The authorized outer caller must pin frames/SEALED.json.
Every actual call verifies its case, original RGB file and mask/camera cache
bytes; camera content is not parsed. Crops receive the original mask array and
native detector box, the only two object fields used by visualZ.crops.
"""
from pathlib import Path
import functools, hashlib, json
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


@functools.lru_cache(maxsize=2)
def frame_seal(folder):
    seal=read(Path(folder)/'SEALED.json')
    assert seal['state']=='R50_FULL5912_ORIGINAL_R12_MASKS_CAMERA_COMPLETE'
    assert seal['n']==5912
    return seal


def image_only_observed(row,runroot,*,with_masks=True):
    """Return (frame RGB, minimal crop scene, original three-SHA binding).

    Point and source-box provenance remains in the already sealed prepared
    stage. It is not re-opened for an image-only computation. This function
    does not inspect row.pointpath, row.metapath or any GT/annotation fields.
    """
    folder=Path(runroot)/'frames';seal=frame_seal(str(folder));sid=row['id']
    casepath=folder/'cases_PRIVATE'/(sid+'.json')
    maskpath=folder/'masks_PRIVATE'/(sid+'.npz')
    camerapath=folder/'camera_PRIVATE'/(sid+'.json')
    actual={}
    for path in (casepath,maskpath,camerapath):
        member=path.relative_to(folder).as_posix()
        actual[path]=sha(path)
        assert actual[path]==seal['files'][member],str(path)
    case=read(casepath)
    assert case['id_PRIVATE']==sid and case['platform']==row['platform']
    assert actual[maskpath]==case['mask_cache_sha256']
    assert actual[camerapath]==case['camera_cache_sha256']
    assert sha(row['imagepath'])==case['image_sha256']
    with np.load(maskpath,allow_pickle=False) as stored:
        H,W=map(int,stored['original_hw'])
        if with_masks:masks=stored['masks_bool'].copy();boxes=np.asarray(stored['boxes_xyxy'],np.float64).copy()
    assert H>0 and W>0 and list(map(int,case['original_hw']))==[H,W]
    objects=[]
    if with_masks:
        assert masks.dtype==np.bool_ and masks.ndim==3 and masks.shape[1:]==(H,W)
        assert boxes.shape==(len(masks),4) and np.isfinite(boxes).all()
        objects=[{'mask_bool':masks[m],'detector_box_xyxy':boxes[m]} for m in range(len(masks))]
    from PIL import Image
    rgb=np.array(Image.open(row['imagepath']).convert('RGB'))
    assert rgb.dtype==np.uint8 and rgb.shape==(H,W,3)
    frame={'rgb':rgb}
    scene={'objects':objects,'image_size_wh':[W,H]}
    binding={name:case[name] for name in ('image_sha256','point_sha256','mask_cache_sha256')}
    return frame,scene,binding
