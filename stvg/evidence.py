"""Frozen Source grids and the same eight selected ReferDINO masks.

The constructors accept raw tensors from native inference or a validated cache.
No historical experiment, temporal interval, model or annotation is accessed.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from stvg._spatial_math import cpu, scoped_single_thread
AFFINITY_TOPK=20
EPS=1e-15

def _require(value, message):
    if not value:
        raise ValueError(message)

def _cpu_tensor(value, dtype=None):
    return torch.as_tensor(value).detach().to(device='cpu', dtype=dtype).clone()

def _valid_grid_shape(pos, mask, visual_count):
    """Infer row-major H,W from y/x position halves, ignoring padded cells.

    Every candidate factorization must have distinct row y vectors, distinct
    column x vectors, and constant valid y/x values within each row/column.
    Ambiguous or unsupported layouts fail instead of silently reshaping tokens.
    """
    candidates = []
    for height in range(1, visual_count + 1):
        if visual_count % height:
            continue
        width = visual_count // height
        okay = True; informative = False
        for view_pos, view_mask in zip(pos, mask):
            # A frame with the most valid cells is enough to constrain one view;
            # all remaining frames are checked again after joint inference.
            count = (~view_mask[:, :visual_count]).sum(1)
            frame = int(count.argmax())
            valid = ~view_mask[frame, :visual_count].reshape(height, width)
            if not bool(valid.any()):
                continue
            informative = True
            grid = view_pos[:visual_count, frame].reshape(height, width, 256)
            ys, xs = [], []
            for row in range(height):
                values = grid[row, valid[row], :128]
                if len(values):
                    if not torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0):
                        okay = False; break
                    ys.append(values[0])
            if not okay:
                break
            for column in range(width):
                values = grid[valid[:, column], column, 128:]
                if len(values):
                    if not torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0):
                        okay = False; break
                    xs.append(values[0])
            if not okay:
                break
            if (len(torch.unique(torch.stack(ys), dim=0)) != len(ys) or
                    len(torch.unique(torch.stack(xs), dim=0)) != len(xs)):
                okay = False; break
        if okay and informative:
            candidates.append((height, width))
    _require(len(candidates) == 1, f'Cannot uniquely validate visual grid H,W: {candidates}')
    return candidates[0]

def extract_visual_grids(spatial_inputs, frame_ids=None):
    """Return physical-node [T,Nvisual,256] features and valid cells on CPU."""
    if frame_ids is None:
        frame_ids = spatial_inputs['frame_ids']
        spatial_inputs = spatial_inputs['spatial_inputs']
    frame_ids = [int(x) for x in frame_ids]
    _require(len(frame_ids) >= 2 and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
             'Physical frame IDs must be strictly increasing')
    views = spatial_inputs
    _require(len(views) == 2, 'Exactly two native offsets are required')
    features, positions, masks, counts = [], [], [], []
    for offset, item in enumerate(views):
        feature = _cpu_tensor(item['encoded_feature'], torch.float32)
        pos = _cpu_tensor(item['encoded_pos'], torch.float32)
        mask = _cpu_tensor(item['encoded_mask'], torch.bool)
        expected_frames = len(frame_ids[offset::2])
        _require(feature.ndim == 3 and feature.shape[-1] == 256 and feature.shape[1] == expected_frames,
                 'Cached feature dimensions disagree with physical offset')
        _require(pos.shape == feature.shape and mask.shape == (expected_frames, feature.shape[0]),
                 'Position/mask dimensions disagree with cached feature')
        visual_flags = pos.abs().sum(-1).any(-1)
        count = int(visual_flags.sum())
        _require(count > 0 and bool(visual_flags[:count].all()) and not bool(visual_flags[count:].any()),
                 'Spatial visual positions must form one nonzero prefix; text positions must be zero')
        features.append(feature); positions.append(pos); masks.append(mask); counts.append(count)
    _require(counts[0] == counts[1], 'Native offsets have incompatible visual token counts')
    visual_count = counts[0]
    height, width = _valid_grid_shape(positions, masks, visual_count)
    for view_pos, view_mask in zip(positions, masks):
        for frame in range(view_pos.shape[1]):
            valid = ~view_mask[frame, :visual_count].reshape(height, width)
            grid = view_pos[:visual_count, frame].reshape(height, width, 256)
            for row in range(height):
                values = grid[row, valid[row], :128]
                _require(not len(values) or torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0),
                         'A frame violates the verified row-major y grid')
            for column in range(width):
                values = grid[valid[:, column], column, 128:]
                _require(not len(values) or torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0),
                         'A frame violates the verified row-major x grid')
    output, valid = [], []
    for node in range(len(frame_ids)):
        view, local = node % 2, node // 2
        v = ~masks[view][local, :visual_count]
        f = features[view][:visual_count, local]
        _require(bool(torch.isfinite(f[v]).all()), 'Nonfinite valid Source features')
        output.append(torch.where(v[:, None], f, torch.zeros_like(f)))
        valid.append(v)
    output, valid = torch.stack(output), torch.stack(valid)
    meta = dict(frames=len(frame_ids), visual_tokens=visual_count, grid_height=height, grid_width=width,
                channels=256, text_tokens_per_offset=[x.shape[0] - visual_count for x in features],
                offset_frames=[x.shape[1] for x in features], physical_interleave='view=t%2, local=t//2',
                valid_visual_cells_min=int(valid.sum(1).min()), valid_visual_cells_max=int(valid.sum(1).max()),
                visual_features='cached post-fusion appearance-position tokens; text/motion token slots excluded',
                pixel_correspondence_training_claim=False)
    return output, valid, frame_ids, (height, width), meta



class SourceEvidence:
    """Interval-independent, detached CPU Source features and teacher seeds."""
    @scoped_single_thread
    def __init__(self, features, valid, frame_ids, positions8, seeds, *, selected_query=None):
        self.features=cpu(features,torch.float32);self.valid=cpu(valid,torch.bool)
        self.frame_ids=cpu(frame_ids,torch.long);self.seeds=cpu(seeds,torch.float64)
        self.positions=[int(x)for x in positions8];self.positions8=self.positions
        _require(self.seeds.ndim==3 and len(self.seeds)==8,'Eight Source-grid soft masks required')
        self.height,self.width=self.seeds.shape[-2:];T,N,D=self.features.shape
        _require(D==256 and N==self.height*self.width and self.valid.shape==(T,N),'Source feature grid mismatch')
        _require(self.frame_ids.shape==(T,)and bool((self.frame_ids[1:]>self.frame_ids[:-1]).all()),'Increasing physical frames required')
        _require(len(set(self.positions))==8 and all(0<=p<T for p in self.positions),'Eight distinct positions required')
        _require(bool(torch.isfinite(self.features).all())and bool(torch.isfinite(self.seeds).all()),'Finite evidence required')
        _require(bool(((self.seeds>=0)&(self.seeds<=1)).all()),'Masks must be soft probabilities')
        self.seeds*=self.valid[self.positions].reshape_as(self.seeds)
        self.observed=torch.zeros(T,dtype=torch.bool);self.observed[self.positions]=True
        self.metadata=dict(schema='boxfixed_source_evidence_v1',GT_read=False,selected_native_query=selected_query,
            grid_height=self.height,grid_width=self.width,temporal_interval_used=False,
            affinity_topk=20,affinity_temperature=.1,source_features='post-fusion visual token prefix; no text/motion token slots')
        self.GT_read=False

    from_arrays=classmethod(lambda cls,features,valid,frame_ids,positions8,seeds,selected_query=None:
        cls(features,valid,frame_ids,positions8,seeds,selected_query=selected_query))

    @classmethod
    @scoped_single_thread
    def from_native_masks(cls,features,valid,frame_ids,positions8,mask_logits,teacher_image_size,grid_shape,*,selected_query=None):
        masks=cpu(mask_logits,torch.float32)
        _require(masks.ndim==3 and len(masks)==8,'Selected native mask logits must be [8,Hm,Wm]')
        _require(len(teacher_image_size)==2 and len(grid_shape)==2,'Explicit teacher and Source grid shapes required')
        full=F.interpolate(masks[:,None],size=tuple(teacher_image_size),mode='bilinear',align_corners=False).sigmoid()
        seeds=F.interpolate(full,size=tuple(grid_shape),mode='area').squeeze(1).double()
        return cls(features,valid,frame_ids,positions8,seeds,selected_query=selected_query)

    @classmethod
    @scoped_single_thread
    def from_capture(cls,spatial_inputs,frame_ids,positions8,mask_logits,teacher_image_size,*,selected_query=None):
        features,valid,frames,shape,meta=extract_visual_grids(spatial_inputs,frame_ids)
        obj=cls.from_native_masks(features,valid,frames,positions8,mask_logits,teacher_image_size,shape,selected_query=selected_query)
        obj.metadata.update(meta);return obj

    def state_dict(self):
        names=('features','valid','frame_ids','seeds','observed')
        return dict(schema='boxfixed_source_evidence_v1',metadata=dict(self.metadata),positions8=list(self.positions),
            **{name:getattr(self,name).clone()for name in names},GT_read=False)
