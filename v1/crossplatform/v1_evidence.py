"""Build the two role-specific anchor sets and frozen decoder inputs."""

import numpy as np
from PIL import Image
import torch

from crossplatform.candidate_frontend import orient_box


def role_masks(caption, graph, document, offsets, valid):
    entities = graph['entities']
    direct_edges = [edge for edge in graph['relations']
                    if 'T' in (edge['subject'], edge['reference'])]
    ref = entities[1] if len(entities) > 1 else None
    predicate = (0, 0)
    if len(direct_edges) == 1:
        edge = direct_edges[0]
        other = edge['reference'] if edge['subject'] == 'T' else edge['subject']
        ref = next((entity for entity in entities if entity['role'] == other), ref)
        words = document['sentences'][edge['sentence']]['words']
        word = next((word for word in words if word['id'] == edge['governor_word']), None)
        if word and word.get('start_char') is not None and word.get('end_char') is not None:
            predicate = (word['start_char'], word['end_char'])
    spans = (entities[0]['span'] if entities else (0, len(caption)),
             ref['span'] if ref else (0, 0), predicate)
    start, end = offsets.T
    content = (end > start) & (start < len(caption))
    masks = np.stack([content & (start < hi) & (end > lo) & (hi > lo) for lo, hi in spans])
    if not masks[0].any():
        masks[0] = content if content.any() else valid
    return masks, ref is not None and masks[1].any()


def visible_pixels(xyz, sampled, target, view, image, calibration):
    from camera_projection import CameraSpec, project_model_points, read_camera_allowlist

    fields = read_camera_allowlist(calibration)
    camera = CameraSpec.from_camera_fields(target, fields, waymo_view_suffix=view,
                                           missing_distortion_policy='assume_zero_declared')
    with Image.open(image) as im:
        size = im.size
    projection = project_model_points(np.concatenate((xyz, sampled)), camera, size)
    valid = projection['in_image']
    projected = np.where(valid[:, None], projection['uv'], 0.)
    width = (size[0] + 3) // 4
    cells = np.floor(projected / 4).astype(int)
    flat = cells[:, 1] * width + cells[:, 0]
    depth = projection['depth']
    counts = len(xyz)
    buffer = np.full(width * ((size[1] + 3) // 4), np.inf)
    np.minimum.at(buffer, flat[counts:][valid[counts:]], depth[counts:][valid[counts:]])
    visible = valid[:counts] & (depth[:counts] <= buffer[flat[:counts]] + .30 + .02 * depth[:counts])
    uv = (projected[:counts] / np.asarray(size)).astype(np.float32)
    return uv, visible


def subject_anchors(boxes, probability, sampled):
    from grounding158 import select_grounding_box
    from reader158 import overlaps

    first, utility = select_grounding_box(boxes, probability)
    order = np.argsort(-utility, kind='stable')
    kernel = overlaps(boxes)
    keep = [first]
    for index in order[1:]:
        if (kernel[index, keep] <= .35).all():
            keep.append(int(index))
            if len(keep) == 4:
                break
    if len(keep) < 4:
        keep.extend(int(index) for index in order if index not in keep)
    selected = np.array(keep[:4], dtype=np.int64)
    log = np.log(np.maximum(utility[selected], 1e-80))
    prior = (log - log[0]).clip(-20., 0.).astype(np.float32)
    anchors = np.stack([orient_box(sampled, boxes[index]) for index in selected])
    return anchors, prior


def reference_anchors(boxes, queries, tokens, roles, has_reference, sampled):
    device = queries.device
    mask = torch.as_tensor(roles[None, 1], device=device).float()
    similarity = (queries @ tokens.transpose(1, 2)) / .07
    reference = (similarity * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
    reference = torch.where(torch.as_tensor([[has_reference]], device=device), reference, 0.)
    prior = torch.as_tensor(boxes[None], device=device)
    prior = torch.cat((prior[..., :3], prior[..., 3:].clamp_min(.03)), -1)
    rank = reference[:, None].expand(-1, 2, -1).clone()
    selected = []
    for _ in range(4):
        index = rank.argmax(-1)
        selected.append(index)
        anchor = prior.gather(1, index[..., None].expand(-1, -1, 6))
        dist = (prior[:, None, :, :3] - anchor[:, :, None, :3]) / (anchor[:, :, None, 3:] * .55 + .4)
        rank = rank - 4. * torch.exp(-.5 * dist.square().sum(-1))
        rank.scatter_(-1, index[..., None], -torch.inf)
    indices = torch.stack(selected, -1)[:, 1]
    anchors = np.stack([orient_box(sampled, boxes[index]) for index in indices[0].cpu().numpy()])
    return anchors, reference.gather(-1, indices)


def decoder_evidence(frame, native, roles, has_reference, valid, image, calibration):
    device = native['xyz'].device
    sampled = frame.sampled_points
    query_boxes = frame.boxes
    uv, visible = visible_pixels(native['xyz'][0].cpu().numpy(), sampled, frame.target,
                                 frame.view_index, image, calibration)
    subject_boxes, subject_prior = subject_anchors(query_boxes, frame.probability, sampled)
    reference_boxes, reference_prior = reference_anchors(
        query_boxes, native['queries'], native['tokens'], roles, has_reference, sampled)
    return dict(
        points=native['points'], xyz=native['xyz'], text=native['text'],
        uv=torch.as_tensor(uv[None], device=device).half(),
        visible=torch.as_tensor(visible[None], device=device),
        valid=torch.as_tensor(valid[None], device=device),
        role_mask=torch.as_tensor(roles[None], device=device),
        has_reference=torch.as_tensor([has_reference], device=device),
        subject_boxes=torch.as_tensor(subject_boxes[None], device=device),
        reference_boxes=torch.as_tensor(reference_boxes[None], device=device),
        base_logit=torch.as_tensor(subject_prior[None], device=device),
        ref_logit=reference_prior)
