import torchvision.transforms as T
from models import build_model
import os
import torch
import misc as utils
import numpy as np
import torch.nn.functional as F
import argparse
import matplotlib.colors
from torchvision.io import read_video
import torchvision.transforms.functional as Func
from ruamel.yaml import YAML
from easydict import EasyDict
from misc import nested_tensor_from_videos_list
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
from rich.progress import track
import imageio.v3 as iio
import cv2
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# colormap
color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

def main(args):
    model = load_model(args)
    infer(model, args.video, args.text, args)

def infer(model, video, text, args):
    assert os.path.exists(video)
    video_name = video.split('/')[-1]
    exp = " ".join(text.lower().split())

    save_name = args.save_name if  args.save_name is not None else video_name
    if not save_name.endswith('.mp4'):
        save_name += '.mp4'

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_video = os.path.join(output_dir, save_name)

    video_frames, _, info = read_video(video, pts_unit='sec')  # (T, H, W, C)
    frames = []
    for i in range(0, len(video_frames), args.frame_step):
        source_frame = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
        frames.append(source_frame) #(C H W)

    video_len = len(frames)

    frames_ids = [x for x in range(video_len)]
    imgs = []
    for t in frames_ids:
        img = frames[t]
        origin_w, origin_h = img.size
        imgs.append(transform(img))

    imgs = torch.stack(imgs, dim=0).to(args.device)
    samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
    img_h, img_w = imgs.shape[-2:]
    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
    target = {"size": size}

    print("begin inference")
    with torch.no_grad():
        with autocast(args.enable_amp):
            outputs = model.infer(samples, [exp], [target])

    pred_logits = outputs["pred_logits"][0]  # [t, q, k]
    pred_masks = outputs["pred_masks"][0]  # [t, q, h, w]
    pred_boxes = outputs["pred_boxes"][0]  # [t, q, 4]

    # according to pred_logits, select the query index
    pred_scores = pred_logits.sigmoid()  # [t, q, k]
    pred_scores = pred_scores.mean(0)  # [q, K]
    max_scores, _ = pred_scores.max(-1)  # [q,]
    _, max_ind = max_scores.max(-1)  # [1,]
    max_inds = max_ind.repeat(video_len)
    pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
    pred_masks = pred_masks.unsqueeze(0)
    pred_boxes = pred_boxes[range(video_len), max_inds].cpu().numpy()  # [t, 4]

    # unpad
    pred_masks = pred_masks[:, :, :img_h, :img_w].cpu()
    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
    pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).cpu().numpy()  # 0.5

    print("saving")

    color = "#DC143C"
    color = (np.array(matplotlib.colors.hex2color(color)) * 255).astype('uint8')

    save_imgs = []
    for t, img in enumerate(frames):
        # draw mask
        img = vis_add_mask(img, pred_masks[t], color, args.mask_edge_width)

        draw = ImageDraw.Draw(img)
        draw_boxes = pred_boxes[t][None]
        draw_boxes = rescale_bboxes(draw_boxes, (origin_w, origin_h)).tolist()

        # draw box
        if args.show_box:
            xmin, ymin, xmax, ymax = draw_boxes[0]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color), width=5)

        # save
        save_dir = os.path.join(output_dir, '{:05d}.png'.format(t))
        if args.save_images:
            img.save(save_dir)
        save_imgs.append(np.asarray(img).copy())

    fps = info['video_fps'] / args.frame_step
    iio.imwrite(save_video, save_imgs, fps=fps)
    print("result video saved to {}".format(save_video))

# Post-process functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    b = np.stack([
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h
    ], axis=1)
    return b

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

def vis_add_mask(img, mask, color, edge_width=3):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    # Increase the edge width using dilation
    kernel = np.ones((edge_width, edge_width), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_mask = mask_dilated & ~mask

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img[edge_mask] = color
    origin_img = Image.fromarray(origin_img)
    return origin_img

def load_model(args):
    model, _, _ = build_model(args)
    device = args.device
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Inference')
    parser.add_argument('--config_path', default='configs/ytvos_swinb.yaml',
                        help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True,
                        help="The checkpoint path")
    parser.add_argument("--frame_step", default=1, type=int, help="Sampling interval of the video")
    parser.add_argument("--output_dir", default="output/demo")
    parser.add_argument("--video", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_images", action='store_true')
    parser.add_argument("--show_box", action='store_true')
    parser.add_argument("--mask_edge_width", default=6, type=int)
    parser.add_argument("--bar_height", default=80, type=int)
    parser.add_argument("--font_size", default=60, type=int)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--tracking_alpha", default=0.1, type=float)
    args = parser.parse_args()

    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)
    args.GroundingDINO.tracking_alpha = args.tracking_alpha
    main(args)