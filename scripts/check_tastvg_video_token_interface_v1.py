"""CPU/source-only interface readback; no weights, pixels, GT or inference."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_accumulation_common_v1 import BASE, PUBLIC, sha, write


def run():
    specifications = [
        ('external/UniversalVTG/feature_extraction/extract_visual_features.py',
         'class PEFeatureExtractor', 'image_features = self.model.encode_image(pixel_values)',
         'The active extractor calls image encoding independently for each batch frame.'),
        ('external/UniversalVTG/perception_models/core/vision_encoder/pe.py',
         '    def forward_features(', '        batch, _, h, w = x.shape',
         'Spatial patch tokens have a 4-D image input; the batch dimension is not a temporal attention axis.'),
        ('external/UniversalVTG/perception_models/core/vision_encoder/pe.py',
         '    def forward(self, x: torch.Tensor, **kwargs):', '        x = self._pool(x)',
         'The visual pool precedes the shared projection returned by the active encoder.'),
        ('external/UniversalVTG/perception_models/core/vision_encoder/pe.py',
         '    def encode_video(', '        video_feats = video_feats.mean(dim=1)',
         'The video convenience method independently encodes b*n frames and averages them; it does not contextualize spatial tokens in time.'),
        ('external/UniversalVTG/universal_vtg_inference.py',
         '    def encode_video(', "        features_dt = features.T.contiguous().float()",
         'The active UniversalVTG video input contract is (feature_dim, T), with spatial coordinates already pooled away.'),
        ('external/UniversalVTG/libs/modeling/model.py',
         '    def _forward_earlyfusion(', '        vid, vid_masks = self.project_video(vid, vid_masks)',
         'Its temporal/fused hidden states are (batch, channels, time), without the spatial patch axis needed to distinguish candidate tube boxes.'),
        ('external/TA-STVG/models/vidswin/video_swin_transformer.py',
         'class VideoSwinTransformerBackbone', "            out[idx] = rearrange(o, 'b c t h w -> (b t) c h w')",
         'VideoSwin does preserve spatial positions after temporal blocks, but its Kinetics-pretrained visual features are not a supplied event-word dual-encoder cosine space.'),
        ('external/TA-STVG/models/pipeline.py',
         '        self.vid = vidswin_model(', '        self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)',
         'TA-STVG projects motion features into its grounding hidden space; equal hidden dimensionality does not supply the requested pretrained lexical cosine interface.'),
        ('external/TA-STVG/models/grounding_model/classifier.py',
         'class SpatialActivation', '            query, att_map = self.layer_ca[i](query, x)',
         'Native ASA uses learned cross-attention, not the requested event-token/patch cosine with a pretrained logit scale.'),
    ]
    records = []
    for relative, start, anchor, conclusion in specifications:
        file = ROOT / relative
        lines = file.read_text().splitlines()
        start_line = next(i for i, line in enumerate(lines) if start in line)
        line = next(i for i in range(start_line, len(lines)) if anchor in lines[i])
        records.append(dict(path=relative, sha256=sha(file), start_line=start_line+1,
                            anchor_line=line+1, anchor=lines[line].strip(), conclusion=conclusion))
    report = dict(status='interface_unavailable_on_inspected_existing_encoders',
                  checks=records, P2_prediction_cells=0,
                  P2_candidate_scores=0, GT_read=False, model_weights_read=False,
                  GPU_used=False, no_download=True,
                  available=['PE-Core static image patch features',
                             'UniversalVTG spatially pooled temporal hidden states',
                             'VideoSwin temporally modeled spatial features'],
                  missing='A supplied trained event-lexical/temporally-contextualized spatial-token cosine space with native logit scale',
                  decision='Retain interface investigation; do not substitute static CLIP or train/invent a projection. No P2 quality claim.',
                  time=time.time())
    write(BASE / 'TOKEN_P2_INTERFACE.json', report)
    write(PUBLIC / 'TOKEN_P2_INTERFACE.json', report)
    print('P2 interface checked; zero qualification inference', len(records), flush=True)


if __name__ == '__main__':
    run()
