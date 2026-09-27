"""Synthetic, CPU-only demonstration; no backbone, dataset, or downloads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2

torch.set_num_threads(2)
torch.manual_seed(17)
model = Desta3DAdapterV2(in_channels=12, query_dim=10, hidden_dim=8).eval()
visual = torch.randn(1, 4, 3, 2, 12)
caption_tokens = torch.randn(1, 5, 10)
caption_mask = torch.ones(1, 5, dtype=torch.bool)
times = torch.tensor([[0., .4, 1.7, 3.]])
with torch.no_grad():
    result = model(visual, caption_tokens, caption_mask, frame_times=times)
for name in ('updated_tokens_spatial', 'updated_tokens_event', 'referent_logits', 'event_logits'):
    print(name, tuple(result[name].shape))
print('This is a synthetic interface example, not a task accuracy result.')

