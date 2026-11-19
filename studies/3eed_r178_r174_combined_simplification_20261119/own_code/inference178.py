"""The registered R174 inference profiles for frozen features and candidate slots."""
import torch

from scope178 import P177,PRIOR,RESULTS,read,sha
from model178 import CombinedDecoder


class R174Inference(CombinedDecoder):
    def __init__(self,profile='preferred'):
        super().__init__()
        self.recipe = read(RESULTS/'DECISION.json')['profiles'][profile]['variant']

    @torch.inference_mode()
    def forward(self,evidence):
        choices,logits = self.run_recipe(evidence,self.recipe)
        i = torch.arange(len(choices),device=choices.device)
        left = evidence['subject_boxes'][i,choices[:,0]]
        right = evidence['reference_boxes'][i,choices[:,1]]
        return dict(choices=choices,boxes=torch.stack((left,right),dim=1),subject_logits=logits)


def load_profile(source,device,profile='preferred'):
    expected = read(P177/'EXECUTION_LOCK.json')['source_models'][source]
    path = PRIOR/source/'full.pth'
    assert sha(path)==expected
    model = R174Inference(profile)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False)
