"""Keep the winning carrier's input box when returning the subject location."""

import torch

from features import encode
from crossplatform.candidate_frontend import sensor_frame_box
from ground_instance import SourceInstanceGrounder, tensorize
from model import NATIVE
from train import SELECTION, decisions


def retained_subject(output, data, indices):
    pool = torch.cat((data['anchors'], output['proposal'][:,NATIVE:]), dim=1)
    row = torch.arange(len(indices), device=indices.device)
    return torch.where((indices==data['v1_index'])[:,None], data['v1_box'], pool[row,indices])


class RetainedGeometryGrounder(SourceInstanceGrounder):
    """Reuse source-selected role scores and return the subject's unmodified carrier."""

    @torch.inference_mode()
    def predict(self, *, readout=None, **frame):
        readout = readout or self.source_readout
        if readout not in SELECTION:
            raise ValueError(f'Unknown readout: {readout}')
        with self._lock:
            feature, _ = encode(self.frozen, frame)
            data = tensorize(feature)
            output = self.head(data)
            _, references, known, indices = decisions(output,data)
            subject = retained_subject(output,data,indices[readout])[0].cpu().numpy()
            reference = references[readout][0].cpu().numpy() if known[readout][0] else None
        view = int(feature['view_index'])
        subject = sensor_frame_box(subject,frame['target'],view)
        if reference is not None:
            reference = sensor_frame_box(reference,frame['target'],view)
        return dict(subject_box=subject.tolist(),reference_box=None if reference is None else reference.tolist(),
                    readout=readout,frame='target_lidar',length_unit='m',yaw_unit='rad')
