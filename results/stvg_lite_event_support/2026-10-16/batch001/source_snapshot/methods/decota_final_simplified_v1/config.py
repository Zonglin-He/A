"""One explicit configuration, without inherited experimental overrides."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MethodConfig:
    direction: str
    spatial_lr: float
    temporal_lr: float
    center_fraction: float
    source_dataset: str
    checkpoint: str
    checkpoint_sha256: str
    spatial_steps: int = 10
    temporal_steps: int = 5
    planned_observations: int = 4
    spatial_selection: str = "best"
    temporal_selection: str = "last"
    eta: float = 0.25
    prior_weight: float = 0.1
    epsilon: float = 1e-6
    margin: float = 0.2
    mode_weight: float = 1.0
    spatial_parameters: int = 1792
    temporal_parameters: int = 66306

    @classmethod
    def for_direction(cls, direction):
        if direction == "vid_to_hc1":
            return cls(direction, .005, .001, 1., "vidstg",
                       "checkpoints/TASTVG_VidSTG.pth",
                       "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83")
        if direction == "hc2_to_vid":
            return cls(direction, .05, .1, .5, "hcstvg2",
                       "checkpoints/TASTVG_HCSTVG2.pth",
                       "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036")
        raise ValueError("direction must be vid_to_hc1 or hc2_to_vid")

    def to_dict(self):
        return asdict(self)


EXPERT_SNAPSHOT = (".cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/"
                   "snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
EXPERT_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
