from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # paths
    BASE_DIR: str = "/home/UFAD/zhou.zhuoyang/hacks/AIDER/image_classifier"
    DATA_DIR: str = "/home/UFAD/zhou.zhuoyang/hacks/AIDER/data/AIDER"  
    OUT_DIR:  str = "/home/UFAD/zhou.zhuoyang/hacks/AIDER/image_classifier/out"

    # data & split
    image_size: int = 256
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # training
    model_name: str = "convnext_tiny" 
    epochs: int = 200             
    batch_size: int = 256
    lr: float = 5e-5
    weight_decay: float = 0.05

    # Loss & Enhancement
    use_focal: bool = False
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # AMP Accuracy
    amp_dtype: str = "bf16"  # "bf16" or "fp16"

    num_workers: int = 32
    ema: bool = True

    # scheduler & early stopping
    use_scheduler: bool = True
    warmup_epochs: int = 3
    cosine_tmax: int = 60         # Cosine annealing main cycle
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0

    early_stop: bool = True
    early_stop_patience: int = 10  

    # logging/checkpoints
    ckpt_best: str = "best.pt"
    ckpt_last: str = "last.pt"

    # classes
    # classes = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
    classes = ["flooded_areas", "normal", "traffic_incident"]
    
    def out_dir(self) -> Path:
        p = Path(self.OUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
