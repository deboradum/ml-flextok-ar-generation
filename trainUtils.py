import yaml
import torch

import torch.optim as optim
from dataclasses import dataclass

from flextok.flextok_wrapper import FlexTokFromHub
from autoregressive.models import ModelArgs, Transformer

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class TrainArgs:
    model: str = "AR_49M"
    epochs: int = 300
    warmup_epochs: int = 30
    warmup_learning_rate: float = 1e-6
    learning_rate_schedule: str = "cosine"
    optimizer: str = "AdamW"
    beta_1: float = 0.9
    beta_2: float = 0.95
    batch_size: int = 1024
    learning_rate: float = 1.2e-3
    final_learning_rate: float = 1.2e-5
    weight_decay: float = 0.05
    gradient_clipping_norm: float = 1.0
    log_every: int = 100
    flextok_model: str = "EPFL-VILAB/flextok_d18_d28_dfn"


def parse_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_args = ModelArgs(**config["modelArgs"])
    train_args = TrainArgs(**config["trainArgs"])

    return model_args, train_args


def get_net(model_config: ModelArgs, train_config: TrainArgs):
    ar_net = Transformer(model_config)

    flextok_net = (
        FlexTokFromHub.from_pretrained(train_config.flextok_model)
        .to(device)
        .eval()
    )
    for param in flextok_net.parameters():
        param.requires_grad = False

    return flextok_net, ar_net


def get_optimizer(config: TrainArgs, net):
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(),
            lr=config.warmup_learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta_1, config.beta_2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.epochs-config.warmup_epochs,
            eta_min=config.final_learning_rate,
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def get_loaders(config: TrainArgs):
    return
