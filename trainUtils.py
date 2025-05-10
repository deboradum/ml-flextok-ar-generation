import os
import yaml
import torch

import torch.optim as optim
import torchvision.transforms as transforms

from torchvision import datasets
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split

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
    dataset: str = "imagenet"
    dataset_path: str = "./data/imagenet"
    checkpoint_path: str = "./checkpoints/c2i_ar49M"  # epoch and extension gets added automatically.
    load_from_path: str = "none"
    skip_epochs: int = -1


def parse_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_args = ModelArgs(**config["modelArgs"])
    train_args = TrainArgs(**config["trainArgs"])

    return model_args, train_args


def get_net(model_config: ModelArgs, train_config: TrainArgs):
    print("Preparing models")
    ar_net = Transformer(model_config).to(device)
    if os.path.isfile(train_config.load_from_path):
        print(f"Loading model weights from {train_config.load_from_path}")
        ar_net.load_state_dict(
            torch.load(train_config.load_from_path, map_location=device)
        )

    flextok_net = (
        FlexTokFromHub.from_pretrained(train_config.flextok_model)
        .to(device)
        .eval()
    )
    for param in flextok_net.parameters():
        param.requires_grad = False

    return flextok_net, ar_net


def get_optimizer(config: TrainArgs, net):
    print("Preparing optimizer")
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


# https://www.kaggle.com/datasets/dimensi0n/imagenet-256
def get_imagenet_loaders(batch_size, data_dir="data/imagenet"):
    transform = transforms.Compose(
        [
            # TODO: Additionally we produce 10 random crops per image prior to tokenization(Sun et al., 2024)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_test_size = len(full_dataset) - train_size
    val_size = val_test_size // 2
    test_size = val_test_size - val_size

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_size, val_test_size]
    )
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


# https://huggingface.co/datasets/deboradum/GeoGuessr-countries-large
def get_geoguessr_loaders(batch_size, data_dir="data/geoguessr"):
    transform = transforms.Compose(
        [
            transforms.RandomCrop((900, 900)),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train/", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/valid/", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test/", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


def get_loaders(config: TrainArgs):
    print("Preparing data loaders")
    if config.dataset == "imagenet":
        return get_imagenet_loaders(config.batch_size, data_dir=config.dataset_path)
    elif config.dataset == "geoguessr":
        return get_geoguessr_loaders(config.batch_size, data_dir=config.dataset_path)
    else:
        raise NotImplementedError
