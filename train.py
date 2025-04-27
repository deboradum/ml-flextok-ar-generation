import time
import torch

from trainUtils import TrainArgs, parse_config, get_net, get_optimizer, get_loaders

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def validate(
    flextok: torch.nn.Module,
    ar_net: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
):
    flextok.eval()
    ar_net.eval()

    running_loss = 0.0
    with torch.no_grad:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # TODO

    return running_loss / len(loader)


def train(
    config: TrainArgs,
    flextok: torch.nn.Module,
    ar_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    warmup_lr = config.warmup_learning_rate
    initial_lr = config.learning_rate
    train_steps = 0
    running_loss = 0.0
    s = time.time()

    for e in config.epochs:
        if e < train_args.warmup_epochs:
            # Linear warmup
            lr = warmup_lr + (initial_lr - warmup_lr) * (e / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

        for X, y in train_loader:
            optimizer.zero_grad()

            with torch.no_grad:
                tokens = flextok.tokenize(X)
            z_indices = tokens.reshape(tokens.shape[0], -1)
            c_indices = y.reshape(-1)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = ar_net(
                    cond_idx=c_indices, idx=z_indices[:, :-1], targets=z_indices
                )
            loss.backward()
            if config.gradient_clipping_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(ar_net.parameters(), config.gradient_clipping_norm)
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

            if train_steps % config.log_every == 0:
                taken = time.time() - s
                avg_loss = running_loss / config.log_every

                # todo: wandb log avg loss

                running_loss = 0.0
                s = time.time()

        val_loss = validate(flextok, ar_net, val_loader)
        # todo: wandb log val loss

    return validate(flextok, ar_net, test_loader)


if __name__ == "__main__":
    assert torch.cuda.is_available()

    model_args, train_args = parse_config("c2i_config.yaml")
    flextok, ar_net = get_net(model_args)
    optimizer, scheduler = get_optimizer(train_args)
    train_loader, val_loader, test_loader = get_loaders(train_args)

    train(
        config=train_args,
        flextok=flextok,
        ar_net=ar_net,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
