import os
import time
import torch
import wandb
import shutil

from dataclasses import asdict

from trainUtils import TrainArgs, parse_config, get_net, get_optimizer, get_loaders

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TARGET_BATCH_SIZE = 1024


def validate(
    flextok: torch.nn.Module,
    ar_net: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
):
    flextok.eval()
    ar_net.eval()

    running_loss = 0.0
    num = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            tokens = flextok.tokenize(X)  # returns a list of [1, L] tensors
            tokens = torch.stack([toks.squeeze() for toks in tokens])  # list to (B, L)
            c_indices = y.reshape(-1)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = ar_net(
                    cond_idx=c_indices, idx=tokens[:, :-1], targets=tokens
                )
            running_loss += loss
            num += 1

    return running_loss / num


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

    accumulation_update_interval = TARGET_BATCH_SIZE // config.batch_size
    assert (
        accumulation_update_interval >= 1
    ), "accumulation_update_interval must be one or greater. (is {accumulation_update_interval})"
    print(
        f"Target batch size is {TARGET_BATCH_SIZE}, training batch size is {config.batch_size}. Updating model parameters every {accumulation_update_interval} steps."
    )

    for e in range(config.epochs):
        if e < train_args.warmup_epochs:
            print("Warmup epoch")
            # Linear warmup
            lr = warmup_lr + (initial_lr - warmup_lr) * (e / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

        if e < config.skip_epochs:
            print(f"Skipping epoch {e}")
            train_steps += len(train_loader)
            continue

        current_lr = optimizer.param_groups[0]["lr"]
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                tokens = flextok.tokenize(X)  # returns a list of [1, L] tensors
                tokens = torch.stack([toks.squeeze() for toks in tokens])  # list to (B, L)
            c_indices = y.reshape(-1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = ar_net(
                    cond_idx=c_indices, idx=tokens[:, :-1], targets=tokens
                )
                loss = loss / accumulation_update_interval
            loss.backward()

            if (i + 1) % accumulation_update_interval == 0:
                if config.gradient_clipping_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(ar_net.parameters(), config.gradient_clipping_norm)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            train_steps += 1

            if train_steps % config.log_every == 0:
                taken = time.time() - s
                avg_loss = running_loss / config.log_every
                ips = config.log_every / taken

                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "epoch": e,
                        "steps": train_steps,
                        "learning_rate": current_lr,
                    }
                )
                print(
                    f"Epoch {e}, step {i}/{len(train_loader)} (global step {train_steps}),",
                    f"Avg Loss: {avg_loss:.4f},",
                    f"Time Taken: {taken:.2f}s, ({ips:.2f} i/s)",
                )

                running_loss = 0.0
                s = time.time()

        val_loss = validate(flextok, ar_net, val_loader)
        wandb.log({"val_loss": val_loss, "epoch": e})

        snapshot_path = os.path.join(f"{config.checkpoint_path}/epoch_{e}.pt")
        try:
            torch.save(ar_net.state_dict(), snapshot_path)
        except Exception as e:
            print(f"Could not save model to {snapshot_path}", e)

    return validate(flextok, ar_net, test_loader)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    config_path = "c2i_config_ar49M.yaml"

    model_args, train_args = parse_config(config_path)
    flextok, ar_net = get_net(model_args, train_args)
    flextok.eval()
    optimizer, scheduler = get_optimizer(train_args, ar_net)
    train_loader, val_loader, test_loader = get_loaders(train_args)

    complete_config = {**asdict(model_args), **asdict(train_args)}
    wandb.init(project="flektok_autoregressive_c21", config=complete_config)

    os.makedirs(train_args.checkpoint_path, exist_ok=True)
    shutil.copy(config_path, train_args.checkpoint_path)

    test_loss = train(
        config=train_args,
        flextok=flextok,
        ar_net=ar_net,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    wandb.log({"test_loss": test_loss})
