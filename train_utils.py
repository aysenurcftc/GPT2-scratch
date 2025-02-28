import tiktoken
import torch
from torch.utils.data import DataLoader
from dataset import GPTDataset
import os


CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")

def save_model(model, optimizer, epoch, global_step, path):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (Epoch: {epoch}, Step: {global_step})")


def load_model(model, optimizer, path, device="cpu"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        print(f"Checkpoint loaded: {path} (Epoch: {start_epoch}, Step: {global_step})")
    else:
        start_epoch, global_step = 0, 0
    return model, optimizer, start_epoch, global_step


def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        checkpoint_path,
):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_path)
    model, optimizer, start_epoch, global_step = load_model(
        model, optimizer, checkpoint_path, device
    )

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Ensure batches are converted to tensors if they are lists
            if isinstance(input_batch, list):
                input_batch = torch.stack(input_batch).to(device)
            else:
                input_batch = input_batch.to(device)

            if isinstance(target_batch, list):
                target_batch = torch.stack(target_batch).to(device)
            else:
                target_batch = target_batch.to(device)

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        save_model(model, optimizer, epoch + 1, global_step, checkpoint_path)

    return train_losses, val_losses, track_tokens_seen



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss



def calc_loss_batch(input_batch, target_batch, model, device):
    if isinstance(input_batch, list):
        input_batch = torch.stack(input_batch).to(device)

    if isinstance(target_batch, list):
        target_batch = torch.stack(target_batch).to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
