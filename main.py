from model import GPTModel
import torch
import tiktoken
from train_utils import create_dataloader_v1, train_model_simple, load_model
from visualization import plot_losses


CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 4,
    "num_epochs": 1,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "checkpoint_path": "model_checkpoint.pth",
}

torch.manual_seed(123)


def load_data():
    """Load train and validation data from files."""
    paths = ["data/train.txt", "data/validation.txt"]
    return [open(p, "r", encoding="utf-8").read() for p in paths]


def create_dataloaders(train_data, val_data):
    """Create train and validation DataLoaders."""

    def get_loader(data, batch_size, shuffle):
        return create_dataloader_v1(
            data,
            batch_size=batch_size,
            max_length=CONFIG["context_length"],
            stride=CONFIG["context_length"],
            drop_last=True,
            shuffle=shuffle,
            num_workers=0,
        )

    return get_loader(train_data, CONFIG["batch_size"], True), get_loader(
        val_data, 4, False
    )


def initialize_model(device):
    """Initialize model and optimizer, load checkpoint if available."""
    model = GPTModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    return load_model(model, optimizer, CONFIG["checkpoint_path"], device)


def main():
    """Main function to train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_data, val_data = load_data()
    train_loader, val_loader = create_dataloaders(train_data, val_data)
    model, optimizer, start_epoch, _ = initialize_model(device)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=CONFIG["num_epochs"],
        eval_freq=CONFIG["eval_freq"],
        eval_iter=CONFIG["eval_iter"],
        checkpoint_path=CONFIG["checkpoint_path"],
    )

    plot_losses(
        torch.linspace(0, CONFIG["num_epochs"], len(train_losses)),
        tokens_seen,
        train_losses,
        val_losses,
    )



if __name__ == "__main__":
    main()
