import argparse
import pickle
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model_utils import build_model, get_device
from predata import analyspromption

MODEL_DIR = Path("./model")
VECTORIZER_PATH = Path("vectorizer.pkl")


def prepare_dataset(path: str, ngram: int = 2):
    """Load, preprocess, and vectorize text data."""

    processor = analyspromption()
    texts, labels = processor.load_text(path)
    cleaned = processor.clean_text(texts)
    tokens = processor.fenci(cleaned)
    vectors, vectorizer = processor.tezheng(tokens, ngram)

    x_tensor = torch.tensor(vectors.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor([int(i) for i in labels], dtype=torch.long)
    return x_tensor, y_tensor, vectorizer


def train_model(
    path: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    save_every: int = 10,
    ngram: int = 2,
):
    device = get_device()
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(exist_ok=True)
    x_tensor, y_tensor, vectorizer = prepare_dataset(path, ngram)
    dataloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = build_model(x_tensor.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
        if save_every and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), MODEL_DIR / f"model{epoch + 1}.pth")

    torch.save(model.state_dict(), MODEL_DIR / "model_latest.pth")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--train-path", default="train/part.0", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--ngram", type=int, default=2, help="N-gram size for CountVectorizer")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        path=args.train_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        ngram=args.ngram,
    )


if __name__ == "__main__":
    main()
