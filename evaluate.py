import argparse
import pickle

import torch

from model_utils import build_model, get_device
from predata import analyspromption


def evaluate(path: str, model_path: str = "./model/model_latest.pth"):
    device = get_device()
    print(f"Using device: {device}")

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    feature_dim = len(vectorizer.get_feature_names_out())
    model = build_model(feature_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型参数加载完成")

    processor = analyspromption()
    texts, labels = processor.load_text(path)
    cleaned = processor.clean_text(texts)
    fenci = processor.fenci(cleaned)

    X_test = vectorizer.transform([' '.join(x) for x in fenci])
    X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
    y_tensor = torch.tensor([int(i) for i in labels], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == y_tensor).sum().item()
        total = len(y_tensor)
        acc = correct / total if total else 0.0

    print(f"验证集样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"验证集准确率: {acc * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--test-path", default="test/part.0", help="Path to test data with labels")
    parser.add_argument("--model-path", default="./model/model_latest.pth", help="Path to trained model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args.test_path, args.model_path)


if __name__ == "__main__":
    main()
