import argparse
import pickle
from typing import List

import torch

from model_utils import build_model, get_device
from predata import analyspromption


def preprocess_texts(processor: analyspromption, texts: List[str]) -> List[str]:
    cleaned = processor.clean_text(texts)
    fenci = processor.fenci(cleaned)
    return [' '.join(x) for x in fenci]


def load_artifacts(model_path: str, vectorizer_path: str):
    device = get_device()
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    feature_dim = len(vectorizer.get_feature_names_out())
    model = build_model(feature_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vectorizer, device


def predict(texts: List[str], model, vectorizer, device) -> List[int]:
    processor = analyspromption()
    processed = preprocess_texts(processor, texts)
    vectors = vectorizer.transform(processed)
    x_tensor = torch.tensor(vectors.toarray(), dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(x_tensor)
        preds = torch.argmax(outputs, dim=1).tolist()
    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment for given texts")
    parser.add_argument("--text", nargs="*", help="Texts to classify. If omitted, reads --input-path instead.")
    parser.add_argument("--input-path", default="test/part.0", help="File containing text and optional labels")
    parser.add_argument("--model-path", default="./model/model_latest.pth", help="Path to trained model weights")
    parser.add_argument("--vectorizer-path", default="vectorizer.pkl", help="Path to saved CountVectorizer")
    return parser.parse_args()


def main():
    args = parse_args()
    model, vectorizer, device = load_artifacts(args.model_path, args.vectorizer_path)

    if args.text:
        predictions = predict(args.text, model, vectorizer, device)
        for text, label in zip(args.text, predictions):
            print(f"{text}\t{label}")
        return

    processor = analyspromption()
    texts, labels = processor.load_text(args.input_path)
    predictions = predict(texts, model, vectorizer, device)

    if labels:
        numeric_labels = [int(i) for i in labels]
        correct = sum(p == l for p, l in zip(predictions, numeric_labels))
        total = len(predictions)
        acc = correct / total if total else 0.0
        print(f"文件样本数: {total}")
        print(f"正确预测数: {correct}")
        print(f"准确率: {acc * 100:.2f}%")
    else:
        for text, label in zip(texts, predictions):
            print(f"{text}\t{label}")


if __name__ == "__main__":
    main()
