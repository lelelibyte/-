# evaluate.py
import torch
from torch import nn
import pickle
from predata import analyspromption

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型和 vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 构建与训练时一致的模型结构
feature_dim = len(vectorizer.get_feature_names_out())
model = nn.Sequential(
    nn.Linear(feature_dim, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
).to(device)

# 加载训练好的权重
state_dict = torch.load("./model/model90.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("模型参数加载完成")

# 读取验证集
path = r'test/part.0'
processor = analyspromption()
texts, labels = processor.load_text(path)
cleaned = processor.clean_text(texts)
fenci = processor.fenci(cleaned)

# 特征转换
X_test = vectorizer.transform([' '.join(x) for x in fenci])
X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
y_tensor = torch.tensor([int(i) for i in labels], dtype=torch.long).to(device)

# 预测并计算准确率
with torch.no_grad():
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == y_tensor).sum().item()
    total = len(y_tensor)
    acc = correct / total

print(f"验证集样本数: {total}")
print(f"正确预测数: {correct}")
print(f"验证集准确率: {acc*100:.2f}%")
